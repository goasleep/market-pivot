from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

from agents.asset_requests import AssetAgentRequest, AssetIntent, AssetRequestResolver
from application.fund_task_compiler import compile_fund_task
from data import open_fund_provider as provider_module
from data.open_fund_provider import AkShareOpenFundDataProvider, normalize_open_fund_category
from harness.bootstrap import build_default_catalog, build_default_validators, load_default_skills
from harness.compiler import harness_task_compiler
from harness.models import EvidenceRecord, SkillManifest, ToolDescriptor
from harness.open_fund_metrics import score_open_fund_candidate
from harness.registry import SkillRegistry, ToolCatalog
from harness.validators import ValidatorRegistry
from models.fund_task import FundDomain, FundProductCategory, FundTaskKind
from models.schemas import AssetType
from models.supervisor import ExecutionMode, TaskRoutingDecision
from tools.open_fund import run_open_fund_nav_backtest, screen_compare_open_funds


def _routing() -> TaskRoutingDecision:
    return TaskRoutingDecision(
        mode=ExecutionMode.EVIDENCE_RESEARCH,
        requires_tools=True,
        allow_research_plan=True,
    )


def _request(message: str, *, intent: AssetIntent = AssetIntent.ANALYZE) -> AssetAgentRequest:
    return AssetAgentRequest(
        message=message,
        history=[],
        intent=intent,
        tickers=(),
        asset_type=AssetType.OPEN_FUND,
        intent_confirmed=True,
    )


def test_product_type_inference_does_not_use_code_prefix_as_final_identity():
    assert AssetRequestResolver._infer_asset_type("分析 ETF 510300", []) == AssetType.ETF
    assert AssetRequestResolver._infer_asset_type("分析基金代码 510300", []) is None
    assert AssetRequestResolver._infer_asset_type("比较债券基金", []) == AssetType.OPEN_FUND


def test_ambiguous_fund_code_requests_product_type_clarification():
    resolver = AssetRequestResolver()
    request = resolver.prepare("分析基金代码 510300")

    _, clarification = resolver.resolve_intent(request)

    assert clarification is not None
    assert clarification["kind"] == "asset_type_clarification"
    assert {option["id"] for option in clarification["options"]} == {"etf", "lof", "open_fund"}


def test_generic_fund_requests_clarification_instead_of_defaulting_to_open_fund():
    resolver = AssetRequestResolver()
    request = resolver.prepare("帮我分析一下这个基金")

    _, clarification = resolver.resolve_intent(request)
    restored = resolver.request_from_payload(resolver.request_payload(request))

    assert request.asset_type_ambiguous is True
    assert AssetRequestResolver._infer_asset_type("帮我分析一下基金", []) is None
    assert resolver.request_payload(request)["asset_type"] is None
    assert restored.asset_type_ambiguous is True
    assert restored.asset_type_candidates == (AssetType.ETF, AssetType.LOF, AssetType.OPEN_FUND)
    assert clarification is not None
    assert {option["id"] for option in clarification["options"]} == {"etf", "lof", "open_fund"}


def test_generic_fund_concept_question_does_not_force_product_selection():
    resolver = AssetRequestResolver()
    request = resolver.prepare("基金是什么，有哪些类型？")

    resolved, clarification = resolver.resolve_intent(request)

    assert clarification is None
    assert resolved.intent == AssetIntent.HELP
    assert resolved.mode.value == "help"


def test_current_asset_type_overrides_history_and_follow_up_can_inherit_history():
    resolver = AssetRequestResolver()
    history = [{"role": "user", "content": "分析 ETF 510300"}]

    current = resolver.prepare("改为筛选债券基金", history)
    follow_up = resolver.prepare("这只基金的风险呢？", history)

    assert current.asset_type == AssetType.OPEN_FUND
    assert current.asset_type_ambiguous is False
    assert follow_up.asset_type == AssetType.ETF
    assert follow_up.asset_type_ambiguous is False


def test_multiple_explicit_product_types_require_scoping_and_api_override_wins():
    resolver = AssetRequestResolver()
    ambiguous = resolver.prepare("比较 ETF 和场外基金")
    _, clarification = resolver.resolve_intent(ambiguous)
    overridden = resolver.prepare("比较这些基金", asset_type="open_fund")
    _, override_interaction = resolver.resolve_intent(overridden)

    assert clarification is not None
    assert {option["id"] for option in clarification["options"]} == {"etf", "open_fund"}
    assert overridden.asset_type == AssetType.OPEN_FUND
    assert overridden.asset_type_explicit is True
    assert override_interaction is None


def test_only_asset_agent_is_public_and_legacy_alias_modules_are_removed():
    from agents.asset_agent import AssetAgent
    from agents.financial_harness_agent import FinancialHarnessAgent

    assert not Path("src/agents/fund_agent.py").exists()
    assert AssetAgent.__bases__ == (FinancialHarnessAgent,)
    assert FinancialHarnessAgent.__bases__ == (AssetRequestResolver,)


def test_open_fund_bond_screening_contract_uses_category_dimensions():
    spec = compile_fund_task("帮我筛选债券基金，哪只更稳", asset_type="open_fund")

    assert spec is not None
    assert spec.task_kind == FundTaskKind.UNIVERSE_RESEARCH
    assert spec.subject.fund_domain == FundDomain.OPEN_FUND
    assert spec.subject.asset_type == "open_fund"
    assert spec.subject.product_category == FundProductCategory.BOND
    assert spec.subject.pricing_basis == "nav"
    assert spec.selection_requirements is not None
    assert spec.selection_requirements.comparison_dimensions == [
        "drawdown",
        "nav_stability",
        "credit_exposure",
        "rate_sensitivity",
        "scale_liquidity",
        "fees",
    ]


def test_open_fund_harness_never_selects_exchange_or_stock_capabilities():
    contract = harness_task_compiler.compile(_request("帮我筛选债券基金，哪只更稳"), _routing())
    registry = load_default_skills(catalog=build_default_catalog())
    selected = registry.resolve_capabilities(
        contract.required_capabilities,
        asset_type=contract.asset_type,
        product_category=contract.product_category,
    )

    assert contract.fund_domain == "open_fund"
    assert contract.product_category == "bond"
    assert contract.pricing_basis == "nav"
    assert contract.required_capabilities == ("open_fund.screen_compare",)
    assert all(skill.domain in {"open_fund", "shared"} for skill in selected)
    assert all(not capability.startswith("exchange_fund.") for skill in selected for capability in skill.capabilities)
    assert "market.quote" in contract.forbidden_capabilities


def test_exchange_fund_screening_requires_formal_scoring_evidence():
    request = AssetAgentRequest(
        message="筛选白酒 ETF，短线首选哪个",
        history=[],
        intent=AssetIntent.ANALYZE,
        tickers=(),
        asset_type=AssetType.ETF,
        intent_confirmed=True,
    )
    contract = harness_task_compiler.compile(request, _routing())
    validator = build_default_validators().get("exchange_fund.screening")
    answer = (
        "候选 512690、515170、159736；对比评分后首选 512690，理由是流动性；"
        "备选 515170，排除 159736。截至 2026-08-26，来源：结构化 Provider。"
    )
    discovery = EvidenceRecord(
        capability_id="exchange_fund.screen_compare",
        tool_name="discover_exchange_fund_candidates",
        source_type="etf_candidate_universe",
        status="available",
        summary=json.dumps({"formal_ranking_eligible": False}),
    )
    formal = EvidenceRecord(
        capability_id="exchange_fund.screen_compare",
        tool_name="screen_compare_exchange_funds",
        source_type="exchange_fund.screen_compare",
        status="available",
        summary=json.dumps({"ranking_is_formal": True}),
    )

    incomplete = validator(contract, answer, (discovery,))
    complete = validator(contract, answer, (discovery, formal))

    assert incomplete.satisfied is False
    assert "formal_screening_evidence" in incomplete.missing
    assert complete.satisfied is True


def test_open_fund_screening_requires_formal_scoring_evidence():
    contract = harness_task_compiler.compile(_request("筛选债券基金，哪只更稳"), _routing())
    validator = build_default_validators().get("open_fund.screening")
    answer = (
        "候选 000001、000002、000003；对比评分后首选 000001，理由是回撤较低；"
        "备选 000002，排除 000003。截至 2026-08-26，来源：结构化 Provider。"
    )
    formal = EvidenceRecord(
        capability_id="open_fund.screen_compare",
        tool_name="screen_compare_open_funds",
        source_type="open_fund.screen_compare",
        status="available",
        summary=json.dumps({"ranking_is_formal": True}),
    )

    incomplete = validator(contract, answer, ())
    complete = validator(contract, answer, (formal,))

    assert incomplete.satisfied is False
    assert "formal_screening_evidence" in incomplete.missing
    assert complete.satisfied is True


def test_registry_exposes_only_new_fund_domain_ids():
    registry = load_default_skills(catalog=build_default_catalog())
    ids = {skill.id for skill in registry.list()}

    assert len({skill.id for skill in registry.list() if skill.domain == "open_fund"}) == 10
    assert not any(skill_id.startswith(("fund.", "etf.")) for skill_id in ids)
    assert {skill.domain for skill in registry.list()} <= {"stock", "exchange_fund", "open_fund", "shared"}


def test_open_fund_scoring_keeps_missing_dimensions_at_zero_without_renormalizing():
    equity = score_open_fund_candidate(
        "equity",
        {"trend": 100, "drawdown": 80, "volatility": 60, "scale_liquidity": 40, "fees": None, "exposure": 20},
    )
    bond = score_open_fund_candidate(
        "bond",
        {
            "drawdown": 100,
            "nav_stability": 80,
            "credit_exposure": 60,
            "rate_sensitivity": 40,
            "scale_liquidity": 20,
            "fees": 0,
        },
    )
    money = score_open_fund_candidate(
        "money_market",
        {"seven_day_yield": 100, "yield_stability": 80, "scale": 60, "redemption": 40, "fees": 20},
    )

    assert equity["score"] == 55.0
    assert equity["missing_dimensions"] == ["fees"]
    assert equity["renormalized"] is False
    assert bond["score"] == 61.0
    assert money["score"] == 67.0


def test_registry_rejects_open_fund_skill_using_exchange_tool():
    catalog = ToolCatalog()
    catalog.register_descriptor(
        ToolDescriptor(
            name="calculate_exchange_fund_premium_discount",
            capability_id="exchange_fund.premium_discount",
            asset_types=("etf", "lof"),
        )
    )
    skill = SkillManifest(
        id="open_fund.bad",
        version="2.0.0",
        title="错误场外能力",
        description="错误引用场内折溢价",
        domain="open_fund",
        asset_types=("open_fund",),
        capabilities=("open_fund.bad",),
        tools=("calculate_exchange_fund_premium_discount",),
    )

    with pytest.raises(ValueError, match="不能依赖场内"):
        SkillRegistry((skill,), catalog=catalog, validators=ValidatorRegistry())


@pytest.mark.asyncio
async def test_open_fund_provider_verifies_identity_and_marks_qdii_limited(monkeypatch):
    async def fake_catalog():
        return [
            {"基金代码": "000001", "基金简称": "华夏成长混合A", "基金类型": "混合型-偏股"},
            {"基金代码": "000002", "基金简称": "海外机会QDII", "基金类型": "QDII-股票型"},
        ]

    async def fake_daily():
        return [{"基金代码": "000001", "申购状态": "开放申购", "赎回状态": "开放赎回"}]

    async def fake_overview(_fund_code: str):
        return [
            {
                "基金代码": "000001（A类）、000011（C类）",
                "基金管理人": "华夏基金",
                "成立日期/规模": "2001年12月18日 / 32.368亿份",
                "净资产规模": "39.38亿元（截止至：2026年06月30日）",
                "管理费率": "1.20%（每年）",
                "托管费率": "0.20%（每年）",
                "销售服务费率": "---（每年）",
            }
        ]

    monkeypatch.setattr(provider_module, "fetch_open_fund_catalog", fake_catalog)
    monkeypatch.setattr(provider_module, "fetch_open_fund_daily", fake_daily)
    monkeypatch.setattr(provider_module, "fetch_open_fund_overview", fake_overview)
    provider = AkShareOpenFundDataProvider()

    profile = await provider.profile("000001")
    qdii = await provider.resolve_instrument("000002")

    assert profile.status == "available"
    assert profile.data is not None and profile.data.verified is True
    assert profile.data.product_category == "hybrid"
    assert profile.data.share_class == "A"
    assert profile.data.manager == "华夏基金"
    assert profile.data.inception_date == "2001-12-18"
    assert profile.data.size_cny == 3_938_000_000
    assert profile.data.management_fee_rate == 1.2
    assert profile.data.custody_fee_rate == 0.2
    assert profile.data.subscription_status == "开放申购"
    assert qdii.status == "limited"
    assert qdii.data is not None and qdii.data.product_category == "qdii"


@pytest.mark.asyncio
async def test_open_fund_profile_reports_source_conflict(monkeypatch):
    async def fake_catalog():
        return [{"基金代码": "000001", "基金简称": "基金A", "基金类型": "混合型"}]

    async def fake_daily():
        return [{"基金代码": "000001", "基金简称": "另一名称A", "申购状态": "开放"}]

    async def fake_overview(_fund_code: str):
        return [{"基金代码": "000001", "基金管理人": "测试基金公司"}]

    monkeypatch.setattr(provider_module, "fetch_open_fund_catalog", fake_catalog)
    monkeypatch.setattr(provider_module, "fetch_open_fund_daily", fake_daily)
    monkeypatch.setattr(provider_module, "fetch_open_fund_overview", fake_overview)

    result = await AkShareOpenFundDataProvider().profile("000001")

    assert result.status == "conflicting"
    assert result.errors[0]["code"] == "profile_source_conflict"


@pytest.mark.asyncio
async def test_open_fund_stale_exposure_is_limited(monkeypatch):
    stale = pd.DataFrame([{"季度": "2020年4季度", "股票代码": "600000", "占净值比例": 8.0}])
    empty = pd.DataFrame()
    monkeypatch.setattr(provider_module.ak, "fund_portfolio_hold_em", lambda **_kwargs: stale)
    monkeypatch.setattr(provider_module.ak, "fund_portfolio_bond_hold_em", lambda **_kwargs: empty)
    monkeypatch.setattr(provider_module.ak, "fund_portfolio_industry_allocation_em", lambda **_kwargs: empty)

    result = await AkShareOpenFundDataProvider().exposure("000001", "2020")

    assert result.status == "limited"
    assert result.as_of == "2020年4季度"
    assert any(error["code"] == "exposure_stale" for error in result.errors)


@pytest.mark.asyncio
async def test_money_fund_uses_yield_series_not_fixed_nav(monkeypatch):
    monkeypatch.setattr(
        provider_module.ak,
        "fund_money_fund_info_em",
        lambda symbol: pd.DataFrame(
            [
                {"净值日期": "2026-08-25", "每万份收益": 0.5123, "7日年化收益率": "1.85%"},
                {"净值日期": "2026-08-26", "每万份收益": 0.4988, "7日年化收益率": "1.82%"},
            ]
        ),
    )

    result = await AkShareOpenFundDataProvider().money_yield_history("000009")

    assert result.status == "available"
    assert result.as_of == "2026-08-26"
    assert result.data is not None
    assert result.data[0] == {
        "date": "2026-08-25",
        "yield_per_10k": 0.5123,
        "seven_day_annualized": 1.85,
    }
    assert "unit_nav" not in result.data[0]


@pytest.mark.asyncio
async def test_open_fund_cross_category_ranking_is_rejected():
    payload = json.loads(
        await screen_compare_open_funds.ainvoke(
            {
                "candidates": [
                    {
                        "fund_code": "000001",
                        "product_category": "equity",
                        "provider_verified": True,
                        "as_of": "2026-08-26",
                    },
                    {
                        "fund_code": "000002",
                        "product_category": "bond",
                        "provider_verified": True,
                        "as_of": "2026-08-26",
                    },
                ],
                "product_category": "equity",
            }
        )
    )

    assert payload["status"] == "data_unavailable"
    assert payload["data"]["ranking_is_formal"] is False
    assert payload["errors"][0]["code"] == "cross_category_ranking_forbidden"


@pytest.mark.asyncio
async def test_open_fund_nav_backtest_uses_next_nav_and_requires_fee():
    nav_points = [
        {"date": "2026-01-01", "unit_nav": 1.0, "cumulative_nav": 1.0},
        {"date": "2026-01-02", "unit_nav": 1.1, "cumulative_nav": 1.2},
        {"date": "2026-01-03", "unit_nav": 1.2, "cumulative_nav": 1.4},
    ]
    payload = json.loads(
        await run_open_fund_nav_backtest.ainvoke(
            {"nav_points": nav_points, "signal_dates": ["2026-01-01", "2026-01-02"], "fee_rate": 0.001}
        )
    )
    missing_fee = json.loads(
        await run_open_fund_nav_backtest.ainvoke(
            {"nav_points": nav_points, "signal_dates": ["2026-01-01", "2026-01-02"]}
        )
    )

    assert payload["status"] == "available"
    assert payload["data"]["execution_rule"] == "next_available_nav"
    assert payload["data"]["executions"][0]["execution_date"] == "2026-01-02"
    assert payload["data"]["nav_field"] == "cumulative_nav"
    assert payload["data"]["lookahead_safe"] is True
    assert missing_fee["status"] == "data_unavailable"
    assert missing_fee["errors"][0]["code"] == "fee_assumption_required"


@pytest.mark.asyncio
async def test_market_price_tools_return_not_applicable_for_open_fund():
    from tools.assets import get_historical_prices, get_realtime_quote

    quote = json.loads(await get_realtime_quote.ainvoke({"ticker": "000001", "asset_type": "open_fund"}))
    history = json.loads(await get_historical_prices.ainvoke({"ticker": "000001", "asset_type": "open_fund"}))

    assert quote["data_status"] == "not_applicable"
    assert quote["error"]["code"] == "market_quote_not_applicable"
    assert history["data_status"] == "not_applicable"
    assert history["error"]["code"] == "market_history_not_applicable"


@pytest.mark.asyncio
async def test_market_price_backtest_api_rejects_open_fund():
    from api.routers.backtest import BacktestRequest, create_backtest_job, run_backtest_api

    request = BacktestRequest(
        ticker="000001",
        asset_type=AssetType.OPEN_FUND,
        start_date="2026-01-01",
        end_date="2026-02-01",
    )

    with pytest.raises(HTTPException, match="open_fund.nav_backtest") as run_error:
        await run_backtest_api(request)
    with pytest.raises(HTTPException, match="open_fund.nav_backtest") as job_error:
        await create_backtest_job(request)

    assert run_error.value.status_code == 422
    assert job_error.value.status_code == 422


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("股票型", "equity"),
        ("混合型", "hybrid"),
        ("债券型", "bond"),
        ("货币型", "money_market"),
        ("指数增强", "enhanced_index"),
        ("QDII-股票型", "qdii"),
        ("FOF-混合型", "fof"),
    ],
)
def test_open_fund_category_normalization(raw: str, expected: str):
    assert normalize_open_fund_category(raw) == expected
