"""Build the trusted runtime catalogs used by each Harness request."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool

from application.fund_completion import validate_fund_response
from harness.models import ToolDescriptor, ValidatorResult
from harness.registry import SkillRegistry, ToolCatalog
from harness.validators import ValidatorRegistry
from models.fund_task import FundIntentSpec

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
PACKAGED_SKILLS_ROOT = _BACKEND_ROOT / "resources" / "agent_skills"
LOCAL_SKILLS_ROOT = _BACKEND_ROOT / "data" / "agent_skills"


def _descriptor(
    name: str,
    capability_id: str,
    *,
    asset_types: tuple[str, ...] = ("any",),
    data_types: tuple[str, ...] = (),
    read_only: bool = True,
    cost: str = "low",
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        capability_id=capability_id,
        asset_types=asset_types,
        data_types=data_types,
        read_only=read_only,
        cost=cost,
    )


DEFAULT_TOOL_DESCRIPTORS = (
    _descriptor("get_realtime_quote", "market.quote", asset_types=("stock", "etf", "lof"), data_types=("market_data",)),
    _descriptor(
        "get_historical_prices", "market.history", asset_types=("stock", "etf", "lof"), data_types=("market_data",)
    ),
    _descriptor(
        "get_exchange_fund_nav_history",
        "exchange_fund.nav",
        asset_types=("etf", "lof"),
        data_types=("fund_nav",),
    ),
    _descriptor("get_fundamentals", "instrument.profile", asset_types=("stock", "etf", "lof"), data_types=("profile",)),
    _descriptor("compare_quotes", "market.compare", asset_types=("stock", "etf", "lof"), data_types=("market_data",)),
    _descriptor("screen_assets", "market.screen", asset_types=("stock",), data_types=("market_data",), cost="medium"),
    _descriptor("search_web", "news.events", data_types=("news",), cost="medium"),
    _descriptor("fetch_web_content", "news.events", data_types=("web_content",), cost="medium"),
    _descriptor("search_market_data_catalog", "market.dataset", data_types=("dataset_catalog",)),
    _descriptor("query_market_data", "market.dataset", data_types=("market_dataset",), cost="high"),
    _descriptor("search_methodology", "methodology.search", data_types=("methodology",)),
    _descriptor(
        "compute_technical_indicators",
        "technical.indicators",
        asset_types=("stock", "etf", "lof"),
        data_types=("technical",),
    ),
    _descriptor("calculate_risk_metrics", "risk.metrics", asset_types=("stock", "etf", "lof"), data_types=("risk",)),
    _descriptor("build_trade_plan", "trade.plan", asset_types=("stock", "etf", "lof"), data_types=("trade_plan",)),
    _descriptor(
        "run_backtest", "backtest.execute", asset_types=("stock", "etf", "lof"), data_types=("backtest",), cost="high"
    ),
    _descriptor(
        "design_and_run_backtest",
        "backtest.execute",
        asset_types=("stock", "etf", "lof"),
        data_types=("backtest",),
        cost="high",
    ),
    _descriptor(
        "compare_strategy_backtests",
        "backtest.execute",
        asset_types=("stock", "etf", "lof"),
        data_types=("backtest",),
        cost="high",
    ),
    _descriptor(
        "design_and_run_sandbox_strategy",
        "strategy.sandbox_research",
        asset_types=("stock", "etf", "lof"),
        data_types=("backtest",),
        cost="high",
    ),
    _descriptor("list_trading_strategies", "strategy.list", data_types=("strategy",)),
    _descriptor("save_artifacts", "artifact.manage", data_types=("artifact",), read_only=False, cost="medium"),
    _descriptor("list_artifacts", "artifact.manage", data_types=("artifact",)),
    _descriptor("read_artifact", "artifact.manage", data_types=("artifact",)),
    _descriptor("create_chart_artifact", "artifact.manage", data_types=("artifact",), read_only=False, cost="medium"),
    _descriptor("get_simulation_portfolio", "simulation.read", data_types=("simulation",)),
    _descriptor("get_simulation_orders", "simulation.read", data_types=("simulation",)),
    _descriptor("list_simulation_accounts", "simulation.read", data_types=("simulation",)),
    _descriptor("list_strategy_deployments", "simulation.read", data_types=("simulation",)),
    _descriptor("create_simulation_account", "simulation.write", data_types=("simulation",), read_only=False),
    _descriptor("deploy_backtest_experiment", "simulation.write", data_types=("simulation",), read_only=False),
    _descriptor("set_strategy_deployment_status", "simulation.write", data_types=("simulation",), read_only=False),
    _descriptor("submit_simulation_order", "simulation.write", data_types=("simulation",), read_only=False),
    _descriptor("cancel_simulation_order", "simulation.write", data_types=("simulation",), read_only=False),
    _descriptor(
        "run_stock_comprehensive_analysis",
        "stock.comprehensive_analysis",
        asset_types=("stock",),
        data_types=("analysis",),
        cost="high",
    ),
    _descriptor("run_research_plan", "research.plan", data_types=("research",), cost="high"),
    _descriptor(
        "discover_exchange_fund_candidates",
        "exchange_fund.screen_compare",
        asset_types=("etf", "lof"),
        data_types=("fund_universe",),
        cost="medium",
    ),
    _descriptor(
        "get_exchange_fund_profile",
        "exchange_fund.profile",
        asset_types=("etf", "lof"),
        data_types=("fund_profile",),
    ),
    _descriptor(
        "get_exchange_fund_exposure",
        "exchange_fund.exposure",
        asset_types=("etf", "lof"),
        data_types=("exposure",),
    ),
    _descriptor(
        "calculate_exchange_fund_tracking_quality",
        "exchange_fund.tracking_quality",
        asset_types=("etf", "lof"),
        data_types=("tracking",),
    ),
    _descriptor(
        "calculate_exchange_fund_liquidity",
        "exchange_fund.liquidity_cost",
        asset_types=("etf", "lof"),
        data_types=("liquidity",),
    ),
    _descriptor(
        "calculate_exchange_fund_premium_discount",
        "exchange_fund.premium_discount",
        asset_types=("etf", "lof"),
        data_types=("premium_discount",),
    ),
    _descriptor(
        "calculate_exchange_fund_relative_strength",
        "exchange_fund.relative_strength",
        asset_types=("etf", "lof"),
        data_types=("relative_strength",),
    ),
    _descriptor(
        "screen_compare_exchange_funds",
        "exchange_fund.screen_compare",
        asset_types=("etf", "lof"),
        data_types=("screening",),
    ),
    _descriptor(
        "calculate_exchange_fund_portfolio_fit",
        "exchange_fund.portfolio_fit",
        asset_types=("etf", "lof"),
        data_types=("portfolio_risk",),
    ),
    _descriptor(
        "get_exchange_fund_event_risk",
        "exchange_fund.event_risk",
        asset_types=("etf", "lof"),
        data_types=("announcement",),
    ),
    _descriptor("get_open_fund_profile", "open_fund.profile", asset_types=("open_fund",), data_types=("fund_profile",)),
    _descriptor("get_open_fund_nav", "open_fund.nav", asset_types=("open_fund",), data_types=("fund_nav",)),
    _descriptor(
        "get_open_fund_money_yield",
        "open_fund.money_yield",
        asset_types=("open_fund",),
        data_types=("money_yield",),
    ),
    _descriptor("get_open_fund_fees", "open_fund.fees", asset_types=("open_fund",), data_types=("fund_fees",)),
    _descriptor(
        "get_open_fund_exposure",
        "open_fund.exposure",
        asset_types=("open_fund",),
        data_types=("exposure",),
    ),
    _descriptor(
        "discover_open_fund_candidates",
        "open_fund.screen_compare",
        asset_types=("open_fund",),
        data_types=("fund_universe",),
        cost="medium",
    ),
    _descriptor(
        "calculate_open_fund_relative_strength",
        "open_fund.relative_strength",
        asset_types=("open_fund",),
        data_types=("relative_strength",),
    ),
    _descriptor(
        "calculate_money_fund_stability",
        "open_fund.money_yield",
        asset_types=("open_fund",),
        data_types=("money_yield",),
    ),
    _descriptor(
        "screen_compare_open_funds",
        "open_fund.screen_compare",
        asset_types=("open_fund",),
        data_types=("screening",),
    ),
    _descriptor(
        "get_open_fund_event_risk",
        "open_fund.event_risk",
        asset_types=("open_fund",),
        data_types=("announcement",),
    ),
    _descriptor(
        "run_open_fund_nav_backtest",
        "open_fund.nav_backtest",
        asset_types=("open_fund",),
        data_types=("backtest",),
        cost="high",
    ),
)


def build_default_catalog(tools: Iterable[StructuredTool] = ()) -> ToolCatalog:
    catalog = ToolCatalog()
    for descriptor in DEFAULT_TOOL_DESCRIPTORS:
        catalog.register_descriptor(descriptor)
    catalog.bind(tools)
    return catalog


def _has_formal_ranking(evidence: Any, *, tool_name: str) -> bool:
    for record in evidence or ():
        record_tool = getattr(record, "tool_name", None)
        status = getattr(record, "status", None)
        summary = getattr(record, "summary", "")
        if isinstance(record, dict):
            record_tool = record.get("tool_name")
            status = record.get("status")
            summary = record.get("summary", "")
        if record_tool != tool_name or status != "available":
            continue
        try:
            payload = json.loads(summary) if isinstance(summary, str) else summary
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and payload.get("ranking_is_formal") is True:
            return True
    return False


def _exchange_fund_screening_validator(
    contract: Any,
    answer: str,
    evidence: Any = None,
) -> ValidatorResult:
    raw_spec = getattr(contract, "source_task_spec", None)
    if raw_spec is None and isinstance(contract, dict):
        raw_spec = contract.get("source_task_spec")
    if not isinstance(raw_spec, dict):
        return ValidatorResult(
            validator_id="exchange_fund.screening",
            satisfied=True,
            reason="任务没有基金筛选合同",
        )
    try:
        spec = FundIntentSpec.model_validate(raw_spec)
    except Exception:
        return ValidatorResult(
            validator_id="exchange_fund.screening",
            satisfied=False,
            missing=("有效的基金任务合同",),
            reason="source_task_spec 无法解析",
        )
    acceptance = validate_fund_response(spec, answer)
    missing = list(acceptance.missing)
    selection = spec.selection_requirements
    requires_formal_ranking = bool(selection and selection.selection_mode == "rank")
    has_formal_ranking = _has_formal_ranking(evidence, tool_name="screen_compare_exchange_funds")
    if requires_formal_ranking and not has_formal_ranking:
        missing.append("formal_screening_evidence")
    satisfied = acceptance.satisfied and (not requires_formal_ranking or has_formal_ranking)
    return ValidatorResult(
        validator_id="exchange_fund.screening",
        satisfied=satisfied,
        missing=tuple(dict.fromkeys(missing)),
        reason="场内基金正式筛选交付已满足" if satisfied else "场内基金筛选缺少正式评分证据或交付不完整",
    )


def _evidence_required_validator(
    contract: Any,
    _answer: str,
    evidence: Any = None,
) -> ValidatorResult:
    requirements = getattr(contract, "evidence_requirements", ())
    records = tuple(evidence or ())
    required = bool(requirements)
    satisfied = not required or bool(records)
    return ValidatorResult(
        validator_id="evidence.required",
        satisfied=satisfied,
        missing=() if satisfied else ("可验证工具证据",),
        reason="证据要求已满足" if satisfied else "任务要求外部证据但没有成功证据记录",
    )


_EXCHANGE_FUND_COMPREHENSIVE_CORE = (
    "market.quote",
    "market.history",
    "exchange_fund.nav",
    "exchange_fund.profile",
    "technical.indicators",
    "risk.metrics",
    "exchange_fund.liquidity_cost",
    "exchange_fund.relative_strength",
)


def _exchange_fund_comprehensive_validator(
    contract: Any,
    _answer: str,
    evidence: Any = None,
) -> ValidatorResult:
    required_capabilities = set(getattr(contract, "required_capabilities", ()))
    if "exchange_fund.comprehensive_analysis" not in required_capabilities:
        return ValidatorResult(
            validator_id="exchange_fund.comprehensive",
            satisfied=True,
            reason="任务未请求基金综合分析",
        )
    records = tuple(evidence or ())
    available = {record.capability_id for record in records if getattr(record, "status", None) == "available"}
    missing = [capability for capability in _EXCHANGE_FUND_COMPREHENSIVE_CORE if capability not in available]
    profile_records = [
        record
        for record in records
        if record.capability_id == "exchange_fund.profile" and record.status == "available"
    ]
    verified = False
    for record in profile_records:
        try:
            payload = json.loads(record.summary or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("verified") is True:
            verified = True
            break
    if not verified:
        missing.append("verified_fund_instrument")
    satisfied = not missing
    return ValidatorResult(
        validator_id="exchange_fund.comprehensive",
        satisfied=satisfied,
        missing=tuple(dict.fromkeys(missing)),
        reason="基金综合分析核心证据齐备" if satisfied else "基金综合分析缺少核心产品证据",
    )


def _open_fund_screening_validator(
    contract: Any,
    answer: str,
    evidence: Any = None,
) -> ValidatorResult:
    raw_spec = getattr(contract, "source_task_spec", None)
    if not isinstance(raw_spec, dict):
        return ValidatorResult(
            validator_id="open_fund.screening",
            satisfied=False,
            missing=("有效的场外基金任务合同",),
            reason="场外基金筛选缺少 source_task_spec",
        )
    spec = FundIntentSpec.model_validate(raw_spec)
    acceptance = validate_fund_response(spec, answer)
    category = str(getattr(contract, "product_category", "unknown"))
    category_ready = category not in {"", "unknown", "qdii", "fof"}
    missing = list(acceptance.missing)
    selection = spec.selection_requirements
    requires_formal_ranking = bool(selection and selection.selection_mode == "rank")
    has_formal_ranking = _has_formal_ranking(evidence, tool_name="screen_compare_open_funds")
    if not category_ready:
        missing.append("可正式筛选的单一场外基金类别")
    if requires_formal_ranking and not has_formal_ranking:
        missing.append("formal_screening_evidence")
    satisfied = acceptance.satisfied and category_ready and (not requires_formal_ranking or has_formal_ranking)
    return ValidatorResult(
        validator_id="open_fund.screening",
        satisfied=satisfied,
        missing=tuple(dict.fromkeys(missing)),
        reason="场外基金同类筛选交付已满足" if satisfied else "场外基金筛选缺少同类口径或必要交付",
    )


def _open_fund_comprehensive_validator(
    contract: Any,
    _answer: str,
    evidence: Any = None,
) -> ValidatorResult:
    records = tuple(evidence or ())
    available_tools = {record.tool_name for record in records if getattr(record, "status", None) == "available"}
    category = str(getattr(contract, "product_category", "unknown"))
    required_tools = {"get_open_fund_profile", "get_open_fund_fees"}
    if category == "money_market":
        required_tools.add("get_open_fund_money_yield")
    elif category not in {"qdii", "fof", "unknown"}:
        required_tools.add("get_open_fund_nav")
    missing = sorted(required_tools - available_tools)
    return ValidatorResult(
        validator_id="open_fund.comprehensive",
        satisfied=not missing,
        missing=tuple(missing),
        reason="场外基金核心证据齐备" if not missing else "场外基金综合分析缺少核心证据",
    )


def _open_fund_nav_backtest_validator(
    _contract: Any,
    _answer: str,
    evidence: Any = None,
) -> ValidatorResult:
    records = tuple(evidence or ())
    valid = any(
        record.tool_name == "run_open_fund_nav_backtest" and getattr(record, "status", None) == "available"
        for record in records
    )
    return ValidatorResult(
        validator_id="open_fund.nav_backtest",
        satisfied=valid,
        missing=() if valid else ("下一可用 NAV 执行的有效回测",),
        reason="场外基金 NAV 回测口径有效" if valid else "NAV 回测缺失、费用缺失或策略不适用",
    )


def build_default_validators() -> ValidatorRegistry:
    validators = ValidatorRegistry()
    validators.register("evidence.required", _evidence_required_validator)
    validators.register("exchange_fund.comprehensive", _exchange_fund_comprehensive_validator)
    validators.register("exchange_fund.screening", _exchange_fund_screening_validator)
    validators.register("open_fund.screening", _open_fund_screening_validator)
    validators.register("open_fund.comprehensive", _open_fund_comprehensive_validator)
    validators.register("open_fund.nav_backtest", _open_fund_nav_backtest_validator)
    return validators


def load_default_skills(
    *,
    catalog: ToolCatalog,
    validators: ValidatorRegistry | None = None,
) -> SkillRegistry:
    return SkillRegistry.load(
        (PACKAGED_SKILLS_ROOT, LOCAL_SKILLS_ROOT),
        catalog=catalog,
        validators=validators or build_default_validators(),
    )
