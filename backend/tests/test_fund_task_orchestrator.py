import json
from collections import Counter
from pathlib import Path

import pytest

import application.fund_response as fund_response_module
from agents.stock_agent import StockAgent
from application.financial_task_planner import compile_financial_task_spec
from application.fund_instruments import resolve_fund_instruments
from application.fund_response import execute_direct_fund_task
from application.fund_task_compiler import compile_fund_task, uses_direct_fund_executor
from data.market_data_catalog import market_data_catalog
from domain.fund_calculations import calculate_from_question
from graph.research_planning import normalize_steps
from models.fund_task import FundTaskKind, InstrumentResolutionStatus, RiskPolicyAction, TaskOutcome
from tools import assets
from tools.registry import build_task_tools

QUESTIONS = Path(__file__).parent / "fixtures" / "fund_agent_questions.txt"


def _compile(index: int):
    question = QUESTIONS.read_text().splitlines()[index - 1]
    request = StockAgent().prepare(question)
    spec = compile_fund_task(
        question,
        tickers=request.tickers,
        asset_type=request.asset_type.value,
        mutation_requested=request.allow_mutating_tools,
    )
    assert spec is not None
    return question, spec


def test_all_122_fund_questions_compile_to_bounded_direct_tasks():
    agent = StockAgent()
    counts = Counter()
    for question in QUESTIONS.read_text().splitlines():
        request = agent.prepare(question)
        spec = compile_fund_task(question, tickers=request.tickers, asset_type=request.asset_type.value)
        assert spec is not None
        assert uses_direct_fund_executor(spec)
        counts[spec.task_kind] += 1

    assert sum(counts.values()) == 122
    assert counts[FundTaskKind.SAFETY_RESPONSE] == 10
    assert counts[FundTaskKind.CALCULATION] >= 8
    assert counts[FundTaskKind.SCENARIO_PLAN] >= 25
    assert counts[FundTaskKind.RULE_DESIGN] >= 20


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (1, FundTaskKind.EDUCATION),
        (11, FundTaskKind.RULE_DESIGN),
        (21, FundTaskKind.RULE_DESIGN),
        (31, FundTaskKind.RULE_DESIGN),
        (41, FundTaskKind.SCENARIO_PLAN),
        (51, FundTaskKind.SCENARIO_PLAN),
        (61, FundTaskKind.SCENARIO_PLAN),
        (71, FundTaskKind.CALCULATION),
        (81, FundTaskKind.RULE_DESIGN),
        (91, FundTaskKind.EDUCATION),
        (101, FundTaskKind.SAFETY_RESPONSE),
        (111, FundTaskKind.SCENARIO_PLAN),
    ],
)
def test_representative_questions_use_expected_task_semantics(index, expected):
    _, spec = _compile(index)
    assert spec.task_kind == expected
    if expected not in {FundTaskKind.INSTRUMENT_RESEARCH, FundTaskKind.UNIVERSE_RESEARCH}:
        assert spec.requires_live_data is False


def test_safety_boundary_questions_are_stopped_before_data_access():
    for index in range(101, 111):
        _, spec = _compile(index)
        assert spec.task_kind == FundTaskKind.SAFETY_RESPONSE
        assert spec.safety_decision.action in {RiskPolicyAction.REFUSE_GUARANTEE, RiskPolicyAction.BLOCK}
        assert spec.allowed_capabilities == []
        assert spec.requires_live_data is False


def test_negated_guarantee_is_not_treated_as_an_unsafe_request():
    _, spec = _compile(89)
    assert spec.task_kind != FundTaskKind.SAFETY_RESPONSE
    assert spec.safety_decision.action == RiskPolicyAction.ALLOW


def test_arbitrary_six_digit_amount_is_not_verified_as_a_fund():
    refs = resolve_fund_instruments(
        "可投资资金100000元，计划配置沪深300指数基金",
        ("100000",),
        asset_type="etf",
    )
    assert refs[0].status == InstrumentResolutionStatus.AMBIGUOUS


def test_explicit_etf_code_is_resolved_for_data_research():
    refs = resolve_fund_instruments("分析 ETF 510300 的最新走势", ("510300",), asset_type="etf")
    assert refs[0].status == InstrumentResolutionStatus.VERIFIED
    spec = compile_fund_task("分析 ETF 510300 的最新走势", tickers=("510300",), asset_type="etf")
    assert spec is not None
    assert spec.task_kind == FundTaskKind.INSTRUMENT_RESEARCH
    assert spec.requires_live_data is True


def test_dividend_catalog_requires_a_stock_domain_anchor():
    false_matches = [7, 23, 24, 57, 65, 72, 73, 77, 78, 79, 88, 101]
    questions = QUESTIONS.read_text().splitlines()
    for index in false_matches:
        assert market_data_catalog.search(questions[index - 1], asset_type="stock") == []
        assert compile_financial_task_spec({"message": questions[index - 1], "asset_type": "stock"}) is None

    canary = "根据A股近6年现金分红数据，筛选每年均有分红的股票，按每股分红排名"
    assert market_data_catalog.search(canary, asset_type="stock")[0].dataset_id == "cn_a_share_cash_dividends"
    assert compile_financial_task_spec({"message": canary, "asset_type": "stock"}) is not None


def test_deterministic_fund_fee_calculations():
    redemption = calculate_from_question("赎回费率为1.5%，买入金额为2万元，请计算赎回费")
    assert redemption is not None
    assert "300.00元" in redemption.result

    commission = calculate_from_question("ETF佣金费率为万分之3，最低5元，买入金额5万元，请计算买卖佣金")
    assert commission is not None
    assert "买卖合计30.00元" in commission.result

    share_class = calculate_from_question(
        "A类份额申购费为0.12%，C类份额年销售服务费为0.40%，预计持有三个月，请比较费用"
    )
    assert share_class is not None
    assert "C类费用较低" in share_class.result


def test_planner_normalizes_common_llm_schema_drift():
    steps = normalize_steps(
        {
            "steps": [
                {
                    "id": "risk",
                    "kind": "risk",
                    "title": "风险",
                    "success_criteria": "形成风险结论",
                    "max_attempts": 3,
                }
            ]
        },
        {"intent": "quote", "asset_type": "stock"},
        "quick",
    )
    assert steps[0]["success_criteria"] == ["形成风险结论"]
    assert steps[0]["max_attempts"] == 2


def test_task_tool_surface_is_allowlisted():
    _, education = _compile(1)
    assert build_task_tools(education, assets.get_realtime_quote, allow_mutating_tools=False) == []

    instrument = compile_fund_task("分析 ETF 510300 的最新走势", tickers=("510300",), asset_type="etf")
    assert instrument is not None
    names = {tool.name for tool in build_task_tools(instrument, assets.get_realtime_quote, allow_mutating_tools=False)}
    assert "get_realtime_quote" in names
    assert "get_fund_nav_history" in names
    assert "screen_assets" not in names
    assert "submit_simulation_order" not in names

    mutation = compile_fund_task(
        "按这个基金方案提交模拟订单",
        tickers=("510300",),
        asset_type="etf",
        mutation_requested=True,
    )
    assert mutation is not None
    mutation_names = {
        tool.name for tool in build_task_tools(mutation, assets.get_realtime_quote, allow_mutating_tools=True)
    }
    assert "submit_simulation_order" in mutation_names
    assert "get_realtime_quote" not in mutation_names


def test_etf_universe_task_does_not_expose_an_unavailable_structured_dataset_query():
    spec = compile_fund_task("筛选近6年每年都分红的ETF", asset_type="etf")
    assert spec is not None
    assert spec.task_kind == FundTaskKind.UNIVERSE_RESEARCH
    assert spec.subject.product_type == "etf"

    names = {tool.name for tool in build_task_tools(spec, assets.get_realtime_quote, allow_mutating_tools=False)}

    assert "search_market_data_catalog" in names
    assert "screen_assets" in names
    assert "query_market_data" not in names


@pytest.mark.asyncio
async def test_safety_response_is_deterministic_and_does_not_call_llm(monkeypatch):
    _, spec = _compile(101)

    class ForbiddenLLM:
        async def chat(self, *args, **kwargs):
            raise AssertionError("safety response must not call LLM")

    monkeypatch.setattr(fund_response_module, "get_llm_service", lambda: ForbiddenLLM())
    answer, acceptance = await execute_direct_fund_task(QUESTIONS.read_text().splitlines()[100], spec)

    assert "不能保证" in answer
    assert "满仓" in answer
    assert acceptance.outcome == TaskOutcome.REFUSED_WITH_ALTERNATIVE


@pytest.mark.asyncio
async def test_direct_agent_answer_has_business_acceptance_and_no_tool_events(monkeypatch):
    class FakeLLM:
        async def chat(self, *args, **kwargs):
            return (
                "结论：趋势继续走弱时应执行退出纪律。\n"
                "价格止损：亏损达到10%减仓。\n"
                "时间止损：5个交易日不能修复则退出。\n"
                "趋势止损：持续在关键均线下且均线下行时退出。"
            )

    monkeypatch.setattr(fund_response_module, "get_llm_service", lambda: FakeLLM())
    question = QUESTIONS.read_text().splitlines()[50]
    events = [event async for event in StockAgent().chat(StockAgent().prepare(question, task_id="fund-direct"))]

    assert not [event for event in events if event.get("type") == "tool"]
    assert events[0]["execution_version"] == 3
    outcome = next(event for event in events if event.get("type") == "task_outcome")
    assert outcome["acceptance"]["outcome"] == "satisfied"
    assert "价格止损" in events[-1]["text"]


def test_fixture_remains_valid_utf8_and_has_no_blank_questions():
    questions = QUESTIONS.read_text().splitlines()
    assert len(questions) == 122
    assert all(question.strip() for question in questions)
    json.dumps(questions, ensure_ascii=False)
