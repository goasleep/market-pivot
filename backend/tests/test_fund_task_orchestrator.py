from pathlib import Path

import pytest

import application.fund_response as fund_response_module
from agents.asset_requests import AssetRequestResolver
from application.financial_task_planner import compile_financial_task_spec
from application.fund_instruments import resolve_fund_instruments
from application.fund_response import execute_direct_fund_task
from application.fund_task_compiler import compile_fund_task, uses_direct_fund_executor
from data.market_data_catalog import market_data_catalog
from domain.fund_calculations import calculate_from_question
from models.fund_task import FundTaskKind, InstrumentResolutionStatus

QUESTIONS = Path(__file__).parent / "fixtures" / "fund_agent_questions.txt"


def _compile(index: int):
    question = QUESTIONS.read_text().splitlines()[index - 1]
    request = AssetRequestResolver().prepare(question)
    spec = compile_fund_task(
        question,
        tickers=request.tickers,
        asset_type=request.asset_type.value,
        mutation_requested=request.allow_mutating_tools,
    )
    assert spec is not None
    return question, spec


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
        (101, FundTaskKind.EDUCATION),
        (111, FundTaskKind.SCENARIO_PLAN),
    ],
)
def test_representative_questions_use_expected_task_semantics(index, expected):
    _, spec = _compile(index)
    assert spec.task_kind == expected
    if expected not in {FundTaskKind.INSTRUMENT_RESEARCH, FundTaskKind.UNIVERSE_RESEARCH}:
        assert spec.requires_live_data is False


def test_former_safety_questions_keep_normal_task_semantics_without_preemptive_refusal():
    kinds = set()
    for index in range(101, 111):
        _, spec = _compile(index)
        kinds.add(spec.task_kind)
        assert spec.allowed_capabilities == []
        assert uses_direct_fund_executor(spec)
    assert FundTaskKind.EDUCATION in kinds
    assert FundTaskKind.SCENARIO_PLAN in kinds


def test_risk_language_does_not_add_a_safety_decision_to_the_task_contract():
    _, spec = _compile(108)
    assert spec.task_kind == FundTaskKind.SCENARIO_PLAN
    assert "safety_decision" not in spec.model_dump()


def test_arbitrary_six_digit_amount_is_not_verified_as_a_fund():
    refs = resolve_fund_instruments(
        "可投资资金100000元，计划配置沪深300指数基金",
        ("100000",),
        asset_type="etf",
    )
    assert refs[0].status == InstrumentResolutionStatus.AMBIGUOUS


def test_explicit_etf_code_is_resolved_for_data_research():
    refs = resolve_fund_instruments("分析 ETF 510300 的最新走势", ("510300",), asset_type="etf")
    assert refs[0].status == InstrumentResolutionStatus.CANDIDATE
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


@pytest.mark.asyncio
async def test_former_safety_response_runs_the_normal_scenario_executor(monkeypatch):
    question, spec = _compile(108)
    calls = []

    class FakeLLM:
        async def chat(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return (
                "结论：按用户设想建立模拟策略。\n"
                "条件：仅在既定信号成立时补仓。\n"
                "执行：分批记录模拟买入。\n"
                "风险上限：达到组合预算后停止新增。\n"
                "失效：趋势或基金机制发生变化时退出。"
            )

    monkeypatch.setattr(fund_response_module, "get_llm_service", lambda: FakeLLM())
    answer, acceptance = await execute_direct_fund_task(question, spec)

    assert calls
    assert "模拟策略" in answer
    assert acceptance.satisfied is True
