from agents.asset_requests import AssetRequestResolver
from models.schemas import AssetType


def test_stock_agent_routes_fund_and_follow_up_context():
    agent = AssetRequestResolver()

    fund_request = agent.resolve("分析 ETF 510300")
    assert fund_request.asset_type == AssetType.ETF
    assert fund_request.ticker == "510300"

    follow_up = agent.resolve(
        "为什么这样判断？",
        [{"role": "user", "content": "分析 ETF 510300"}],
    )
    assert follow_up.intent.value == "analyze"
    assert follow_up.asset_type == AssetType.ETF
    assert follow_up.ticker == "510300"


def test_request_resolver_does_not_verify_product_type_from_ticker_prefix():
    agent = AssetRequestResolver()

    assert agent.prepare("回测 510300").asset_type == AssetType.STOCK
    assert agent.prepare("分析 159915").asset_type == AssetType.STOCK
    assert agent.prepare("查看 166009").asset_type == AssetType.STOCK
    assert agent.prepare("分析股票 510300").asset_type == AssetType.STOCK


def test_backtest_takes_precedence_over_comparison_wording():
    agent = AssetRequestResolver()
    request = agent.prepare("给 510300 执行几个策略回测，并对比盈利情况")

    resolved, interaction = agent.resolve_intent(request)

    assert interaction is None
    assert resolved.intent.value == "backtest"


def test_stock_agent_routes_lof():
    request = AssetRequestResolver().resolve("查看 LOF 166009 历史走势")
    assert request.asset_type == AssetType.LOF
    assert request.intent.value == "history"


def test_stock_agent_uses_coarse_research_gate_and_gates_mutations():
    agent = AssetRequestResolver()

    request = agent.prepare("510300")
    resolved, interaction = agent.resolve_intent(request)
    assert resolved.intent.value == "analyze"
    assert resolved.mode.value == "financial_research"
    assert interaction is None

    clear_request = agent.prepare("分析 ETF 510300")
    resolved, interaction = agent.resolve_intent(clear_request)
    assert resolved.intent.value == "analyze"
    assert interaction is None

    recommendation = agent.prepare("分析 ETF 510300，给我买入建议")
    assert recommendation.allow_mutating_tools is False
    execution = agent.prepare("按这个方案提交模拟订单")
    assert execution.allow_mutating_tools is True
    resolved, interaction = agent.resolve_intent(execution)
    assert resolved.mode.value == "simulation_mutation"
    assert interaction is None
