from agents.stock_agent import StockAgent
from models.schemas import AssetType


def test_stock_agent_routes_fund_and_follow_up_context():
    agent = StockAgent()

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


def test_stock_agent_routes_lof():
    request = StockAgent().resolve("查看 LOF 166009 历史走势")
    assert request.asset_type == AssetType.LOF
    assert request.intent.value == "history"


def test_stock_agent_clarifies_ambiguous_requests_and_gates_mutations():
    agent = StockAgent()

    request = agent.prepare("510300")
    resolved, interaction = agent.resolve_intent(request)
    assert resolved.intent.value == "analyze"
    assert interaction is not None
    assert {option["id"] for option in interaction["options"]} == {
        "quote",
        "history",
        "analyze",
        "backtest",
    }

    clear_request = agent.prepare("分析 ETF 510300")
    resolved, interaction = agent.resolve_intent(clear_request)
    assert resolved.intent.value == "analyze"
    assert interaction is None

    recommendation = agent.prepare("分析 ETF 510300，给我买入建议")
    assert recommendation.allow_mutating_tools is False
    execution = agent.prepare("按这个方案提交模拟订单")
    assert execution.allow_mutating_tools is True
