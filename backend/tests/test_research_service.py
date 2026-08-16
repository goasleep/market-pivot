import pytest

from application.research import ResearchService
from models.schemas import AssetType, Decision, TradeDecision


class FakeWorkflow:
    def __init__(self):
        self.states = []

    async def ainvoke(self, state, config=None):
        self.states.append(state)
        return {
            **state,
            "final_decision": TradeDecision(
                ticker=state["ticker"],
                asset_type=AssetType(state["asset_type"]),
                decision=Decision.HOLD,
            ),
        }


@pytest.mark.asyncio
async def test_research_service_builds_one_canonical_workflow_state():
    service = ResearchService()
    workflow = FakeWorkflow()

    result = await service.run(
        "510300",
        strategy="bull_trend",
        asset_type=AssetType.ETF,
        investor_context={"available_capital": 100_000},
        conversation_history=[{"role": "user", "content": "分析 ETF 510300"}],
        workflow_override=workflow,
    )

    assert result["final_decision"].asset_type == AssetType.ETF
    assert workflow.states == [
        {
            "ticker": "510300",
            "asset_type": "etf",
            "current_price": 0.0,
            "progress": [],
            "strategy_name": "bull_trend",
            "investor_context": {"available_capital": 100_000},
            "conversation_history": [{"role": "user", "content": "分析 ETF 510300"}],
        }
    ]
