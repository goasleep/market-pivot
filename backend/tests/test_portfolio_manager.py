import pytest

from agents.portfolio_manager import _buy_plan_has_evidence, decide
from models.schemas import AgentReport, AssetType, Decision, DecisionDashboard, MarketContext


def test_buy_plan_requires_traceable_ordered_price_levels():
    incomplete = DecisionDashboard.model_validate(
        {
            "battle_plan": {
                "entry_price": 3.86,
                "stop_loss": 3.72,
                "take_profit": 4.16,
                "entry_explanation": "等待 MA20 附近回调",
                "stop_loss_explanation": "跌破支撑位",
                "take_profit_explanation": "接近前期高点",
                "price_evidence": [{"metric": "MA20", "value": 3.84}],
            }
        }
    )
    assert not _buy_plan_has_evidence(incomplete)

    complete = DecisionDashboard.model_validate(
        {
            "battle_plan": {
                "entry_price": 3.86,
                "stop_loss": 3.72,
                "take_profit": 4.16,
                "entry_explanation": "等待 MA20 附近回调",
                "stop_loss_explanation": "跌破支撑位",
                "take_profit_explanation": "接近前期高点",
                "price_evidence": [
                    {
                        "metric": "MA20",
                        "value": 3.84,
                        "source": "market_context/history",
                        "as_of": "2026-08-15",
                    }
                ],
            }
        }
    )
    assert _buy_plan_has_evidence(complete)


@pytest.mark.asyncio
async def test_decision_downgrades_buy_without_price_evidence():
    class FakeLLM:
        async def chat_json(self, prompt: str, system: str) -> dict:
            return {
                "decision": "buy",
                "confidence": 0.9,
                "entry_price": 3.86,
                "target_price": 4.16,
                "stop_loss": 3.72,
                "position_size": 0.1,
                "reasoning": "测试买入",
                "dashboard": {
                    "battle_plan": {
                        "entry_price": 3.86,
                        "stop_loss": 3.72,
                        "take_profit": 4.16,
                    }
                },
            }

    decision = await decide(
        "510300",
        {"technical": AgentReport(agent_name="technical", reasoning="test")},
        current_price=3.9,
        asset_type=AssetType.ETF,
        market_context=MarketContext(
            ticker="510300",
            asset_type=AssetType.ETF,
            current_price=3.9,
            data_status={"source": "test"},
        ),
        llm=FakeLLM(),
    )

    assert decision.decision == Decision.HOLD
    assert "降级为观望" in decision.reasoning
