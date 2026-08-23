import pandas as pd
from strategy_helpers import compare_expression, strategy_mapping

from agents.asset_agent import AssetAgent
from engine.strategy_runtime import evaluate_strategy_intent
from engine.trading_engine import TradingEngine
from models.schemas import AssetType, Decision, SimulationAccountConfig, TradeDecision
from strategies.compiler import strategy_from_mapping


def test_stock_and_fund_share_one_trade_plan_shape():
    decision = TradeDecision(
        ticker="510300",
        asset_type=AssetType.ETF,
        decision=Decision.BUY,
        entry_price=4.0,
        target_price=4.4,
        stop_loss=3.8,
        position_size=0.2,
    )

    assert decision.plan.entry_price == 4.0
    assert decision.plan.take_profit == 4.4
    assert decision.target_price == 4.4
    assert decision.position_size == 0.2
    assert "plan" in decision.model_dump()


def test_etf_uses_fund_trading_costs_and_auto_exit_levels():
    config = SimulationAccountConfig(initial_cash=100_000, asset_type=AssetType.ETF)
    rules = config.effective_trading_rules(AssetType.ETF)
    assert rules.stamp_tax_rate == 0
    assert rules.transfer_fee_rate == 0

    engine = TradingEngine(100_000, config, current_date="2026-01-02")
    trade = engine.buy("510300", 1_000, 4.0, "2026-01-02", take_profit=4.4)
    assert trade is not None
    engine.set_date("2026-01-03")
    engine.update_prices({"510300": 4.4})

    assert not engine.portfolio.positions
    assert engine.portfolio.trades[-1].action == Decision.SELL
    assert engine.portfolio.trades[-1].tax == 0


def test_llm_strategy_spec_compiles_into_deterministic_conditions():
    spec = strategy_from_mapping(
        strategy_mapping(
            "etf_momentum",
            entry=compare_expression("return_pct", "gt", 0, 5),
            stop_loss_pct=0.05,
            take_profit_pct=0.1,
        ),
        source="llm",
    )
    history = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"],
            "close": [1, 1, 1, 1, 1, 1.1],
            "volume": [100] * 6,
        }
    )
    intent, _state, result = evaluate_strategy_intent(spec, history, asset_type=AssetType.ETF)
    assert intent.decision == Decision.BUY
    assert result["expression_traces"]["entry"]["left"] > 0


def test_asset_agent_is_the_canonical_chat_alias():
    assert AssetAgent.__name__ == "AssetAgent"
