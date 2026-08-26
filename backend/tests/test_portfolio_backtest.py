import pandas as pd
import pytest
from strategy_helpers import compare_expression, strategy_mapping

from engine.backtester import run_pool_backtest


def _frame(start: float, days: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="D").strftime("%Y-%m-%d")
    close = [start + index * 0.1 for index in range(days)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": [value - 0.02 for value in close],
            "high": [value + 0.05 for value in close],
            "low": [value - 0.05 for value in close],
            "close": close,
            "volume": [1000] * days,
            "amount": [1000 * value for value in close],
        }
    )


@pytest.mark.asyncio
async def test_portfolio_backtest_allocates_and_rebalances_with_shared_cash(monkeypatch):
    import engine.backtester as backtester

    frames = {"510300": _frame(4.0), "159915": _frame(2.0), "512100": _frame(1.0), "515000": _frame(3.0)}

    async def fake_history(ticker, **_kwargs):
        return frames[ticker].copy()

    async def fake_context(*_args, **_kwargs):
        return {"historical": True}

    monkeypatch.setattr(backtester, "async_get_exchange_fund_history", fake_history)
    monkeypatch.setattr(backtester, "build_market_context", fake_context)

    result = await run_pool_backtest(
        tickers=list(frames),
        start_date="2026-01-01",
        end_date="2026-01-30",
        asset_type="etf",
        decision_interval=1,
        fill_time="same_close",
        strategy_spec=strategy_mapping(
            "portfolio_test",
            entry=compare_expression("return_pct", "gt", -1, 1),
        ),
        portfolio_spec={
            "allocation_method": "equal_weight",
            "rebalance_frequency": "weekly",
            "max_position_weight": 0.4,
            "max_positions": 3,
            "cash_reserve": 0.1,
        },
    )

    assert result["mode"] == "portfolio"
    assert result["portfolio_spec"]["max_positions"] == 3
    assert result["portfolio_history"]
    assert result["target_weights_history"]
    assert len(result["symbol_metrics"]) == 4
    for snapshot in result["portfolio_history"]:
        assert len(snapshot["positions"]) <= 3
        assert all(position["weight"] <= 0.4 + 1e-8 for position in snapshot["positions"])
    assert result["total_trades"] > 0
