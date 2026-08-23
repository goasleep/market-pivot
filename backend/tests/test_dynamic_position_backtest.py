import math

import pandas as pd
import pytest

from application.strategy_comparison import standard_strategy_suite
from engine.backtester import run_backtest


def _prepared_history():
    dates = pd.bdate_range("2023-01-02", periods=360).strftime("%Y-%m-%d")
    close = [10 + index * 0.01 + math.sin(index / 3) * 1.5 for index in range(len(dates))]
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [value * 1.001 for value in close],
            "high": [value * 1.01 for value in close],
            "low": [value * 0.99 for value in close],
            "close": close,
            "volume": [2_000_000] * len(dates),
        }
    )
    snapshot = {
        "sha256": "d" * 64,
        "actual_start_date": dates[0],
        "actual_end_date": dates[-1],
    }
    return frame, snapshot


@pytest.mark.asyncio
async def test_dynamic_exposure_uses_warmup_but_only_trades_in_evaluation_period():
    prepared = _prepared_history()
    evaluation_start = str(prepared[0].iloc[252]["date"])
    strategy = next(item for item in standard_strategy_suite("etf") if item.name == "volatility_target_15")

    result = await run_backtest(
        ticker="510300",
        start_date=str(prepared[0].iloc[0]["date"]),
        end_date=str(prepared[0].iloc[-1]["date"]),
        asset_type="etf",
        strategy_spec=strategy.model_dump(mode="json"),
        prepared_data=prepared,
        evaluation_start_date=evaluation_start,
    )

    assert result["equity_curve"][0]["date"] == evaluation_start
    assert result["equity_curve"][0]["value"] == 1_000_000
    assert result["price_curve"][0] == {
        "date": evaluation_start,
        "value": round(float(prepared[0].iloc[252]["close"]), 6),
    }
    assert result["price_curve"][-1]["date"] == str(prepared[0].iloc[-1]["date"])
    assert all(trade["date"] >= evaluation_start for trade in result["trades"])
    targets = [point["target_exposure"] for point in result["signal_curve"]]
    assert targets and all(0 <= value <= 0.95 for value in targets)
    assert any(0 < value < 0.95 for value in targets)


@pytest.mark.asyncio
async def test_buy_hold_does_not_enter_during_warmup():
    prepared = _prepared_history()
    evaluation_start = str(prepared[0].iloc[252]["date"])
    strategy = next(item for item in standard_strategy_suite("etf") if item.name == "buy_hold")

    result = await run_backtest(
        ticker="510300",
        start_date=str(prepared[0].iloc[0]["date"]),
        end_date=str(prepared[0].iloc[-1]["date"]),
        asset_type="etf",
        strategy_spec=strategy.model_dump(mode="json"),
        prepared_data=prepared,
        evaluation_start_date=evaluation_start,
    )

    assert result["equity_curve"][0]["value"] == 1_000_000
    assert result["trades"][0]["date"] > evaluation_start
