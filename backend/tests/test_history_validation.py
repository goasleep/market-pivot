import pandas as pd
import pytest

import data.history_validation as validation_module
from data.backtest_data import BacktestDataError
from data.history_validation import prepare_cross_validated_backtest_data


def _history(source: str, *, scale: float = 1.0, shock: float = 0.0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=280).strftime("%Y-%m-%d")
    close = [(10 + index * 0.02) * scale for index in range(len(dates))]
    if shock:
        close = [value * (1 + shock if index % 3 == 0 else 1) for index, value in enumerate(close)]
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": [value * 1.01 for value in close],
            "low": [value * 0.99 for value in close],
            "close": close,
            "volume": [1_000_000] * len(dates),
        }
    )
    frame.attrs["source_metadata"] = {"source_id": source, "source_name": source}
    return frame


@pytest.mark.asyncio
async def test_cross_validation_compares_returns_and_accepts_different_price_scales(monkeypatch):
    left = _history("eastmoney")
    right = _history("tencent", scale=2.5)

    async def fake_fetch(*_args, **_kwargs):
        return [("primary", left), ("tencent", right)]

    monkeypatch.setattr(validation_module, "_fetch_candidates", fake_fetch)
    frame, snapshot, report = await prepare_cross_validated_backtest_data(
        ticker="510300",
        start_date="2024-01-02",
        end_date="2025-01-27",
        asset_type="etf",
    )

    assert report["status"] == "verified"
    assert report["comparison"]["median_abs_return_diff"] == 0
    assert report["selected_source"] == "eastmoney"
    assert snapshot["sha256"]
    assert len(frame) == 280


@pytest.mark.asyncio
async def test_cross_validation_stops_close_score_severe_conflict(monkeypatch):
    left = _history("eastmoney")
    right = _history("tencent", shock=0.08)

    async def fake_fetch(*_args, **_kwargs):
        return [("primary", left), ("tencent", right)]

    monkeypatch.setattr(validation_module, "_fetch_candidates", fake_fetch)
    with pytest.raises(BacktestDataError, match="严重冲突"):
        await prepare_cross_validated_backtest_data(
            ticker="510300",
            start_date="2024-01-02",
            end_date="2025-01-27",
            asset_type="etf",
        )


@pytest.mark.asyncio
async def test_cross_validation_continues_unverified_when_one_source_fails(monkeypatch):
    left = _history("eastmoney")

    async def fake_fetch(*_args, **_kwargs):
        return [("primary", left), ("tencent", RuntimeError("source unavailable"))]

    monkeypatch.setattr(validation_module, "_fetch_candidates", fake_fetch)
    _, _, report = await prepare_cross_validated_backtest_data(
        ticker="510300",
        start_date="2024-01-02",
        end_date="2025-01-27",
        asset_type="etf",
    )

    assert report["status"] == "unverified"
    assert any(not candidate["eligible"] for candidate in report["candidates"])
