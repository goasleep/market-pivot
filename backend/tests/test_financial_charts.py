import pandas as pd
import pytest

from charts.financial import (
    ChartDataUnavailableError,
    calculate_market_risk_metrics,
    render_fund_structure_chart,
    render_risk_chart,
    render_technical_chart,
)


def _history(periods: int = 140) -> list[dict]:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=periods, freq="D").strftime("%Y-%m-%d"),
            "open": [10 + index * 0.02 for index in range(periods)],
            "high": [10.3 + index * 0.02 for index in range(periods)],
            "low": [9.8 + index * 0.02 for index in range(periods)],
            "close": [10.1 + index * 0.02 for index in range(periods)],
            "volume": [100_000 + index * 100 for index in range(periods)],
        }
    )
    return frame.to_dict(orient="records")


def test_financial_charts_render_pngs_with_bounded_metadata():
    technical = render_technical_chart("510300", "etf", _history())
    risk = render_risk_chart("510300", "etf", _history())

    assert technical.content.startswith(b"\x89PNG")
    assert risk.content.startswith(b"\x89PNG")
    assert technical.metadata["row_count"] == 120
    assert risk.metadata["metrics"] == calculate_market_risk_metrics(_history())


def test_fund_structure_chart_uses_only_overlapping_price_nav_dates():
    history = _history(12)
    nav = [
        {"date": row["date"], "unit_nav": row["close"] * 0.99}
        for row in history[-8:]
    ]

    rendered = render_fund_structure_chart("510300", "etf", history, nav)

    assert rendered.content.startswith(b"\x89PNG")
    assert rendered.metadata["row_count"] == 8
    assert rendered.metadata["latest_premium_pct"] == pytest.approx(1.0101, rel=1e-4)


def test_fund_structure_chart_rejects_insufficient_overlap():
    with pytest.raises(ChartDataUnavailableError, match="不足 5 日"):
        render_fund_structure_chart("510300", "etf", _history(8), [{"date": "2026-01-01", "unit_nav": 10}])
