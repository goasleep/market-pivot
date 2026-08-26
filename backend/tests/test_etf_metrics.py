import json

import pytest

from harness.exchange_fund_metrics import liquidity_metrics, premium_discount, score_candidate, tracking_metrics
from tools.exchange_fund import calculate_exchange_fund_premium_discount, screen_compare_exchange_funds


def test_tracking_metrics_exact_fixture():
    result = tracking_metrics(
        ticker="510300",
        fund_returns=[0.01, -0.02, 0.03],
        benchmark_returns=[0.009, -0.018, 0.028],
        periods_per_year=252,
    )
    assert result.tracking_difference == pytest.approx(0.084)
    assert result.tracking_error == pytest.approx(0.03304542)


def test_premium_discount_and_liquidity_exact_fixture():
    premium = premium_discount(ticker="510300", price=1.01, nav_or_iopv=1.0)
    liquidity = liquidity_metrics(
        ticker="510300",
        average_amount=10_000_000,
        bid=0.999,
        ask=1.001,
        planned_amount=100_000,
    )
    assert premium.premium_discount_rate == 0.01
    assert liquidity.spread_bps == 20.0
    assert liquidity.order_participation_rate == 0.01
    assert liquidity.impact_risk == "low"


def test_missing_dimensions_are_zero_not_renormalized():
    result = score_candidate(
        {"liquidity": 100, "spread_premium": None, "trend": 50, "drawdown": 50, "tracking_fee": 50},
        horizon="short_term",
    )
    assert result["score"] == 57.5
    assert result["missing_dimensions"] == ["spread_premium"]
    assert result["renormalized"] is False


@pytest.mark.asyncio
async def test_qdii_stale_nav_never_outputs_precise_premium():
    payload = json.loads(
        await calculate_exchange_fund_premium_discount.ainvoke(
            {
                "ticker": "513100",
                "price": 1.2,
                "nav_or_iopv": 1.0,
                "price_at": "2026-08-26",
                "nav_at": "2026-08-25",
                "is_qdii": True,
                "markets_comparable": True,
            }
        )
    )
    assert payload["status"] == "limited"
    assert payload["data"]["premium_discount_rate"] is None


@pytest.mark.asyncio
async def test_screening_refuses_formal_ranking_without_verification_or_as_of():
    payload = json.loads(
        await screen_compare_exchange_funds.ainvoke(
            {
                "candidates": [
                    {
                        "ticker": "510300",
                        "verified": True,
                        "liquidity": 100,
                        "as_of": "2026-08-26",
                        "dimensions": {"liquidity": 90},
                    },
                    {
                        "ticker": "159919",
                        "verified": False,
                        "liquidity": 90,
                        "as_of": "2026-08-26",
                        "dimensions": {"liquidity": 80},
                    },
                ]
            }
        )
    )
    assert payload["status"] == "limited"
    assert payload["data"]["ranking"] == []
    assert payload["data"]["ranking_is_formal"] is False


@pytest.mark.asyncio
async def test_screening_accepts_provider_verified_snapshot_liquidity():
    payload = json.loads(
        await screen_compare_exchange_funds.ainvoke(
            {
                "candidates": [
                    {
                        "ticker": "512690",
                        "provider_verified": True,
                        "amount": 200_000_000,
                        "as_of": "2026-08-26T14:00:00+08:00",
                        "dimensions": {"liquidity": 100, "trend": 50},
                    },
                    {
                        "ticker": "515170",
                        "provider_verified": True,
                        "amount": 40_000_000,
                        "as_of": "2026-08-26T14:00:00+08:00",
                        "dimensions": {"liquidity": 60, "trend": 50},
                    },
                ]
            }
        )
    )

    assert payload["status"] == "available"
    assert payload["data"]["ranking_is_formal"] is True
    assert [item["ticker"] for item in payload["data"]["ranking"]] == ["512690", "515170"]
