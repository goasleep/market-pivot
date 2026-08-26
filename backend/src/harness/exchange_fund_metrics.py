"""Deterministic ETF/LOF calculations and transparent screening scores."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any, Literal

from models.fund_data import LiquidityMetrics, PremiumDiscountMetrics, TrackingMetrics


def premium_discount(
    *,
    ticker: str,
    price: float | None,
    nav_or_iopv: float | None,
    price_at: str | None = None,
    nav_at: str | None = None,
    comparable: bool = True,
    reason: str = "",
) -> PremiumDiscountMetrics:
    valid = comparable and price is not None and nav_or_iopv is not None and nav_or_iopv > 0
    rate = (price / nav_or_iopv - 1) if valid else None
    return PremiumDiscountMetrics(
        ticker=ticker,
        price=price,
        nav_or_iopv=nav_or_iopv,
        premium_discount_rate=round(rate, 8) if rate is not None else None,
        price_at=price_at,
        nav_at=nav_at,
        comparable=valid,
        reason=reason if not valid else "",
    )


def tracking_metrics(
    *,
    ticker: str,
    fund_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    periods_per_year: int = 252,
    sample_start: str | None = None,
    sample_end: str | None = None,
    benchmark_source: str | None = None,
) -> TrackingMetrics:
    if len(fund_returns) != len(benchmark_returns) or len(fund_returns) < 2:
        return TrackingMetrics(ticker=ticker, observations=min(len(fund_returns), len(benchmark_returns)))
    differences = [float(left) - float(right) for left, right in zip(fund_returns, benchmark_returns, strict=True)]
    return TrackingMetrics(
        ticker=ticker,
        tracking_difference=round(statistics.fmean(differences) * periods_per_year, 8),
        tracking_error=round(statistics.stdev(differences) * math.sqrt(periods_per_year), 8),
        sample_start=sample_start,
        sample_end=sample_end,
        observations=len(differences),
        benchmark_source=benchmark_source,
    )


def liquidity_metrics(
    *,
    ticker: str,
    average_amount: float | None,
    bid: float | None = None,
    ask: float | None = None,
    turnover_rate: float | None = None,
    planned_amount: float | None = None,
) -> LiquidityMetrics:
    midpoint = (bid + ask) / 2 if bid and ask and bid > 0 and ask >= bid else None
    spread_bps = ((ask - bid) / midpoint * 10_000) if midpoint else None
    participation = (
        planned_amount / average_amount if planned_amount and average_amount and average_amount > 0 else None
    )
    if participation is None:
        risk: Literal["low", "medium", "high", "unknown"] = "unknown"
    elif participation <= 0.01:
        risk = "low"
    elif participation <= 0.05:
        risk = "medium"
    else:
        risk = "high"
    return LiquidityMetrics(
        ticker=ticker,
        average_amount=average_amount,
        turnover_rate=turnover_rate,
        spread_bps=round(spread_bps, 4) if spread_bps is not None else None,
        order_participation_rate=round(participation, 8) if participation is not None else None,
        impact_risk=risk,
    )


_WEIGHTS = {
    "short_term": {
        "liquidity": 0.35,
        "spread_premium": 0.20,
        "trend": 0.20,
        "drawdown": 0.15,
        "tracking_fee": 0.10,
    },
    "medium_term": {
        "tracking": 0.25,
        "trend": 0.20,
        "drawdown": 0.20,
        "liquidity": 0.15,
        "fee_size": 0.10,
        "exposure": 0.10,
    },
}


def score_candidate(
    dimensions: dict[str, float | None],
    *,
    horizon: Literal["short_term", "medium_term"],
) -> dict[str, Any]:
    weights = _WEIGHTS[horizon]
    missing = [name for name in weights if dimensions.get(name) is None]
    score = sum(max(0.0, min(100.0, float(dimensions.get(name) or 0.0))) * weight for name, weight in weights.items())
    return {"score": round(score, 4), "weights": weights, "missing_dimensions": missing, "renormalized": False}
