"""Category-aware deterministic metrics for off-exchange open funds."""

from __future__ import annotations

import math
import statistics
from typing import Any

_WEIGHTS: dict[str, dict[str, float]] = {
    "equity": {
        "trend": 0.20,
        "drawdown": 0.20,
        "volatility": 0.15,
        "scale_liquidity": 0.15,
        "fees": 0.10,
        "exposure": 0.20,
    },
    "hybrid": {
        "trend": 0.20,
        "drawdown": 0.20,
        "volatility": 0.15,
        "scale_liquidity": 0.15,
        "fees": 0.10,
        "exposure": 0.20,
    },
    "index": {
        "trend": 0.20,
        "drawdown": 0.20,
        "volatility": 0.15,
        "scale_liquidity": 0.15,
        "fees": 0.10,
        "exposure": 0.20,
    },
    "enhanced_index": {
        "trend": 0.20,
        "drawdown": 0.20,
        "volatility": 0.15,
        "scale_liquidity": 0.15,
        "fees": 0.10,
        "exposure": 0.20,
    },
    "bond": {
        "drawdown": 0.25,
        "nav_stability": 0.20,
        "credit_exposure": 0.20,
        "rate_sensitivity": 0.15,
        "scale_liquidity": 0.10,
        "fees": 0.10,
    },
    "money_market": {
        "seven_day_yield": 0.25,
        "yield_stability": 0.25,
        "scale": 0.20,
        "redemption": 0.20,
        "fees": 0.10,
    },
}


def score_open_fund_candidate(product_category: str, dimensions: dict[str, float | None]) -> dict[str, Any]:
    if product_category not in _WEIGHTS:
        raise ValueError(f"不支持的场外基金评分类别: {product_category}")
    weights = _WEIGHTS[product_category]
    missing = [name for name in weights if dimensions.get(name) is None]
    score = sum(max(0.0, min(100.0, float(dimensions.get(name) or 0.0))) * weight for name, weight in weights.items())
    return {
        "score": round(score, 4),
        "weights": weights,
        "missing_dimensions": missing,
        "renormalized": False,
    }


def nav_performance(values: list[float]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and float(value) > 0]
    if len(clean) < 2:
        return {"return": None, "volatility": None, "max_drawdown": None}
    returns = [clean[index] / clean[index - 1] - 1 for index in range(1, len(clean))]
    peak = clean[0]
    max_drawdown = 0.0
    for value in clean:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1)
    volatility = statistics.stdev(returns) * math.sqrt(252) if len(returns) >= 2 else 0.0
    return {
        "return": round(clean[-1] / clean[0] - 1, 8),
        "volatility": round(volatility, 8),
        "max_drawdown": round(max_drawdown, 8),
    }
