"""Deterministic risk and trade-plan calculations exposed to the Agent."""

from __future__ import annotations

from typing import Any


def calculate_risk_metrics(
    current_price: float,
    *,
    entry_price: float | None = None,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.16,
    position_size_pct: float = 0.2,
    available_capital: float | None = None,
    max_loss_pct: float | None = None,
    min_lot: int = 100,
) -> dict[str, Any]:
    """Calculate bounded levels, risk/reward, and an optional share estimate."""
    price = float(current_price)
    if price <= 0:
        raise ValueError("current_price 必须大于 0")
    entry = float(entry_price if entry_price is not None else price)
    if entry <= 0:
        raise ValueError("entry_price 必须大于 0")
    stop_pct = min(max(float(stop_loss_pct), 0.0), 0.95)
    target_pct = max(float(take_profit_pct), 0.0)
    position_pct = min(max(float(position_size_pct), 0.0), 1.0)
    stop_loss = entry * (1 - stop_pct)
    take_profit = entry * (1 + target_pct)
    risk_per_share = max(entry - stop_loss, 0.0)
    reward_per_share = max(take_profit - entry, 0.0)
    risk_reward = reward_per_share / risk_per_share if risk_per_share else None

    result: dict[str, Any] = {
        "current_price": round(price, 6),
        "entry_price": round(entry, 6),
        "stop_loss": round(stop_loss, 6),
        "take_profit": round(take_profit, 6),
        "stop_loss_pct": round(stop_pct, 6),
        "take_profit_pct": round(target_pct, 6),
        "position_size_pct": round(position_pct, 6),
        "risk_per_share": round(risk_per_share, 6),
        "reward_per_share": round(reward_per_share, 6),
        "risk_reward_ratio": round(risk_reward, 6) if risk_reward is not None else None,
    }
    if available_capital is not None:
        capital = max(float(available_capital), 0.0)
        position_value = capital * position_pct
        shares = int(position_value / entry)
        result["available_capital"] = round(capital, 6)
        result["position_value"] = round(position_value, 6)
        result["shares"] = (shares // max(int(min_lot), 1)) * max(int(min_lot), 1)
        result["estimated_loss"] = round(result["shares"] * risk_per_share, 6)
        if max_loss_pct is not None:
            loss_limit = capital * min(max(float(max_loss_pct), 0.0), 1.0)
            result["max_loss_limit"] = round(loss_limit, 6)
            result["within_max_loss"] = result["estimated_loss"] <= loss_limit
    return result


def build_trade_plan(
    ticker: str,
    current_price: float,
    *,
    asset_type: str = "stock",
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.16,
    position_size_pct: float = 0.2,
    available_capital: float | None = None,
    max_loss_pct: float | None = None,
) -> dict[str, Any]:
    """Build a traceable, deterministic trade-plan payload."""
    metrics = calculate_risk_metrics(
        current_price,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        position_size_pct=position_size_pct,
        available_capital=available_capital,
        max_loss_pct=max_loss_pct,
    )
    as_of = "输入价格时点"
    metrics["ticker"] = ticker
    metrics["asset_type"] = asset_type
    metrics["price_evidence"] = [
        {
            "metric": "entry_price",
            "value": metrics["entry_price"],
            "source": "tool_input/current_price",
            "as_of": as_of,
            "calculation": "按输入当前价格作为计划入场价",
        },
        {
            "metric": "stop_loss",
            "value": metrics["stop_loss"],
            "source": "tool_input/stop_loss_pct",
            "as_of": as_of,
            "calculation": f"entry_price × (1 - {metrics['stop_loss_pct']})",
        },
        {
            "metric": "take_profit",
            "value": metrics["take_profit"],
            "source": "tool_input/take_profit_pct",
            "as_of": as_of,
            "calculation": f"entry_price × (1 + {metrics['take_profit_pct']})",
        },
    ]
    return metrics
