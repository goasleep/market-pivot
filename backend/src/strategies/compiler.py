"""Compile and evaluate the small, deterministic strategy DSL."""

from __future__ import annotations

from typing import Any

import pandas as pd

from models.schemas import AssetType, StrategyCondition, StrategySpec


def strategy_from_mapping(data: dict[str, Any], *, source: str | None = None) -> StrategySpec:
    """Convert YAML/LLM mappings into a validated strategy definition."""
    payload = dict(data)
    payload.setdefault("name", "generated_strategy")
    if source:
        payload["source"] = source
    conditions = payload.get("entry_conditions", [])
    exits = payload.get("exit_conditions", [])
    payload["entry_conditions"] = [
        item
        if isinstance(item, dict)
        else {"indicator": "close", "operator": "gt", "value": 0, "description": str(item)}
        for item in conditions
    ]
    payload["exit_conditions"] = [
        item
        if isinstance(item, dict)
        else {"indicator": "close", "operator": "lt", "value": 0, "description": str(item)}
        for item in exits
    ]
    if not payload.get("asset_types"):
        payload["asset_types"] = [AssetType.ETF, AssetType.LOF]
    return StrategySpec.model_validate(payload)


def _indicator(history: pd.DataFrame, name: str, window: int | None) -> float | None:
    if history.empty or "close" not in history:
        return None
    close = pd.to_numeric(history["close"], errors="coerce").dropna()
    if close.empty:
        return None
    if name == "close":
        return float(close.iloc[-1])
    if name == "return_pct":
        periods = window or 1
        if len(close) <= periods:
            return None
        return float((close.iloc[-1] / close.iloc[-periods - 1] - 1) * 100)
    if name.startswith("ma"):
        periods = window or int(name.removeprefix("ma"))
        return float(close.rolling(periods).mean().iloc[-1]) if len(close) >= periods else None
    if name == "volume_ratio":
        if "volume" not in history or len(history) < (window or 20):
            return None
        volume = pd.to_numeric(history["volume"], errors="coerce").dropna()
        periods = window or 20
        if len(volume) < periods:
            return None
        average = volume.tail(periods).mean()
        return float(volume.iloc[-1] / average) if average else None
    return None


def _matches(value: float | None, condition: StrategyCondition) -> bool:
    if value is None:
        return False
    target = condition.value
    if condition.operator == "between":
        if not isinstance(target, list) or len(target) != 2:
            return False
        return target[0] <= value <= target[1]
    target_value = float(target) if not isinstance(target, list) else float(target[0])
    return {
        "gt": value > target_value,
        "gte": value >= target_value,
        "lt": value < target_value,
        "lte": value <= target_value,
        "eq": value == target_value,
    }[condition.operator]


def evaluate_strategy(spec: StrategySpec, history: pd.DataFrame, *, asset_type: AssetType | str) -> dict[str, Any]:
    """Evaluate all conditions without any LLM call or future data."""
    asset_type = AssetType(asset_type)
    if asset_type not in spec.asset_types:
        return {"matched": False, "reason": "asset_type_not_supported", "conditions": []}
    def evaluate_conditions(conditions: list[StrategyCondition]) -> list[dict[str, Any]]:
        results = []
        for condition in conditions:
            value = _indicator(history, condition.indicator, condition.window)
            results.append(
                {
                    "condition": condition.model_dump(mode="json"),
                    "value": value,
                    "matched": _matches(value, condition),
                }
            )
        return results

    results = evaluate_conditions(spec.entry_conditions)
    exit_results = evaluate_conditions(spec.exit_conditions)
    matched = bool(results) and all(item["matched"] for item in results)
    exit_matched = bool(exit_results) and all(item["matched"] for item in exit_results)
    return {
        "matched": matched,
        "exit_matched": exit_matched,
        "reason": "all_entry_conditions_matched" if matched else "entry_conditions_not_matched",
        "conditions": results,
        "exit_conditions": exit_results,
    }
