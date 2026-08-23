"""Compile and evaluate the small, deterministic strategy DSL."""

from __future__ import annotations

from typing import Any

import pandas as pd

from models.schemas import (
    AssetType,
    IndicatorSpec,
    StrategyExpression,
    StrategyOperand,
    StrategySpec,
)
from strategies.plugin_registry import get_strategy_plugin

SUPPORTED_INDICATORS = frozenset(
    {
        "close",
        "return_pct",
        "ma",
        "ema",
        "price_vs_ma_pct",
        "ma_spread_pct",
        "bollinger_zscore",
        "rolling_breakout_pct",
        "volume_ratio",
        "rsi",
        "atr",
        "volatility",
    }
)


def available_indicators() -> list[dict[str, Any]]:
    """Return the public indicator contract exposed to the strategy Agent."""
    return [
        {"name": "close", "source": "close", "requires": ["close"], "default_window": None},
        {"name": "return_pct", "source": "close", "requires": ["close"], "default_window": 1},
        {"name": "ma", "source": "close", "requires": ["close"], "default_window": 20},
        {"name": "ema", "source": "close", "requires": ["close"], "default_window": 20},
        {
            "name": "price_vs_ma_pct",
            "source": "close",
            "requires": ["close"],
            "default_window": 20,
        },
        {
            "name": "ma_spread_pct",
            "source": "close",
            "requires": ["close"],
            "default_window": None,
            "params": {"fast_window": 5, "slow_window": 20},
        },
        {
            "name": "bollinger_zscore",
            "source": "close",
            "requires": ["close"],
            "default_window": 20,
        },
        {
            "name": "rolling_breakout_pct",
            "source": "close",
            "requires": ["close"],
            "default_window": 20,
        },
        {"name": "volume_ratio", "source": "volume", "requires": ["volume"], "default_window": 20},
        {"name": "rsi", "source": "close", "requires": ["close"], "default_window": 14},
        {"name": "atr", "source": "ohlcv", "requires": ["high", "low", "close"], "default_window": 14},
        {
            "name": "volatility",
            "source": "close",
            "requires": ["close"],
            "default_window": 20,
            "annualized": True,
        },
    ]


def _normalize_ratio(value: Any) -> Any:
    """Accept both fractional ratios and human-friendly percentage values."""
    if value is None or isinstance(value, bool):
        return value
    try:
        text = str(value).strip().lower().replace("percent", "").replace("百分比", "")
        number = float(text.rstrip("%"))
    except (TypeError, ValueError):
        return value
    if abs(number) > 1 and abs(number) <= 100:
        return number / 100
    return number


def _normalize_indicator_specs(value: Any) -> list[dict[str, Any]]:
    """Normalize common LLM indicator shapes into the executable list contract."""
    if value is None:
        return []
    if isinstance(value, dict):
        if any(key in value for key in ("name", "indicator", "type", "metric")):
            items: list[Any] = [value]
        else:
            items = [{**(item if isinstance(item, dict) else {}), "name": key} for key, item in value.items()]
    elif isinstance(value, list):
        items = value
    else:
        items = [value]

    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            item = {"name": item}
        elif not isinstance(item, dict):
            item = {"name": str(item)}
        else:
            item = dict(item)
        item.setdefault("name", item.get("indicator") or item.get("type") or item.get("metric") or "")
        item.pop("indicator", None)
        item.pop("type", None)
        item.pop("metric", None)
        if "window" not in item:
            item["window"] = item.get("period") or item.get("length") or item.get("lookback")
        if item.get("source") == "volume":
            item["source"] = "ohlcv"
        role_aliases = {
            "trend": "filter",
            "momentum": "confirmation",
            "signal": "entry",
            "entry_signal": "entry",
            "exit_signal": "exit",
            "volume": "filter",
            "risk_management": "risk",
            "stop_loss": "risk",
        }
        role = item.get("role")
        if isinstance(role, str):
            item["role"] = role_aliases.get(role.strip().lower(), role.strip().lower())
        if not isinstance(item.get("params"), dict):
            item["params"] = {}
        normalized.append(item)
    return normalized


def strategy_from_mapping(data: dict[str, Any], *, source: str | None = None) -> StrategySpec:
    """Convert YAML/LLM mappings into a validated strategy definition.

    The Agent contract is strict, but this boundary also accepts a few
    unambiguous LLM variants (percentage values and keyed indicator maps) so a
    harmless formatting difference cannot prevent an otherwise safe strategy
    from reaching the deterministic validator.
    """
    payload = dict(data)
    payload.setdefault("name", "generated_strategy")
    if source:
        payload["source"] = source
    if isinstance(payload.get("asset_types"), str):
        payload["asset_types"] = [payload["asset_types"]]
    payload["indicator_specs"] = _normalize_indicator_specs(payload.get("indicator_specs"))
    for field in ("stop_loss_pct", "take_profit_pct"):
        if field in payload:
            payload[field] = _normalize_ratio(payload[field])
    if not payload.get("asset_types"):
        payload["asset_types"] = [AssetType.ETF, AssetType.LOF]
    spec = StrategySpec.model_validate(payload)
    errors = validate_strategy_spec(spec)
    if errors:
        raise ValueError("策略包含不受支持的指标或组件: " + "; ".join(errors))
    return spec


def _indicator_definition(specs: list[IndicatorSpec], name: str) -> tuple[str, int | None, dict[str, Any]]:
    normalized = name.strip().lower()
    for spec in specs:
        if normalized in {spec.name.lower(), (spec.alias or "").lower()}:
            configured_window = spec.window
            raw_window = spec.params.get("window")
            if configured_window is None and raw_window is not None:
                try:
                    configured_window = int(raw_window)
                except (TypeError, ValueError):
                    configured_window = None
            return spec.name.lower(), configured_window, dict(spec.params)
    if normalized.startswith(("ma", "ema")) and normalized[2:].isdigit():
        return normalized[:2], int(normalized[2:]), {}
    return normalized, None, {}


def _indicator(
    history: pd.DataFrame,
    name: str,
    window: int | None,
    indicator_specs: list[IndicatorSpec] | None = None,
) -> float | None:
    canonical, configured_window, params = _indicator_definition(indicator_specs or [], name)
    window = window or configured_window
    if history.empty or "close" not in history:
        return None
    close = pd.to_numeric(history["close"], errors="coerce").dropna()
    if close.empty:
        return None
    if canonical == "close":
        return float(close.iloc[-1])
    if canonical == "return_pct":
        periods = window or 1
        if len(close) <= periods:
            return None
        return float((close.iloc[-1] / close.iloc[-periods - 1] - 1) * 100)
    if canonical == "ma":
        periods = window or 20
        return float(close.rolling(periods).mean().iloc[-1]) if len(close) >= periods else None
    if canonical == "ema":
        periods = window or 20
        return float(close.ewm(span=periods, adjust=False).mean().iloc[-1]) if len(close) >= periods else None
    if canonical == "price_vs_ma_pct":
        periods = window or 20
        if len(close) < periods:
            return None
        average = close.rolling(periods).mean().iloc[-1]
        return float((close.iloc[-1] / average - 1) * 100) if average else None
    if canonical == "ma_spread_pct":
        fast = int(params.get("fast_window", 5))
        slow = int(params.get("slow_window", window or 20))
        if fast < 1 or slow <= fast or len(close) < slow:
            return None
        fast_value = close.rolling(fast).mean().iloc[-1]
        slow_value = close.rolling(slow).mean().iloc[-1]
        return float((fast_value / slow_value - 1) * 100) if slow_value else None
    if canonical == "bollinger_zscore":
        periods = window or 20
        if len(close) < periods:
            return None
        sample = close.tail(periods)
        deviation = sample.std(ddof=0)
        return float((close.iloc[-1] - sample.mean()) / deviation) if deviation else 0.0
    if canonical == "rolling_breakout_pct":
        periods = window or 20
        if len(close) <= periods:
            return None
        prior_high = close.iloc[-periods - 1 : -1].max()
        return float((close.iloc[-1] / prior_high - 1) * 100) if prior_high else None
    if canonical == "volume_ratio":
        if "volume" not in history or len(history) < (window or 20):
            return None
        volume = pd.to_numeric(history["volume"], errors="coerce").dropna()
        periods = window or 20
        if len(volume) < periods:
            return None
        average = volume.tail(periods).mean()
        return float(volume.iloc[-1] / average) if average else None
    if canonical == "rsi":
        periods = window or 14
        if len(close) <= periods:
            return None
        changes = close.diff()
        gains = changes.clip(lower=0).rolling(periods).mean().iloc[-1]
        losses = -changes.clip(upper=0).rolling(periods).mean().iloc[-1]
        if losses == 0:
            return 100.0
        return float(100 - (100 / (1 + gains / losses)))
    if canonical == "atr":
        if not {"high", "low", "close"}.issubset(history.columns):
            return None
        periods = window or 14
        if len(history) < periods:
            return None
        high = pd.to_numeric(history["high"], errors="coerce")
        low = pd.to_numeric(history["low"], errors="coerce")
        previous_close = pd.to_numeric(history["close"], errors="coerce").shift(1)
        true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(
            axis=1
        )
        value = true_range.rolling(periods).mean().iloc[-1]
        return float(value) if pd.notna(value) else None
    if canonical == "volatility":
        periods = window or 20
        if len(close) <= periods:
            return None
        returns = close.pct_change().dropna().tail(periods)
        return float(returns.std() * (252**0.5) * 100) if len(returns) >= periods else None
    return None


def indicator_value(
    history: pd.DataFrame,
    operand: StrategyOperand,
    indicator_specs: list[IndicatorSpec] | None = None,
) -> float | list[float] | None:
    """Resolve an expression operand without exposing the private indicator implementation."""

    if operand.type == "constant":
        return operand.value
    return _indicator(history, operand.indicator or "", operand.window, indicator_specs)


def _expression_indicators(expression: StrategyExpression, *, depth: int = 1) -> tuple[list[str], list[str]]:
    if depth > 12:
        return [], ["表达式嵌套深度不能超过 12"]
    names = []
    errors = []
    for operand in (expression.left, expression.right):
        if operand is not None and operand.type == "indicator" and operand.indicator:
            names.append(operand.indicator)
    for child in expression.children:
        child_names, child_errors = _expression_indicators(child, depth=depth + 1)
        names.extend(child_names)
        errors.extend(child_errors)
    if expression.expression is not None:
        child_names, child_errors = _expression_indicators(expression.expression, depth=depth + 1)
        names.extend(child_names)
        errors.extend(child_errors)
    return names, errors


def validate_strategy_spec(
    spec: StrategySpec,
    *,
    available_columns: set[str] | None = None,
) -> list[str]:
    """Validate indicators before a strategy is allowed to run."""
    errors: list[str] = []
    definitions = {item["name"].lower(): item for item in available_indicators()}
    aliases = {item.alias.lower(): item.name.lower() for item in spec.indicator_specs if item.alias}
    requested = [item.name for item in spec.indicator_specs]
    for component in spec.components:
        if component.expression is not None:
            names, expression_errors = _expression_indicators(component.expression)
            requested.extend(names)
            errors.extend(expression_errors)
        if component.type == "python":
            try:
                get_strategy_plugin(component.plugin or "", component.plugin_version)
            except ValueError as exc:
                errors.append(str(exc))
    for name in requested:
        canonical = aliases.get(name.lower(), name.lower())
        if canonical.startswith(("ma", "ema")) and canonical[2:].isdigit():
            canonical = canonical[:2]
        if canonical not in definitions:
            errors.append(name)
            continue
        if available_columns is not None:
            required = set(definitions[canonical]["requires"])
            missing = sorted(required - available_columns)
            if missing:
                errors.append(f"{name} 缺少字段: {', '.join(missing)}")
    return list(dict.fromkeys(errors))


def _compare_values(
    left: float | list[float] | None,
    operator: str,
    right: float | list[float] | None,
) -> bool:
    if left is None or isinstance(left, list) or right is None:
        return False
    if operator == "between":
        return isinstance(right, list) and len(right) == 2 and float(right[0]) <= left <= float(right[1])
    if isinstance(right, list):
        return False
    return {
        "gt": left > right,
        "gte": left >= right,
        "lt": left < right,
        "lte": left <= right,
        "eq": left == right,
    }[operator]


def evaluate_expression(
    expression: StrategyExpression,
    history: pd.DataFrame,
    *,
    indicator_specs: list[IndicatorSpec] | None = None,
) -> dict[str, Any]:
    """Evaluate a recursive expression against the supplied historical prefix."""

    def evaluate(item: StrategyExpression, frame: pd.DataFrame) -> dict[str, Any]:
        if item.type == "compare":
            left = indicator_value(frame, item.left, indicator_specs) if item.left else None
            right = indicator_value(frame, item.right, indicator_specs) if item.right else None
            matched = _compare_values(left, item.operator or "eq", right)
            return {"type": item.type, "matched": matched, "left": left, "right": right, "operator": item.operator}
        if item.type in {"crosses_above", "crosses_below"}:
            current_left = indicator_value(frame, item.left, indicator_specs) if item.left else None
            current_right = indicator_value(frame, item.right, indicator_specs) if item.right else None
            previous = frame.iloc[:-1]
            previous_left = indicator_value(previous, item.left, indicator_specs) if item.left else None
            previous_right = indicator_value(previous, item.right, indicator_specs) if item.right else None
            values = (previous_left, previous_right, current_left, current_right)
            if any(value is None or isinstance(value, list) for value in values):
                matched = False
            elif item.type == "crosses_above":
                matched = previous_left <= previous_right and current_left > current_right
            else:
                matched = previous_left >= previous_right and current_left < current_right
            return {
                "type": item.type,
                "matched": matched,
                "previous": {"left": previous_left, "right": previous_right},
                "current": {"left": current_left, "right": current_right},
            }
        if item.type in {"all", "any"}:
            children = [evaluate(child, frame) for child in item.children]
            matched = (
                all(child["matched"] for child in children)
                if item.type == "all"
                else any(child["matched"] for child in children)
            )
            return {"type": item.type, "matched": matched, "children": children}
        if item.type == "not":
            child = evaluate(item.expression, frame) if item.expression else {"matched": False}
            return {"type": item.type, "matched": not child["matched"], "child": child}
        if item.type in {"sustained", "count"}:
            bars = item.bars or 1
            outcomes = []
            for offset in range(bars, 0, -1):
                prefix = frame.iloc[: len(frame) - offset + 1]
                outcomes.append(evaluate(item.expression, prefix) if item.expression else {"matched": False})
            matched_count = sum(bool(outcome["matched"]) for outcome in outcomes)
            matched = matched_count == bars if item.type == "sustained" else matched_count >= (item.minimum or bars)
            return {
                "type": item.type,
                "matched": matched,
                "bars": bars,
                "matched_count": matched_count,
                "outcomes": outcomes,
            }
        return {"type": item.type, "matched": False}

    return evaluate(expression, history)
