"""Allowlisted Python components for reproducible hybrid strategies.

Plugins receive only an immutable historical prefix, declared parameters and
persisted strategy variables. They return signals; order creation remains in
the trusted trading runtime.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from math import sqrt, tanh
from typing import Any

import pandas as pd

from models.schemas import StrategySignal

StrategyPlugin = Callable[[pd.DataFrame, dict[str, Any], dict[str, Any]], StrategySignal | dict[str, Any]]

_PLUGINS: dict[tuple[str, str], StrategyPlugin] = {}


def register_strategy_plugin(name: str, version: str = "1.0.0"):
    """Register trusted code by stable name and version; dynamic imports are deliberately unsupported."""

    def decorator(function: StrategyPlugin) -> StrategyPlugin:
        key = (name, version)
        if key in _PLUGINS and _PLUGINS[key] is not function:
            raise ValueError(f"策略插件已注册: {name}@{version}")
        _PLUGINS[key] = function
        return function

    return decorator


def get_strategy_plugin(name: str, version: str = "1.0.0") -> StrategyPlugin:
    try:
        return _PLUGINS[(name, version)]
    except KeyError as exc:
        raise ValueError(f"未注册的策略插件: {name}@{version}") from exc


def strategy_plugin_manifest(name: str, version: str = "1.0.0") -> dict[str, str]:
    """Return a content hash suitable for immutable deployment snapshots."""

    plugin = get_strategy_plugin(name, version)
    try:
        source = inspect.getsource(plugin)
    except (OSError, TypeError):
        source = f"{plugin.__module__}:{plugin.__qualname__}"
    return {
        "name": name,
        "version": version,
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def strategy_plugins_manifest(components) -> list[dict[str, str]]:
    items = [
        strategy_plugin_manifest(component.plugin or "", component.plugin_version)
        for component in components
        if component.type == "python"
    ]
    return sorted(items, key=lambda item: (item["name"], item["version"]))


@register_strategy_plugin("core.trend_score", "1.0.0")
def trend_score(
    history: pd.DataFrame,
    params: dict[str, Any],
    _state: dict[str, Any],
) -> dict[str, Any]:
    """Continuous moving-average trend strength in [-1, 1]."""

    fast_window = int(params.get("fast_window", 10))
    slow_window = int(params.get("slow_window", 60))
    sensitivity = float(params.get("sensitivity", 20.0))
    close = pd.to_numeric(history.get("close"), errors="coerce").dropna()
    if fast_window < 1 or slow_window <= fast_window or len(close) < slow_window:
        return {"score": 0.0, "confidence": 0.0, "reasons": ["趋势指标预热不足"]}
    fast = float(close.tail(fast_window).mean())
    slow = float(close.tail(slow_window).mean())
    spread = fast / slow - 1 if slow else 0.0
    return {
        "score": float(tanh(spread * sensitivity)),
        "confidence": min(1.0, len(close) / (slow_window * 2)),
        "metrics": {"fast_ma": fast, "slow_ma": slow, "spread_pct": spread * 100},
        "reasons": [f"MA{fast_window}/MA{slow_window} 连续趋势强度"],
    }


@register_strategy_plugin("core.market_regime", "1.0.0")
def market_regime(
    history: pd.DataFrame,
    params: dict[str, Any],
    _state: dict[str, Any],
) -> dict[str, Any]:
    """Long-only trend router: 1 in an uptrend, 0 in a defensive regime."""

    lookback = int(params.get("lookback", 120))
    close = pd.to_numeric(history.get("close"), errors="coerce").dropna()
    if len(close) < lookback:
        return {"score": 0.0, "confidence": 0.0, "reasons": ["市场状态指标预热不足"]}
    average = float(close.tail(lookback).mean())
    current = float(close.iloc[-1])
    bullish = current > average
    return {
        "score": 1.0 if bullish else -1.0,
        "confidence": min(1.0, abs(current / average - 1) * 20) if average else 0.0,
        "metrics": {"regime_ma": average, "current_close": current},
        "state_updates": {"market_regime": "trend" if bullish else "defensive"},
        "reasons": ["价格位于市场状态均线上方" if bullish else "价格位于市场状态均线下方"],
    }


@register_strategy_plugin("core.volatility_target", "1.0.0")
def volatility_target(
    history: pd.DataFrame,
    params: dict[str, Any],
    _state: dict[str, Any],
) -> dict[str, Any]:
    """Continuous volatility target with an optional moving-average trend gate."""

    volatility_window = int(params.get("volatility_window", 20))
    target_volatility = float(params.get("target_volatility", 0.15))
    max_exposure = float(params.get("max_exposure", 0.95))
    trend_window = int(params.get("trend_window", 0))
    close = pd.to_numeric(history.get("close"), errors="coerce").dropna()
    required = max(volatility_window + 1, trend_window)
    if len(close) < required:
        return {"score": 0.0, "target_exposure": 0.0, "confidence": 0.0, "reasons": ["波动率预热不足"]}
    if trend_window and float(close.iloc[-1]) <= float(close.tail(trend_window).mean()):
        return {
            "score": -1.0,
            "target_exposure": 0.0,
            "confidence": 1.0,
            "reasons": [f"价格位于 MA{trend_window} 下方"],
        }
    returns = close.pct_change(fill_method=None).dropna().tail(volatility_window)
    realized = float(returns.std(ddof=0) * sqrt(252))
    exposure = max_exposure if realized <= 0 else min(max_exposure, target_volatility / realized)
    return {
        "score": 1.0,
        "target_exposure": max(0.0, exposure),
        "confidence": 1.0,
        "metrics": {"realized_volatility": realized, "target_volatility": target_volatility},
        "reasons": [f"{volatility_window} 日波动率目标仓位"],
    }
