"""Application facade for historical backtest use cases."""

from __future__ import annotations

from typing import Any

from engine.backtester import run_backtest as _run_backtest
from engine.backtester import run_pool_backtest as _run_pool_backtest


async def run_backtest(**params: Any) -> dict[str, Any]:
    """Run a single-symbol backtest without exposing the engine to routers."""
    return await _run_backtest(**params)


async def run_pool_backtest(**params: Any) -> dict[str, Any]:
    """Run a pooled backtest without exposing the engine to routers."""
    return await _run_pool_backtest(**params)
