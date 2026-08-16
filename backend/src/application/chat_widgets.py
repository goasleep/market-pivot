"""Application facade for chat-facing market widgets.

Routers should expose HTTP contracts, not know which data provider or widget
renderer is needed to build a surface.
"""

from __future__ import annotations

import asyncio
from typing import Any

from data.akshare_provider import async_get_stock_history, get_breaker_status
from strategies.skill_manager import list_strategies
from widgets.a2ui import (
    CATALOG,
    render_breaker_status,
    render_mini_chart,
    render_strategy_selector,
)


class ChatWidgetService:
    """Build protocol-compliant widget payloads for chat endpoints."""

    @staticmethod
    def _surface(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"protocol": "a2ui", "catalog": CATALOG, "messages": messages}

    @staticmethod
    def catalog() -> list[dict[str, Any]]:
        return CATALOG

    async def strategies(self) -> dict[str, Any]:
        strategies = await asyncio.to_thread(list_strategies)
        return self._surface(render_strategy_selector(strategies))

    async def breakers(self) -> dict[str, Any]:
        breakers = await asyncio.to_thread(get_breaker_status)
        return self._surface(render_breaker_status(breakers))

    async def mini_chart(self, ticker: str) -> dict[str, Any]:
        history = await async_get_stock_history(ticker)
        prices = [] if history.empty else history.tail(30)["close"].tolist()
        return self._surface(render_mini_chart(prices))


chat_widget_service = ChatWidgetService()
