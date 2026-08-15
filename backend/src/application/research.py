"""Use cases for running the Stock Agent research workflow."""

from __future__ import annotations

from typing import Any, AsyncIterator

from graph.workflow import workflow
from models.schemas import AssetType


class ResearchService:
    """Application boundary around LangGraph; routers do not build graph state."""

    async def run(
        self,
        ticker: str,
        strategy: str | None = None,
        asset_type: AssetType | str = AssetType.STOCK,
        investor_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "ticker": ticker,
            "asset_type": AssetType(asset_type).value,
            "current_price": 0.0,
            "progress": [],
        }
        if strategy:
            state["strategy_name"] = strategy
        if investor_context:
            state["investor_context"] = investor_context
        return await workflow.ainvoke(state)

    async def stream(
        self,
        ticker: str,
        strategy: str | None = None,
        asset_type: AssetType | str = AssetType.STOCK,
        investor_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        state: dict[str, Any] = {
            "ticker": ticker,
            "asset_type": AssetType(asset_type).value,
            "current_price": 0.0,
            "progress": [],
        }
        if strategy:
            state["strategy_name"] = strategy
        if investor_context:
            state["investor_context"] = investor_context
        accumulated: dict[str, Any] = dict(state)
        async for update in workflow.astream(state, stream_mode="updates"):
            node, node_update = next(iter(update.items()))
            if isinstance(node_update, dict):
                accumulated.update(node_update)
            yield {"node": node, "update": node_update, "state": accumulated}


research_service = ResearchService()
