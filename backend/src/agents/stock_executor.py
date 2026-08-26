"""Controlled executor boundary for the stock comprehensive-analysis graph."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain_core.tools import StructuredTool

from agents.stock_analysis import StockAnalysisRuntime


class StockComprehensiveExecutor:
    """Expose the legacy stock graph only as the trusted stock capability."""

    def __init__(self) -> None:
        self._runtime = StockAnalysisRuntime()

    def build_tool(
        self,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        *,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> StructuredTool:
        return self._runtime.build_tool(
            progress_callback,
            conversation_id=conversation_id,
            task_id=task_id,
        )


stock_comprehensive_executor = StockComprehensiveExecutor()
