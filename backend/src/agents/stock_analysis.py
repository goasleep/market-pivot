"""Controlled runtime for the stock comprehensive-analysis graph."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import StructuredTool
from loguru import logger

from agents.asset_requests import AssetAgentRequest, AssetRequestResolver
from application.research import research_service
from graph.checkpointing import checkpoint_manager
from models.schemas import AssetType
from observability import build_trace_config


class StockAnalysisRuntime(AssetRequestResolver):
    """Build and execute only the trusted comprehensive-analysis capability."""

    def build_tool(
        self,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        *,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> StructuredTool:
        async def run_analysis(
            ticker: str,
            config: RunnableConfig,
            asset_type: Literal["stock", "etf", "lof"],
            strategy: str | None = None,
        ) -> str:
            """运行综合研究分析，适合用户要求趋势、买卖、风险或交易建议时使用。"""
            normalized_tickers = self.extract_tickers(ticker)
            if len(normalized_tickers) != 1:
                raise ValueError("ticker 必须是单个六位 A 股代码，例如 600519 或 510300")
            try:
                normalized_asset_type = AssetType(asset_type)
            except ValueError as exc:
                raise ValueError("asset_type 必须是 stock、etf 或 lof") from exc
            request = self.prepare(
                f"分析 {normalized_tickers[0]}",
                strategy=strategy,
                asset_type=normalized_asset_type.value,
                conversation_id=conversation_id,
                task_id=task_id,
            )
            if progress_callback is None:
                _, result = await self.analyze(request, config=config)
            else:
                result: dict[str, Any] = {}
                async for update in self.analyze_stream(
                    request,
                    config=config,
                    progress_callback=progress_callback,
                ):
                    result = update.get("state", result)
            decision = result.get("final_decision")
            if decision is None:
                return "{}"
            market_context = result.get("market_context")
            try:
                report_artifacts = await research_service.create_artifacts(
                    decision,
                    market_context,
                    source="chat-tool-analysis",
                    conversation_id=request.conversation_id,
                    task_id=request.task_id,
                    execution_key=f"{request.task_id}:comprehensive-report" if request.task_id else None,
                )
            except Exception as exc:
                logger.warning("Analysis report artifact generation failed; returning decision only: {}", exc)
                report_artifacts = []
            artifacts = [*(result.get("visual_artifacts") or []), *report_artifacts]
            payload = research_service.decision_payload(decision, market_context, artifacts=artifacts)
            return json.dumps(payload, ensure_ascii=False)

        return StructuredTool.from_function(
            coroutine=run_analysis,
            name="run_stock_comprehensive_analysis",
            description=(
                "运行短中期股票、ETF或LOF研究分析。只有用户明确需要分析、判断、策略或风险建议时调用。"
                "必须同时传入 ticker 和 asset_type；asset_type 只能是 stock、etf 或 lof。"
            ),
        )

    def _analysis_tool(
        self,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        *,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> StructuredTool:
        """Compatibility alias while the legacy Supervisor is being removed."""
        return self.build_tool(
            progress_callback,
            conversation_id=conversation_id,
            task_id=task_id,
        )

    async def analyze(
        self,
        request: AssetAgentRequest,
        *,
        config: RunnableConfig | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the multi-agent stock analysis workflow with tracing metadata."""
        if not request.ticker:
            raise ValueError("A stock code is required for analysis")

        run_config = self._analysis_trace_config(request, config)
        result = await research_service.run(
            request.ticker,
            strategy=request.strategy,
            asset_type=request.asset_type,
            conversation_history=request.history,
            conversation_id=request.conversation_id,
            task_id=request.task_id,
            trace_config=run_config,
        )
        context = result.get("market_context")
        return context.realtime if context else {}, result

    async def analyze_stream(
        self,
        request: AssetAgentRequest,
        *,
        config: RunnableConfig | None = None,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream actual LangGraph node updates for the chat UI."""
        if not request.ticker:
            raise ValueError("A stock code is required for analysis")

        run_config = self._analysis_trace_config(request, config)
        async for update in research_service.stream(
            request.ticker,
            strategy=request.strategy,
            asset_type=request.asset_type,
            conversation_history=request.history,
            conversation_id=request.conversation_id,
            task_id=request.task_id,
            trace_config=run_config,
        ):
            context = update.get("state", {}).get("market_context")
            event = {
                **update,
                "realtime": context.realtime if context else {},
                "data_status": context.data_status if context else {},
            }
            if progress_callback is not None:
                await progress_callback(event)
            yield event

    @staticmethod
    def _analysis_trace_config(
        request: AssetAgentRequest,
        config: RunnableConfig | None,
    ) -> RunnableConfig:
        """Reuse the parent chat trace when analysis is invoked as a tool."""
        if config is None:
            trace_config = build_trace_config(
                "asset-agent-analysis",
                tags=["asset-agent", "chat", request.intent.value],
                metadata={
                    "ticker": request.ticker or "",
                    "intent": request.intent.value,
                    "strategy": request.strategy or "auto",
                    "conversation_id": request.conversation_id or "",
                },
                session_id=request.conversation_id,
            )
        else:
            trace_config = {
                **config,
                "run_name": "asset-agent-analysis",
                "tags": [*config.get("tags", []), "asset-agent", "analysis"],
                "metadata": {
                    **config.get("metadata", {}),
                    "ticker": request.ticker or "",
                    "intent": request.intent.value,
                    "strategy": request.strategy or "auto",
                    "conversation_id": request.conversation_id or "",
                },
            }
        if request.task_id and checkpoint_manager.saver is not None:
            return checkpoint_manager.graph_config(
                f"{request.task_id}:research:comprehensive",
                trace_config,
            )
        return trace_config
