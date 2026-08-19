"""Application boundary for the multi-agent research workflow.

The HTTP layer, chat Agent, automation service, and backtester all need the
same workflow semantics.  Keeping state construction and workflow invocation
here prevents each caller from growing its own slightly different pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from artifacts.service import artifact_service
from graph.workflow import workflow as compiled_workflow
from models.schemas import AssetType, MarketContext, TradeDecision
from observability import build_trace_config


class ResearchService:
    """Application boundary around LangGraph; callers do not build graph state."""

    workflow = compiled_workflow

    @staticmethod
    def build_state(
        ticker: str,
        *,
        strategy: str | None = None,
        asset_type: AssetType | str = AssetType.STOCK,
        investor_context: dict[str, Any] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        market_context: MarketContext | None = None,
        current_price: float | None = None,
        as_of_date: str | None = None,
        is_backtest: bool = False,
    ) -> dict[str, Any]:
        """Create the canonical state accepted by ``graph.workflow``."""
        normalized_asset_type = AssetType(asset_type)
        state: dict[str, Any] = {
            "ticker": ticker,
            "asset_type": normalized_asset_type.value,
            "current_price": (
                current_price
                if current_price is not None
                else (market_context.current_price if market_context else 0.0)
            ),
            "progress": [],
        }
        if strategy:
            state["strategy_name"] = strategy
        if investor_context:
            state["investor_context"] = investor_context
        if conversation_history:
            state["conversation_history"] = conversation_history[-12:]
        if market_context is not None:
            state["market_context"] = market_context
        if as_of_date is not None:
            state["as_of_date"] = as_of_date
        if is_backtest:
            state["is_backtest"] = True
        return state

    @staticmethod
    def build_trace_config(
        ticker: str,
        asset_type: AssetType | str,
        *,
        stream: bool = False,
        run_name: str = "research-analysis",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_asset_type = AssetType(asset_type).value
        trace_tags = ["research", "analysis"]
        if stream:
            trace_tags.append("stream")
        if tags:
            trace_tags.extend(tags)
        trace_metadata = {"ticker": ticker, "asset_type": normalized_asset_type}
        if metadata:
            trace_metadata.update(metadata)
        return build_trace_config(
            run_name,
            tags=trace_tags,
            metadata=trace_metadata,
        )

    @staticmethod
    def _invoke_options(config: dict[str, Any] | None) -> dict[str, Any]:
        """Keep run metadata attached; fall back for minimal test doubles."""
        return {"config": config} if config else {}

    async def run_state(
        self,
        state: dict[str, Any],
        *,
        trace_config: dict[str, Any] | None = None,
        workflow_override: Any | None = None,
    ) -> dict[str, Any]:
        """Invoke the compiled workflow for an already prepared state.

        ``workflow_override`` is intentionally narrow and exists for backtest
        doubles and isolated tests; production callers use the shared graph.
        """
        runner = workflow_override or self.workflow
        options = self._invoke_options(trace_config)
        try:
            return await runner.ainvoke(state, **options)
        except TypeError as exc:
            if options and "unexpected keyword argument 'config'" in str(exc):
                return await runner.ainvoke(state)
            raise

    async def stream_state(
        self,
        state: dict[str, Any],
        *,
        trace_config: dict[str, Any] | None = None,
        workflow_override: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream canonical workflow updates with an accumulated state."""
        runner = workflow_override or self.workflow
        options = self._invoke_options(trace_config)
        accumulated = dict(state)
        try:
            updates = runner.astream(state, stream_mode="updates", **options)
            async for update in updates:
                node, node_update = next(iter(update.items()))
                if isinstance(node_update, dict):
                    accumulated.update(node_update)
                yield {"node": node, "update": node_update, "state": accumulated}
        except TypeError as exc:
            if not options or "unexpected keyword argument 'config'" not in str(exc):
                raise
            accumulated = dict(state)
            async for update in runner.astream(state, stream_mode="updates"):
                node, node_update = next(iter(update.items()))
                if isinstance(node_update, dict):
                    accumulated.update(node_update)
                yield {"node": node, "update": node_update, "state": accumulated}

    async def run(
        self,
        ticker: str,
        strategy: str | None = None,
        asset_type: AssetType | str = AssetType.STOCK,
        investor_context: dict[str, Any] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        market_context: MarketContext | None = None,
        current_price: float | None = None,
        as_of_date: str | None = None,
        is_backtest: bool = False,
        trace_config: dict[str, Any] | None = None,
        workflow_override: Any | None = None,
    ) -> dict[str, Any]:
        state = self.build_state(
            ticker,
            strategy=strategy,
            asset_type=asset_type,
            investor_context=investor_context,
            conversation_history=conversation_history,
            market_context=market_context,
            current_price=current_price,
            as_of_date=as_of_date,
            is_backtest=is_backtest,
        )
        return await self.run_state(
            state,
            trace_config=trace_config or self.build_trace_config(ticker, asset_type),
            workflow_override=workflow_override,
        )

    async def stream(
        self,
        ticker: str,
        strategy: str | None = None,
        asset_type: AssetType | str = AssetType.STOCK,
        investor_context: dict[str, Any] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        market_context: MarketContext | None = None,
        current_price: float | None = None,
        as_of_date: str | None = None,
        is_backtest: bool = False,
        trace_config: dict[str, Any] | None = None,
        workflow_override: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        state = self.build_state(
            ticker,
            strategy=strategy,
            asset_type=asset_type,
            investor_context=investor_context,
            conversation_history=conversation_history,
            market_context=market_context,
            current_price=current_price,
            as_of_date=as_of_date,
            is_backtest=is_backtest,
        )
        async for update in self.stream_state(
            state,
            trace_config=trace_config or self.build_trace_config(ticker, asset_type, stream=True),
            workflow_override=workflow_override,
        ):
            yield update

    async def create_artifacts(
        self,
        decision: TradeDecision,
        market_context: MarketContext | None,
        *,
        source: str,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Create report files through the dedicated ReportAgent."""
        return await asyncio.to_thread(
            artifact_service.create_analysis_artifacts,
            decision,
            market_context,
            source=source,
            conversation_id=conversation_id,
            task_id=task_id,
        )

    @staticmethod
    def decision_payload(
        decision: TradeDecision,
        market_context: MarketContext | None = None,
        *,
        show_reasoning: bool = True,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Serialize one decision consistently for HTTP and tool consumers."""
        return {
            "ticker": decision.ticker,
            "asset_type": decision.asset_type.value,
            "decision": decision.decision.value,
            "confidence": decision.confidence,
            "entry_price": decision.entry_price,
            "target_price": decision.target_price,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "position_size": decision.position_size,
            "plan": decision.plan.model_dump(mode="json"),
            "reasoning": decision.reasoning if show_reasoning else "",
            "agent_reports": decision.agent_reports if show_reasoning else {},
            "dashboard": decision.dashboard.model_dump() if decision.dashboard else None,
            "instrument": market_context.instrument.model_dump(mode="json") if market_context else None,
            "fund_data": (
                market_context.fund_data.model_dump(mode="json")
                if market_context and market_context.fund_data
                else None
            ),
            "data_status": market_context.data_status if market_context else {},
            "artifacts": artifacts or [],
        }


research_service = ResearchService()
