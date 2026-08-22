"""Streaming adapter for the checkpointed market Research Plan graph."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from graph.checkpointing import checkpoint_manager
from graph.research_plan import ResearchPlanContext, get_research_plan_graph
from models.research_plan import ResearchPlan, StepResult

RESEARCH_GRAPH_NAME = "market-research-plan"


def _plan_snapshot(
    values: dict[str, Any],
    *,
    status: str = "running",
    running_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    raw_plan = values.get("plan")
    if not isinstance(raw_plan, dict):
        return None
    plan = ResearchPlan.model_validate(raw_plan)
    results = {
        key: StepResult.model_validate(value)
        for key, value in (values.get("step_results") or {}).items()
        if isinstance(value, dict)
    }
    steps = []
    for step in plan.steps:
        result = results.get(step.id)
        steps.append(
            {
                "id": step.id,
                "kind": step.kind,
                "title": step.title,
                "status": result.status if result else "running" if step.id in (running_ids or set()) else "pending",
                "summary": result.summary if result else "",
                "error": result.error if result else None,
            }
        )
    completed = sum(item["status"] in {"completed", "failed", "skipped"} for item in steps)
    failed = any(item["status"] == "failed" for item in steps)
    if status == "completed" and failed:
        status = "completed_with_gaps"
    return {
        "plan_id": plan.plan_id,
        "objective": plan.objective,
        "asset_type": plan.asset_type.value,
        "tickers": plan.tickers,
        "as_of_date": plan.as_of_date,
        "depth": plan.depth,
        "revision": plan.revision,
        "status": status,
        "progress": round(completed / len(steps) * 100) if steps else 0,
        "steps": steps,
    }


class ResearchPlanService:
    """Run or resume research while exposing only serializable public progress."""

    @staticmethod
    def _context(tools: Iterable[StructuredTool]) -> ResearchPlanContext:
        return ResearchPlanContext(tools={tool.name: tool for tool in tools})

    @staticmethod
    def _config(thread_id: str, config: RunnableConfig | None = None) -> RunnableConfig:
        return checkpoint_manager.graph_config(thread_id, config)

    async def stream(
        self,
        request: dict[str, Any],
        tools: Iterable[StructuredTool],
        *,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        thread_id = str(request.get("task_id") or "")
        if not thread_id:
            raise ValueError("Research Plan 需要稳定的 task_id 作为 checkpoint thread_id")
        async for event in self._stream_graph(
            {"request": request},
            tools,
            config=self._config(thread_id, config),
        ):
            yield event

    async def resume(
        self,
        request: dict[str, Any],
        tools: Iterable[StructuredTool],
        *,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        thread_id = str(request.get("task_id") or "")
        if not thread_id:
            raise ValueError("Research Plan 恢复缺少 task_id")
        graph = get_research_plan_graph()
        resolved_config = self._config(thread_id, config)
        snapshot = await graph.aget_state(resolved_config)
        values = dict(snapshot.values or {})
        if values and not snapshot.next:
            plan = _plan_snapshot(values, status="completed")
            if plan:
                yield {"type": "plan_update", "plan": plan, "create": True}
            final_response = str(values.get("final_response") or "")
            if final_response:
                yield {"type": "text", "text": final_response}
            return
        async for event in self._stream_graph(
            None if values else {"request": request},
            tools,
            config=resolved_config,
        ):
            yield event

    async def _stream_graph(
        self,
        graph_input: dict[str, Any] | None,
        tools: Iterable[StructuredTool],
        *,
        config: RunnableConfig,
    ) -> AsyncIterator[dict[str, Any]]:
        graph = get_research_plan_graph()
        emitted_plan = False
        emitted_results: set[tuple[str, int, str]] = set()
        final_values: dict[str, Any] = {}
        async for update in graph.astream(
            graph_input,
            config=config,
            context=self._context(tools),
            stream_mode="updates",
        ):
            snapshot = await graph.aget_state(config)
            values = dict(snapshot.values or {})
            final_values = values
            running_ids: set[str] = set()
            if "dispatch" in update and isinstance(values.get("plan"), dict):
                plan = ResearchPlan.model_validate(values["plan"])
                results = {
                    key: StepResult.model_validate(value)
                    for key, value in (values.get("step_results") or {}).items()
                    if isinstance(value, dict)
                }
                completed = {key for key, result in results.items() if result.status == "completed"}
                running_ids = {
                    step.id
                    for step in plan.steps
                    if step.id not in results and set(step.depends_on) <= completed
                }
            plan_update = _plan_snapshot(values, running_ids=running_ids)
            if plan_update:
                yield {"type": "plan_update", "plan": plan_update, "create": not emitted_plan}
                emitted_plan = True
            for result_value in (values.get("step_results") or {}).values():
                if not isinstance(result_value, dict):
                    continue
                result = StepResult.model_validate(result_value)
                key = (result.step_id, result.attempt, result.status)
                if result.status not in {"completed", "failed"} or key in emitted_results:
                    continue
                emitted_results.add(key)
                plan = ResearchPlan.model_validate(values["plan"])
                step = next(item for item in plan.steps if item.id == result.step_id)
                tool_name = {
                    "market_snapshot": "get_realtime_quote",
                    "price_history": "get_historical_prices",
                    "fund_nav": "get_fund_nav_history",
                    "fundamentals": "get_fundamentals",
                    "news": "search_web",
                    "methodology": "search_methodology",
                    "backtest": "design_and_run_backtest",
                    "comparison": "compare_quotes",
                    "comprehensive_analysis": "run_fund_or_stock_analysis",
                    "risk": "calculate_risk_metrics",
                    "report": "save_artifacts",
                }.get(step.kind, step.kind)
                yield {
                    "type": "tool",
                    "name": tool_name,
                    "status": result.status,
                    "result": json.dumps(result.output or {"error": result.error}, ensure_ascii=False, default=str),
                }
            if "finish" in update:
                final_plan = _plan_snapshot(values, status="completed")
                if final_plan:
                    yield {"type": "plan_update", "plan": final_plan, "create": not emitted_plan}
                    emitted_plan = True

        final_response = str(final_values.get("final_response") or "")
        if final_response:
            yield {"type": "text", "text": final_response}


research_plan_service = ResearchPlanService()
