"""Checkpointed orchestrator-worker graph for market Deep Research."""

from __future__ import annotations

import asyncio
import json
import operator
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, TypedDict, get_args
from uuid import uuid4

from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from loguru import logger

from application.financial_task_planner import compile_financial_task_spec
from graph.agent_loop import tool_timeout_seconds
from graph.research_evidence import (
    _classify_failure,
    _compact,
    _comparison_synthesis_text,
    _deterministic_synthesis_fallback,
    _evidence,
    _find_price,
    _result_summaries,
    _synthesis_request,
)
from graph.research_planning import (
    DEPTH_BUDGETS,
    classify_depth,
    derive_task_contract,
)
from graph.research_planning import (
    effective_max_attempts as _effective_max_attempts,
)
from graph.research_planning import (
    fallback_steps as _fallback_steps,
)
from graph.research_planning import (
    normalize_steps as _normalize_steps,
)
from graph.research_planning import (
    validate_plan_contract as _validate_plan_contract,
)
from llm.context import (
    TokenCounter,
    compact_json_value,
    context_safe_error,
    get_context_budget,
    is_context_overflow_error,
)
from llm.service import get_llm_service
from models.research_plan import (
    ResearchBudget,
    ResearchPlan,
    ResearchStep,
    ResearchStepKind,
    StepRecovery,
    StepResult,
)
from models.schemas import AssetType


def merge_step_results(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {**(left or {}), **(right or {})}


class ResearchPlanState(TypedDict, total=False):
    request: dict[str, Any]
    depth: str
    budget: dict[str, Any]
    deadline_at: str
    plan: dict[str, Any]
    step: dict[str, Any]
    step_results: Annotated[dict[str, dict[str, Any]], merge_step_results]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    replan_count: int
    tool_calls: Annotated[int, operator.add]
    needs_replan: bool
    final_response: str
    task_contract: dict[str, Any]
    financial_task_spec: dict[str, Any]


@dataclass(frozen=True)
class ResearchPlanContext:
    tools: dict[str, StructuredTool]


class ResearchToolExecutionError(RuntimeError):
    """A tool observation that retains safe call context for recovery."""

    def __init__(self, tool_name: str, args: dict[str, Any], message: str):
        super().__init__(message)
        self.tool_name = tool_name
        self.tool_args = _compact(args)


RECOVERY_INPUT_RULES: dict[str, dict[str, str]] = {
    "price_history": {
        "limit": "20 到 500 的整数",
        "start_date": "YYYY-MM-DD，必须早于或等于 end_date",
        "end_date": "YYYY-MM-DD，必须晚于或等于 start_date",
    },
    "fund_nav": {"limit": "20 到 500 的整数"},
    "technical": {"limit": "20 到 500 的整数"},
    "news": {
        "query": "不超过 500 字的检索词",
        "num_results": "1 到 20 的整数",
        "freshness": "搜索工具支持的时间过滤表达式",
    },
    "methodology": {"query": "不超过 500 字的检索词", "limit": "1 到 10 的整数"},
    "backtest": {
        "start_date": "YYYY-MM-DD，必须早于 end_date",
        "end_date": "YYYY-MM-DD，不能晚于研究截止日",
        "objective": "保留用户原目标，并补充解决错误所需的约束，不超过 2000 字",
        "execution_mode": "agent、comparison 或 sandbox",
        "initial_capital": "1000 到 1000000000 的数值",
        "decision_interval": "1 到 250 的整数",
    },
    "risk": {
        "stop_loss_pct": "0 到 0.5 的小数",
        "take_profit_pct": "0 到 2 的小数",
        "position_size_pct": "0 到 1 的小数",
    },
}


async def scope_research(state: ResearchPlanState) -> dict[str, Any]:
    depth = classify_depth(state["request"])
    budget = DEPTH_BUDGETS[depth]
    deadline = datetime.now(timezone.utc) + timedelta(seconds=budget.deadline_seconds)
    contract = derive_task_contract(state["request"])
    financial_spec = compile_financial_task_spec(state["request"])
    return {
        "depth": depth,
        "budget": budget.model_dump(mode="json"),
        "deadline_at": deadline.isoformat(),
        "step_results": {},
        "evidence": [],
        "replan_count": 0,
        "tool_calls": 0,
        "needs_replan": False,
        "task_contract": contract.model_dump(mode="json"),
        "financial_task_spec": financial_spec.model_dump(mode="json") if financial_spec else {},
    }


async def plan_research(state: ResearchPlanState) -> dict[str, Any]:
    request = state["request"]
    depth = state["depth"]
    budget = ResearchBudget.model_validate(state["budget"])
    financial_spec = state.get("financial_task_spec") or {}
    if financial_spec:
        steps = [
            {
                "id": "data_catalog",
                "kind": "data_catalog",
                "title": "解析数据集、字段与口径",
                "depends_on": [],
                "inputs": {},
                "success_criteria": ["找到满足任务字段与时间口径的数据集"],
                "max_attempts": 2,
            },
            {
                "id": "data_query",
                "kind": "data_query",
                "title": "查询结构化数据并执行确定性变换",
                "depends_on": ["data_catalog"],
                "inputs": {},
                "success_criteria": ["返回覆盖率、预览、来源与完整结果产物"],
                "max_attempts": 2,
            },
            {
                "id": "data_validation",
                "kind": "data_validation",
                "title": "核验业务验收条件",
                "depends_on": ["data_query"],
                "inputs": {},
                "success_criteria": ["明确 satisfied、partial、data_unavailable 或 invalid_result"],
                "max_attempts": 1,
            },
            {
                "id": "synthesis",
                "kind": "synthesis",
                "title": "生成带口径与覆盖说明的结果表",
                "depends_on": ["data_validation"],
                "inputs": {},
                "success_criteria": ["回答只引用已通过验收的程序结果"],
                "max_attempts": 1,
            },
        ]
    else:
        prompt = json.dumps(
        {
            "objective": request.get("message"),
            "intent": request.get("intent"),
            "tickers": request.get("tickers", []),
            "asset_type": request.get("asset_type", "stock"),
            "depth": depth,
            "max_steps": budget.max_steps,
            "allowed_kinds": list(get_args(ResearchStepKind)),
            "rules": [
                "只输出公开的操作计划，不输出内部思维链",
                "步骤必须构成无环依赖图",
                "市场数值必须由结构化数据步骤提供",
                "最终结论必须依赖证据步骤",
                "回测和综合分析步骤 max_attempts=2，报告步骤 max_attempts=1",
            ],
        },
        ensure_ascii=False,
    )
        try:
            raw = await get_llm_service().chat_json(
                prompt,
                system=(
                    "你是市场研究 Planner。仅返回 JSON 对象："
                    "{steps:[{id,kind,title,depends_on,inputs,success_criteria,max_attempts}]}。"
                ),
            )
            steps = _normalize_steps(raw, request, depth)
        except Exception as exc:
            logger.warning("Research planner fell back to deterministic template: {}", exc)
            steps = _fallback_steps(request, depth)

    plan_kwargs = {
        "plan_id": f"research-{uuid4().hex[:16]}",
        "objective": str(request.get("message", "市场研究")),
        "asset_type": AssetType(str(request.get("asset_type", "stock"))),
        "tickers": [str(item) for item in request.get("tickers", [])],
        "as_of_date": str(request.get("as_of_date") or date.today().isoformat()),
        "depth": depth,
        "task_contract": state.get("task_contract"),
    }
    try:
        plan = ResearchPlan(**plan_kwargs, steps=[ResearchStep.model_validate(item) for item in steps])
        _validate_plan_contract(plan, request, budget)
    except (TypeError, ValueError) as exc:
        logger.warning("Research planner output was invalid; using deterministic template: {}", exc)
        plan = ResearchPlan(
            **plan_kwargs,
            steps=[ResearchStep.model_validate(item) for item in _fallback_steps(request, depth)],
        )
        _validate_plan_contract(plan, request, budget)
    return {"plan": plan.model_dump(mode="json")}


async def validate_plan(state: ResearchPlanState) -> dict[str, Any]:
    plan = ResearchPlan.model_validate(state["plan"])
    budget = ResearchBudget.model_validate(state["budget"])
    _validate_plan_contract(plan, state["request"], budget)
    return {"plan": plan.model_dump(mode="json")}


async def dispatch_ready(state: ResearchPlanState) -> dict[str, Any]:
    return {}


def route_dispatch(state: ResearchPlanState) -> list[Send] | str:
    plan = ResearchPlan.model_validate(state["plan"])
    results = {key: StepResult.model_validate(value) for key, value in (state.get("step_results") or {}).items()}
    completed = {step_id for step_id, result in results.items() if result.status == "completed"}
    ready = [
        step
        for step in plan.steps
        if (step.id not in results or results[step.id].status == "pending") and set(step.depends_on) <= completed
    ]
    if not ready:
        return "verify"
    budget = ResearchBudget.model_validate(state["budget"])
    remaining_calls = max(0, budget.max_tool_calls - state.get("tool_calls", 0))
    if remaining_calls == 0:
        return "verify"
    return [
        Send(
            "worker",
            {
                "request": state["request"],
                "depth": state["depth"],
                "budget": state["budget"],
                "deadline_at": state["deadline_at"],
                "plan": state["plan"],
                "step": step.model_dump(mode="json"),
                "step_results": state.get("step_results", {}),
                "replan_count": state.get("replan_count", 0),
                "tool_calls": state.get("tool_calls", 0),
                "task_contract": state.get("task_contract", {}),
                "financial_task_spec": state.get("financial_task_spec", {}),
            },
        )
        for step in ready[: min(budget.max_parallel, remaining_calls)]
    ]


def _bounded_number(value: Any, minimum: float, maximum: float, *, integer: bool = False) -> int | float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not minimum <= number <= maximum:
        return None
    return int(number) if integer else number


def _sanitize_recovery_patch(
    step: ResearchStep,
    raw_patch: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Keep LLM recovery changes inside the public, read-only research contract."""
    if not isinstance(raw_patch, dict):
        return {}
    allowed = RECOVERY_INPUT_RULES.get(step.kind, {})
    patch = {key: value for key, value in raw_patch.items() if key in allowed}
    sanitized: dict[str, Any] = {}

    limit_ranges = {
        "price_history": (20, 500),
        "fund_nav": (20, 500),
        "technical": (20, 500),
        "methodology": (1, 10),
    }
    if "limit" in patch and step.kind in limit_ranges:
        value = _bounded_number(patch["limit"], *limit_ranges[step.kind], integer=True)
        if value is not None:
            sanitized["limit"] = value
    if step.kind == "price_history":
        proposed_start = patch.get("start_date")
        proposed_end = patch.get("end_date")
        try:
            start = date.fromisoformat(str(proposed_start)) if proposed_start else None
            end = date.fromisoformat(str(proposed_end)) if proposed_end else None
        except ValueError:
            start = None
            end = None
        if start is not None and (end is None or start <= end):
            sanitized["start_date"] = start.isoformat()
        if end is not None and (start is None or start <= end):
            sanitized["end_date"] = end.isoformat()
    if step.kind == "news":
        if isinstance(patch.get("query"), str) and patch["query"].strip():
            sanitized["query"] = patch["query"].strip()[:500]
        value = _bounded_number(patch.get("num_results"), 1, 20, integer=True)
        if value is not None:
            sanitized["num_results"] = value
        if isinstance(patch.get("freshness"), str) and patch["freshness"].strip():
            sanitized["freshness"] = patch["freshness"].strip()[:100]
    if step.kind == "methodology" and isinstance(patch.get("query"), str) and patch["query"].strip():
        sanitized["query"] = patch["query"].strip()[:500]
    if step.kind == "backtest":
        original_objective = str(request.get("message", "")).strip()
        if isinstance(patch.get("objective"), str) and patch["objective"].strip():
            proposed = patch["objective"].strip()
            if original_objective and original_objective not in proposed:
                proposed = f"{original_objective}\n工具修复约束：{proposed}"
            sanitized["objective"] = proposed[:2000]
        mode = str(patch.get("execution_mode", "")).strip().lower()
        tickers = [str(item) for item in request.get("tickers", [])]
        compares_strategies = "策略" in original_objective and any(
            term in original_objective for term in ("不同", "多个", "几个", "多种", "对比", "比较")
        )
        if mode in {"agent", "comparison", "sandbox"}:
            if len(tickers) != 1 and mode != "agent":
                mode = ""
            elif compares_strategies and mode != "comparison":
                mode = ""
        if mode:
            sanitized["execution_mode"] = mode
        capital = _bounded_number(patch.get("initial_capital"), 1_000, 1_000_000_000)
        if capital is not None:
            sanitized["initial_capital"] = capital
        interval = _bounded_number(patch.get("decision_interval"), 1, 250, integer=True)
        if interval is not None:
            sanitized["decision_interval"] = interval

        cutoff_text = str(request.get("as_of_date") or date.today().isoformat())
        current_end = str(step.inputs.get("end_date") or cutoff_text)
        proposed_end = str(patch.get("end_date") or current_end)
        proposed_start = str(patch.get("start_date") or step.inputs.get("start_date") or "")
        try:
            cutoff = date.fromisoformat(cutoff_text)
            end = date.fromisoformat(proposed_end)
            start = date.fromisoformat(proposed_start) if proposed_start else None
        except ValueError:
            start = None
            end = None
            cutoff = None
        if end is not None and cutoff is not None and end <= cutoff:
            if "end_date" in patch:
                sanitized["end_date"] = end.isoformat()
            if start is not None and start < end and "start_date" in patch:
                sanitized["start_date"] = start.isoformat()
    if step.kind == "risk":
        for key, minimum, maximum in (
            ("stop_loss_pct", 0, 0.5),
            ("take_profit_pct", 0, 2),
            ("position_size_pct", 0, 1),
        ):
            value = _bounded_number(patch.get(key), minimum, maximum)
            if value is not None:
                sanitized[key] = value
    return sanitized


async def _reflect_on_failure(
    step: ResearchStep,
    result: StepResult,
    state: ResearchPlanState,
) -> StepRecovery:
    error = str(result.error or "工具执行失败")[:500]
    classification = _classify_failure(error)
    if classification == "terminal":
        return StepRecovery(
            attempt=result.attempt,
            classification="terminal",
            action="abort",
            summary="该错误涉及权限、用户拒绝或不可用能力，已停止自动重试。",
            error=error,
        )
    if classification == "transient":
        return StepRecovery(
            attempt=result.attempt,
            classification="transient",
            action="retry",
            summary="检测到临时网络或服务错误，将在预算内使用原参数重试一次。",
            error=error,
        )

    allowed_inputs = RECOVERY_INPUT_RULES.get(step.kind, {})
    prompt = json.dumps(
        {
            "user_objective": state.get("request", {}).get("message", ""),
            "asset_type": state.get("request", {}).get("asset_type", "stock"),
            "tickers": state.get("request", {}).get("tickers", []),
            "failed_step": step.model_dump(mode="json"),
            "failed_tool_call": result.failure_context,
            "attempt": result.attempt,
            "error": error,
            "classification": classification,
            "allowed_input_patch": allowed_inputs,
        },
        ensure_ascii=False,
        default=str,
    )
    try:
        raw = await get_llm_service().chat_json(
            prompt,
            system=(
                "你是工具失败恢复控制器。只返回 JSON："
                "{action:'retry|adjust|abort',summary:'公开的简短调整说明',input_patch:{}}。"
                "不要输出内部思维链。必须保持用户标的、资产类型和研究目标，不得请求实盘交易或扩大权限；"
                "input_patch 只能使用 allowed_input_patch 中的字段。参数可修正时选 adjust；"
                "原样重试可能成功时选 retry；无法安全恢复时选 abort。"
            ),
        )
    except Exception as exc:
        logger.warning("Research recovery reflection failed; using bounded retry: {}", exc)
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    action = str(raw.get("action") or "retry").lower()
    if action not in {"retry", "adjust", "abort"}:
        action = "retry"
    patch = _sanitize_recovery_patch(step, raw.get("input_patch"), state.get("request", {}))
    if action == "adjust" and not patch:
        action = "retry"
    summary = str(raw.get("summary") or "").strip()[:500] or "已检查失败原因，将在安全预算内重试一次。"
    return StepRecovery(
        attempt=result.attempt,
        classification=classification,
        action=action,
        summary=summary,
        error=error,
        input_patch=patch if action == "adjust" else {},
    )


async def _call_tool(context: ResearchPlanContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    tool = context.tools.get(name)
    if tool is None:
        raise ResearchToolExecutionError(name, args, f"研究步骤需要的工具不可用: {name}")
    timeout = tool_timeout_seconds(name)
    try:
        raw = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError as exc:
        raise ResearchToolExecutionError(name, args, f"工具 {name} 执行超时") from exc
    except Exception as exc:
        raise ResearchToolExecutionError(name, args, str(exc)[:500] or type(exc).__name__) from exc
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError:
        payload = {"value": str(raw)}
    if isinstance(payload, dict) and payload.get("ok") is False:
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        raise ResearchToolExecutionError(name, args, str(message or "工具返回失败状态"))
    return payload if isinstance(payload, dict) else {"value": payload}


async def _execute_step(
    step: ResearchStep,
    state: ResearchPlanState,
    context: ResearchPlanContext,
) -> dict[str, Any]:
    request = state["request"]
    tickers = [str(item) for item in request.get("tickers", [])]
    ticker = tickers[0] if tickers else ""
    asset_type = str(request.get("asset_type", "stock"))
    common = {"ticker": ticker, "asset_type": asset_type}
    if step.kind == "data_catalog":
        return await _call_tool(
            context,
            "search_market_data_catalog",
            {"query": str(request.get("message") or ""), "asset_type": asset_type, "limit": 5},
        )
    if step.kind == "data_query":
        spec = state.get("financial_task_spec") or {}
        if not spec:
            raise ValueError("结构化数据步骤缺少 FinancialTaskSpec")
        return await _call_tool(context, "query_market_data", {"task_spec": spec})
    if step.kind == "data_validation":
        query_output = next(
            (
                item.get("output")
                for item in state.get("step_results", {}).values()
                if isinstance(item, dict)
                and isinstance(item.get("output"), dict)
                and item["output"].get("data_type") == "market_dataset"
            ),
            {},
        )
        acceptance = query_output.get("acceptance") or {}
        return {
            "data_type": "task_acceptance",
            "outcome_status": acceptance.get("status", "invalid_result"),
            "satisfied": acceptance.get("satisfied", False),
            "checks": acceptance.get("checks", []),
            "issues": acceptance.get("issues", []),
            "coverage": query_output.get("coverage", {}),
            "provenance": query_output.get("provenance", []),
        }
    if step.kind in {"instrument_profile", "fundamentals"}:
        return await _call_tool(context, "get_fundamentals", common)
    if step.kind in {"market_snapshot", "liquidity"}:
        if len(tickers) > 1:
            return await _call_tool(context, "compare_quotes", {"tickers": tickers, "asset_type": asset_type})
        return await _call_tool(context, "get_realtime_quote", common)
    if step.kind == "price_history":
        limit = int(step.inputs.get("limit") or 120)
        date_args = {
            key: value
            for key in ("start_date", "end_date")
            if (value := request.get(key) or step.inputs.get(key))
        }
        payloads = await asyncio.gather(
            *(
                _call_tool(
                    context,
                    "get_historical_prices",
                    {"ticker": item, "asset_type": asset_type, "limit": limit, **date_args},
                )
                for item in tickers[:10]
            )
        )
        return {"data_type": "price_history_collection", "items": payloads}
    if step.kind == "fund_nav":
        limit = int(step.inputs.get("limit") or 120)
        payloads = await asyncio.gather(
            *(
                _call_tool(
                    context,
                    "get_fund_nav_history",
                    {"ticker": item, "asset_type": asset_type, "limit": limit},
                )
                for item in tickers[:10]
            )
        )
        return {"data_type": "fund_nav_collection", "items": payloads}
    if step.kind == "technical":
        limit = int(step.inputs.get("limit") or 120)
        payloads = await asyncio.gather(
            *(
                _call_tool(
                    context,
                    "compute_technical_indicators",
                    {"ticker": item, "asset_type": asset_type, "limit": limit},
                )
                for item in tickers[:10]
            )
        )
        return {"data_type": "technical_collection", "items": payloads}
    if step.kind == "news":
        query = str(
            step.inputs.get("query")
            or f"{' '.join(tickers)} {asset_type} 最新新闻 公告 风险 催化 {request.get('message', '')}"
        )
        return await _call_tool(
            context,
            "search_web",
            {
                "query": query,
                "num_results": int(step.inputs.get("num_results") or 10),
                "freshness": str(step.inputs.get("freshness") or "qdr:m"),
            },
        )
    if step.kind == "methodology":
        if str(request.get("intent")) == "strategies":
            return await _call_tool(context, "list_trading_strategies", {})
        return await _call_tool(
            context,
            "search_methodology",
            {
                "query": str(step.inputs.get("query") or request.get("message", "")),
                "asset_type": asset_type,
                "limit": int(step.inputs.get("limit") or 5),
            },
        )
    if step.kind == "comparison":
        return await _call_tool(context, "compare_quotes", {"tickers": tickers, "asset_type": asset_type})
    if step.kind == "backtest":
        original_objective = str(request.get("message", ""))
        explicit_dates = re.findall(r"\d{4}-\d{2}-\d{2}", original_objective)
        explicit_start = explicit_end = None
        if len(explicit_dates) >= 2:
            try:
                first_date = date.fromisoformat(explicit_dates[0])
                second_date = date.fromisoformat(explicit_dates[1])
                if first_date < second_date:
                    explicit_start, explicit_end = first_date.isoformat(), second_date.isoformat()
            except ValueError:
                pass
        end_date = str(
            explicit_end or step.inputs.get("end_date") or request.get("as_of_date") or date.today().isoformat()
        )
        objective = str(step.inputs.get("objective") or request.get("message", ""))
        contract_operation = str((state.get("task_contract") or {}).get("operation") or "")
        compares_strategies = contract_operation == "strategy_comparison" or (
            "策略" in original_objective
            and any(term in original_objective for term in ("不同", "多个", "几个", "多种", "对比", "比较"))
        )
        sandbox_requested = contract_operation == "sandbox_research" or any(
            term in original_objective.lower() for term in ("python", "代码", "沙盒", "自定义因子")
        )
        execution_mode = str(step.inputs.get("execution_mode") or "").lower()
        if contract_operation not in {"strategy_comparison", "sandbox_research"}:
            if execution_mode == "comparison":
                compares_strategies = True
                sandbox_requested = False
            elif execution_mode == "sandbox":
                sandbox_requested = True
                compares_strategies = False
            elif execution_mode == "agent":
                sandbox_requested = False
                compares_strategies = False
        default_days = 3652 if compares_strategies else 1826 if sandbox_requested else 365
        start_date = str(
            explicit_start
            or step.inputs.get("start_date")
            or (date.fromisoformat(end_date) - timedelta(days=default_days)).isoformat()
        )
        shared_backtest_args = {
            "start_date": start_date,
            "end_date": end_date,
            "asset_type": asset_type,
        }
        if "initial_capital" in step.inputs:
            shared_backtest_args["initial_capital"] = float(step.inputs["initial_capital"])
        interval_args = (
            {"decision_interval": int(step.inputs["decision_interval"])} if "decision_interval" in step.inputs else {}
        )
        if len(tickers) == 1 and sandbox_requested:
            return await _call_tool(
                context,
                "design_and_run_sandbox_strategy",
                {
                    "objective": objective,
                    "ticker": ticker,
                    **shared_backtest_args,
                },
            )
        if len(tickers) == 1 and compares_strategies:
            return await _call_tool(
                context,
                "compare_strategy_backtests",
                {
                    "ticker": ticker,
                    "objective": objective,
                    **interval_args,
                    **shared_backtest_args,
                },
            )
        if len(tickers) > 1:
            return await _call_tool(
                context,
                "design_and_run_backtest",
                {
                    "objective": objective,
                    "tickers": tickers,
                    "mode": "pool",
                    **interval_args,
                    **shared_backtest_args,
                },
            )
        return await _call_tool(
            context,
            "design_and_run_backtest",
            {
                "objective": objective,
                "ticker": ticker,
                **interval_args,
                **shared_backtest_args,
            },
        )
    if step.kind == "comprehensive_analysis":
        return await _call_tool(context, "run_fund_or_stock_analysis", common)
    if step.kind == "risk":
        price = _find_price(state.get("step_results", {}))
        if not price:
            return {"data_type": "risk", "status": "partial", "message": "缺少有效当前价格，无法计算价格型风险指标"}
        return await _call_tool(
            context,
            "calculate_risk_metrics",
            {
                "current_price": price,
                **{
                    key: step.inputs[key]
                    for key in ("stop_loss_pct", "take_profit_pct", "position_size_pct")
                    if key in step.inputs
                },
            },
        )
    if step.kind == "synthesis":
        dataset_result = next(
            (
                item.get("output")
                for item in state.get("step_results", {}).values()
                if isinstance(item, dict)
                and isinstance(item.get("output"), dict)
                and item["output"].get("data_type") == "market_dataset"
            ),
            None,
        )
        if dataset_result is not None:
            acceptance = dataset_result.get("acceptance") or {}
            coverage = dataset_result.get("coverage") or {}
            preview = [item for item in dataset_result.get("preview", []) if isinstance(item, dict)]
            columns = [str(item) for item in dataset_result.get("schema_fields", [])]
            periods = [str(item) for item in coverage.get("requested_periods", [])]
            status = str(acceptance.get("status") or "invalid_result")
            if status != "satisfied":
                issues = "；".join(str(item) for item in acceptance.get("issues", [])) or "业务验收条件未全部满足"
                text = (
                    f"本次结构化查询已执行，但业务结果状态为 `{status}`，因此不输出可能误导的完整排名。\n\n"
                    f"覆盖情况：请求 {periods or '—'}，实际返回 {coverage.get('returned_periods') or '—'}；{issues}。"
                )
            else:
                headers = columns or (list(preview[0]) if preview else [])
                lines = [
                    "结构化筛选与业务验收均已通过。以下按近6个完整财年的累计税前每股现金分红从高到低排列：",
                    "",
                    "| 排名 | " + " | ".join(headers) + " |",
                    "| ---: | " + " | ".join("---" for _ in headers) + " |",
                ]
                for rank, row in enumerate(preview, 1):
                    values = []
                    for column in headers:
                        value = row.get(column, "")
                        values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
                    lines.append(f"| {rank} | " + " | ".join(values) + " |")
                artifacts = dataset_result.get("artifacts") or []
                lines.extend(
                    [
                        "",
                        f"覆盖：{coverage.get('requested_periods')}；源记录 {coverage.get('source_rows', 0)} 条；"
                        f"满足连续分红条件 {coverage.get('result_rows', 0)} 只。",
                        "口径：仅统计已实施的税前现金分红，单位为元/股；同一报告年度多次实施方案合并求和；"
                        "送股、转增和仅预案未实施的方案不计入。",
                        "来源：AkShare / 东方财富结构化分红数据。"
                        + (f"完整 CSV 已生成（{len(artifacts)} 个产物）。" if artifacts else "本次未生成完整 CSV。"),
                    ]
                )
                text = "\n".join(lines)
            return {
                "data_type": "synthesis",
                "text": text,
                "outcome_status": status,
                "acceptance": acceptance,
                "provenance": {"source": "deterministic_dataset_synthesis"},
            }
        comparison = next(
            (
                item.get("output")
                for item in state.get("step_results", {}).values()
                if isinstance(item, dict)
                and isinstance(item.get("output"), dict)
                and item["output"].get("data_type") == "strategy_backtest_comparison"
            ),
            None,
        )
        if comparison:
            return {
                "data_type": "synthesis",
                "text": _comparison_synthesis_text(comparison),
                "provenance": {"source": "deterministic_comparison_conclusion"},
            }
        evidence = _result_summaries(state.get("step_results", {}))
        system = (
            "基于给定证据生成简洁中文市场研究结论。明确数据日期、缺失和风险；"
            "直接给出你的首选建议、证据和适用条件，最终选择交给用户；"
            "不要用‘不存在唯一最好方案’、‘不同指标代表不同取舍’等常识性段落代替建议；"
            "工具已正常返回但 evidence_status 为 limited 或 unavailable 时，应说明限制并继续基于其余证据判断，"
            "不得把无数据或验收条件未满足描述成工具调用失败；"
            "面向短中期基金交易研究，不承诺收益；股票分析只能表述为底层资产研究。"
        )
        try:
            context_budget = get_context_budget()
            counter = TokenCounter(context_budget.model)
            compact_request = _synthesis_request(request)
            empty_prompt = json.dumps(
                {"request": compact_request, "evidence": []},
                ensure_ascii=False,
                default=str,
            )
            fixed_tokens = counter.count_messages(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": empty_prompt},
                ]
            )
            evidence_budget = max(256, context_budget.input_limit - fixed_tokens - 512)
            compacted_evidence = compact_json_value(evidence, evidence_budget, counter=counter)
            prompt = json.dumps(
                {"request": compact_request, "evidence": compacted_evidence},
                ensure_ascii=False,
                default=str,
            )
            text = await get_llm_service().chat(prompt, system=system)
            source = "derived"
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise
            logger.warning("Synthesis context overflow; using deterministic evidence summary: {}", exc)
            text = _deterministic_synthesis_fallback(evidence)
            source = "deterministic_context_fallback"
        return {"data_type": "synthesis", "text": text, "provenance": {"source": source}}
    if step.kind == "report":
        summaries = _result_summaries(state.get("step_results", {}))
        content = "# 市场研究报告\n\n" + "\n\n".join(
            f"## {item['step_id']}\n{item.get('summary') or ''}" for item in summaries
        )
        return await _call_tool(
            context,
            "save_artifacts",
            {
                "artifacts": [
                    {
                        "name": f"{ResearchPlan.model_validate(state['plan']).plan_id}-研究报告.md",
                        "format": "md",
                        "content": content,
                        "description": "可审计市场研究计划报告",
                        "asset_type": asset_type,
                        "ticker": ticker or None,
                    }
                ],
                "execution_key": f"{request.get('task_id')}:{step.id}",
            },
        )
    raise ValueError(f"不支持的研究步骤: {step.kind}")


async def run_worker(
    state: ResearchPlanState,
    runtime: Runtime[ResearchPlanContext],
) -> dict[str, Any]:
    step = ResearchStep.model_validate(state["step"])
    prior = StepResult.model_validate(
        (state.get("step_results") or {}).get(step.id)
        or {
            "step_id": step.id,
            "status": "pending",
        }
    )
    attempt = prior.attempt + 1
    budget = ResearchBudget.model_validate(state["budget"])
    if state.get("tool_calls", 0) >= budget.max_tool_calls:
        result = StepResult(step_id=step.id, status="skipped", attempt=attempt, error="研究任务已耗尽工具调用预算")
        return {"step_results": {step.id: result.model_dump(mode="json")}}
    if datetime.now(timezone.utc) > datetime.fromisoformat(state["deadline_at"]):
        result = StepResult(step_id=step.id, status="skipped", attempt=attempt, error="研究任务已超过执行截止时间")
        return {"step_results": {step.id: result.model_dump(mode="json")}}
    try:
        payload = await _execute_step(step, state, runtime.context)
        evidence = _evidence(payload, step.kind)
        artifacts = [
            str(item.get("artifact_id"))
            for item in payload.get("artifacts", [])
            if isinstance(item, dict) and item.get("artifact_id")
        ]
        summary = str(payload.get("text") or payload.get("message") or payload.get("data_type") or step.title)[:1000]
        result = StepResult(
            step_id=step.id,
            status="completed",
            attempt=attempt,
            summary=summary,
            evidence=evidence,
            artifact_ids=artifacts,
            output=_compact(payload),
            failure_context=prior.failure_context,
            recovery_history=prior.recovery_history,
        )
        return {
            "step_results": {step.id: result.model_dump(mode="json")},
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "tool_calls": 1,
        }
    except Exception as exc:
        failure_context = (
            {"tool_name": exc.tool_name, "args": exc.tool_args} if isinstance(exc, ResearchToolExecutionError) else {}
        )
        _, error_message = context_safe_error(exc, str(exc)[:500])
        result = StepResult(
            step_id=step.id,
            status="failed",
            attempt=attempt,
            summary=f"{step.title}执行失败",
            error=error_message,
            failure_context=failure_context,
            recovery_history=prior.recovery_history,
        )
        return {
            "step_results": {step.id: result.model_dump(mode="json")},
            "tool_calls": 1,
        }


async def verify_evidence(state: ResearchPlanState) -> dict[str, Any]:
    """Annotate evidence quality without rewriting successful tool calls as failures."""
    plan = ResearchPlan.model_validate(state["plan"])
    results = {key: StepResult.model_validate(value) for key, value in (state.get("step_results") or {}).items()}
    evidence_kinds = {
        "instrument_profile",
        "market_snapshot",
        "price_history",
        "fund_nav",
        "liquidity",
        "technical",
        "fundamentals",
        "news",
        "methodology",
        "backtest",
        "comparison",
        "comprehensive_analysis",
        "data_catalog",
        "data_query",
        "data_validation",
    }
    updates: dict[str, dict[str, Any]] = {}
    for step in plan.steps:
        result = results.get(step.id)
        if result is None or result.status != "completed" or step.kind not in evidence_kinds:
            continue
        output = result.output
        issues: list[str] = []
        unavailable = output.get("available") is False
        if not result.evidence:
            issues.append("缺少可审计来源信息")
        elif any(not item.source or not item.retrieved_at for item in result.evidence):
            issues.append("部分证据缺少来源或检索时间")
        evidence_statuses = {item.data_status for item in result.evidence}
        if evidence_statuses & {"unavailable", "failed"}:
            issues.append("一个或多个数据源未返回可用数据")
        elif evidence_statuses & {"degraded", "partial", "unverified"}:
            issues.append("一个或多个数据源仅提供降级或部分数据")

        if unavailable:
            error = output.get("error")
            message = error.get("message") if isinstance(error, dict) else error
            issues.append(str(message or output.get("message") or "工具正常返回，但没有可用数据"))

        items = output.get("items") or []
        if items:
            missing_items = [item for item in items if isinstance(item, dict) and item.get("available") is False]
            if missing_items:
                unavailable = len(missing_items) == len(items)
                issues.append(
                    "所有标的均无可用数据" if unavailable else f"{len(missing_items)} 个标的缺少可用数据"
                )

        if step.kind == "news":
            if not (output.get("results") or output.get("news")):
                unavailable = True
                issues.append("未检索到相关资讯")
        elif step.kind == "price_history":
            covered = [item for item in items if isinstance(item, dict) and item.get("history")]
            if not covered:
                unavailable = True
                issues.append("未取得可用历史价格")
            elif len(covered) < len(items):
                issues.append(f"{len(items) - len(covered)} 个标的缺少历史价格")
        elif step.kind == "fund_nav":
            covered = [item for item in items if isinstance(item, dict) and item.get("history")]
            if not covered:
                unavailable = True
                issues.append("未取得可用基金净值")
            elif len(covered) < len(items):
                issues.append(f"{len(items) - len(covered)} 个标的缺少基金净值")
        elif step.kind == "technical":
            covered = [item for item in items if isinstance(item, dict) and item.get("indicators")]
            if not covered:
                unavailable = True
                issues.append("未取得可用技术指标")
            elif len(covered) < len(items):
                issues.append(f"{len(items) - len(covered)} 个标的缺少技术指标")
        elif step.kind == "methodology":
            if not (output.get("results") or output.get("strategies")):
                unavailable = True
                issues.append("未检索到相关方法论")
        elif step.kind == "market_snapshot":
            if not (output.get("quote") or output.get("quotes") or output.get("results")):
                unavailable = True
                issues.append("未取得可用行情快照")
        elif step.kind == "comparison":
            quotes = output.get("quotes") or []
            covered = [item for item in quotes if isinstance(item, dict) and item.get("quote")]
            if not covered:
                unavailable = True
                issues.append("所有对比标的均缺少行情")
            elif len(covered) < len(quotes):
                issues.append(f"{len(quotes) - len(covered)} 个对比标的缺少行情")
        elif step.kind == "backtest" and output.get("data_type") == "strategy_backtest_comparison":
            acceptance = output.get("acceptance") or {}
            if acceptance.get("satisfied") is False:
                missing = [str(item) for item in acceptance.get("missing", [])]
                issues.append("任务验收条件未满足" + (f": {', '.join(missing)}" if missing else ""))
        elif step.kind == "backtest" and output.get("data_type") == "sandbox_strategy_candidate":
            sandbox_backtest = output.get("result", {}).get("backtest", {})
            if output.get("validation", {}).get("passed") is not True:
                issues.append("沙盒候选未通过全部验证")
            if not output.get("candidate_id") or sandbox_backtest.get("final_value") is None:
                unavailable = True
                issues.append("沙盒候选缺少可信回测结果")
        elif step.kind == "data_catalog":
            if not output.get("matches"):
                unavailable = True
                issues.append("数据目录未找到匹配数据集")
        elif step.kind == "data_query":
            acceptance = output.get("acceptance") or {}
            if acceptance.get("status") == "data_unavailable":
                unavailable = True
                issues.append("结构化数据源没有返回请求周期数据")
            elif acceptance.get("status") != "satisfied":
                issues.extend(str(item) for item in acceptance.get("issues", []))
        elif step.kind == "data_validation":
            if output.get("outcome_status") == "data_unavailable":
                unavailable = True
            elif output.get("outcome_status") != "satisfied":
                issues.extend(str(item) for item in output.get("issues", []))
        requires_as_of = step.kind in {"market_snapshot", "price_history", "fund_nav", "news", "comparison"}
        has_as_of = not requires_as_of or all(item.as_of for item in result.evidence)
        if not has_as_of:
            issues.append("部分证据缺少数据截止时间")
        issues = list(dict.fromkeys(issues))
        evidence_status = "unavailable" if unavailable else "limited" if issues else "sufficient"
        if result.evidence_status != evidence_status or result.evidence_issues != issues:
            updates[step.id] = result.model_copy(
                update={"evidence_status": evidence_status, "evidence_issues": issues}
            ).model_dump(mode="json")
    results.update({key: StepResult.model_validate(value) for key, value in updates.items()})
    failed = [step for step in plan.steps if step.id in results and results[step.id].status == "failed"]
    retryable = [step for step in failed if results[step.id].attempt < _effective_max_attempts(step)]
    budget = ResearchBudget.model_validate(state["budget"])
    exhausted = state.get("tool_calls", 0) >= budget.max_tool_calls
    expired = datetime.now(timezone.utc) > datetime.fromisoformat(state["deadline_at"])
    if exhausted or expired:
        reason = "研究任务已耗尽工具调用预算" if exhausted else "研究任务已超过执行截止时间"
        for step in plan.steps:
            result = results.get(step.id)
            if result is None or result.status == "pending":
                updates[step.id] = StepResult(
                    step_id=step.id,
                    status="skipped",
                    attempt=result.attempt if result else 0,
                    error=reason,
                ).model_dump(mode="json")
        results.update({key: StepResult.model_validate(value) for key, value in updates.items()})
        failed = [step for step in plan.steps if step.id in results and results[step.id].status == "failed"]
        retryable = []
    return {
        "step_results": updates,
        "needs_replan": bool(retryable and state.get("replan_count", 0) < budget.max_replans),
    }


def route_after_verify(state: ResearchPlanState) -> str:
    if state.get("needs_replan"):
        return "replan"
    plan = ResearchPlan.model_validate(state["plan"])
    results = {key: StepResult.model_validate(value) for key, value in (state.get("step_results") or {}).items()}
    terminal = {"completed", "failed", "skipped"}
    if all(step.id in results and results[step.id].status in terminal for step in plan.steps):
        return "finish"
    completed = {step_id for step_id, result in results.items() if result.status == "completed"}
    if any(
        (step.id not in results or results[step.id].status == "pending") and set(step.depends_on) <= completed
        for step in plan.steps
    ):
        return "dispatch"
    return "skip_blocked"


async def replan(state: ResearchPlanState) -> dict[str, Any]:
    plan = ResearchPlan.model_validate(state["plan"])
    results = {key: StepResult.model_validate(value) for key, value in (state.get("step_results") or {}).items()}
    updates: dict[str, dict[str, Any]] = {}
    revised_steps: list[ResearchStep] = []
    for step in plan.steps:
        result = results.get(step.id)
        max_attempts = _effective_max_attempts(step)
        if result and result.status == "failed" and result.attempt < max_attempts:
            recovery = await _reflect_on_failure(step, result, state)
            history = [*result.recovery_history, recovery]
            if recovery.action == "abort":
                updates[step.id] = result.model_copy(
                    update={
                        "attempt": max_attempts,
                        "summary": recovery.summary,
                        "recovery_history": history,
                    }
                ).model_dump(mode="json")
            else:
                if recovery.action == "adjust":
                    step = step.model_copy(update={"inputs": {**step.inputs, **recovery.input_patch}})
                updates[step.id] = result.model_copy(
                    update={
                        "status": "pending",
                        "summary": recovery.summary,
                        "error": None,
                        "recovery_history": history,
                    }
                ).model_dump(mode="json")
        revised_steps.append(step)
    plan = plan.model_copy(update={"steps": tuple(revised_steps), "revision": plan.revision + 1})
    _validate_plan_contract(plan, state["request"], ResearchBudget.model_validate(state["budget"]))
    return {
        "plan": plan.model_dump(mode="json"),
        "step_results": updates,
        "replan_count": state.get("replan_count", 0) + 1,
        "needs_replan": False,
    }


async def skip_blocked(state: ResearchPlanState) -> dict[str, Any]:
    plan = ResearchPlan.model_validate(state["plan"])
    results = {key: StepResult.model_validate(value) for key, value in (state.get("step_results") or {}).items()}
    updates = {}
    for step in plan.steps:
        if step.id not in results or results[step.id].status == "pending":
            updates[step.id] = StepResult(
                step_id=step.id,
                status="skipped",
                error="依赖步骤未成功，已跳过",
            ).model_dump(mode="json")
    return {"step_results": updates}


async def finish_research(state: ResearchPlanState) -> dict[str, Any]:
    plan = ResearchPlan.model_validate(state["plan"])
    results = {key: StepResult.model_validate(value) for key, value in (state.get("step_results") or {}).items()}
    synthesis = next(
        (
            result.output.get("text")
            for step in plan.steps
            if step.kind == "synthesis"
            and (result := results.get(step.id)) is not None
            and result.status == "completed"
        ),
        None,
    )
    if synthesis:
        response = str(synthesis)
    else:
        completed = [
            step.title for step in plan.steps if results.get(step.id) and results[step.id].status == "completed"
        ]
        failed = [
            step.title for step in plan.steps if not results.get(step.id) or results[step.id].status != "completed"
        ]
        response = f"已完成：{'、'.join(completed) or '无'}。"
        if failed:
            response += f" 数据不足或失败：{'、'.join(failed)}。"
        response += "以上仅用于短中期研究与模拟交易，不构成收益承诺。"
    return {"final_response": response}


def build_research_plan_graph(checkpointer: Any | None = None):
    graph = StateGraph(ResearchPlanState, context_schema=ResearchPlanContext)
    graph.add_node("scope", scope_research)
    graph.add_node("planner", plan_research)
    graph.add_node("validate", validate_plan)
    graph.add_node("dispatch", dispatch_ready)
    graph.add_node("worker", run_worker)
    graph.add_node("verify", verify_evidence)
    graph.add_node("replan", replan)
    graph.add_node("skip_blocked", skip_blocked)
    graph.add_node("finish", finish_research)
    graph.add_edge(START, "scope")
    graph.add_edge("scope", "planner")
    graph.add_edge("planner", "validate")
    graph.add_edge("validate", "dispatch")
    graph.add_conditional_edges("dispatch", route_dispatch, ["worker", "verify"])
    graph.add_edge("worker", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"replan": "replan", "dispatch": "dispatch", "skip_blocked": "skip_blocked", "finish": "finish"},
    )
    graph.add_edge("replan", "dispatch")
    graph.add_edge("skip_blocked", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


research_plan_graph = build_research_plan_graph()


def configure_research_plan_graph(checkpointer: Any | None) -> None:
    global research_plan_graph
    research_plan_graph = build_research_plan_graph(checkpointer)


def get_research_plan_graph():
    return research_plan_graph
