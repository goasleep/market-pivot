"""Reusable LangGraph agent loop for LLM-directed tool use."""

from __future__ import annotations

import asyncio
import json
import operator
import re
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from loguru import logger

from application.fund_completion import validate_fund_response
from llm.context import context_safe_error, is_context_overflow_error
from llm.service import get_llm_service
from models.fund_task import FundIntentSpec, FundTaskKind
from models.supervisor import CompletionResult, SupervisorOutcome
from tools.policies import tool_requires_confirmation

DEFAULT_MAX_STEPS = 100
MAX_EXECUTION_MINUTES = 30
MAX_EXECUTION_SECONDS = MAX_EXECUTION_MINUTES * 60
LLM_TIMEOUT_SECONDS = MAX_EXECUTION_SECONDS
TOOL_TIMEOUT_SECONDS = MAX_EXECUTION_SECONDS
RESEARCH_PLAN_TIMEOUT_SECONDS = MAX_EXECUTION_SECONDS
LONG_RUNNING_TOOL_TIMEOUT_SECONDS = MAX_EXECUTION_SECONDS
_UNFINISHED_ANSWER = re.compile(r"下一步(?:需要|要)|进一步(?:校准|确认)?需|仍需(?:查询|查找|核对)|待(?:查询|核对|确认)")
_TOOL_OBSERVATION_LIMIT = 6000
_REUSABLE_BACKTEST_TOOLS = {
    "run_backtest",
    "design_and_run_backtest",
    "compare_strategy_backtests",
    "design_and_run_sandbox_strategy",
}


def _compact_json_value(value: Any, *, depth: int = 0) -> Any:
    """Keep model-facing observations bounded while preserving audit results elsewhere."""
    if depth >= 5:
        return str(value)[:500]
    if isinstance(value, dict):
        return {str(key): _compact_json_value(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) <= 8:
            return [_compact_json_value(item, depth=depth + 1) for item in value]
        return {
            "_truncated": True,
            "count": len(value),
            "first": [_compact_json_value(item, depth=depth + 1) for item in value[:2]],
            "latest": [_compact_json_value(item, depth=depth + 1) for item in value[-5:]],
        }
    if isinstance(value, str) and len(value) > 1000:
        return f"{value[:900]}…[truncated {len(value) - 900} chars]"
    return value


def compact_tool_observation(result: str, *, limit: int = _TOOL_OBSERVATION_LIMIT) -> str:
    """Compress only the ToolMessage sent back to the LLM; tool events retain the full payload."""
    if len(result) <= limit:
        return result
    try:
        payload = json.loads(result)
    except (TypeError, ValueError, json.JSONDecodeError):
        return f"{result[: limit - 80]}…[full result retained in Evidence Store]"
    compacted = json.dumps(_compact_json_value(payload), ensure_ascii=False, default=str)
    if len(compacted) <= limit:
        return compacted
    metadata = {
        key: payload.get(key)
        for key in (
            "data_type",
            "status",
            "data_status",
            "available",
            "ticker",
            "asset_type",
            "as_of",
            "fetched_at",
            "freshness",
            "sources",
            "provenance",
            "errors",
            "error",
        )
        if isinstance(payload, dict) and key in payload
    }
    metadata["_observation_truncated"] = True
    metadata["summary"] = compacted[: max(0, limit - 1200)]
    return json.dumps(metadata, ensure_ascii=False, default=str)[:limit]


def tool_timeout_seconds(name: str) -> int:
    """Return the execution budget for a tool invocation."""
    if name == "run_research_plan":
        return RESEARCH_PLAN_TIMEOUT_SECONDS
    if name in {
        "run_stock_comprehensive_analysis",
        "run_backtest",
        "design_and_run_backtest",
        "compare_strategy_backtests",
        "design_and_run_sandbox_strategy",
        "query_market_data",
    }:
        return LONG_RUNNING_TOOL_TIMEOUT_SECONDS
    return TOOL_TIMEOUT_SECONDS


def tool_attempts(name: str) -> int:
    """Return retry count without duplicating expensive or mutating work."""
    if name in {
        "run_research_plan",
        "run_stock_comprehensive_analysis",
        "run_backtest",
        "design_and_run_backtest",
        "compare_strategy_backtests",
        "design_and_run_sandbox_strategy",
        "query_market_data",
    }:
        return 1
    return 1 if name.startswith(("submit_", "cancel_", "create_", "fill_")) else 2


def _tool_result_is_reusable(event: dict[str, Any]) -> bool:
    if event.get("status") != "completed":
        return False
    try:
        payload = json.loads(str(event.get("result") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return not (
        payload.get("available") is False
        or payload.get("data_status") == "unavailable"
        or payload.get("error")
    )


def _same_backtest_scope(designed_args: dict[str, Any], requested_args: dict[str, Any]) -> bool:
    requested_ticker = str(requested_args.get("ticker") or "")
    designed_tickers = {
        str(item)
        for item in [designed_args.get("ticker"), *(designed_args.get("tickers") or [])]
        if item
    }
    if (
        not requested_ticker
        or designed_tickers != {requested_ticker}
        or str(designed_args.get("mode") or "auto") not in {"auto", "single"}
    ):
        return False
    return all(
        str(designed_args.get(field) or default) == str(requested_args.get(field) or default)
        for field, default in (
            ("start_date", ""),
            ("end_date", ""),
            ("asset_type", "stock"),
            ("initial_capital", 1_000_000),
            ("decision_interval", 1),
        )
    )


def _find_reusable_tool_event(
    name: str,
    args: dict[str, Any],
    tool_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return prior evidence that makes an identical expensive call redundant."""
    if name not in _REUSABLE_BACKTEST_TOOLS:
        return None
    for event in reversed(tool_events):
        if not _tool_result_is_reusable(event):
            continue
        prior_name = str(event.get("name") or "")
        prior_args = event.get("args")
        if not isinstance(prior_args, dict):
            continue
        if prior_name == name and prior_args == args:
            return event
        if (
            name == "run_backtest"
            and prior_name == "design_and_run_backtest"
            and _same_backtest_scope(prior_args, args)
        ):
            return event
    return None


class AgentLoopState(TypedDict, total=False):
    messages: list[Any]
    step: int
    max_steps: int
    tool_events: Annotated[list[dict[str, Any]], operator.add]
    reasoning_events: Annotated[list[dict[str, Any]], operator.add]
    final_response: str
    candidate_response: str
    task_contract: dict[str, Any]
    completion_result: dict[str, Any]
    judge_events: Annotated[list[dict[str, Any]], operator.add]
    max_steps_reached: bool
    pending_tool_confirmation: dict[str, Any]
    checkpoint_messages: list[dict[str, Any]]


@dataclass(frozen=True)
class AgentLoopContext:
    """Run-scoped dependencies that must never be written to checkpoints."""

    tools: list[StructuredTool]
    tool_map: dict[str, StructuredTool]
    native_interrupts: bool = False
    task_id: str | None = None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content or "")


def _fund_screening_acceptance(contract: dict[str, Any], candidate: str):
    raw_spec = contract.get("source_task_spec")
    if not isinstance(raw_spec, dict):
        return None
    try:
        spec = FundIntentSpec.model_validate(raw_spec)
    except Exception:
        return None
    if spec.task_kind != FundTaskKind.UNIVERSE_RESEARCH:
        return None
    return spec, validate_fund_response(spec, candidate)


def _screened_candidate_count(tool_events: list[dict[str, Any]]) -> int | None:
    counts: list[int] = []
    for event in tool_events:
        if event.get("name") != "screen_assets" or event.get("status") != "completed":
            continue
        try:
            payload = json.loads(str(event.get("result") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("count"), int):
            counts.append(payload["count"])
    return max(counts) if counts else None


def _candidate_shortage_is_disclosed(candidate: str) -> bool:
    return bool(re.search(r"(?:只有|仅有|仅|不足|没有更多|暂无更多).{0,16}(?:只|个|候选|符合|合格)", candidate))


def _message_objects(messages: list[Any]) -> list[Any]:
    """Normalize the app's role dictionaries before checkpoint serialization."""
    normalized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            normalized.append(SystemMessage(content=content))
        elif role == "assistant":
            normalized.append(AIMessage(content=content, tool_calls=message.get("tool_calls", [])))
        elif role == "tool":
            normalized.append(ToolMessage(content=content, tool_call_id=message.get("tool_call_id", "")))
        else:
            normalized.append(HumanMessage(content=content))
    return normalized


async def decide_next_action(
    state: AgentLoopState,
    runtime: Runtime[AgentLoopContext],
) -> dict[str, Any]:
    """Ask the model whether to answer or call one or more tools."""
    try:
        response = await asyncio.wait_for(
            get_llm_service().chat_with_tools(
                state["messages"],
                runtime.context.tools,
                temperature=0.2,
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        if not is_context_overflow_error(exc):
            raise
        logger.warning("Agent context recovery exhausted; returning a deterministic continuation: {}", exc)
        has_tool_results = any(isinstance(message, ToolMessage) for message in state["messages"])
        fallback = (
            "已完成的数据获取结果保留在上方。本轮不再追加可能失真的综合判断，"
            "你可以直接基于结构化结果继续追问某个指标或风险点。"
            if has_tool_results
            else "为了给出可靠结论，我需要先聚焦分析目标。请告诉我最关注的标的、时间范围或风险指标。"
        )
        response = AIMessage(content=fallback)
    step = state.get("step", 0) + 1
    candidate_response = _content_text(response.content)
    max_steps_reached = bool(response.tool_calls and step >= state.get("max_steps", DEFAULT_MAX_STEPS))
    reasoning_events: list[dict[str, Any]] = []
    if response.tool_calls and not max_steps_reached:
        tool_names = [str(call.get("name", "数据工具")) for call in response.tool_calls]
        model_summary = candidate_response.strip().splitlines()[0][:240] if candidate_response.strip() else ""
        summary = model_summary or f"第 {step} 轮：需要先获取 {', '.join(tool_names)} 的数据，再继续判断。"
        reasoning_events.append({"step": step, "text": summary})
    if max_steps_reached:
        candidate_response = (
            f"{candidate_response}\n\n"
            f"已达到 Agent 最大执行轮数（{state.get('max_steps', DEFAULT_MAX_STEPS)} 轮），"
            "无法继续调用工具；以下结论仅覆盖已经取得的证据。"
        ).strip()
        response = AIMessage(content=candidate_response)
    return {
        "messages": [*state["messages"], response],
        "step": step,
        "candidate_response": candidate_response if not response.tool_calls else "",
        "final_response": "",
        "reasoning_events": reasoning_events,
        "max_steps_reached": max_steps_reached,
    }


async def judge_completion(state: AgentLoopState) -> dict[str, Any]:
    """Decide whether the candidate actually satisfies the request contract."""
    candidate = str(state.get("candidate_response") or "").strip()
    contract = state.get("task_contract") or {}
    tool_events = state.get("tool_events") or []
    completed_tools = [event for event in tool_events if event.get("status") == "completed"]
    budget_exhausted = state.get("step", 0) >= state.get("max_steps", DEFAULT_MAX_STEPS)

    if not candidate:
        result = CompletionResult(
            outcome=SupervisorOutcome.PARTIAL if budget_exhausted else SupervisorOutcome.FAILED,
            satisfied=False,
            terminal=budget_exhausted,
            missing=["最终回答"],
            next_action="基于已有工具结果生成完整回答",
            reason="模型没有生成候选答案",
        )
    elif contract.get("requires_tools") and not completed_tools and not budget_exhausted:
        result = CompletionResult(
            outcome=SupervisorOutcome.PARTIAL,
            satisfied=False,
            terminal=False,
            missing=["可验证的工具证据"],
            next_action="调用合适的原子工具或研究子能力取得证据，不要只描述下一步",
            reason="任务需要外部或结构化数据，但尚无成功工具结果",
        )
    elif _UNFINISHED_ANSWER.search(candidate) and not budget_exhausted:
        result = CompletionResult(
            outcome=SupervisorOutcome.PARTIAL,
            satisfied=False,
            terminal=False,
            missing=["候选答案中声明尚待执行的工作"],
            next_action="立即执行答案中尚待查询或核对的工作，再重新生成完整答案",
            reason="候选答案仍在描述下一步，尚未真正完成任务",
        )
    else:
        deterministic_acceptance = _fund_screening_acceptance(contract, candidate)
        if deterministic_acceptance is not None and not deterministic_acceptance[1].satisfied:
            spec, acceptance = deterministic_acceptance
            selection = spec.selection_requirements
            screened_count = _screened_candidate_count(tool_events)
            disclosed_shortage = bool(
                selection is not None
                and screened_count is not None
                and screened_count < selection.minimum_candidates
                and _candidate_shortage_is_disclosed(candidate)
            )
            result = CompletionResult(
                outcome=SupervisorOutcome.PARTIAL,
                satisfied=False,
                terminal=budget_exhausted or disclosed_shortage,
                missing=acceptance.missing,
                next_action="补充候选池、横向对比、首选与备选、排除原因及数据日期；候选必须达到合同要求的数量",
                reason=(
                    f"结构化筛选仅返回 {screened_count} 个候选，少于合同要求，答案已明确披露数据不足"
                    if disclosed_shortage
                    else "基金筛选答案未通过确定性交付校验"
                ),
            )
        else:
            prompt = json.dumps(
                {
                    "task_contract": contract,
                    "candidate_answer": candidate,
                    "tool_observations": tool_events[-12:],
                    "budget_exhausted": budget_exhausted,
                },
                ensure_ascii=False,
                default=str,
            )
            try:
                payload = await asyncio.wait_for(
                    get_llm_service().chat_json(
                        prompt,
                        system=(
                            "你是 Supervisor 的完成判定器，不负责回答原问题。判断候选答案是否完成任务合同。"
                            "采用交付优先：用户明确要求的最小交付项已完成时可以结束；可选证据缺失应披露，"
                            "但不要为了追求完美阻止一个已经有用、诚实且可执行的回答。"
                            "若关键交付项仍缺失且现有工具可取得，返回terminal=false。"
                            "只有确实需要用户私有信息/授权时才 needs_input。"
                            "数据已尽力查询但不可得时可 data_unavailable。"
                            "返回字段：outcome(satisfied|partial|needs_input|data_unavailable|failed)、"
                            "satisfied、terminal、missing、next_action、reason。"
                        ),
                    ),
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                result = CompletionResult.model_validate(payload)
            except Exception as exc:
                logger.warning("Completion judge failed; using conservative fallback: {}", exc)
                result = CompletionResult(
                    outcome=SupervisorOutcome.PARTIAL,
                    satisfied=False,
                    terminal=True,
                    missing=["完成判定结果不可用"],
                    reason="完成判定器异常；已保留候选答案，但不能默认判定为完成",
                )

    if budget_exhausted and not result.terminal:
        result = result.model_copy(
            update={
                "outcome": SupervisorOutcome.PARTIAL,
                "satisfied": False,
                "terminal": True,
                "reason": f"{result.reason}；执行预算已耗尽".strip("；"),
            }
        )

    result_payload = result.model_dump(mode="json")
    if result.terminal:
        return {
            "completion_result": result_payload,
            "judge_events": [result_payload],
            "final_response": candidate,
        }

    continuation = result.next_action or "继续完成任务合同中尚未覆盖的部分"
    return {
        "messages": [
            *state["messages"],
            SystemMessage(
                content=(
                    "完成判定器认为任务尚未结束。请继续执行，不要向用户只陈述计划。"
                    f"尚缺：{'、'.join(result.missing) or '任务合同未完全满足'}。"
                    f"下一步：{continuation}。"
                )
            ),
        ],
        "completion_result": result_payload,
        "judge_events": [result_payload],
        "candidate_response": "",
        "final_response": "",
    }


async def execute_tool_calls(
    state: AgentLoopState,
    runtime: Runtime[AgentLoopContext],
    config: RunnableConfig,
) -> dict[str, Any]:
    """Execute the model's selected tools and append observations.

    LangGraph injects the current node config into this function. Passing it to
    each tool invocation keeps the LangChain callback manager attached to the
    ``execute_tools`` node, so Langfuse can build the observation hierarchy
    instead of recording tools as unparented siblings.
    """
    response = state["messages"][-1]
    if not isinstance(response, AIMessage):
        return {"tool_events": [{"name": "unknown", "status": "invalid model response"}]}

    requested_calls = response.tool_calls or []
    budget = (state.get("task_contract") or {}).get("budget") or {}
    maximum_calls = int(budget.get("max_tool_calls") or 200)
    remaining_calls = max(0, maximum_calls - len(state.get("tool_events") or []))
    calls = requested_calls[:remaining_calls]
    rejected_calls = requested_calls[remaining_calls:]
    approvals: dict[str, bool] = {}
    for call in calls:
        name = str(call.get("name", ""))
        if tool_requires_confirmation(name):
            args = call.get("args", {})
            if not isinstance(args, dict):
                args = {}
            call_id = call.get("id", "") or f"tool-call-{state.get('step', 0)}-0"
            if runtime.context.native_interrupts:
                answer = interrupt(
                    {
                        "kind": "tool_confirmation",
                        "question": "Agent 准备执行一个需要用户确认的工具操作，是否继续？",
                        "tool_name": name,
                        "tool_call_id": call_id,
                        "args": args,
                    }
                )
                approvals[call_id] = bool(
                    answer is True
                    or answer == "approve"
                    or (isinstance(answer, dict) and answer.get("approved") is True)
                )
                continue
            return {
                "pending_tool_confirmation": {
                    "tool_name": name,
                    "tool_call_id": call_id,
                    "args": args,
                },
                "checkpoint_messages": messages_to_dict(_message_objects(state["messages"])),
            }

    semaphore = asyncio.Semaphore(4)

    async def invoke_call(
        index: int,
        call: dict[str, Any],
    ) -> tuple[int, dict[str, Any] | None, ToolMessage]:
        name = call.get("name", "")
        tool = runtime.context.tool_map.get(name)
        call_id = call.get("id", "") or f"tool-call-{state.get('step', 0)}-{index}"
        if tool_requires_confirmation(name) and not approvals.get(call_id, False):
            result = json.dumps(
                {"ok": False, "error": {"code": "user_denied", "message": "用户拒绝执行该工具"}},
                ensure_ascii=False,
            )
            return index, {"name": name, "status": "failed", "result": result}, ToolMessage(
                content=result, tool_call_id=call_id
            )
        if not tool:
            result = json.dumps(
                {"ok": False, "error": {"code": "unknown_tool", "message": f"未知工具: {name}"}},
                ensure_ascii=False,
            )
            return index, {"name": name or "unknown", "status": "failed", "result": result}, ToolMessage(
                content=result, tool_call_id=call_id
            )

        args = call.get("args", {})
        if not isinstance(args, dict):
            result = json.dumps(
                {"ok": False, "error": {"code": "invalid_arguments", "message": "工具参数必须是对象"}},
                ensure_ascii=False,
            )
            return index, {"name": name, "status": "failed", "result": result}, ToolMessage(
                content=result, tool_call_id=call_id
            )
        args = dict(args)
        reusable = _find_reusable_tool_event(name, args, state.get("tool_events") or [])
        if reusable is not None:
            prior_name = str(reusable.get("name") or name)
            logger.info("Reusing completed tool evidence: {} -> {}", prior_name, name)
            result = json.dumps(
                {
                    "ok": True,
                    "reused": True,
                    "prior_tool": prior_name,
                    "message": "匹配的回测已完成；直接使用上一条工具证据生成最终回答，不要重复执行。",
                },
                ensure_ascii=False,
            )
            return index, None, ToolMessage(content=result, tool_call_id=call_id)
        if runtime.context.task_id and name in {
            "save_artifacts",
            "submit_simulation_order",
            "create_simulation_account",
            "deploy_backtest_experiment",
        }:
            args.setdefault("execution_key", f"{runtime.context.task_id}:{call_id}")

        attempts = tool_attempts(name)
        timeout_seconds = tool_timeout_seconds(name)
        result: str | None = None
        error_payload: dict[str, Any] | None = None
        async with semaphore:
            for attempt in range(attempts):
                try:
                    value = await asyncio.wait_for(
                        tool.ainvoke(args, config=config),
                        timeout=timeout_seconds,
                    )
                    result = str(value)
                    break
                except asyncio.TimeoutError:
                    error_payload = {
                        "code": "tool_timeout",
                        "message": f"工具 {name} 执行超过 {timeout_seconds} 秒",
                        "timeout_seconds": timeout_seconds,
                        "attempt": attempt + 1,
                        "attempts": attempts,
                    }
                except Exception as exc:  # Tool failures are observations, not model failures.
                    error_code, error_message = context_safe_error(exc, str(exc)[:500])
                    error_payload = {
                        "code": error_code,
                        "message": error_message,
                        "attempt": attempt + 1,
                        "attempts": attempts,
                    }
                if attempt + 1 < attempts:
                    await asyncio.sleep(0.2 * (attempt + 1))
        if result is None:
            result = json.dumps(
                {"ok": False, "error": error_payload or {"code": "tool_error", "message": "工具执行失败"}},
                ensure_ascii=False,
            )
            event = {"name": name, "status": "failed", "result": result, "args": args}
        else:
            event = {"name": name, "status": "completed", "result": result, "args": args}
        model_result = (
            compact_tool_observation(result)
            if (state.get("task_contract") or {}).get("compact_tool_results")
            else result
        )
        return index, event, ToolMessage(content=model_result, tool_call_id=call_id)

    indexed_calls = list(enumerate(calls))
    read_calls = [
        (index, call)
        for index, call in indexed_calls
        if not tool_requires_confirmation(call.get("name", ""))
    ]
    write_calls = [(index, call) for index, call in indexed_calls if tool_requires_confirmation(call.get("name", ""))]
    completed: list[tuple[int, dict[str, Any] | None, ToolMessage]] = []
    if read_calls:
        completed.extend(await asyncio.gather(*(invoke_call(index, call) for index, call in read_calls)))
    for index, call in write_calls:
        completed.append(await invoke_call(index, call))
    for offset, call in enumerate(rejected_calls, start=len(calls)):
        name = str(call.get("name") or "unknown")
        call_id = str(call.get("id") or f"tool-call-{state.get('step', 0)}-{offset}")
        result = json.dumps(
            {
                "ok": False,
                "error": {"code": "tool_budget_exhausted", "message": f"工具调用预算上限为 {maximum_calls}"},
            },
            ensure_ascii=False,
        )
        completed.append(
            (
                offset,
                {"name": name, "status": "failed", "result": result},
                ToolMessage(content=result, tool_call_id=call_id),
            )
        )
    completed.sort(key=lambda item: item[0])
    events = [item[1] for item in completed if item[1] is not None]
    tool_messages = [item[2] for item in completed]

    return {
        "messages": [*state["messages"], *tool_messages],
        "tool_events": events,
    }


def route_after_decision(state: AgentLoopState) -> str:
    """Continue only when the model requested tools and the budget remains."""
    last = state["messages"][-1]
    if (
        isinstance(last, AIMessage)
        and last.tool_calls
        and state.get("step", 0) < state.get("max_steps", DEFAULT_MAX_STEPS)
    ):
        return "execute_tools"
    return "judge"


def route_after_judge(state: AgentLoopState) -> str:
    result = state.get("completion_result") or {}
    return END if result.get("terminal") else "decide"


def route_after_tools(state: AgentLoopState) -> str:
    """Stop the graph when a tool call needs a durable user confirmation."""
    return END if state.get("pending_tool_confirmation") else "decide"


def build_agent_loop(checkpointer: Any | None = None):
    """Compile the cyclic LLM/tool graph.

    Tools are request-scoped state so the same graph can be reused for stock,
    ETF, LOF, and future capabilities without global mutable tool registries.
    """
    graph = StateGraph(AgentLoopState, context_schema=AgentLoopContext)
    graph.add_node("decide", decide_next_action)
    graph.add_node("execute_tools", execute_tool_calls)
    graph.add_node("judge", judge_completion)
    graph.set_entry_point("decide")
    graph.add_conditional_edges(
        "decide",
        route_after_decision,
        {"execute_tools": "execute_tools", "judge": "judge"},
    )
    graph.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {"decide": "decide", END: END},
    )
    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {"decide": "decide", END: END},
    )
    return graph.compile(checkpointer=checkpointer)


agent_loop = build_agent_loop()


def configure_agent_loop(checkpointer: Any | None) -> None:
    """Recompile the shared graph when the application checkpoint backend starts."""
    global agent_loop
    agent_loop = build_agent_loop(checkpointer)


def get_agent_loop():
    return agent_loop


def _loop_context(
    tools: list[StructuredTool],
    *,
    native_interrupts: bool = False,
    task_id: str | None = None,
) -> AgentLoopContext:
    return AgentLoopContext(
        tools=tools,
        tool_map={tool.name: tool for tool in tools},
        native_interrupts=native_interrupts,
        task_id=task_id,
    )


async def run_agent_loop(
    messages: list[Any],
    tools: list[StructuredTool],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    config: dict[str, Any] | None = None,
    task_id: str | None = None,
    task_contract: dict[str, Any] | None = None,
) -> AgentLoopState:
    """Run the single LangGraph chat loop and return its final state."""
    return await agent_loop.ainvoke(
        {
            "messages": messages,
            "step": 0,
            "max_steps": max_steps,
            "tool_events": [],
            "reasoning_events": [],
            "judge_events": [],
            "task_contract": task_contract or {},
        },
        config=config,
        context=_loop_context(tools, task_id=task_id),
    )


async def stream_agent_loop(
    messages: list[Any],
    tools: list[StructuredTool],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    config: dict[str, Any] | None = None,
    native_interrupts: bool = False,
    task_id: str | None = None,
    task_contract: dict[str, Any] | None = None,
):
    """Stream updates from the single LangGraph chat loop."""
    async for update in _stream_manual_agent_loop(
        messages,
        tools,
        max_steps=max_steps,
        config=config,
        native_interrupts=native_interrupts,
        task_id=task_id,
        task_contract=task_contract,
    ):
        yield update


async def _stream_manual_agent_loop(
    messages: list[Any],
    tools: list[StructuredTool],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    config: dict[str, Any] | None = None,
    native_interrupts: bool = False,
    task_id: str | None = None,
    task_contract: dict[str, Any] | None = None,
):
    """Compatibility loop used by offline tests and no-key local development."""
    async for update in agent_loop.astream(
        {
            "messages": messages,
            "step": 0,
            "max_steps": max_steps,
            "tool_events": [],
            "reasoning_events": [],
            "judge_events": [],
            "task_contract": task_contract or {},
        },
        config=config,
        context=_loop_context(tools, native_interrupts=native_interrupts, task_id=task_id),
        stream_mode="updates",
    ):
        yield update


async def resume_native_agent_loop(
    tools: list[StructuredTool],
    *,
    approved: bool,
    config: dict[str, Any],
    task_id: str | None = None,
):
    """Resume the latest native LangGraph interrupt for one task thread."""
    async for update in agent_loop.astream(
        Command(resume={"approved": approved}),
        config=config,
        context=_loop_context(tools, native_interrupts=True, task_id=task_id),
        stream_mode="updates",
    ):
        yield update


async def resume_checkpoint_agent_loop(
    tools: list[StructuredTool],
    *,
    config: dict[str, Any],
    task_id: str | None = None,
):
    """Continue an interrupted native thread from its latest durable super-step."""
    async for update in agent_loop.astream(
        None,
        config=config,
        context=_loop_context(tools, native_interrupts=True, task_id=task_id),
        stream_mode="updates",
    ):
        yield update


async def resume_agent_loop(
    checkpoint_messages: list[dict[str, Any]],
    tools: list[StructuredTool],
    pending_tool_call: dict[str, Any],
    *,
    approved: bool,
    max_steps: int = DEFAULT_MAX_STEPS,
    config: dict[str, Any] | None = None,
    task_id: str | None = None,
):
    """Resume a loop after a persisted tool confirmation decision.

    The checkpoint contains the model's original AI tool-call message. We
    execute that exact call once when approved, append its observation, and
    then re-enter the normal loop so the model can continue from the same
    conversation state. A rejection becomes a durable tool observation and
    never invokes the underlying tool.
    """
    messages = messages_from_dict(checkpoint_messages)
    name = str(pending_tool_call.get("tool_name", ""))
    call_id = str(pending_tool_call.get("tool_call_id", ""))
    args = pending_tool_call.get("args", {})
    if not isinstance(args, dict):
        args = {}
    if not approved:
        result = json.dumps(
            {"ok": False, "error": {"code": "user_denied", "message": "用户拒绝执行该工具"}},
            ensure_ascii=False,
        )
        messages.append(ToolMessage(content=result, tool_call_id=call_id))
        async for update in stream_agent_loop(
            messages,
            tools,
            max_steps=max_steps,
            config=config,
            task_id=task_id,
        ):
            yield update
        return

    tool_map = {tool.name: tool for tool in tools}
    tool = tool_map.get(name)
    if tool is None:
        result = json.dumps(
            {"ok": False, "error": {"code": "unknown_tool", "message": f"未知工具: {name}"}},
            ensure_ascii=False,
        )
        status = "failed"
    else:
        result = None
        error_payload: dict[str, Any] | None = None
        attempts = tool_attempts(name)
        timeout_seconds = tool_timeout_seconds(name)
        for attempt in range(attempts):
            try:
                invoke_args = dict(args)
                if task_id and name in {
                    "save_artifacts",
                    "submit_simulation_order",
                    "create_simulation_account",
                    "deploy_backtest_experiment",
                }:
                    invoke_args.setdefault("execution_key", f"{task_id}:{call_id}")
                value = await asyncio.wait_for(tool.ainvoke(invoke_args, config=config), timeout=timeout_seconds)
                result = str(value)
                break
            except asyncio.TimeoutError:
                error_payload = {
                    "code": "tool_timeout",
                    "message": f"工具 {name} 执行超过 {timeout_seconds} 秒",
                    "timeout_seconds": timeout_seconds,
                    "attempt": attempt + 1,
                    "attempts": attempts,
                }
            except Exception as exc:  # Tool failures are observations, not model failures.
                error_code, error_message = context_safe_error(exc, str(exc)[:500])
                error_payload = {
                    "code": error_code,
                    "message": error_message,
                    "attempt": attempt + 1,
                    "attempts": attempts,
                }
            if attempt + 1 < attempts:
                await asyncio.sleep(0.2 * (attempt + 1))
        if result is None:
            result = json.dumps(
                {"ok": False, "error": error_payload or {"code": "tool_error", "message": "工具执行失败"}},
                ensure_ascii=False,
            )
            status = "failed"
        else:
            status = "completed"

    messages.append(ToolMessage(content=result, tool_call_id=call_id))
    async for update in stream_agent_loop(
        messages,
        tools,
        max_steps=max_steps,
        config=config,
        task_id=task_id,
    ):
        for node_update in update.values():
            if isinstance(node_update, dict) and status:
                # The normal stream owns subsequent events. The synthetic
                # tool event is emitted before it so the UI sees one audit
                # entry for the confirmed call.
                node_update.setdefault("tool_events", [])
                node_update["tool_events"] = [
                    {"name": name, "status": status, "result": result},
                    *node_update["tool_events"],
                ]
                status = ""
                break
        yield update
