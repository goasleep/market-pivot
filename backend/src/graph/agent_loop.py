"""Reusable LangGraph agent loop for LLM-directed tool use."""

from __future__ import annotations

import asyncio
import json
import operator
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

from llm.context import select_messages_for_model
from llm.service import get_llm_service
from tools.policies import tool_requires_confirmation

DEFAULT_MAX_STEPS = 100
TOOL_TIMEOUT_SECONDS = 60
LONG_RUNNING_TOOL_TIMEOUT_SECONDS = 900
LLM_TIMEOUT_SECONDS = 90


def tool_timeout_seconds(name: str) -> int:
    """Return the execution budget for a tool invocation."""
    if name in {
        "run_fund_or_stock_analysis",
        "run_backtest",
        "design_and_run_backtest",
        "compare_strategy_backtests",
        "design_and_run_sandbox_strategy",
    }:
        return LONG_RUNNING_TOOL_TIMEOUT_SECONDS
    return TOOL_TIMEOUT_SECONDS


def tool_attempts(name: str) -> int:
    """Return retry count without duplicating expensive or mutating work."""
    if name in {
        "run_fund_or_stock_analysis",
        "run_backtest",
        "design_and_run_backtest",
        "compare_strategy_backtests",
        "design_and_run_sandbox_strategy",
    }:
        return 1
    return 1 if name.startswith(("submit_", "cancel_", "create_", "fill_")) else 2


class AgentLoopState(TypedDict, total=False):
    messages: list[Any]
    step: int
    max_steps: int
    tool_events: Annotated[list[dict[str, Any]], operator.add]
    reasoning_events: Annotated[list[dict[str, Any]], operator.add]
    final_response: str
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
    context = select_messages_for_model(state["messages"], tools=runtime.context.tools)
    response = await asyncio.wait_for(
        get_llm_service().chat_with_tools(
            context.messages,
            runtime.context.tools,
            temperature=0.2,
        ),
        timeout=LLM_TIMEOUT_SECONDS,
    )
    step = state.get("step", 0) + 1
    final_response = _content_text(response.content)
    max_steps_reached = bool(response.tool_calls and step >= state.get("max_steps", DEFAULT_MAX_STEPS))
    reasoning_events: list[dict[str, Any]] = []
    if response.tool_calls and not max_steps_reached:
        tool_names = [str(call.get("name", "数据工具")) for call in response.tool_calls]
        model_summary = final_response.strip().splitlines()[0][:240] if final_response.strip() else ""
        summary = model_summary or f"第 {step} 轮：需要先获取 {', '.join(tool_names)} 的数据，再继续判断。"
        reasoning_events.append({"step": step, "text": summary})
    if max_steps_reached:
        final_response = (
            f"{final_response}\n\n"
            f"已达到 Agent 最大执行轮数（{state.get('max_steps', DEFAULT_MAX_STEPS)} 轮），"
            "为避免无限调用，已停止继续执行。请缩小问题范围后重试。"
        ).strip()
    return {
        "messages": [*state["messages"], response],
        "step": step,
        "final_response": final_response,
        "reasoning_events": reasoning_events,
        "max_steps_reached": max_steps_reached,
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

    calls = response.tool_calls or []
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

    tool_messages: list[ToolMessage] = []
    events: list[dict[str, Any]] = []
    for call in response.tool_calls or []:
        name = call.get("name", "")
        tool = runtime.context.tool_map.get(name)
        call_id = call.get("id", "") or f"tool-call-{state.get('step', 0)}-{len(events)}"
        if tool_requires_confirmation(name) and not approvals.get(call_id, False):
            result = json.dumps(
                {"ok": False, "error": {"code": "user_denied", "message": "用户拒绝执行该工具"}},
                ensure_ascii=False,
            )
            events.append({"name": name, "status": "failed", "result": result})
            tool_messages.append(ToolMessage(content=result, tool_call_id=call_id))
            continue
        if not tool:
            result = json.dumps(
                {"ok": False, "error": {"code": "unknown_tool", "message": f"未知工具: {name}"}},
                ensure_ascii=False,
            )
            events.append({"name": name or "unknown", "status": "failed", "result": result})
            tool_messages.append(ToolMessage(content=result, tool_call_id=call_id))
            continue

        args = call.get("args", {})
        if not isinstance(args, dict):
            result = json.dumps(
                {"ok": False, "error": {"code": "invalid_arguments", "message": "工具参数必须是对象"}},
                ensure_ascii=False,
            )
            events.append({"name": name, "status": "failed", "result": result})
            tool_messages.append(ToolMessage(content=result, tool_call_id=call_id))
            continue
        args = dict(args)
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
                error_payload = {
                    "code": "tool_error",
                    "message": str(exc)[:500],
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
            events.append({"name": name, "status": "failed", "result": result})
        else:
            events.append({"name": name, "status": "completed", "result": result})
        tool_messages.append(ToolMessage(content=result, tool_call_id=call_id))

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
    return END


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
    graph.set_entry_point("decide")
    graph.add_conditional_edges(
        "decide",
        route_after_decision,
        {"execute_tools": "execute_tools", END: END},
    )
    graph.add_conditional_edges(
        "execute_tools",
        route_after_tools,
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
) -> AgentLoopState:
    """Run the single LangGraph chat loop and return its final state."""
    return await agent_loop.ainvoke(
        {
            "messages": messages,
            "step": 0,
            "max_steps": max_steps,
            "tool_events": [],
            "reasoning_events": [],
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
):
    """Stream updates from the single LangGraph chat loop."""
    async for update in _stream_manual_agent_loop(
        messages,
        tools,
        max_steps=max_steps,
        config=config,
        native_interrupts=native_interrupts,
        task_id=task_id,
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
):
    """Compatibility loop used by offline tests and no-key local development."""
    async for update in agent_loop.astream(
        {
            "messages": messages,
            "step": 0,
            "max_steps": max_steps,
            "tool_events": [],
            "reasoning_events": [],
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
                error_payload = {
                    "code": "tool_error",
                    "message": str(exc)[:500],
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
