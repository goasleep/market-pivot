"""Reusable LangGraph agent loop for LLM-directed tool use."""

from __future__ import annotations

import asyncio
import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph

from llm.service import get_llm_service

DEFAULT_MAX_STEPS = 100
TOOL_TIMEOUT_SECONDS = 60
LONG_RUNNING_TOOL_TIMEOUT_SECONDS = 300
LLM_TIMEOUT_SECONDS = 90


def tool_timeout_seconds(name: str) -> int:
    """Return the execution budget for a tool invocation."""
    if name == "run_fund_or_stock_analysis":
        return LONG_RUNNING_TOOL_TIMEOUT_SECONDS
    return TOOL_TIMEOUT_SECONDS


def tool_attempts(name: str) -> int:
    """Return retry count without duplicating expensive or mutating work."""
    if name == "run_fund_or_stock_analysis":
        return 1
    return 1 if name.startswith(("submit_", "cancel_", "create_", "fill_")) else 2


class AgentLoopState(TypedDict, total=False):
    messages: list[Any]
    step: int
    max_steps: int
    tools: list[StructuredTool]
    tool_map: dict[str, StructuredTool]
    tool_events: Annotated[list[dict[str, Any]], operator.add]
    reasoning_events: Annotated[list[dict[str, Any]], operator.add]
    final_response: str
    max_steps_reached: bool


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content or "")


async def decide_next_action(state: AgentLoopState) -> dict[str, Any]:
    """Ask the model whether to answer or call one or more tools."""
    response = await asyncio.wait_for(
        get_llm_service().chat_with_tools(
            state["messages"],
            state["tools"],
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

    tool_messages: list[ToolMessage] = []
    events: list[dict[str, Any]] = []
    for call in response.tool_calls or []:
        name = call.get("name", "")
        tool = state["tool_map"].get(name)
        call_id = call.get("id", "") or f"tool-call-{state.get('step', 0)}-{len(events)}"
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


def build_agent_loop():
    """Compile the cyclic LLM/tool graph.

    Tools are request-scoped state so the same graph can be reused for stock,
    ETF, LOF, and future capabilities without global mutable tool registries.
    """
    graph = StateGraph(AgentLoopState)
    graph.add_node("decide", decide_next_action)
    graph.add_node("execute_tools", execute_tool_calls)
    graph.set_entry_point("decide")
    graph.add_conditional_edges(
        "decide",
        route_after_decision,
        {"execute_tools": "execute_tools", END: END},
    )
    graph.add_edge("execute_tools", "decide")
    return graph.compile()


agent_loop = build_agent_loop()


async def run_agent_loop(
    messages: list[Any],
    tools: list[StructuredTool],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    config: dict[str, Any] | None = None,
) -> AgentLoopState:
    """Run a bounded LLM/tool loop and return its full trace state."""
    return await agent_loop.ainvoke(
        {
            "messages": messages,
            "tools": tools,
            "tool_map": {tool.name: tool for tool in tools},
            "step": 0,
            "max_steps": max_steps,
            "tool_events": [],
            "reasoning_events": [],
        },
        config=config,
    )


async def stream_agent_loop(
    messages: list[Any],
    tools: list[StructuredTool],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    config: dict[str, Any] | None = None,
):
    """Stream node updates so chat clients can show each tool round."""
    async for update in agent_loop.astream(
        {
            "messages": messages,
            "tools": tools,
            "tool_map": {tool.name: tool for tool in tools},
            "step": 0,
            "max_steps": max_steps,
            "tool_events": [],
            "reasoning_events": [],
        },
        config=config,
        stream_mode="updates",
    ):
        yield update
