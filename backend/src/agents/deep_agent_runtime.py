"""Shared Deep Agents runtime adapters.

Deep Agents are intentionally kept behind this module.  The application owns
business tools, task persistence, and A2UI rendering; this adapter translates
Deep Agent graph updates into the existing application event vocabulary.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, TypeVar

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from loguru import logger

from llm.service import get_llm_service

StructuredT = TypeVar("StructuredT")

# A process-local checkpointer lets the application use native Deep Agent HITL
# while the durable chat store continues to own task and UI event persistence.
# A future deployment can replace this with a database-backed saver without
# changing callers of this module.
deep_agent_checkpointer = MemorySaver()


def deep_agents_enabled() -> bool:
    """Return whether the configured runtime can make a real model call.

    Tests and offline tooling intentionally run without an API key.  They keep
    the existing deterministic/fake seams, while configured production runs
    use the Deep Agent path.
    """
    try:
        from config import get_llm_config

        return bool(get_llm_config().get("api_key")) and hasattr(get_llm_service(), "get_model")
    except Exception:
        return False


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif block:
                parts.append(str(block))
        return "".join(parts)
    return str(content or "")


def _message_list(value: Any) -> list[BaseMessage]:
    if isinstance(value, dict):
        value = value.get("messages", [])
    if not isinstance(value, list):
        return []
    return [message for message in value if isinstance(message, BaseMessage)]


def _decode_tool_result(message: ToolMessage) -> str:
    content = _content_text(message.content)
    return content


def build_deep_agent(
    *,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    response_format: Any | None = None,
    subagents: Sequence[dict[str, Any]] | None = None,
    interrupt_on: dict[str, bool | dict[str, Any]] | None = None,
    checkpointer: Any = None,
    name: str | None = None,
    model: BaseChatModel | None = None,
):
    """Build one Deep Agent using a tool-call-compatible configured model.

    DeepSeek V4's Thinking mode rejects forced or specific ``tool_choice``
    values used by Deep Agents and ToolStrategy.  Deep Agents are therefore
    built with Thinking disabled; regular chat calls retain the provider
    default and can still use Thinking.
    """
    return create_deep_agent(
        model=model or get_llm_service().get_model(thinking=False),
        tools=list(tools or []),
        system_prompt=system_prompt,
        response_format=response_format,
        subagents=list(subagents or []),
        interrupt_on=interrupt_on,
        checkpointer=checkpointer,
        name=name,
    )


def _structured_value(result: dict[str, Any], response_format: type[StructuredT]) -> StructuredT:
    value = result.get("structured_response")
    if isinstance(value, response_format):
        return value
    if isinstance(value, dict):
        return response_format.model_validate(value)

    messages = _message_list(result)
    for message in reversed(messages):
        text = _content_text(getattr(message, "content", "")).strip()
        if not text:
            continue
        try:
            return response_format.model_validate(json.loads(text))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    raise ValueError(f"Deep Agent 未返回可验证的 {response_format.__name__} 结构")


async def invoke_structured(
    agent: Any,
    prompt: str,
    response_format: type[StructuredT],
    *,
    config: dict[str, Any] | None = None,
) -> StructuredT:
    """Invoke a Deep Agent and validate its structured response."""
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    )
    return _structured_value(result, response_format)


def _interrupt_payload(value: Any) -> dict[str, Any] | None:
    """Normalize LangGraph interrupt values into the app's confirmation shape."""
    if not value:
        return None
    raw = value
    raw = getattr(raw, "value", raw)
    if isinstance(value, (list, tuple)):
        raw = value[0] if value else None
        raw = getattr(raw, "value", raw)
    if isinstance(raw, dict):
        action_requests = raw.get("action_requests")
        if isinstance(action_requests, list) and action_requests:
            raw = action_requests[0]
        action = raw.get("action") or raw.get("tool_name") or raw.get("name")
        args = raw.get("args") or raw.get("arguments") or {}
        if action:
            return {"tool_name": str(action), "args": args if isinstance(args, dict) else {}}
    action = getattr(raw, "action", None) or getattr(raw, "tool_name", None)
    args = getattr(raw, "args", {})
    if action:
        return {"tool_name": str(action), "args": args if isinstance(args, dict) else {}}
    return None


def _update_events(update: dict[str, Any], seen: set[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for node_name, payload in update.items():
        if node_name == "__interrupt__":
            continue
        for message in _message_list(payload):
            message_id = str(getattr(message, "id", "") or f"{node_name}:{id(message)}")
            if message_id in seen:
                continue
            seen.add(message_id)
            if isinstance(message, AIMessage):
                if message.tool_calls:
                    names = ", ".join(str(call.get("name", "工具")) for call in message.tool_calls)
                    events.append(
                        {
                            "type": "reasoning",
                            "text": f"Deep Agent 正在调用：{names}",
                        }
                    )
                elif message.content:
                    events.append({"type": "text", "text": _content_text(message.content)})
            elif isinstance(message, ToolMessage):
                events.append(
                    {
                        "type": "tool",
                        "name": str(message.name or message.tool_call_id or "unknown"),
                        "status": "completed",
                        "result": _decode_tool_result(message),
                    }
                )
    return events


async def stream_deep_agent(
    agent: Any,
    messages: list[Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
    resume: Any | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream normalized Deep Agent events for the chat/A2UI boundary."""
    input_value: Any
    if resume is not None:
        input_value = Command(resume=resume)
    else:
        input_value = {"messages": messages or []}

    seen: set[str] = set()
    async for update in agent.astream(input_value, config=config, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        events = _update_events(update, seen)
        interrupt = _interrupt_payload(update.get("__interrupt__"))
        if interrupt:
            interrupt["deep_agent"] = True
            interrupt["thread_id"] = (config or {}).get("configurable", {}).get("thread_id", "")
            yield {
                "type": "interaction_required",
                "kind": "tool_confirmation",
                "question": "Deep Agent 准备执行一个需要用户确认的工具操作，是否继续？",
                "options": [
                    {"id": "approve", "label": "确认执行"},
                    {"id": "reject", "label": "取消执行"},
                ],
                "pending_tool_call": interrupt,
            }
        for event in events:
            yield event


def log_deep_agent_failure(exc: Exception) -> None:
    """Log adapter failures without hiding real model/tool errors."""
    logger.exception("Deep Agent execution failed: {}", exc)
