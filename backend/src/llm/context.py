"""Token-aware prompt construction with explicit non-compressible context."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger


class ContextWindowExceededError(ValueError):
    """Raised when non-compressible context cannot fit in the selected model."""


@dataclass(frozen=True)
class ContextBudget:
    model: str
    context_window: int
    output_reserve: int
    safety_margin: int
    input_limit: int


@dataclass(frozen=True)
class ContextSelection:
    messages: list[Any]
    model: str
    input_limit: int
    raw_tokens: int
    selected_tokens: int
    protected_tokens: int
    tool_tokens: int
    dropped_messages: int

    @property
    def compacted(self) -> bool:
        return self.dropped_messages > 0


class TokenCounter:
    """Count chat and tool-schema tokens, using a conservative fallback for unknown models."""

    _MESSAGE_OVERHEAD = 4
    _REPLY_OVERHEAD = 2

    def __init__(self, model: str):
        self.model = model
        self._approximate = False
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")
            self._approximate = True

    def count_text(self, text: str) -> int:
        count = len(self._encoding.encode(text or ""))
        return math.ceil(count * 1.15) if self._approximate else count

    def count_message(self, message: Any) -> int:
        payload = [_message_role(message), _content_text(_message_content(message))]
        tool_calls = _message_tool_calls(message)
        if tool_calls:
            payload.append(json.dumps(tool_calls, ensure_ascii=False, sort_keys=True, default=str))
        tool_call_id = _message_tool_call_id(message)
        if tool_call_id:
            payload.append(tool_call_id)
        return self._MESSAGE_OVERHEAD + self.count_text("\n".join(payload))

    def count_messages(self, messages: Sequence[Any]) -> int:
        if not messages:
            return 0
        return self._REPLY_OVERHEAD + sum(self.count_message(message) for message in messages)

    def count_tools(self, tools: Sequence[Any] | None) -> int:
        if not tools:
            return 0
        payloads: list[dict[str, Any]] = []
        for tool in tools:
            schema: dict[str, Any] = {}
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None:
                if hasattr(args_schema, "model_json_schema"):
                    schema = args_schema.model_json_schema()
                elif isinstance(args_schema, dict):
                    schema = args_schema
            payloads.append(
                {
                    "name": str(getattr(tool, "name", "")),
                    "description": str(getattr(tool, "description", "")),
                    "parameters": schema,
                }
            )
        return self.count_text(json.dumps(payloads, ensure_ascii=False, sort_keys=True, default=str)) + 8 * len(
            payloads
        )


def get_context_budget(
    *,
    model: str | None = None,
    max_output_tokens: int | None = None,
) -> ContextBudget:
    """Resolve the effective input budget from the request-scoped LLM profile."""
    from config import get_llm_config

    config = get_llm_config(model=model)
    selected_model = str(model or config["model"])
    context_window = int(config["context_window"])
    configured_output = int(max_output_tokens or config["max_tokens"])
    output_reserve = configured_output
    safety_margin = max(1024, math.ceil(context_window * 0.05))
    input_limit = context_window - output_reserve - safety_margin
    if input_limit <= 0:
        raise ValueError("LLM context window is too small for its configured output reserve")
    return ContextBudget(
        model=selected_model,
        context_window=context_window,
        output_reserve=output_reserve,
        safety_margin=safety_margin,
        input_limit=input_limit,
    )


def select_conversation_history(
    history: Sequence[Any],
    *,
    p0_messages: Sequence[Any],
    tools: Sequence[Any] | None = None,
    model: str | None = None,
    max_output_tokens: int | None = None,
    budget: ContextBudget | None = None,
) -> ContextSelection:
    """Select recent complete turns while keeping P0 messages and pending interactions."""
    effective_budget = budget or get_context_budget(model=model, max_output_tokens=max_output_tokens)
    counter = TokenCounter(effective_budget.model)
    tool_tokens = counter.count_tools(tools)
    pending_indices = {index for index, message in enumerate(history) if _has_pending_interaction(message)}
    protected_history = [message for index, message in enumerate(history) if index in pending_indices]
    protected_tokens = counter.count_messages([*p0_messages, *protected_history]) + tool_tokens
    _require_p0_fits(protected_tokens, effective_budget)

    selected_indices = set(pending_indices)
    used_tokens = protected_tokens
    groups = _conversation_turn_groups(history)
    for group in reversed(groups):
        candidates = [index for index in group if index not in selected_indices]
        if not candidates:
            continue
        candidate_messages = [history[index] for index in candidates]
        candidate_tokens = sum(counter.count_message(message) for message in candidate_messages)
        if used_tokens + candidate_tokens > effective_budget.input_limit:
            break
        selected_indices.update(candidates)
        used_tokens += candidate_tokens

    selected = [message for index, message in enumerate(history) if index in selected_indices]
    return _selection(
        messages=selected,
        raw_messages=[*p0_messages, *history],
        selected_accounting_messages=[*p0_messages, *selected],
        protected_messages=[*p0_messages, *protected_history],
        counter=counter,
        budget=effective_budget,
        tool_tokens=tool_tokens,
        dropped_messages=len(history) - len(selected),
    )


def select_messages_for_model(
    messages: Sequence[Any],
    *,
    tools: Sequence[Any] | None = None,
    model: str | None = None,
    max_output_tokens: int | None = None,
    budget: ContextBudget | None = None,
) -> ContextSelection:
    """Fit messages to the model while preserving every P0 message and atomic tool exchange."""
    effective_budget = budget or get_context_budget(model=model, max_output_tokens=max_output_tokens)
    counter = TokenCounter(effective_budget.model)
    tool_tokens = counter.count_tools(tools)
    protected_indices = _protected_indices(messages)
    protected_messages = [message for index, message in enumerate(messages) if index in protected_indices]
    protected_tokens = counter.count_messages(protected_messages) + tool_tokens
    _require_p0_fits(protected_tokens, effective_budget)

    selected_indices = set(protected_indices)
    used_tokens = protected_tokens
    for group in reversed(_atomic_message_groups(messages)):
        candidates = [index for index in group if index not in selected_indices]
        if not candidates:
            continue
        candidate_messages = [messages[index] for index in candidates]
        candidate_tokens = sum(counter.count_message(message) for message in candidate_messages)
        if used_tokens + candidate_tokens > effective_budget.input_limit:
            break
        selected_indices.update(candidates)
        used_tokens += candidate_tokens

    selected = [message for index, message in enumerate(messages) if index in selected_indices]
    return _selection(
        messages=selected,
        raw_messages=list(messages),
        selected_accounting_messages=selected,
        protected_messages=protected_messages,
        counter=counter,
        budget=effective_budget,
        tool_tokens=tool_tokens,
        dropped_messages=len(messages) - len(selected),
    )


def _selection(
    *,
    messages: list[Any],
    raw_messages: list[Any],
    selected_accounting_messages: list[Any],
    protected_messages: list[Any],
    counter: TokenCounter,
    budget: ContextBudget,
    tool_tokens: int,
    dropped_messages: int,
) -> ContextSelection:
    selection = ContextSelection(
        messages=messages,
        model=budget.model,
        input_limit=budget.input_limit,
        raw_tokens=counter.count_messages(raw_messages) + tool_tokens,
        selected_tokens=counter.count_messages(selected_accounting_messages) + tool_tokens,
        protected_tokens=counter.count_messages(protected_messages) + tool_tokens,
        tool_tokens=tool_tokens,
        dropped_messages=dropped_messages,
    )
    log = logger.info if selection.compacted else logger.debug
    log(
        "LLM context model={} selected={}/{} tokens dropped_messages={} p0={} tools={} limit={}",
        selection.model,
        selection.selected_tokens,
        selection.raw_tokens,
        selection.dropped_messages,
        selection.protected_tokens,
        selection.tool_tokens,
        selection.input_limit,
    )
    return selection


def _require_p0_fits(protected_tokens: int, budget: ContextBudget) -> None:
    if protected_tokens <= budget.input_limit:
        return
    raise ContextWindowExceededError(
        f"不可压缩上下文需要 {protected_tokens} tokens，但模型 {budget.model} 的输入预算只有 "
        f"{budget.input_limit} tokens"
    )


def _conversation_turn_groups(messages: Sequence[Any]) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    for index, message in enumerate(messages):
        if _message_role(message) == "user" and current:
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    return groups


def _atomic_message_groups(messages: Sequence[Any]) -> list[list[int]]:
    """Group tool calls with observations and ordinary user/assistant turns."""
    groups: list[list[int]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = _message_tool_calls(message)
        if tool_calls:
            call_ids = {str(call.get("id", "")) for call in tool_calls if isinstance(call, dict)}
            group = [index]
            cursor = index + 1
            while cursor < len(messages):
                candidate = messages[cursor]
                if _message_role(candidate) != "tool" or _message_tool_call_id(candidate) not in call_ids:
                    break
                group.append(cursor)
                cursor += 1
            groups.append(group)
            index = cursor
            continue
        if _message_role(message) == "user" and index + 1 < len(messages):
            following = messages[index + 1]
            if _message_role(following) == "assistant" and not _message_tool_calls(following):
                groups.append([index, index + 1])
                index += 2
                continue
        groups.append([index])
        index += 1
    return groups


def _protected_indices(messages: Sequence[Any]) -> set[int]:
    protected = {
        index
        for index, message in enumerate(messages)
        if _message_role(message) == "system" or _has_pending_interaction(message)
    }
    user_indices = [index for index, message in enumerate(messages) if _message_role(message) == "user"]
    if user_indices:
        protected.add(user_indices[-1])

    current_user_index = user_indices[-1] if user_indices else -1
    latest_tool_ai = next(
        (
            index
            for index in range(len(messages) - 1, current_user_index, -1)
            if _message_role(messages[index]) == "assistant" and _message_tool_calls(messages[index])
        ),
        None,
    )
    if latest_tool_ai is not None:
        protected.add(latest_tool_ai)
        call_ids = {
            str(call.get("id", "")) for call in _message_tool_calls(messages[latest_tool_ai]) if isinstance(call, dict)
        }
        for index in range(latest_tool_ai + 1, len(messages)):
            if _message_role(messages[index]) == "tool" and _message_tool_call_id(messages[index]) in call_ids:
                protected.add(index)
    return protected


def _has_pending_interaction(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    for part in message.get("parts") or []:
        if not isinstance(part, dict) or part.get("type") != "interaction":
            continue
        content = part.get("content")
        if isinstance(content, dict) and content.get("status") == "pending":
            return True
    return False


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        role = str(message.get("role", "user"))
        return "assistant" if role == "ai" else role
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, ToolMessage):
        return "tool"
    if isinstance(message, AIMessage):
        return "assistant"
    return str(getattr(message, "type", "user"))


def _message_content(message: Any) -> Any:
    return message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")


def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
    value = message.get("tool_calls", []) if isinstance(message, dict) else getattr(message, "tool_calls", [])
    return list(value or [])


def _message_tool_call_id(message: Any) -> str:
    value = message.get("tool_call_id", "") if isinstance(message, dict) else getattr(message, "tool_call_id", "")
    return str(value or "")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
