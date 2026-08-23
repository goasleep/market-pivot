"""Token-aware prompt construction with explicit non-compressible context."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from config import get_llm_config


class ContextWindowExceededError(ValueError):
    """Raised when non-compressible context cannot fit in the selected model."""


def is_context_overflow_error(exc: Exception) -> bool:
    """Recognize local and common provider context-window failures."""
    if isinstance(exc, ContextWindowExceededError):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "context length",
            "context window",
            "maximum context",
            "too many tokens",
            "token limit",
            "上下文",
        )
    )


def context_safe_error(exc: Exception, default: str) -> tuple[str, str]:
    """Return a stable error code/message without leaking model context details."""
    if is_context_overflow_error(exc):
        return "result_unavailable", "本步骤未能形成可靠结果，其他已获取数据不受影响"
    return "tool_error", default


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
    compacted_messages: int = 0

    @property
    def compacted(self) -> bool:
        return self.dropped_messages > 0 or self.compacted_messages > 0


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

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Return a tokenizer-safe prefix within the requested estimated budget."""
        if max_tokens <= 0:
            return ""
        raw_limit = math.floor(max_tokens / 1.15) if self._approximate else max_tokens
        tokens = self._encoding.encode(text or "")
        return self._encoding.decode(tokens[:raw_limit])

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


_CONTEXT_KEY_PRIORITY = (
    "data_type",
    "status",
    "available",
    "error",
    "message",
    "summary",
    "text",
    "ticker",
    "tickers",
    "asset_type",
    "as_of",
    "data_date",
    "provenance",
    "decision",
    "signal",
    "confidence",
    "reasoning",
    "conclusion",
    "metrics",
    "key_data",
    "quote",
    "acceptance",
    "artifacts",
    "results",
    "news",
    "history",
)


def compact_json_value(
    value: Any,
    max_tokens: int,
    *,
    model: str | None = None,
    counter: TokenCounter | None = None,
) -> Any:
    """Create a deterministic JSON-compatible projection within a token budget."""
    token_counter = counter or TokenCounter(model or get_context_budget().model)
    original_text = _content_text(value)
    original_tokens = token_counter.count_text(original_text)
    if original_tokens <= max_tokens:
        return value

    levels = (
        (4, 20, 12, 2000),
        (3, 14, 8, 1200),
        (3, 10, 5, 700),
        (2, 8, 3, 400),
        (2, 5, 2, 200),
    )
    for max_depth, max_dict_items, max_list_items, max_string_chars in levels:
        projected = _project_json_value(
            value,
            depth=0,
            max_depth=max_depth,
            max_dict_items=max_dict_items,
            max_list_items=max_list_items,
            max_string_chars=max_string_chars,
        )
        envelope = {
            "_context_compacted": True,
            "original_tokens": original_tokens,
            "data": projected,
        }
        if token_counter.count_text(_content_text(envelope)) <= max_tokens:
            return envelope

    envelope = {
        "_context_compacted": True,
        "original_tokens": original_tokens,
        "preview": "",
    }
    base_tokens = token_counter.count_text(_content_text(envelope))
    preview_budget = max(0, max_tokens - base_tokens - 8)
    envelope["preview"] = token_counter.truncate_text(original_text, preview_budget)
    if token_counter.count_text(_content_text(envelope)) <= max_tokens:
        return envelope
    return {"_context_compacted": True, "original_tokens": original_tokens}


def get_context_budget(
    *,
    model: str | None = None,
    max_output_tokens: int | None = None,
) -> ContextBudget:
    """Resolve the effective input budget from the request-scoped LLM profile."""
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
    replacements = _compact_protected_tool_messages(
        messages,
        protected_indices=protected_indices,
        counter=counter,
        tool_tokens=tool_tokens,
        input_limit=effective_budget.input_limit,
    )
    protected_messages = [
        replacements.get(index, message) for index, message in enumerate(messages) if index in protected_indices
    ]
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

    selected = [replacements.get(index, message) for index, message in enumerate(messages) if index in selected_indices]
    return _selection(
        messages=selected,
        raw_messages=list(messages),
        selected_accounting_messages=selected,
        protected_messages=protected_messages,
        counter=counter,
        budget=effective_budget,
        tool_tokens=tool_tokens,
        dropped_messages=len(messages) - len(selected),
        compacted_messages=len(replacements),
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
    compacted_messages: int = 0,
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
        compacted_messages=compacted_messages,
    )
    log = logger.info if selection.compacted else logger.debug
    log(
        "LLM context model={} selected={}/{} tokens dropped_messages={} compacted_messages={} p0={} tools={} limit={}",
        selection.model,
        selection.selected_tokens,
        selection.raw_tokens,
        selection.dropped_messages,
        selection.compacted_messages,
        selection.protected_tokens,
        selection.tool_tokens,
        selection.input_limit,
    )
    return selection


def _compact_protected_tool_messages(
    messages: Sequence[Any],
    *,
    protected_indices: set[int],
    counter: TokenCounter,
    tool_tokens: int,
    input_limit: int,
) -> dict[int, Any]:
    """Project oversized current tool observations without mutating durable graph state."""
    tool_indices = [index for index in sorted(protected_indices) if _message_role(messages[index]) == "tool"]
    if not tool_indices:
        return {}

    protected = [messages[index] for index in sorted(protected_indices)]
    if counter.count_messages(protected) + tool_tokens <= input_limit:
        return {}

    empty_replacements = {index: _replace_message_content(messages[index], "") for index in tool_indices}
    minimum = [empty_replacements.get(index, messages[index]) for index in sorted(protected_indices)]
    minimum_tokens = counter.count_messages(minimum) + tool_tokens
    if minimum_tokens > input_limit:
        return {}

    content_budget = max(0, input_limit - minimum_tokens - 16 * len(tool_indices))
    per_message_budget = max(16, content_budget // len(tool_indices))
    replacements: dict[int, Any] = {}
    for index in tool_indices:
        original_content = _message_content(messages[index])
        compacted = compact_json_value(original_content, per_message_budget, counter=counter)
        replacements[index] = _replace_message_content(messages[index], _content_text(compacted))

    projected = [replacements.get(index, messages[index]) for index in sorted(protected_indices)]
    if counter.count_messages(projected) + tool_tokens <= input_limit:
        return replacements
    return empty_replacements


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


def _replace_message_content(message: Any, content: str) -> Any:
    if isinstance(message, dict):
        return {**message, "content": content}
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    return message


def _project_json_value(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    max_dict_items: int,
    max_list_items: int,
    max_string_chars: int,
) -> Any:
    if isinstance(value, str):
        if depth == 0:
            try:
                decoded = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return value[:max_string_chars]
            return _project_json_value(
                decoded,
                depth=depth,
                max_depth=max_depth,
                max_dict_items=max_dict_items,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
            )
        return value[:max_string_chars]
    if depth >= max_depth:
        if isinstance(value, dict):
            return {"_omitted_fields": len(value)}
        if isinstance(value, list):
            return {"_omitted_items": len(value)}
        return value
    if isinstance(value, dict):
        keys = list(value)
        ordered = [key for key in _CONTEXT_KEY_PRIORITY if key in value]
        ordered.extend(key for key in keys if key not in ordered)
        selected_keys = ordered[:max_dict_items]
        result = {
            str(key): _project_json_value(
                value[key],
                depth=depth + 1,
                max_depth=max_depth,
                max_dict_items=max_dict_items,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
            )
            for key in selected_keys
        }
        if len(keys) > len(selected_keys):
            result["_omitted_fields"] = len(keys) - len(selected_keys)
        return result
    if isinstance(value, list):
        if len(value) <= max_list_items:
            selected = value
            omitted = 0
        else:
            head_count = (max_list_items + 1) // 2
            tail_count = max_list_items - head_count
            selected = [*value[:head_count], *value[-tail_count:]] if tail_count else value[:head_count]
            omitted = len(value) - len(selected)
        result = [
            _project_json_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_dict_items=max_dict_items,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
            )
            for item in selected
        ]
        if omitted:
            result.insert(len(result) // 2, {"_omitted_items": omitted})
        return result
    return value
