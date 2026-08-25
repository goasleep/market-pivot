"""Application-level LLM service built on LangChain chat-model abstractions."""

import asyncio
import json
from typing import Any
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger

from config import get_llm_config
from llm.context import (
    ContextBudget,
    ContextWindowExceededError,
    TokenCounter,
    compact_json_value,
    get_context_budget,
    is_context_overflow_error,
    select_messages_for_model,
)
from llm.factory import get_chat_model
from llm_catalog import supports_vision

_GENERATED_TEXT_REPLACEMENTS = (
    ("性交易", "性的交易"),
    ("性交", "相关行为"),
)
LLM_CALL_TIMEOUT_SECONDS = 1800.0
VISION_CALL_TIMEOUT_SECONDS = LLM_CALL_TIMEOUT_SECONDS


def _consume_background_task_result(task: asyncio.Task[Any]) -> None:
    """Consume a late provider result after a hard deadline to avoid task warnings."""
    try:
        task.result()
    except BaseException:
        pass


async def _ainvoke_with_timeout(
    model: Any,
    messages: list[Any],
    *,
    timeout: float | None = None,
) -> Any:
    """Apply a hard per-call limit without imposing an outer task deadline.

    ``asyncio.wait_for`` waits for cancellation to finish. Some OpenAI-compatible
    transports defer or suppress that cancellation, allowing a nominal timeout to
    overrun indefinitely. ``asyncio.wait`` lets the Agent stop waiting at the
    deadline; the provider task is cancelled best-effort and observed separately.
    """
    call_timeout = LLM_CALL_TIMEOUT_SECONDS if timeout is None else timeout
    task = asyncio.ensure_future(model.ainvoke(messages))
    done, _ = await asyncio.wait({task}, timeout=call_timeout)
    if task in done:
        return task.result()
    task.cancel()
    task.add_done_callback(_consume_background_task_result)
    raise TimeoutError(f"LLM call exceeded the {call_timeout:g}-second hard limit")


def _normalize_generated_financial_text(text: str) -> str:
    """Rewrite moderation-prone substrings in model-generated financial prose.

    The gateway performs substring matching, so ordinary phrases such as
    ``技术性交易`` can be mistaken for unrelated sensitive content.  Apply
    semantic, visible rewrites only to generated text; user input and source
    evidence remain untouched.
    """
    normalized = text
    for source, replacement in _GENERATED_TEXT_REPLACEMENTS:
        normalized = normalized.replace(source, replacement)
    return normalized


def _normalize_generated_message(message: AIMessage) -> AIMessage:
    """Normalize textual AIMessage content while preserving tool calls and metadata."""
    content = message.content
    if isinstance(content, str):
        normalized_content: Any = _normalize_generated_financial_text(content)
    elif isinstance(content, list):
        normalized_content = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                normalized_content.append(
                    {
                        **block,
                        "text": _normalize_generated_financial_text(block["text"]),
                    }
                )
            else:
                normalized_content.append(block)
    else:
        return message
    return message.model_copy(update={"content": normalized_content})


def _message_text(message: AIMessage) -> str:
    """Normalize LangChain message content to the string expected by callers."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            elif block:
                parts.append(str(block))
        return "".join(parts)
    return str(content or "")


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Parse one model response while preserving the existing tolerant behavior."""
    normalized = _normalize_generated_financial_text(raw).strip()
    if normalized.startswith("```"):
        normalized = "\n".join(line for line in normalized.splitlines() if not line.startswith("```"))
    try:
        parsed = JsonOutputParser().parse(normalized)
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON response is not an object")
        return parsed
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("JSON parse failed: {}\nRaw: {}", exc, normalized[:500])
        return {"error": "json_parse_failed", "raw": normalized[:500]}


def _to_messages(messages: list[Any]) -> list[Any]:
    """Convert the application's role/content dictionaries to LangChain messages."""
    converted = []
    for message in messages:
        if not isinstance(message, dict):
            converted.append(message)
            continue
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        elif role == "tool":
            converted.append(ToolMessage(content=content, tool_call_id=message.get("tool_call_id", "")))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def _recovery_budget(*, model: str | None = None, max_tokens: int | None = None) -> ContextBudget:
    """Reserve extra headroom for one transparent retry after a context rejection."""
    budget = get_context_budget(model=model, max_output_tokens=max_tokens)
    reduced_limit = max(
        1,
        min(
            budget.input_limit - 1,
            max(256, int(budget.input_limit * 0.75)),
        ),
    )
    return ContextBudget(
        model=budget.model,
        context_window=budget.context_window,
        output_reserve=budget.output_reserve,
        safety_margin=budget.safety_margin + (budget.input_limit - reduced_limit),
        input_limit=reduced_limit,
    )


def _project_application_prompt(prompt: str, system: str, budget: ContextBudget) -> list[dict[str, str]]:
    """Project an application-built prompt; the primary Agent's user message does not use this path."""
    counter = TokenCounter(budget.model)
    empty_messages: list[dict[str, str]] = []
    if system:
        empty_messages.append({"role": "system", "content": system})
    empty_messages.append({"role": "user", "content": ""})
    prompt_budget = budget.input_limit - counter.count_messages(empty_messages) - 128
    if prompt_budget <= 0:
        raise ContextWindowExceededError(
            f"固定系统上下文需要至少 {budget.input_limit - prompt_budget} tokens，"
            f"但模型 {budget.model} 的恢复预算只有 {budget.input_limit} tokens"
        )
    projected = compact_json_value(prompt, prompt_budget, counter=counter)
    content = projected if isinstance(projected, str) else json.dumps(projected, ensure_ascii=False, default=str)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return select_messages_for_model(messages, budget=budget).messages


class LLMService:
    """Stable application interface for chat and JSON model calls.

    Agents depend on this service instead of a provider-specific SDK or module.
    The underlying LangChain model is resolved on each call to preserve the
    existing hot-reload behavior.
    """

    def get_model(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        profile_id: str | None = None,
        route: str | None = None,
    ) -> BaseChatModel:
        return get_chat_model(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            profile_id=profile_id,
            route=route,
        )

    def supports_vision(
        self,
        model: str | None = None,
        profile_id: str | None = None,
        route: str | None = None,
    ) -> bool:
        """Use the explicit model allowlist instead of probing provider capabilities."""
        config = get_llm_config(profile_id=profile_id, model=model, route=route)
        return supports_vision(model or str(config.get("model") or ""))

    async def chat(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        profile_id: str | None = None,
        route: str | None = None,
    ) -> str:
        """Invoke the configured chat model and return its text content."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        chat_model = self.get_model(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            profile_id=profile_id,
            route=route,
        )
        try:
            context = select_messages_for_model(messages, model=model, max_output_tokens=max_tokens)
            response = await _ainvoke_with_timeout(chat_model, _to_messages(context.messages))
        except Exception as exc:
            if not is_context_overflow_error(exc):
                logger.error("LLM call failed: {}", exc)
                raise
            logger.warning("LLM context rejected; retrying with projected application prompt: {}", exc)
            recovery_budget = _recovery_budget(model=model, max_tokens=max_tokens)
            recovery_messages = _project_application_prompt(prompt, system, recovery_budget)
            response = await _ainvoke_with_timeout(chat_model, _to_messages(recovery_messages))
        try:
            content = _normalize_generated_financial_text(_message_text(response))
            logger.debug("LLM response ({}): {} chars", model or "configured", len(content))
            return content
        except Exception as exc:
            logger.error("LLM response normalization failed: {}", exc)
            raise

    def chat_sync(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        profile_id: str | None = None,
        route: str | None = None,
    ) -> str:
        """Synchronous model call for blocking artifact-generation boundaries."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        chat_model = self.get_model(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            profile_id=profile_id,
            route=route,
        )
        try:
            context = select_messages_for_model(messages, model=model, max_output_tokens=max_tokens)
            response = chat_model.invoke(_to_messages(context.messages))
        except Exception as exc:
            if not is_context_overflow_error(exc):
                logger.error("Synchronous LLM call failed: {}", exc)
                raise
            logger.warning("Synchronous LLM context rejected; retrying with projected application prompt: {}", exc)
            recovery_budget = _recovery_budget(model=model, max_tokens=max_tokens)
            recovery_messages = _project_application_prompt(prompt, system, recovery_budget)
            response = chat_model.invoke(_to_messages(recovery_messages))
        try:
            content = _normalize_generated_financial_text(_message_text(response))
            logger.debug("Synchronous LLM response ({}): {} chars", model or "configured", len(content))
            return content
        except Exception as exc:
            logger.error("Synchronous LLM response normalization failed: {}", exc)
            raise

    async def chat_json(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        profile_id: str | None = None,
        route: str | None = None,
    ) -> dict[str, Any]:
        """Invoke the model and parse a JSON object using LangChain's parser."""
        json_instruction = "You must respond with valid JSON only, no markdown, no explanation."
        system = f"{system}\n\n{json_instruction}" if system else json_instruction
        raw = await self.chat(
            prompt,
            system=system,
            model=model,
            temperature=0.0,
            profile_id=profile_id,
            route=route,
        )
        return _parse_json_object(raw)

    async def chat_json_with_images(
        self,
        prompt: str,
        image_urls: list[str],
        system: str = "",
        model: str | None = None,
        profile_id: str | None = None,
        route: str | None = None,
        detail: str = "high",
    ) -> dict[str, Any]:
        """Invoke a vision-capable model with HTTP(S) image artifacts and parse JSON."""
        if not image_urls:
            return await self.chat_json(
                prompt,
                system=system,
                model=model,
                profile_id=profile_id,
                route=route,
            )
        if detail not in {"low", "high", "original", "auto"}:
            raise ValueError("image detail must be low, high, original, or auto")
        for image_url in image_urls:
            parsed = urlparse(image_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("model image inputs must use absolute HTTP(S) URLs")

        json_instruction = "You must respond with valid JSON only, no markdown, no explanation."
        effective_system = f"{system}\n\n{json_instruction}" if system else json_instruction
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": image_url, "detail": detail},
            }
            for image_url in image_urls
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": content},
        ]
        chat_model = self.get_model(
            model=model,
            temperature=0.0,
            profile_id=profile_id,
            route=route,
        )
        context = select_messages_for_model(messages, model=model)
        try:
            response = await _ainvoke_with_timeout(
                chat_model,
                _to_messages(context.messages),
                timeout=VISION_CALL_TIMEOUT_SECONDS,
            )
            return _parse_json_object(_message_text(response))
        except Exception as exc:
            logger.warning("Vision model call failed or timed out; falling back to text input: {}", exc)
            return await self.chat_json(
                prompt,
                system=system,
                model=model,
                profile_id=profile_id,
                route=route,
            )

    async def chat_langchain(
        self,
        messages: list[Any],
        model: str | None = None,
        temperature: float | None = None,
        profile_id: str | None = None,
        route: str | None = None,
    ) -> str:
        """Invoke the underlying LangChain model with role/content messages."""
        chat_model = self.get_model(
            model=model,
            temperature=temperature,
            profile_id=profile_id,
            route=route,
        )
        try:
            context = select_messages_for_model(messages, model=model)
            response = await _ainvoke_with_timeout(chat_model, _to_messages(context.messages))
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise
            logger.warning("LangChain context rejected; retrying with reduced context budget: {}", exc)
            context = select_messages_for_model(messages, budget=_recovery_budget(model=model))
            response = await _ainvoke_with_timeout(chat_model, _to_messages(context.messages))
        return _normalize_generated_financial_text(_message_text(response))

    async def chat_with_tools(
        self,
        messages: list[Any],
        tools: list[Any],
        model: str | None = None,
        temperature: float | None = None,
        profile_id: str | None = None,
        route: str | None = None,
    ) -> AIMessage:
        """Invoke the configured model with tool definitions.

        Tool selection is intentionally owned by the model. The caller is
        responsible for executing returned tool calls and feeding ToolMessage
        results back into the conversation.
        """
        chat_model = self.get_model(
            model=model,
            temperature=temperature,
            profile_id=profile_id,
            route=route,
        )
        bound_model = chat_model.bind_tools(tools) if tools else chat_model
        try:
            context = select_messages_for_model(messages, tools=tools, model=model)
            response = await _ainvoke_with_timeout(bound_model, _to_messages(context.messages))
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise
            logger.warning("Tool-chat context rejected; retrying with reduced context budget: {}", exc)
            context = select_messages_for_model(messages, tools=tools, budget=_recovery_budget(model=model))
            response = await _ainvoke_with_timeout(bound_model, _to_messages(context.messages))
        return _normalize_generated_message(response)


_default_service = LLMService()


def get_llm_service() -> LLMService:
    """Return the stateless application LLM service."""
    return _default_service


async def chat(*args, **kwargs) -> str:
    return await _default_service.chat(*args, **kwargs)


async def chat_json(*args, **kwargs) -> dict[str, Any]:
    return await _default_service.chat_json(*args, **kwargs)


async def chat_langchain(*args, **kwargs) -> str:
    return await _default_service.chat_langchain(*args, **kwargs)
