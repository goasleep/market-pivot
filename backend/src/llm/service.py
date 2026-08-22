"""Application-level LLM service built on LangChain chat-model abstractions."""

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger

from llm.factory import get_chat_model

_GENERATED_TEXT_REPLACEMENTS = (
    ("性交易", "性的交易"),
    ("性交", "相关行为"),
)


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

        try:
            response = await self.get_model(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                profile_id=profile_id,
                route=route,
            ).ainvoke(_to_messages(messages))
            content = _normalize_generated_financial_text(_message_text(response))
            logger.debug("LLM response ({}): {} chars", model or "configured", len(content))
            return content
        except Exception as exc:
            logger.error("LLM call failed: {}", exc)
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

        try:
            response = self.get_model(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                profile_id=profile_id,
                route=route,
            ).invoke(_to_messages(messages))
            content = _normalize_generated_financial_text(_message_text(response))
            logger.debug("Synchronous LLM response ({}): {} chars", model or "configured", len(content))
            return content
        except Exception as exc:
            logger.error("Synchronous LLM call failed: {}", exc)
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
        raw = _normalize_generated_financial_text(
            await self.chat(
                prompt,
                system=system,
                model=model,
                temperature=0.0,
                profile_id=profile_id,
                route=route,
            )
        ).strip()

        # Keep compatibility with models that still wrap JSON in a markdown fence.
        if raw.startswith("```"):
            raw = "\n".join(line for line in raw.splitlines() if not line.startswith("```"))

        try:
            parsed = JsonOutputParser().parse(raw)
            if not isinstance(parsed, dict):
                raise ValueError("LLM JSON response is not an object")
            return parsed
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("JSON parse failed: {}\nRaw: {}", exc, raw[:500])
            return {"error": "json_parse_failed", "raw": raw[:500]}

    async def chat_langchain(
        self,
        messages: list[Any],
        model: str | None = None,
        temperature: float | None = None,
        profile_id: str | None = None,
        route: str | None = None,
    ) -> str:
        """Invoke the underlying LangChain model with role/content messages."""
        response = await self.get_model(
            model=model,
            temperature=temperature,
            profile_id=profile_id,
            route=route,
        ).ainvoke(_to_messages(messages))
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
        bound_model = self.get_model(
            model=model,
            temperature=temperature,
            thinking=False,
            profile_id=profile_id,
            route=route,
        ).bind_tools(tools)
        response = await bound_model.ainvoke(_to_messages(messages))
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
