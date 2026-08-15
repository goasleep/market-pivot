"""Application-level LLM service built on LangChain chat-model abstractions."""

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger

from llm.deepseek import get_chat_model


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
    ) -> BaseChatModel:
        return get_chat_model(model=model, temperature=temperature, max_tokens=max_tokens)

    async def chat(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
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
            ).ainvoke(_to_messages(messages))
            content = _message_text(response)
            logger.debug("LLM response ({}): {} chars", model or "configured", len(content))
            return content
        except Exception as exc:
            logger.error("LLM call failed: {}", exc)
            raise

    async def chat_json(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Invoke the model and parse a JSON object using LangChain's parser."""
        json_instruction = "You must respond with valid JSON only, no markdown, no explanation."
        system = f"{system}\n\n{json_instruction}" if system else json_instruction
        raw = (await self.chat(prompt, system=system, model=model, temperature=0.0)).strip()

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
    ) -> str:
        """Invoke the underlying LangChain model with role/content messages."""
        response = await self.get_model(model=model, temperature=temperature).ainvoke(_to_messages(messages))
        return _message_text(response)

    async def chat_with_tools(
        self,
        messages: list[Any],
        tools: list[Any],
        model: str | None = None,
        temperature: float | None = None,
    ) -> AIMessage:
        """Invoke the configured model with tool definitions.

        Tool selection is intentionally owned by the model. The caller is
        responsible for executing returned tool calls and feeding ToolMessage
        results back into the conversation.
        """
        bound_model = self.get_model(model=model, temperature=temperature).bind_tools(tools)
        response = await bound_model.ainvoke(_to_messages(messages))
        return response


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
