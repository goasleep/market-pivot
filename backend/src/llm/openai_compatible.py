"""OpenAI and OpenAI-compatible chat model provider."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import get_llm_config


def get_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    profile_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Create a ChatOpenAI instance for an OpenAI-compatible endpoint."""
    cfg = config or get_llm_config(profile_id=profile_id, model=model)
    info = cfg.get("model_info", {})
    selected_model = model or cfg["model"]
    effective_temperature = temperature if temperature is not None else info.get("temperature", cfg["temperature"])
    effective_max_tokens = max_tokens if max_tokens is not None else info.get("max_tokens", cfg["max_tokens"])
    return ChatOpenAI(
        model=selected_model,
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        temperature=effective_temperature,
        max_tokens=effective_max_tokens,
        max_retries=2,
    )
