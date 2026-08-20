"""Provider-neutral model factory."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from config import get_llm_config
from llm.deepseek import get_chat_model as get_deepseek_chat_model
from llm.openai_compatible import get_chat_model as get_openai_chat_model


def get_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    thinking: bool | None = None,
    profile_id: str | None = None,
    route: str | None = None,
) -> BaseChatModel:
    cfg: dict[str, Any] = get_llm_config(profile_id=profile_id, model=model, route=route)
    provider_type = cfg.get("type", "deepseek")
    if provider_type == "deepseek":
        return get_deepseek_chat_model(
            model=cfg["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            config=cfg,
        )
    if provider_type == "openai_compatible":
        return get_openai_chat_model(
            model=cfg["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            config=cfg,
        )
    raise ValueError(f"不支持的 LLM Provider 类型: {provider_type}")
