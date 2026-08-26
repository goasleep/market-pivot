"""DeepSeek provider implementation backed by LangChain's ChatDeepSeek."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

from config import get_llm_config
from llm_catalog import MODEL_CONFIGS


def get_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    thinking: bool | None = None,
    profile_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Create a LangChain chat model from the current application config.

    A fresh model is created for every logical call so changes made through
    the settings API take effect without restarting the backend.
    """
    cfg = config or (
        get_llm_config() if profile_id is None and model is None else get_llm_config(profile_id=profile_id, model=model)
    )
    selected_model = model or cfg["model"]
    preset = cfg.get("model_info") or MODEL_CONFIGS.get(selected_model, {})

    effective_temperature = (
        temperature if temperature is not None else cfg.get("temperature", preset.get("temperature", 0.3))
    )
    effective_max_tokens = (
        max_tokens if max_tokens is not None else cfg.get("max_tokens", preset.get("max_tokens", 8192))
    )

    # DeepSeek V4 enables Thinking by default.  Its Thinking mode supports
    # model-selected tool calls, but rejects forced/specific tool_choice values
    # used by structured-output and agent runtimes.  Keep normal chat calls on
    # the provider default, while allowing tool orchestration to explicitly
    # disable Thinking for the affected V4 models.
    extra_body = None
    if thinking is not None and selected_model.startswith("deepseek-v4-"):
        extra_body = {"thinking": {"type": "enabled" if thinking else "disabled"}}

    model_kwargs = {
        "model": selected_model,
        "api_key": cfg["api_key"],
        "base_url": cfg["base_url"],
        "temperature": effective_temperature,
        "max_tokens": effective_max_tokens,
        "max_retries": 2,
    }
    if extra_body is not None:
        model_kwargs["extra_body"] = extra_body

    return ChatDeepSeek(**model_kwargs)
