"""DeepSeek provider implementation backed by LangChain's ChatDeepSeek."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

from config import get_llm_config

# Model presets are provider metadata. The active model and runtime parameters
# still come from the hot-reloadable LLM configuration.
MODEL_CONFIGS = {
    "deepseek-v4-flash": {
        "max_tokens": 8192,
        "temperature": 0.3,
        "description": "DeepSeek V4 Flash for fast general-purpose analysis",
    },
    "deepseek-chat": {
        "max_tokens": 8192,
        "temperature": 0.3,
        "description": "General purpose chat model (V3)",
    },
    "deepseek-reasoner": {
        "max_tokens": 16384,
        "temperature": 0.0,
        "description": "Reasoning model (R1) for complex analysis",
    },
}


def get_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Create a LangChain chat model from the current application config.

    A fresh model is created for every logical call so changes made through
    the settings API take effect without restarting the backend.
    """
    cfg = get_llm_config()
    selected_model = model or cfg["model"]
    preset = MODEL_CONFIGS.get(selected_model, {})

    effective_temperature = (
        temperature if temperature is not None else cfg.get("temperature", preset.get("temperature", 0.3))
    )
    effective_max_tokens = (
        max_tokens if max_tokens is not None else cfg.get("max_tokens", preset.get("max_tokens", 8192))
    )

    return ChatDeepSeek(
        model=selected_model,
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        temperature=effective_temperature,
        max_tokens=effective_max_tokens,
        max_retries=2,
    )
