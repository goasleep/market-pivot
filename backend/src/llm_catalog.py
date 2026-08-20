"""Built-in model metadata for configurable LLM profiles."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEEPSEEK_MODELS: dict[str, dict[str, Any]] = {
    "deepseek-v4-flash": {
        "max_tokens": 8192, "temperature": 0.3,
        "description": "DeepSeek V4 Flash for fast general-purpose analysis",
        "supports_tools": True, "supports_reasoning": True,
    },
    "deepseek-chat": {
        "max_tokens": 8192, "temperature": 0.3,
        "description": "General purpose chat model (V3)",
        "supports_tools": True, "supports_reasoning": False,
    },
    "deepseek-reasoner": {
        "max_tokens": 16384, "temperature": 0.0,
        "description": "Reasoning model (R1) for complex analysis",
        "supports_tools": True, "supports_reasoning": True,
    },
}

OPENAI_COMPATIBLE_MODELS: dict[str, dict[str, Any]] = {
    "gpt-4o-mini": {
        "max_tokens": 8192, "temperature": 0.3,
        "description": "OpenAI GPT-4o mini",
        "supports_tools": True, "supports_reasoning": False,
    },
    "gpt-4o": {
        "max_tokens": 8192, "temperature": 0.3,
        "description": "OpenAI GPT-4o",
        "supports_tools": True, "supports_reasoning": False,
    },
}


def default_profiles(
    *,
    deepseek_api_key: str = "",
    deepseek_base_url: str = "https://api.deepseek.com/v1",
    deepseek_model: str = "deepseek-v4-flash",
) -> dict[str, dict[str, Any]]:
    return {
        "deepseek": {
            "id": "deepseek", "name": "DeepSeek", "type": "deepseek",
            "api_key": deepseek_api_key, "base_url": deepseek_base_url,
            "model": deepseek_model,
            "temperature": DEEPSEEK_MODELS.get(deepseek_model, {}).get("temperature", 0.3),
            "max_tokens": DEEPSEEK_MODELS.get(deepseek_model, {}).get("max_tokens", 8192),
            "models": deepcopy(DEEPSEEK_MODELS),
        },
        "openai": {
            "id": "openai", "name": "OpenAI Compatible", "type": "openai_compatible",
            "api_key": "", "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 8192,
            "models": deepcopy(OPENAI_COMPATIBLE_MODELS),
        },
    }


def model_info(profile: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    selected = model or str(profile.get("model", ""))
    models = profile.get("models") if isinstance(profile.get("models"), dict) else {}
    info = models.get(selected) if isinstance(models, dict) else None
    if isinstance(info, dict):
        return dict(info)
    return {
        "max_tokens": int(profile.get("max_tokens", 8192)),
        "temperature": float(profile.get("temperature", 0.3)),
        "description": "Custom model", "supports_tools": True, "supports_reasoning": False,
    }


MODEL_CONFIGS = DEEPSEEK_MODELS
