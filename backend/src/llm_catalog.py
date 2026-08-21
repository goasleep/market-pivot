"""Built-in model metadata for configurable LLM profiles."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEEPSEEK_MODELS: dict[str, dict[str, Any]] = {
    "deepseek-v4-flash": {
        "max_tokens": 8192,
        "temperature": 0.3,
        "description": "DeepSeek V4 Flash for fast general-purpose analysis",
        "supports_tools": True,
        "supports_reasoning": True,
    },
    "deepseek-chat": {
        "max_tokens": 8192,
        "temperature": 0.3,
        "description": "General purpose chat model (V3)",
        "supports_tools": True,
        "supports_reasoning": False,
    },
    "deepseek-reasoner": {
        "max_tokens": 16384,
        "temperature": 0.0,
        "description": "Reasoning model (R1) for complex analysis",
        "supports_tools": True,
        "supports_reasoning": True,
    },
}

OPENAI_COMPATIBLE_MODELS: dict[str, dict[str, Any]] = {
    "gpt-5.6-sol": {
        "max_tokens": 128000,
        "temperature": 0.3,
        "description": "OpenAI GPT-5.6 Sol for frontier reasoning and coding",
        "supports_tools": True,
        "supports_reasoning": True,
    },
    "gpt-5.6-terra": {
        "max_tokens": 128000,
        "temperature": 0.3,
        "description": "OpenAI GPT-5.6 Terra for balanced intelligence and cost",
        "supports_tools": True,
        "supports_reasoning": True,
    },
    "gpt-5.6-luna": {
        "max_tokens": 128000,
        "temperature": 0.3,
        "description": "OpenAI GPT-5.6 Luna for cost-sensitive, high-volume workloads",
        "supports_tools": True,
        "supports_reasoning": True,
    },
    "gpt-4o-mini": {
        "max_tokens": 8192,
        "temperature": 0.3,
        "description": "OpenAI GPT-4o mini",
        "supports_tools": True,
        "supports_reasoning": False,
    },
    "gpt-4o": {
        "max_tokens": 8192,
        "temperature": 0.3,
        "description": "OpenAI GPT-4o",
        "supports_tools": True,
        "supports_reasoning": False,
    },
}


def default_profiles(
    *,
    deepseek_model: str = "deepseek-v4-flash",
) -> dict[str, dict[str, Any]]:
    return {
        "deepseek": {
            "id": "deepseek",
            "name": "DeepSeek",
            "type": "deepseek",
            "model": deepseek_model,
            "temperature": DEEPSEEK_MODELS.get(deepseek_model, {}).get("temperature", 0.3),
            "max_tokens": DEEPSEEK_MODELS.get(deepseek_model, {}).get("max_tokens", 8192),
        },
        "openai": {
            "id": "openai",
            "name": "OpenAI Compatible",
            "type": "openai_compatible",
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 8192,
        },
    }


def models_for_profile(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog = DEEPSEEK_MODELS if profile.get("type") == "deepseek" else OPENAI_COMPATIBLE_MODELS
    models = deepcopy(catalog)
    selected = str(profile.get("model") or "")
    if selected and selected not in models:
        models[selected] = {
            "max_tokens": int(profile.get("max_tokens", 8192)),
            "temperature": float(profile.get("temperature", 0.3)),
            "description": "Custom model",
            "supports_tools": True,
            "supports_reasoning": False,
        }
    return models


def model_info(profile: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    selected = model or str(profile.get("model", ""))
    info = models_for_profile(profile).get(selected)
    if isinstance(info, dict):
        return dict(info)
    return {
        "max_tokens": int(profile.get("max_tokens", 8192)),
        "temperature": float(profile.get("temperature", 0.3)),
        "description": "Custom model",
        "supports_tools": True,
        "supports_reasoning": False,
    }


MODEL_CONFIGS = DEEPSEEK_MODELS
