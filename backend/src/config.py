"""Configuration management for A-Share Agent backend."""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_catalog import default_profiles, model_info
from llm_runtime import current_llm_profile

# --- LLM config defaults (runtime state is loaded by the application startup) ---

_LLM_CONFIG_DEFAULTS = {
    "active_profile_id": "deepseek",
    "profiles": {},
    "routing": {
        "enabled": False,
        "routes": {
            "chat": {"profile_id": "deepseek", "model": "deepseek-v4-flash"},
            "analysis": {"profile_id": "deepseek", "model": "deepseek-v4-flash"},
            "report": {"profile_id": "deepseek", "model": "deepseek-chat"},
        },
    },
}


_runtime_llm_config: dict[str, Any] = {}


def _normalise_state(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Migrate the old flat DeepSeek config into the profile format."""
    value = raw if isinstance(raw, dict) else {}
    if isinstance(value.get("profiles"), dict) and value["profiles"]:
        state = deepcopy(value)
    else:
        legacy = value
        profiles = default_profiles(
            deepseek_api_key=str(legacy.get("api_key", settings.deepseek_api_key) or ""),
            deepseek_base_url=str(legacy.get("base_url", settings.deepseek_base_url) or ""),
            deepseek_model=str(legacy.get("model", settings.deepseek_model) or ""),
        )
        if "temperature" in legacy:
            profiles["deepseek"]["temperature"] = legacy["temperature"]
        if "max_tokens" in legacy:
            profiles["deepseek"]["max_tokens"] = legacy["max_tokens"]
        state = {
            "active_profile_id": "deepseek",
            "profiles": profiles,
            "routing": deepcopy(_LLM_CONFIG_DEFAULTS["routing"]),
        }

    state.setdefault("active_profile_id", "deepseek")
    state.setdefault("routing", deepcopy(_LLM_CONFIG_DEFAULTS["routing"]))
    routing = state["routing"]
    if not isinstance(routing, dict):
        routing = deepcopy(_LLM_CONFIG_DEFAULTS["routing"])
        state["routing"] = routing
    routing.setdefault("enabled", False)
    routing.setdefault("routes", {})
    for route, target in _LLM_CONFIG_DEFAULTS["routing"]["routes"].items():
        routing["routes"].setdefault(route, deepcopy(target))

    profiles = state["profiles"]
    for profile_id, profile in list(profiles.items()):
        if not isinstance(profile, dict):
            del profiles[profile_id]
            continue
        profile.setdefault("id", profile_id)
        profile.setdefault("name", profile_id)
        profile.setdefault("type", "openai_compatible")
        profile.setdefault("api_key", "")
        profile.setdefault("base_url", "")
        profile.setdefault("model", "")
        profile.setdefault("temperature", 0.3)
        profile.setdefault("max_tokens", 8192)
        profile.setdefault("models", {})
        if not isinstance(profile["models"], dict):
            profile["models"] = {}
        for model, info in profile["models"].items():
            if isinstance(info, dict):
                info.setdefault("max_tokens", profile["max_tokens"])
                info.setdefault("temperature", profile["temperature"])
                info.setdefault("description", "Configured model")
                info.setdefault("supports_tools", True)
                info.setdefault("supports_reasoning", False)

    if state["active_profile_id"] not in profiles:
        state["active_profile_id"] = next(iter(profiles), "deepseek")
    return state


def get_llm_state() -> dict[str, Any]:
    """Get the complete hot-reloadable profile configuration."""
    return deepcopy(_runtime_llm_config)


def get_llm_config(
    profile_id: str | None = None,
    model: str | None = None,
    route: str | None = None,
) -> dict[str, Any]:
    """Get one effective profile, respecting request-scoped selection."""
    scoped = current_llm_profile()
    if scoped is not None and profile_id is None and model is None and route is None:
        return scoped
    state = get_llm_state()
    profiles = state["profiles"]
    selected_id = profile_id or state["active_profile_id"]
    if route and state.get("routing", {}).get("enabled") and profile_id is None and model is None:
        target = state.get("routing", {}).get("routes", {}).get(route, {})
        selected_id = target.get("profile_id") or selected_id
        model = model or target.get("model")
    profile = deepcopy(profiles.get(selected_id) or profiles[state["active_profile_id"]])
    selected_model = model or profile.get("model", "")
    profile["profile_id"] = profile.get("id", selected_id)
    profile["model"] = selected_model
    info = model_info(profile, selected_model)
    profile["temperature"] = info.get("temperature", profile.get("temperature", 0.3))
    profile["max_tokens"] = info.get("max_tokens", profile.get("max_tokens", 8192))
    profile["model_info"] = info
    return profile


def resolve_llm_profile(
    profile_id: str | None = None,
    model: str | None = None,
    *,
    route: str = "chat",
    auto: bool = False,
) -> dict[str, Any]:
    """Resolve and snapshot a profile for one task."""
    return get_llm_config(
        profile_id=None if auto else profile_id,
        model=None if auto else model,
        route=route if auto else None,
    )


def save_llm_config(updates: dict) -> dict[str, Any]:
    """Update the in-memory LLM configuration.

    Args:
        updates: Partial dict of config keys to update.
                  api_key="" or api_key=None means "keep existing".

    Returns:
        The full updated config dict.
    """
    current = _normalise_state(updates if "profiles" in updates else _runtime_llm_config)
    if "profiles" in updates:
        current = _normalise_state(updates)
    else:
        profile_id = str(updates.get("profile_id") or current["active_profile_id"])
        profile = current["profiles"].setdefault(
            profile_id,
            {
                "id": profile_id,
                "name": profile_id,
                "type": "openai_compatible",
                "api_key": "",
                "base_url": "",
                "model": "",
                "temperature": 0.3,
                "max_tokens": 8192,
                "models": {},
            },
        )
        for key in ("name", "type", "base_url", "model", "temperature", "max_tokens", "models"):
            if key in updates and updates[key] is not None:
                profile[key] = updates[key]
        if updates.get("api_key"):
            profile["api_key"] = updates["api_key"]
        if "active_profile_id" in updates and updates["active_profile_id"] in current["profiles"]:
            current["active_profile_id"] = updates["active_profile_id"]
        if "routing" in updates:
            current["routing"] = updates["routing"]
    _runtime_llm_config.clear()
    _runtime_llm_config.update(_normalise_state(current))
    active = get_llm_config()
    logger.info("LLM config updated (profile={}, model={})", active["profile_id"], active["model"])
    return get_llm_state()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DeepSeek API
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-flash"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Live trading safety gate.  Keep disabled unless a reviewed broker
    # adapter and an isolated live account have been configured.
    live_trading_enabled: bool = False

    # Local SQLite database for single-node mode. DATABASE_URL selects the
    # PostgreSQL runtime backend for durable multi-node deployments.
    database_path: str = "./data/cache.db"
    data_cache_path: str | None = None  # backwards-compatible legacy environment variable
    database_url: str | None = None  # optional Tortoise URL for shared persistence

    # S3-compatible artifact storage.  The backend proxies preview/download
    # requests, so objects do not need to be public.
    s3_endpoint_url: str = ""
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_session_token: str = ""
    s3_addressing_style: str = "path"
    s3_artifacts_prefix: str = "a-share-agent/artifacts"

    # Serper web search. The search tool stays disabled until a key is set.
    serper_api_key: str = ""
    serper_base_url: str = "https://google.serper.dev"
    serper_gl: str = "cn"
    serper_hl: str = "zh-cn"
    # AnySearch web search. The API key is optional; without one the provider
    # uses AnySearch's anonymous tier when explicitly selected.
    anysearch_api_key: str = ""
    anysearch_base_url: str = "https://api.anysearch.com"
    anysearch_zone: str = "cn"
    anysearch_language: str = "zh-CN"
    ddgs_region: str = "cn-zh"
    ddgs_safesearch: str = "moderate"

    # LangSmith observability
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "a-share-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Langfuse observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @property
    def database_file_path(self) -> Path:
        p = Path(self.data_cache_path or self.database_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chat_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite://{self.database_file_path.expanduser().resolve()}"


settings = Settings()
_runtime_llm_config.update(
    _normalise_state(
        {
            "active_profile_id": "deepseek",
            "profiles": default_profiles(
                deepseek_api_key=settings.deepseek_api_key,
                deepseek_base_url=settings.deepseek_base_url,
                deepseek_model=settings.deepseek_model,
            ),
            "routing": deepcopy(_LLM_CONFIG_DEFAULTS["routing"]),
        }
    )
)


def configure_langsmith() -> None:
    """Expose settings to LangChain/LangGraph's automatic LangSmith tracer."""
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"  # compatibility with older LangChain releases
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    logger.info(f"LangSmith tracing enabled for project: {settings.langsmith_project}")


def configure_langfuse() -> None:
    """Expose Langfuse settings to its SDK and LangChain callback handler."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return

    # BaseSettings reads .env without mutating os.environ, while the Langfuse
    # SDK reads its standard environment variables directly.
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_base_url
    logger.info("Langfuse tracing enabled")


configure_langsmith()
configure_langfuse()
