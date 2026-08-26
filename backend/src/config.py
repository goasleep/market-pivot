"""Configuration management for A-Share Agent backend."""

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_catalog import model_info
from llm_runtime import current_llm_profile

# --- Environment-owned LLM configuration ---

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _BACKEND_ROOT.parent
_ENV_FILES = (_REPOSITORY_ROOT / ".env", _BACKEND_ROOT / ".env")
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_ENVIRONMENT_PROFILE_ID = "environment"
_SUPPORTED_LLM_PROVIDERS = {"deepseek", "openai_compatible"}


def _environment_value(name: str, fallback: str) -> str:
    value = os.getenv(name)
    return (value if value is not None else fallback).strip()


def get_llm_provider_type() -> str:
    """Return the provider adapter selected by LLM_PROVIDER."""
    provider = _environment_value("LLM_PROVIDER", settings.llm_provider).lower()
    if provider not in _SUPPORTED_LLM_PROVIDERS:
        supported = ", ".join(sorted(_SUPPORTED_LLM_PROVIDERS))
        raise ValueError(f"LLM_PROVIDER must be one of: {supported}")
    return provider


def get_llm_model() -> str:
    """Return the model ID selected by LLM_MODEL."""
    model = _environment_value("LLM_MODEL", settings.llm_model)
    if not model:
        raise ValueError("LLM_MODEL must not be empty")
    return model


def get_llm_temperature() -> float:
    """Return the generation temperature selected by LLM_TEMPERATURE."""
    value = _environment_value("LLM_TEMPERATURE", str(settings.llm_temperature))
    try:
        temperature = float(value)
    except ValueError as exc:
        raise ValueError("LLM_TEMPERATURE must be a number") from exc
    if not 0 <= temperature <= 2:
        raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
    return temperature


def get_llm_max_tokens() -> int:
    """Return the requested output limit, preferring the unambiguous new variable."""
    preferred = os.getenv("LLM_MAX_OUTPUT_TOKENS")
    if preferred is None and settings.llm_max_output_tokens is not None:
        preferred = str(settings.llm_max_output_tokens)
    value = preferred if preferred is not None else _environment_value("LLM_MAX_TOKENS", str(settings.llm_max_tokens))
    value = value.strip()
    try:
        max_tokens = int(value)
    except ValueError as exc:
        raise ValueError("LLM_MAX_OUTPUT_TOKENS must be an integer") from exc
    if max_tokens < 256:
        raise ValueError("LLM_MAX_OUTPUT_TOKENS must be at least 256")
    return max_tokens


def get_llm_context_window(default: int) -> int:
    """Return the model input/output context window, with an optional environment override."""
    configured_default = settings.llm_context_window or default
    value = _environment_value("LLM_CONTEXT_WINDOW", str(configured_default))
    try:
        context_window = int(value)
    except ValueError as exc:
        raise ValueError("LLM_CONTEXT_WINDOW must be an integer") from exc
    if context_window < 4096:
        raise ValueError("LLM_CONTEXT_WINDOW must be at least 4096")
    return context_window


def _environment_profile() -> dict[str, Any]:
    requested_output_tokens = get_llm_max_tokens()
    profile = {
        "id": _ENVIRONMENT_PROFILE_ID,
        "name": "Environment",
        "type": get_llm_provider_type(),
        "model": get_llm_model(),
        "temperature": get_llm_temperature(),
        "max_tokens": requested_output_tokens,
    }
    profile["model_info"] = model_info(profile, profile["model"])
    profile["context_window"] = get_llm_context_window(int(profile["model_info"]["context_window"]))
    profile["model_info"]["context_window"] = profile["context_window"]
    model_output_limit = int(profile["model_info"].get("max_output_tokens", requested_output_tokens))
    effective_output_tokens = min(requested_output_tokens, model_output_limit)
    if effective_output_tokens != requested_output_tokens:
        logger.warning(
            "Configured output limit {} exceeds model catalog limit {}; using {}",
            requested_output_tokens,
            model_output_limit,
            effective_output_tokens,
        )
    profile["max_tokens"] = effective_output_tokens
    profile["model_info"]["max_tokens"] = effective_output_tokens
    profile["model_info"]["max_output_tokens"] = effective_output_tokens
    return profile


def get_llm_state() -> dict[str, Any]:
    """Return the secret-free LLM state derived exclusively from the environment."""
    profile = _environment_profile()
    public_profile = {key: value for key, value in profile.items() if key != "model_info"}
    return {
        "active_profile_id": _ENVIRONMENT_PROFILE_ID,
        "profiles": {_ENVIRONMENT_PROFILE_ID: public_profile},
        "routing": {"enabled": False, "routes": {}},
    }


def _read_api_key() -> str:
    """Resolve the shared LLM key from OPENAI_API_KEY."""
    value = os.getenv("OPENAI_API_KEY")
    if value is not None:
        return value.strip()
    return settings.openai_api_key.strip()


def llm_api_key_is_set() -> bool:
    """Return whether OPENAI_API_KEY has a value."""
    return bool(_read_api_key())


def get_llm_base_url() -> str:
    """Resolve the shared LLM endpoint from OPENAI_BASE_URL."""
    value = os.getenv("OPENAI_BASE_URL")
    configured = value if value is not None else settings.openai_base_url
    return configured.strip() or _DEFAULT_OPENAI_BASE_URL


def _validate_llm_base_url(provider_type: str, base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OPENAI_BASE_URL must be an absolute HTTP(S) URL")
    if provider_type == "openai_compatible" and parsed.path.rstrip("/") == "":
        raise ValueError("OPENAI_BASE_URL for an OpenAI-compatible provider must include its API path, usually /v1")


def get_llm_config(
    profile_id: str | None = None,
    model: str | None = None,
    route: str | None = None,
) -> dict[str, Any]:
    """Get the environment-owned profile, respecting a task snapshot when present."""
    scoped = current_llm_profile()
    if scoped is not None:
        return scoped
    _ = (profile_id, model, route)  # Kept for compatibility with existing callers; overrides are intentionally ignored.
    profile = _environment_profile()
    profile["profile_id"] = _ENVIRONMENT_PROFILE_ID
    profile["api_key"] = _read_api_key()
    profile["base_url"] = get_llm_base_url()
    _validate_llm_base_url(profile["type"], profile["base_url"])
    return profile


def resolve_llm_profile(
    profile_id: str | None = None,
    model: str | None = None,
    *,
    route: str = "chat",
    auto: bool = False,
) -> dict[str, Any]:
    """Resolve and snapshot the environment-owned profile for one task."""
    _ = (profile_id, model, route, auto)
    return get_llm_config()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment-owned LLM settings. The frontend exposes these as read-only status.
    llm_provider: str = "openai_compatible"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 8192
    llm_max_output_tokens: int | None = None
    llm_context_window: int = 0

    # Shared environment-owned connection settings.
    openai_api_key: str = ""
    openai_base_url: str = _DEFAULT_OPENAI_BASE_URL

    # Server
    host: str = "0.0.0.0"
    port: int = 18000

    # Live trading safety gate.  Keep disabled unless a reviewed broker
    # adapter and an isolated live account have been configured.
    live_trading_enabled: bool = False
    automation_max_concurrency: int = 3

    # Local SQLite database for single-node mode. DATABASE_URL selects the
    # PostgreSQL runtime backend for durable multi-node deployments.
    database_path: str = "./data/cache.db"
    database_url: str | None = None  # optional Tortoise URL for shared persistence

    # LangGraph owns these checkpoint tables/files.  Local mode deliberately
    # uses a separate SQLite file so graph super-step writes do not contend
    # with chat and market-cache transactions.
    checkpoint_database_path: str = "./data/checkpoints.db"
    checkpoint_database_url: str | None = None
    checkpoint_retention_days: int = 30

    # S3-compatible artifact storage.  The backend proxies preview/download
    # requests, so objects do not need to be public.
    s3_endpoint_url: str = ""
    s3_public_endpoint_url: str = ""
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_session_token: str = ""
    s3_addressing_style: str = "path"
    s3_artifacts_prefix: str = "a-share-agent/artifacts"

    # Dedicated S3-compatible storage for immutable historical market data.
    # Keep this separate from generated artifacts so credentials, retention,
    # and storage lifecycle policies can evolve independently.
    market_history_cache_enabled: bool = False
    market_history_s3_endpoint_url: str = ""
    market_history_s3_bucket: str = ""
    market_history_s3_region: str = "us-east-1"
    market_history_s3_access_key_id: str = ""
    market_history_s3_secret_access_key: str = ""
    market_history_s3_session_token: str = ""
    market_history_s3_addressing_style: str = "path"
    market_history_s3_prefix: str = "a-share-agent/market-history/v1"

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
        p = Path(self.database_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chat_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite://{self.database_file_path.expanduser().resolve()}"

    @property
    def checkpoint_file_path(self) -> Path:
        path = Path(self.checkpoint_database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.expanduser().resolve()

    @property
    def resolved_checkpoint_database_url(self) -> str | None:
        """Return the configured PostgreSQL checkpoint URL, if any."""
        return self.checkpoint_database_url or self.database_url


settings = Settings()


def configure_langsmith() -> None:
    """Expose settings to LangChain/LangGraph's automatic LangSmith tracer."""
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        return

    os.environ["LANGSMITH_TRACING"] = "true"
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
