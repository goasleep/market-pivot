"""Configuration management for A-Share Agent backend."""

import os
from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- LLM config defaults (runtime state is loaded by the application startup) ---

_LLM_CONFIG_DEFAULTS = {
    "api_key": "",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-v4-flash",
    "temperature": 0.3,
    "max_tokens": 8192,
}


_runtime_llm_config: dict = {}


def get_llm_config() -> dict:
    """Get the hot-reloadable in-memory LLM configuration."""
    return dict(_runtime_llm_config)


def save_llm_config(updates: dict) -> dict:
    """Update the in-memory LLM configuration.

    Args:
        updates: Partial dict of config keys to update.
                  api_key="" or api_key=None means "keep existing".

    Returns:
        The full updated config dict.
    """
    current = get_llm_config()

    for key in _LLM_CONFIG_DEFAULTS:
        if key in updates:
            val = updates[key]
            # Skip empty api_key (don't overwrite existing key with empty)
            if key == "api_key" and not val:
                continue
            current[key] = val

    _runtime_llm_config.update(current)
    logger.info(f"LLM config updated (model={current['model']})")

    return current


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
_runtime_llm_config.update(_LLM_CONFIG_DEFAULTS)
_runtime_llm_config.update(
    {
        "api_key": settings.deepseek_api_key,
        "base_url": settings.deepseek_base_url,
        "model": settings.deepseek_model,
    }
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
