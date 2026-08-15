"""Configuration management for A-Share Agent backend."""

import json
import os
from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from data.database import SQLiteDatabase

# --- LLM config persistence (SQLite, hot-reloadable) ---

_LEGACY_LLM_CONFIG_PATH = Path(__file__).parent.parent / "data" / "llm_config.json"
_LLM_SETTINGS_KEY = "llm_config"

_LLM_CONFIG_DEFAULTS = {
    "api_key": "",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-v4-flash",
    "temperature": 0.3,
    "max_tokens": 8192,
}


def _load_llm_config() -> dict:
    """Load LLM config from SQLite, falling back to env vars / defaults."""
    # Start with env defaults
    config = dict(_LLM_CONFIG_DEFAULTS)
    config["api_key"] = settings.deepseek_api_key
    config["base_url"] = settings.deepseek_base_url
    config["model"] = settings.deepseek_model

    # Override with the persisted SQLite setting
    persisted = database.get_setting(_LLM_SETTINGS_KEY)
    if isinstance(persisted, dict):
        for key in _LLM_CONFIG_DEFAULTS:
            if key in persisted:
                config[key] = persisted[key]

    return config


def get_llm_config() -> dict:
    """Get current LLM configuration (hot-reads from SQLite each time)."""
    return _load_llm_config()


def save_llm_config(updates: dict) -> dict:
    """Update and persist LLM configuration.

    Args:
        updates: Partial dict of config keys to update.
                  api_key="" or api_key=None means "keep existing".

    Returns:
        The full updated config dict.
    """
    current = _load_llm_config()

    for key in _LLM_CONFIG_DEFAULTS:
        if key in updates:
            val = updates[key]
            # Skip empty api_key (don't overwrite existing key with empty)
            if key == "api_key" and not val:
                continue
            current[key] = val

    # Persist alongside the data cache in the shared SQLite database
    database.set_setting(_LLM_SETTINGS_KEY, current)
    logger.info(f"LLM config saved to SQLite (model={current['model']})")

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

    # Unified SQLite database (cache, settings, and future trading records)
    database_path: str = "./data/cache.db"
    data_cache_path: str | None = None  # backwards-compatible legacy environment variable

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


settings = Settings()
database = SQLiteDatabase(settings.database_file_path)


def _migrate_legacy_llm_config() -> None:
    """Import the old JSON config once, then remove the legacy file."""
    if database.get_setting(_LLM_SETTINGS_KEY) is not None or not _LEGACY_LLM_CONFIG_PATH.exists():
        return

    try:
        legacy = json.loads(_LEGACY_LLM_CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(legacy, dict):
            return
        migrated = {key: legacy[key] for key in _LLM_CONFIG_DEFAULTS if key in legacy}
        database.set_setting(_LLM_SETTINGS_KEY, migrated)
        _LEGACY_LLM_CONFIG_PATH.unlink()
        logger.info("Migrated LLM config from JSON into the shared SQLite database")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed to migrate legacy LLM config: {exc}")


_migrate_legacy_llm_config()


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
