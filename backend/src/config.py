"""Configuration management for A-Share Agent backend."""

import json
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger


# --- LLM config persistence (JSON file, hot-reloadable) ---

_LLM_CONFIG_PATH = Path(__file__).parent.parent / "data" / "llm_config.json"

_LLM_CONFIG_DEFAULTS = {
    "api_key": "",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "temperature": 0.3,
    "max_tokens": 8192,
}


def _load_llm_config() -> dict:
    """Load LLM config from JSON file, falling back to env vars / defaults."""
    # Start with env defaults
    config = dict(_LLM_CONFIG_DEFAULTS)
    config["api_key"] = settings.deepseek_api_key
    config["base_url"] = settings.deepseek_base_url
    config["model"] = settings.deepseek_model

    # Override with persisted JSON if it exists
    if _LLM_CONFIG_PATH.exists():
        try:
            persisted = json.loads(_LLM_CONFIG_PATH.read_text(encoding="utf-8"))
            for key in _LLM_CONFIG_DEFAULTS:
                if key in persisted:
                    config[key] = persisted[key]
            logger.debug(f"Loaded LLM config from {_LLM_CONFIG_PATH}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load LLM config file: {e}, using env defaults")

    return config


def get_llm_config() -> dict:
    """Get current LLM configuration (hot-reads from file each time)."""
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

    # Persist to file
    _LLM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LLM_CONFIG_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"LLM config saved to {_LLM_CONFIG_PATH} (model={current['model']})")

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
    deepseek_model: str = "deepseek-chat"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Data cache
    data_cache_path: str = "./data/cache.db"

    # LangSmith observability
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "a-share-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    @property
    def cache_db_path(self) -> Path:
        p = Path(self.data_cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()


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


configure_langsmith()
