"""LLM configuration router — GET/PUT /api/config/llm.

Allows the frontend to read and update LLM settings (API key, model, etc.)
at runtime. Changes are persisted to a JSON file and take effect immediately.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger

from config import get_llm_config, save_llm_config
from llm.deepseek import MODEL_CONFIGS

router = APIRouter()


def _mask_key(key: str) -> str:
    """Mask API key for safe display: show first 4 and last 4 chars."""
    if not key or len(key) <= 8:
        return "***" if key else ""
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


class LLMConfigResponse(BaseModel):
    api_key_masked: str
    api_key_set: bool
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    available_models: dict[str, dict]


class LLMConfigUpdate(BaseModel):
    api_key: str = ""  # empty = keep existing
    base_url: str = ""
    model: str = ""
    temperature: float | None = None
    max_tokens: int | None = None


@router.get("/llm", response_model=LLMConfigResponse)
async def get_llm_settings():
    """Get current LLM configuration (API key masked)."""
    cfg = get_llm_config()
    return LLMConfigResponse(
        api_key_masked=_mask_key(cfg["api_key"]),
        api_key_set=bool(cfg["api_key"]),
        base_url=cfg["base_url"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        available_models={
            k: {"description": v["description"], "max_tokens": v["max_tokens"]} for k, v in MODEL_CONFIGS.items()
        },
    )


@router.put("/llm", response_model=LLMConfigResponse)
async def update_llm_settings(update: LLMConfigUpdate):
    """Update LLM configuration. Persists to file, hot-reloads immediately."""
    # Build updates dict, skipping empty/null fields
    updates = {}
    if update.api_key:
        updates["api_key"] = update.api_key
    if update.base_url:
        updates["base_url"] = update.base_url
    if update.model:
        updates["model"] = update.model
    if update.temperature is not None:
        updates["temperature"] = update.temperature
    if update.max_tokens is not None:
        updates["max_tokens"] = update.max_tokens

    if not updates:
        return await get_llm_settings()

    cfg = save_llm_config(updates)
    logger.info(f"LLM config updated via API: model={cfg['model']}, base_url={cfg['base_url']}")

    return LLMConfigResponse(
        api_key_masked=_mask_key(cfg["api_key"]),
        api_key_set=bool(cfg["api_key"]),
        base_url=cfg["base_url"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        available_models={
            k: {"description": v["description"], "max_tokens": v["max_tokens"]} for k, v in MODEL_CONFIGS.items()
        },
    )
