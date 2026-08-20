"""LLM configuration router — GET/PUT /api/config/llm.

Allows the frontend to read and update LLM settings (API key, model, etc.)
at runtime. Changes are persisted to the shared SQLite database and take effect immediately.
"""

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel

from config import get_llm_config, get_llm_state
from data.settings_store import update_llm_config

router = APIRouter()


def _mask_key(key: str) -> str:
    """Mask API key for safe display: show first 4 and last 4 chars."""
    if not key or len(key) <= 8:
        return "***" if key else ""
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


class LLMConfigResponse(BaseModel):
    active_profile_id: str
    profiles: dict[str, dict]
    routing: dict
    # Legacy active-profile fields retained for older clients.
    api_key_masked: str
    api_key_set: bool
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    available_models: dict[str, dict]


class LLMConfigUpdate(BaseModel):
    active_profile_id: str = ""
    profile_id: str = ""
    profile_name: str = ""
    provider_type: str = ""
    routing: dict | None = None
    models: dict[str, dict] | None = None
    # Legacy active-profile fields.
    api_key: str = ""  # empty = keep existing
    base_url: str = ""
    model: str = ""
    temperature: float | None = None
    max_tokens: int | None = None


def _profile_response(profile: dict) -> dict:
    return {
        "id": profile.get("id", ""),
        "name": profile.get("name", profile.get("id", "")),
        "type": profile.get("type", "openai_compatible"),
        "api_key_masked": _mask_key(str(profile.get("api_key", ""))),
        "api_key_set": bool(profile.get("api_key")),
        "base_url": profile.get("base_url", ""),
        "model": profile.get("model", ""),
        "temperature": profile.get("temperature", 0.3),
        "max_tokens": profile.get("max_tokens", 8192),
        "available_models": {
            key: {
                "description": value.get("description", "Configured model"),
                "max_tokens": value.get("max_tokens", profile.get("max_tokens", 8192)),
                "temperature": value.get("temperature", profile.get("temperature", 0.3)),
                "supports_tools": value.get("supports_tools", True),
                "supports_reasoning": value.get("supports_reasoning", False),
            }
            for key, value in (profile.get("models", {}) or {}).items()
            if isinstance(value, dict)
        },
    }


def _response() -> LLMConfigResponse:
    state = get_llm_state()
    profiles = {key: _profile_response(value) for key, value in state["profiles"].items()}
    active_id = state["active_profile_id"]
    active = profiles[active_id]
    return LLMConfigResponse(
        active_profile_id=active_id,
        profiles=profiles,
        routing=state.get("routing", {}),
        api_key_masked=active["api_key_masked"],
        api_key_set=active["api_key_set"],
        base_url=active["base_url"],
        model=active["model"],
        temperature=active["temperature"],
        max_tokens=active["max_tokens"],
        available_models=active["available_models"],
    )


@router.get("/llm", response_model=LLMConfigResponse)
async def get_llm_settings():
    """Get current LLM configuration (API key masked)."""
    return _response()


@router.put("/llm", response_model=LLMConfigResponse)
async def update_llm_settings(update: LLMConfigUpdate):
    """Update LLM configuration. Persists to SQLite, hot-reloads immediately."""
    # Build updates dict, skipping empty/null fields. A profile_id selects the
    # profile being edited; active_profile_id changes the default profile.
    updates = {}
    if update.active_profile_id:
        updates["active_profile_id"] = update.active_profile_id
    if update.profile_id:
        updates["profile_id"] = update.profile_id
    if update.profile_name:
        updates["name"] = update.profile_name
    if update.provider_type:
        updates["type"] = update.provider_type
    if update.routing is not None:
        updates["routing"] = update.routing
    if update.models is not None:
        updates["models"] = update.models
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
        return _response()

    await update_llm_config(updates)
    active = get_llm_config()
    logger.info(
        "LLM config updated via API: profile={}, model={}, base_url={}",
        active["profile_id"],
        active["model"],
        active["base_url"],
    )
    return _response()
