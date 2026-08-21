"""LLM configuration router — GET/PUT /api/config/llm.

Allows the frontend to read and update non-secret LLM settings at runtime.
API keys and the base URL are resolved exclusively from environment variables.
"""

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, ConfigDict

from config import get_llm_base_url, get_llm_config, get_llm_state, llm_api_key_is_set
from data.settings_store import update_llm_config
from llm_catalog import models_for_profile

router = APIRouter()


class LLMConfigResponse(BaseModel):
    active_profile_id: str
    profiles: dict[str, dict]
    routing: dict
    api_key_set: bool
    base_url: str


class LLMConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_profile_id: str = ""
    profile_id: str = ""
    profile_name: str = ""
    provider_type: str = ""
    routing: dict | None = None
    model: str = ""
    temperature: float | None = None
    max_tokens: int | None = None


def _profile_response(profile: dict) -> dict:
    return {
        "id": profile.get("id", ""),
        "name": profile.get("name", profile.get("id", "")),
        "type": profile.get("type", "openai_compatible"),
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
            for key, value in models_for_profile(profile).items()
            if isinstance(value, dict)
        },
    }


def _response() -> LLMConfigResponse:
    state = get_llm_state()
    profiles = {key: _profile_response(value) for key, value in state["profiles"].items()}
    active_id = state["active_profile_id"]
    return LLMConfigResponse(
        active_profile_id=active_id,
        profiles=profiles,
        routing=state.get("routing", {}),
        api_key_set=llm_api_key_is_set(),
        base_url=get_llm_base_url(),
    )


@router.get("/llm", response_model=LLMConfigResponse)
async def get_llm_settings():
    """Get current non-secret LLM configuration and environment key status."""
    return _response()


@router.put("/llm", response_model=LLMConfigResponse)
async def update_llm_settings(update: LLMConfigUpdate):
    """Persist non-secret LLM configuration and hot-reload it immediately."""
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
