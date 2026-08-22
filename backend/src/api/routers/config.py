"""Read-only environment-backed LLM configuration router."""

from fastapi import APIRouter
from pydantic import BaseModel

from config import get_llm_config, llm_api_key_is_set

router = APIRouter()


class LLMConfigResponse(BaseModel):
    config_source: str
    provider_type: str
    model: str
    temperature: float
    context_window: int
    max_tokens: int
    api_key_set: bool
    base_url: str


def _response() -> LLMConfigResponse:
    config = get_llm_config()
    return LLMConfigResponse(
        config_source="environment",
        provider_type=config["type"],
        model=config["model"],
        temperature=config["temperature"],
        context_window=config["context_window"],
        max_tokens=config["max_tokens"],
        api_key_set=llm_api_key_is_set(),
        base_url=config["base_url"],
    )


@router.get("/llm", response_model=LLMConfigResponse)
async def get_llm_settings():
    """Get the effective non-secret LLM environment configuration."""
    return _response()
