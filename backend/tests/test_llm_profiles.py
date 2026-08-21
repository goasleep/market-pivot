from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from api.routers.config import LLMConfigUpdate, _profile_response, _response
from config import get_llm_config, get_llm_state, resolve_llm_profile
from data import settings_store
from llm import factory, openai_compatible
from llm_catalog import OPENAI_COMPATIBLE_MODELS, default_profiles, models_for_profile


def test_openai_compatible_profile_builds_chat_model(monkeypatch):
    captured: list[dict] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(openai_compatible, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        openai_compatible,
        "get_llm_config",
        lambda **_: {
            "profile_id": "openai",
            "type": "openai_compatible",
            "api_key": "test-key",
            "base_url": "https://example.test/v1",
            "model": "custom-model",
            "temperature": 0.2,
            "max_tokens": 2048,
            "model_info": {"temperature": 0.2, "max_tokens": 2048},
        },
    )

    openai_compatible.get_chat_model()

    assert captured[-1] == {
        "model": "custom-model",
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
        "temperature": 0.2,
        "max_tokens": 2048,
        "max_retries": 2,
    }


def test_default_state_contains_profiles_and_routes():
    state = get_llm_state()
    assert {"deepseek", "openai"}.issubset(state["profiles"])
    assert state["profiles"]["openai"]["type"] == "openai_compatible"
    assert "analysis" in state["routing"]["routes"]

    selected = resolve_llm_profile("openai", "gpt-4o-mini")
    assert selected["profile_id"] == "openai"
    assert selected["model"] == "gpt-4o-mini"
    assert get_llm_config(profile_id="openai")["type"] == "openai_compatible"


def test_openai_profile_keeps_gpt_5_6_catalog_without_connection_settings():
    profile = default_profiles()["openai"]
    models = models_for_profile(profile)

    assert "base_url" not in profile
    assert "models" not in profile
    assert {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}.issubset(models)
    for model_id in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        assert OPENAI_COMPATIBLE_MODELS[model_id]["max_tokens"] == 128000
        assert OPENAI_COMPATIBLE_MODELS[model_id]["supports_tools"] is True
        assert OPENAI_COMPATIBLE_MODELS[model_id]["supports_reasoning"] is True


def test_all_profiles_resolve_shared_openai_api_key(monkeypatch):
    original = get_llm_state()
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    try:
        from config import save_llm_config

        state = save_llm_config(original)

        assert all("api_key" not in profile for profile in state["profiles"].values())
        assert get_llm_config(profile_id="openai")["api_key"] == "environment-secret"
        assert get_llm_config(profile_id="deepseek")["api_key"] == "environment-secret"
    finally:
        save_llm_config(original)


def test_all_profiles_resolve_shared_openai_base_url(monkeypatch):
    original = get_llm_state()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    try:
        from config import save_llm_config

        state = save_llm_config(original)

        assert all("base_url" not in profile for profile in state["profiles"].values())
        assert get_llm_config(profile_id="openai")["base_url"] == "https://gateway.example/v1"
        assert get_llm_config(profile_id="deepseek")["base_url"] == "https://gateway.example/v1"
    finally:
        save_llm_config(original)


def test_blank_openai_base_url_uses_official_default(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "  ")

    assert get_llm_config(profile_id="openai")["base_url"] == "https://api.openai.com/v1"


def test_settings_response_exposes_environment_connection_status(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    profile = _profile_response(get_llm_state()["profiles"]["openai"])
    response = _response()

    assert response.api_key_set is True
    assert response.base_url == "https://gateway.example/v1"
    assert set(response.model_dump()) == {"active_profile_id", "profiles", "routing", "api_key_set", "base_url"}
    assert "api_key_set" not in profile
    assert "base_url" not in profile


@pytest.mark.asyncio
async def test_settings_store_ignores_non_profile_payload(monkeypatch):
    original = get_llm_state()

    async def fake_init_database():
        return None

    class FakeAppSetting:
        @classmethod
        async def get_or_none(cls, **_kwargs):
            return SimpleNamespace(value='{"model":"unsupported-flat-config"}')

    monkeypatch.setattr(settings_store, "init_database", fake_init_database)
    monkeypatch.setattr(settings_store, "AppSetting", FakeAppSetting)

    assert await settings_store.load_llm_config() == original


def test_settings_rejects_plaintext_api_key():
    with pytest.raises(ValidationError) as exc_info:
        LLMConfigUpdate(api_key="must-not-be-persisted")

    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


def test_settings_rejects_persisted_base_url():
    with pytest.raises(ValidationError) as exc_info:
        LLMConfigUpdate(base_url="https://must-not-be-persisted.example/v1")

    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


def test_deepseek_named_model_does_not_override_openai_compatible_provider(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        factory,
        "get_llm_config",
        lambda **_: {
            "type": "openai_compatible",
            "model": "deepseek-chat",
        },
    )
    monkeypatch.setattr(factory, "get_openai_chat_model", lambda **_: sentinel)

    def fail_if_native_adapter_is_selected(**_kwargs):
        raise AssertionError("DeepSeek native adapter must not be selected from the model name")

    monkeypatch.setattr(factory, "get_deepseek_chat_model", fail_if_native_adapter_is_selected)

    assert factory.get_chat_model() is sentinel
