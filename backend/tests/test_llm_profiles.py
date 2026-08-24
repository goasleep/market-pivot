import pytest

from api.routers.config import _response, router
from config import get_llm_config, get_llm_state, resolve_llm_profile
from llm import factory, openai_compatible
from llm_catalog import OPENAI_COMPATIBLE_MODELS, default_profiles, models_for_profile, supports_vision


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
            "profile_id": "environment",
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


def test_environment_state_contains_one_effective_profile(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "4096")

    state = get_llm_state()
    profile = state["profiles"]["environment"]

    assert state["active_profile_id"] == "environment"
    assert state["routing"] == {"enabled": False, "routes": {}}
    assert profile == {
        "context_window": 128000,
        "id": "environment",
        "name": "Environment",
        "type": "openai_compatible",
        "model": "gpt-5.6-sol",
        "temperature": 0.2,
        "max_tokens": 4096,
    }


def test_environment_context_window_override_is_exposed(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "32768")

    assert get_llm_config()["context_window"] == 32768
    assert get_llm_state()["profiles"]["environment"]["context_window"] == 32768


def test_request_overrides_do_not_replace_environment_config(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")

    selected = resolve_llm_profile("deepseek", "deepseek-chat", route="analysis", auto=True)

    assert selected["profile_id"] == "environment"
    assert selected["type"] == "openai_compatible"
    assert selected["model"] == "gpt-4o-mini"


def test_openai_profile_keeps_gpt_5_6_catalog_without_connection_settings():
    profile = default_profiles()["openai"]
    models = models_for_profile(profile)

    assert "base_url" not in profile
    assert "models" not in profile
    assert {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}.issubset(models)
    for model_id in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        assert OPENAI_COMPATIBLE_MODELS[model_id]["context_window"] == 128000
        assert OPENAI_COMPATIBLE_MODELS[model_id]["max_output_tokens"] == 8192
        assert OPENAI_COMPATIBLE_MODELS[model_id]["max_tokens"] == 128000
        assert OPENAI_COMPATIBLE_MODELS[model_id]["supports_tools"] is True
        assert OPENAI_COMPATIBLE_MODELS[model_id]["supports_reasoning"] is True


def test_vision_support_uses_explicit_gpt_5_6_allowlist():
    for model_id in ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        assert supports_vision(model_id) is True
    for model_id in ("gpt-4o", "gpt-4o-mini", "deepseek-chat", "custom-vision-model"):
        assert supports_vision(model_id) is False


def test_environment_profile_resolves_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    assert get_llm_config()["api_key"] == "environment-secret"
    assert "api_key" not in get_llm_state()["profiles"]["environment"]


def test_environment_profile_resolves_openai_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

    assert get_llm_config()["base_url"] == "https://gateway.example/v1"
    assert "base_url" not in get_llm_state()["profiles"]["environment"]


def test_blank_openai_base_url_uses_official_default(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "  ")

    assert get_llm_config()["base_url"] == "https://api.openai.com/v1"


def test_openai_compatible_base_url_requires_api_path(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example")

    with pytest.raises(ValueError, match="usually /v1"):
        get_llm_config()


def test_settings_response_exposes_read_only_environment_status(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

    response = _response()

    assert response.config_source == "environment"
    assert response.provider_type == "openai_compatible"
    assert response.model == "gpt-5.6-sol"
    assert response.api_key_set is True
    assert response.base_url == "https://gateway.example/v1"
    assert set(response.model_dump()) == {
        "config_source",
        "provider_type",
        "model",
        "temperature",
        "context_window",
        "max_tokens",
        "api_key_set",
        "base_url",
    }


def test_settings_router_is_read_only():
    llm_routes = [route for route in router.routes if route.path == "/llm"]

    assert llm_routes
    assert all("PUT" not in route.methods for route in llm_routes)


def test_invalid_environment_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        get_llm_config()


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
