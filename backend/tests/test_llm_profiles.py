from config import get_llm_config, get_llm_state, resolve_llm_profile
from llm import openai_compatible


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
