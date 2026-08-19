from llm import deepseek


def test_v4_tool_model_disables_thinking(monkeypatch):
    captured: list[dict] = []

    class FakeChatDeepSeek:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(
        deepseek,
        "get_llm_config",
        lambda: {
            "api_key": "test-key",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-v4-flash",
            "temperature": 0.1,
            "max_tokens": 1024,
        },
    )
    monkeypatch.setattr(deepseek, "ChatDeepSeek", FakeChatDeepSeek)

    deepseek.get_chat_model(thinking=False)
    assert captured[-1]["extra_body"] == {"thinking": {"type": "disabled"}}

    deepseek.get_chat_model()
    assert "extra_body" not in captured[-1]
