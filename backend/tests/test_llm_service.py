import pytest
from langchain_core.messages import AIMessage

from llm.service import LLMService, _normalize_generated_financial_text


def test_normalize_generated_financial_text_rewrites_gateway_false_positives():
    text = "这是一次技术性交易，不涉及性交内容。"

    normalized = _normalize_generated_financial_text(text)

    assert normalized == "这是一次技术性的交易，不涉及相关行为内容。"
    assert "性交易" not in normalized
    assert "性交" not in normalized


@pytest.mark.asyncio
async def test_chat_json_normalizes_generated_strings(monkeypatch):
    class FakeModel:
        async def ainvoke(self, _messages):
            return AIMessage(
                content='{"reasoning":"技术性交易", "details":{"note":"性交"}}'
            )

    service = LLMService()
    monkeypatch.setattr(service, "get_model", lambda **_kwargs: FakeModel())

    result = await service.chat_json("prompt")

    assert result == {
        "reasoning": "技术性的交易",
        "details": {"note": "相关行为"},
    }


@pytest.mark.asyncio
async def test_chat_with_tools_normalizes_generated_content(monkeypatch):
    class FakeModel:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(content="技术性交易与性交")

    service = LLMService()
    monkeypatch.setattr(service, "get_model", lambda **_kwargs: FakeModel())

    result = await service.chat_with_tools([], [])

    assert result.content == "技术性的交易与相关行为"
