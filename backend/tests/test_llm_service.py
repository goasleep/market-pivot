import json

import pytest
from langchain_core.messages import AIMessage

import llm.context as context_module
import llm.service as service_module
from llm.context import ContextBudget
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


@pytest.mark.asyncio
async def test_chat_transparently_retries_provider_context_rejection(monkeypatch):
    class FakeModel:
        calls = 0

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("maximum context length exceeded")
            return AIMessage(content="恢复后的结果")

    model = FakeModel()
    service = LLMService()
    monkeypatch.setattr(service, "get_model", lambda **_kwargs: model)

    result = await service.chat('{"request":"分析 510300"}', system="研究助手")

    assert result == "恢复后的结果"
    assert model.calls == 2


@pytest.mark.asyncio
async def test_chat_projects_oversized_application_json_before_model_call(monkeypatch):
    budget = ContextBudget(
        model="gpt-4o-mini",
        context_window=2048,
        output_reserve=512,
        safety_margin=512,
        input_limit=1024,
    )
    captured = []

    class FakeModel:
        async def ainvoke(self, messages):
            captured.extend(messages)
            return AIMessage(content="已处理")

    service = LLMService()
    monkeypatch.setattr(service, "get_model", lambda **_kwargs: FakeModel())
    monkeypatch.setattr(context_module, "get_context_budget", lambda **_kwargs: budget)
    monkeypatch.setattr(service_module, "get_context_budget", lambda **_kwargs: budget)
    prompt = json.dumps({"evidence": ["大量证据" * 5000]}, ensure_ascii=False)

    result = await service.chat(prompt, system="研究助手")

    assert result == "已处理"
    assert len(captured) == 2
    assert "_context_compacted" in str(captured[-1].content)
