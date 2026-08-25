import asyncio
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
async def test_chat_json_with_images_sends_absolute_urls_as_content_blocks(monkeypatch):
    captured = []

    class FakeModel:
        async def ainvoke(self, messages):
            captured.extend(messages)
            return AIMessage(content='{"signal":"hold"}')

    service = LLMService()
    monkeypatch.setattr(service, "get_model", lambda **_kwargs: FakeModel())

    result = await service.chat_json_with_images(
        "分析图表",
        ["https://objects.example.test/charts/technical.png?signature=test"],
        system="研究助手",
    )

    assert result == {"signal": "hold"}
    user_content = captured[-1].content
    assert user_content[0] == {"type": "text", "text": "分析图表"}
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("https://")


@pytest.mark.asyncio
async def test_chat_json_with_images_rejects_non_http_input():
    with pytest.raises(ValueError, match="absolute HTTP"):
        await LLMService().chat_json_with_images("分析", ["data:image/png;base64,AAAA"])


@pytest.mark.asyncio
async def test_chat_json_with_images_times_out_to_text_path(monkeypatch):
    class SlowVisionModel:
        async def ainvoke(self, _messages):
            import asyncio

            await asyncio.sleep(1)

    class TextModel:
        async def ainvoke(self, _messages):
            return AIMessage(content='{"signal":"hold","fallback":true}')

    models = iter((SlowVisionModel(), TextModel()))
    service = LLMService()
    monkeypatch.setattr(service, "get_model", lambda **_kwargs: next(models))
    monkeypatch.setattr(service_module, "VISION_CALL_TIMEOUT_SECONDS", 0.01)

    result = await service.chat_json_with_images(
        "分析图表",
        ["https://objects.example.test/chart.png"],
    )

    assert result == {"signal": "hold", "fallback": True}


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
async def test_async_model_calls_use_thirty_minute_timeout(monkeypatch):
    captured: list[float] = []
    original_wait = service_module.asyncio.wait

    async def capture_wait(awaitables, timeout):
        captured.append(timeout)
        return await original_wait(awaitables, timeout=timeout)

    class FakeModel:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(content="完成")

    service = LLMService()
    monkeypatch.setattr(service, "get_model", lambda **_kwargs: FakeModel())
    monkeypatch.setattr(service_module.asyncio, "wait", capture_wait)

    await service.chat("问题")
    await service.chat_langchain([])
    await service.chat_with_tools([], [])

    assert captured == [1800, 1800, 1800]


@pytest.mark.asyncio
async def test_async_model_hard_timeout_does_not_wait_for_provider_cancellation(monkeypatch):
    finished = asyncio.Event()

    class CancellationDelayingModel:
        async def ainvoke(self, _messages):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0.08)
                finished.set()
                return AIMessage(content="迟到结果")

    service = LLMService()
    monkeypatch.setattr(service, "get_model", lambda **_kwargs: CancellationDelayingModel())
    monkeypatch.setattr(service_module, "LLM_CALL_TIMEOUT_SECONDS", 0.01)

    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError, match="hard limit"):
        await service.chat("问题")
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.05
    await asyncio.wait_for(finished.wait(), timeout=0.2)


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
