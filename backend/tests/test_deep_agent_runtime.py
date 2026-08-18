import pytest
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from agents.deep_agent_runtime import build_deep_agent, invoke_structured, stream_deep_agent


class BindableFakeChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self


class StructuredAnswer(BaseModel):
    value: str


@tool
def ping(value: str) -> str:
    """Return a deterministic tool observation."""
    return f"pong:{value}"


@pytest.mark.asyncio
async def test_deep_agent_stream_adapts_tool_and_text_events():
    model = BindableFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "ping", "args": {"value": "x"}, "id": "call-1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="完成"),
        ]
    )
    agent = build_deep_agent(model=model, tools=[ping], name="runtime-test")

    events = [
        event
        async for event in stream_deep_agent(
            agent,
            [{"role": "user", "content": "执行"}],
            config={"configurable": {"thread_id": "runtime-test"}},
        )
    ]

    assert any(event["type"] == "reasoning" for event in events)
    assert {event["type"] for event in events} >= {"tool", "text"}
    assert next(event for event in events if event["type"] == "tool")["result"] == "pong:x"
    assert events[-1] == {"type": "text", "text": "完成"}


@pytest.mark.asyncio
async def test_deep_agent_human_confirmation_resumes_with_same_thread():
    model = BindableFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "ping", "args": {"value": "x"}, "id": "call-2", "type": "tool_call"}
                ],
            ),
            AIMessage(content="已确认"),
        ]
    )
    from langgraph.checkpoint.memory import MemorySaver

    agent = build_deep_agent(
        model=model,
        tools=[ping],
        interrupt_on={"ping": True},
        checkpointer=MemorySaver(),
        name="hitl-test",
    )
    config = {"configurable": {"thread_id": "hitl-test"}}
    paused = [
        event
        async for event in stream_deep_agent(
            agent,
            [{"role": "user", "content": "执行"}],
            config=config,
        )
    ]

    interaction = next(event for event in paused if event["type"] == "interaction_required")
    assert interaction["pending_tool_call"]["thread_id"] == "hitl-test"

    resumed = [
        event
        async for event in stream_deep_agent(
            agent,
            [],
            config=config,
            resume={"decisions": [{"type": "approve"}]},
        )
    ]
    assert next(event for event in resumed if event["type"] == "tool")["result"] == "pong:x"
    assert resumed[-1] == {"type": "text", "text": "已确认"}


@pytest.mark.asyncio
async def test_deep_agent_tool_strategy_returns_validated_model():
    model = BindableFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "StructuredAnswer", "args": {"value": "ok"}, "id": "call-3", "type": "tool_call"}
                ],
            )
        ]
    )
    agent = build_deep_agent(
        model=model,
        response_format=ToolStrategy(StructuredAnswer),
        name="structured-test",
    )

    result = await invoke_structured(agent, "返回结果", StructuredAnswer)

    assert result == StructuredAnswer(value="ok")
