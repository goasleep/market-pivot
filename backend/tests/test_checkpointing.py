import os
from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph

import graph.agent_loop as agent_loop
import graph.checkpointing as checkpointing
from application.research import ResearchService
from graph.checkpointing import CheckpointManager, _checkpoint_serializer, _psycopg_url
from models.schemas import AgentReport, AssetType, Decision, MarketContext, SignalType, TradeDecision


def test_checkpoint_graph_config_and_postgres_url_normalization():
    config = CheckpointManager.graph_config("task-1", {"configurable": {"tenant": "local"}})
    assert config["configurable"] == {"tenant": "local", "thread_id": "task-1"}
    assert config["recursion_limit"] == 120
    assert _psycopg_url("postgresql+asyncpg://user:pass@db/app") == "postgresql://user:pass@db/app"


def test_checkpoint_serializer_round_trips_explicit_application_types():
    serializer = _checkpoint_serializer()
    values = (
        AssetType.ETF,
        MarketContext(ticker="510300", asset_type=AssetType.ETF),
        Decision.HOLD,
        AgentReport(agent_name="technical"),
        SignalType.WATCH,
        TradeDecision(ticker="510300", asset_type=AssetType.ETF),
    )

    for value in values:
        decoded = serializer.loads_typed(serializer.dumps_typed(value))
        assert type(decoded) is type(value)
        assert decoded == value


def test_background_research_does_not_allocate_checkpoints(monkeypatch):
    monkeypatch.setattr(checkpointing.checkpoint_manager, "saver", object())
    options, checkpointed = ResearchService._invoke_options({"tags": ["automation"]})
    assert options == {"config": {"tags": ["automation"]}}
    assert checkpointed is False

    options, checkpointed = ResearchService._invoke_options(
        {"configurable": {"thread_id": "chat-task"}, "tags": ["chat"]}
    )
    assert options["config"]["configurable"]["thread_id"] == "chat-task"
    assert checkpointed is True


@pytest.mark.asyncio
async def test_checkpoint_manager_uses_independent_sqlite(monkeypatch, tmp_path):
    path = tmp_path / "checkpoints.db"
    monkeypatch.setattr(checkpointing.settings, "checkpoint_database_url", None)
    monkeypatch.setattr(checkpointing.settings, "database_url", None)
    monkeypatch.setattr(checkpointing.settings, "checkpoint_database_path", str(path))
    manager = CheckpointManager()
    try:
        saver = await manager.start()
        assert isinstance(saver, AsyncSqliteSaver)
        assert path.exists()

        class CounterState(TypedDict):
            count: int
            asset_type: AssetType

        async def increment(state: CounterState):
            return {"count": state["count"] + 1}

        builder = StateGraph(CounterState)
        builder.add_node("increment", increment)
        builder.add_edge(START, "increment")
        graph = builder.compile(checkpointer=saver)
        config = manager.graph_config("sqlite-integration")
        result = await graph.ainvoke({"count": 0, "asset_type": AssetType.ETF}, config=config)
        assert result == {"count": 1, "asset_type": AssetType.ETF}
        rebuilt = builder.compile(checkpointer=saver)
        assert (await rebuilt.aget_state(config)).values["count"] == 1
        assert (await rebuilt.aget_state(config)).values["asset_type"] is AssetType.ETF
    finally:
        await manager.stop()
    assert manager.saver is None


@pytest.mark.asyncio
async def test_checkpoint_manager_prefers_postgres_and_runs_setup(monkeypatch):
    observed = {"url": "", "setup": False, "closed": False, "serde": None}

    class FakeSaver:
        async def setup(self):
            observed["setup"] = True

    saver = FakeSaver()

    class FakeContext:
        async def __aenter__(self):
            return saver

        async def __aexit__(self, exc_type, exc, traceback):
            observed["closed"] = True

    class FakePostgresSaver:
        @staticmethod
        def from_conn_string(url, *, serde=None):
            observed["url"] = url
            observed["serde"] = serde
            return FakeContext()

    monkeypatch.setattr(checkpointing, "AsyncPostgresSaver", FakePostgresSaver)
    monkeypatch.setattr(
        checkpointing.settings,
        "checkpoint_database_url",
        "postgresql+asyncpg://user:pass@db/app",
    )
    manager = CheckpointManager()
    assert await manager.start() is saver
    assert observed["url"] == "postgresql://user:pass@db/app"
    assert observed["setup"] is True
    assert observed["closed"] is False
    assert observed["serde"].__class__.__name__ == "JsonPlusSerializer"
    await manager.stop()
    assert observed["closed"] is True


@pytest.mark.asyncio
async def test_native_interrupt_survives_graph_recompile(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    async def submit(ticker: str, execution_key: str | None = None) -> str:
        calls.append((ticker, execution_key))
        return "submitted"

    tool = StructuredTool.from_function(
        coroutine=submit,
        name="submit_simulation_order",
        description="Submit a paper order.",
    )

    class FakeLLM:
        async def chat_with_tools(self, messages, tools, temperature=0.2):
            del tools, temperature
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content="done")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_simulation_order",
                        "args": {"ticker": "510300"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )

    monkeypatch.setattr(agent_loop, "get_llm_service", lambda: FakeLLM())
    saver = MemorySaver()
    config = {"configurable": {"thread_id": "task-interrupt"}, "recursion_limit": 120}
    agent_loop.configure_agent_loop(saver)
    try:
        first = [
            update
            async for update in agent_loop.stream_agent_loop(
                [{"role": "user", "content": "下模拟单"}],
                [tool],
                config=config,
                native_interrupts=True,
                task_id="task-interrupt",
            )
        ]
        assert any("__interrupt__" in update for update in first)
        assert calls == []

        agent_loop.configure_agent_loop(saver)
        resumed = [
            update
            async for update in agent_loop.resume_native_agent_loop(
                [tool],
                approved=True,
                config=config,
                task_id="task-interrupt",
            )
        ]
        assert resumed
        assert calls == [("510300", "task-interrupt:call-1")]
    finally:
        agent_loop.configure_agent_loop(None)


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL is not configured")
async def test_postgres_checkpoint_integration(monkeypatch):
    monkeypatch.setattr(checkpointing.settings, "checkpoint_database_url", os.environ["TEST_POSTGRES_URL"])
    manager = CheckpointManager()
    try:
        saver = await manager.start()
        assert isinstance(saver, AsyncPostgresSaver)
    finally:
        await manager.stop()
