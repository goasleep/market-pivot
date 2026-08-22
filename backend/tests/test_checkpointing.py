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
from graph.checkpointing import CheckpointManager, _psycopg_url


def test_checkpoint_graph_config_and_postgres_url_normalization():
    config = CheckpointManager.graph_config("task-1", {"configurable": {"tenant": "local"}})
    assert config["configurable"] == {"tenant": "local", "thread_id": "task-1"}
    assert config["recursion_limit"] == 120
    assert _psycopg_url("postgresql+asyncpg://user:pass@db/app") == "postgresql://user:pass@db/app"


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

        async def increment(state: CounterState):
            return {"count": state["count"] + 1}

        builder = StateGraph(CounterState)
        builder.add_node("increment", increment)
        builder.add_edge(START, "increment")
        graph = builder.compile(checkpointer=saver)
        config = manager.graph_config("sqlite-integration")
        assert (await graph.ainvoke({"count": 0}, config=config))["count"] == 1
        rebuilt = builder.compile(checkpointer=saver)
        assert (await rebuilt.aget_state(config)).values["count"] == 1
    finally:
        await manager.stop()
    assert manager.saver is None


@pytest.mark.asyncio
async def test_checkpoint_manager_prefers_postgres_and_runs_setup(monkeypatch):
    observed = {"url": "", "setup": False, "closed": False}

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
        def from_conn_string(url):
            observed["url"] = url
            return FakeContext()

    monkeypatch.setattr(checkpointing, "AsyncPostgresSaver", FakePostgresSaver)
    monkeypatch.setattr(
        checkpointing.settings,
        "checkpoint_database_url",
        "postgresql+asyncpg://user:pass@db/app",
    )
    manager = CheckpointManager()
    assert await manager.start() is saver
    assert observed == {
        "url": "postgresql://user:pass@db/app",
        "setup": True,
        "closed": False,
    }
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
