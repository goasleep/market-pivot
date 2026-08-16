import json
import sqlite3

import pytest
import pytest_asyncio

from application.chat_service import ChatStore


@pytest_asyncio.fixture
async def store(tmp_path):
    chat_store = ChatStore(tmp_path / "chat.db")
    await chat_store.init()
    yield chat_store
    await chat_store.close()


@pytest.mark.asyncio
async def test_chat_store_persists_partial_and_cancelled_turn(store):
    _, assistant_id = await store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="分析 ETF 510300",
        history=[],
    )

    await store.append_part(assistant_id, {"type": "text", "content": "开始分析"})
    await store.update_task("task-1", "cancelled")

    conversation = await store.get_conversation("conversation-1")
    assert conversation is not None
    assert conversation["messages"][-1]["status"] == "cancelled"
    assert conversation["messages"][-1]["loading"] is False
    assert conversation["messages"][-1]["parts"][0]["content"] == "开始分析"
    task = await store.get_task("task-1")
    assert task is not None
    assert task["status"] == "cancelled"


@pytest.mark.asyncio
async def test_chat_store_replaces_history_for_edit(store):
    await store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="第一问",
        history=[],
    )
    await store.update_task("task-1", "completed")
    await store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-2",
        message="修改后的第一问",
        history=[],
    )

    conversation = await store.get_conversation("conversation-1")
    assert conversation is not None
    assert [message["role"] for message in conversation["messages"]] == ["user", "assistant"]
    assert conversation["messages"][0]["parts"][0]["content"] == "修改后的第一问"


@pytest.mark.asyncio
async def test_chat_task_creation_is_idempotent_and_events_resume_from_cursor(store):
    _, assistant_id = await store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="查询 ETF 510300",
        history=[],
    )

    _, retry_assistant_id = await store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="查询 ETF 510300",
        history=[],
    )
    assert retry_assistant_id == assistant_id
    conversation = await store.get_conversation("conversation-1")
    assert conversation is not None
    assert len(conversation["messages"]) == 2

    await store.append_event("task-1", "a2ui", '{"a2ui": {"type": "createSurface"}}')
    await store.append_event("task-1", "a2ui", '{"a2ui": {"type": "updateComponents"}}')
    events = await store.list_events("task-1", after_sequence=1)
    assert [event["id"] for event in events] == ["2"]

    with pytest.raises(ValueError, match="其他会话"):
        await store.prepare_task(
            conversation_id="conversation-2",
            task_id="task-1",
            message="冲突请求",
            history=[],
        )


@pytest.mark.asyncio
async def test_cancelled_task_rejects_late_parts(store):
    _, assistant_id = await store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="查询模拟盘",
        history=[],
    )
    assert await store.request_cancel("task-1") == "cancel_requested"
    assert not await store.append_part(
        assistant_id,
        {"type": "text", "content": "不应写入"},
        task_id="task-1",
    )
    assert await store.mark_cancelled("task-1")


@pytest.mark.asyncio
async def test_chat_store_searches_message_content(store):
    await store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="分析 ETF 510300 的短期趋势",
        history=[],
    )
    await store.update_task("task-1", "completed")
    await store.prepare_task(
        conversation_id="conversation-2",
        task_id="task-2",
        message="查询模拟盘账户",
        history=[],
    )
    await store.update_task("task-2", "completed")
    results = await store.list_conversations(query="短期趋势")
    assert [item["conversation_id"] for item in results] == ["conversation-1"]


@pytest.mark.asyncio
async def test_chat_store_rebuilds_search_index_for_legacy_sqlite(tmp_path):
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE chat_conversations (
            conversation_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE chat_messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            parts_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed',
            task_id TEXT,
            position INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE chat_tasks (
            task_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE chat_task_events (
            task_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            event TEXT NOT NULL,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (task_id, sequence)
        );
        CREATE TABLE chat_message_references (
            message_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            reference_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (message_id, position)
        );
        """
    )
    timestamp = "2026-01-01T00:00:00+00:00"
    connection.execute(
        "INSERT INTO chat_conversations VALUES (?, ?, ?, ?)",
        ("legacy-1", "旧标题", timestamp, timestamp),
    )
    connection.execute(
        "INSERT INTO chat_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "legacy-msg",
            "legacy-1",
            "user",
            json.dumps([{"type": "text", "content": "历史内容搜索"}], ensure_ascii=False),
            "completed",
            None,
            0,
            timestamp,
            timestamp,
        ),
    )
    connection.commit()
    connection.close()

    store = ChatStore(db_path)
    await store.init()
    try:
        results = await store.list_conversations(query="历史内容")
        assert [item["conversation_id"] for item in results] == ["legacy-1"]
    finally:
        await store.close()
