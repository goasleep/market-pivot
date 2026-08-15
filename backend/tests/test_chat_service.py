import pytest

from application.chat_service import ChatStore


def test_chat_store_persists_partial_and_cancelled_turn(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    _, assistant_id = store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="分析 ETF 510300",
        history=[],
    )

    store.append_part(assistant_id, {"type": "text", "content": "开始分析"})
    store.update_task("task-1", "cancelled")

    conversation = store.get_conversation("conversation-1")
    assert conversation is not None
    assert conversation["messages"][-1]["status"] == "cancelled"
    assert conversation["messages"][-1]["loading"] is False
    assert conversation["messages"][-1]["parts"][0]["content"] == "开始分析"
    assert store.get_task("task-1")["status"] == "cancelled"


def test_chat_store_replaces_history_for_edit(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="第一问",
        history=[],
    )
    store.update_task("task-1", "completed")
    store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-2",
        message="修改后的第一问",
        history=[],
    )

    conversation = store.get_conversation("conversation-1")
    assert conversation is not None
    assert [message["role"] for message in conversation["messages"]] == ["user", "assistant"]
    assert conversation["messages"][0]["parts"][0]["content"] == "修改后的第一问"


def test_chat_task_creation_is_idempotent_and_events_resume_from_cursor(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    _, assistant_id = store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="查询 ETF 510300",
        history=[],
    )

    _, retry_assistant_id = store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="查询 ETF 510300",
        history=[],
    )
    assert retry_assistant_id == assistant_id
    assert len(store.get_conversation("conversation-1")["messages"]) == 2

    store.append_event("task-1", "a2ui", '{"a2ui": {"type": "createSurface"}}')
    store.append_event("task-1", "a2ui", '{"a2ui": {"type": "updateComponents"}}')
    events = store.list_events("task-1", after_sequence=1)
    assert [event["id"] for event in events] == ["2"]

    with pytest.raises(ValueError, match="其他会话"):
        store.prepare_task(
            conversation_id="conversation-2",
            task_id="task-1",
            message="冲突请求",
            history=[],
        )


def test_cancelled_task_rejects_late_parts(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    _, assistant_id = store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="查询模拟盘",
        history=[],
    )
    assert store.request_cancel("task-1") == "cancel_requested"
    assert not store.append_part(
        assistant_id,
        {"type": "text", "content": "不应写入"},
        task_id="task-1",
    )
    assert store.mark_cancelled("task-1")
