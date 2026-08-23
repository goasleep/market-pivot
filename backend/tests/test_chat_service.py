import json
import sqlite3

import pytest
import pytest_asyncio

from application import chat_service
from application.chat_service import ChatStore, ChatTaskInput, ChatTaskManager
from data.chat_models import ChatMessageSearch


def test_public_task_error_hides_provider_details():
    message = chat_service._public_task_error(RuntimeError("sensitive_words_detected request-id-secret"))

    assert message == "模型服务拒绝了最终文字生成；已获取的结构化数据仍可参考，请稍后重试。"
    assert "request-id-secret" not in message


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
async def test_chat_store_clears_stale_error_when_task_is_reclaimed(store):
    await store.prepare_task(
        conversation_id="conversation-reclaim",
        task_id="task-reclaim",
        message="分析 ETF 510300",
    )
    await store.update_task("task-reclaim", "interrupted", "节点正在关闭")

    assert await store.begin_task("task-reclaim")
    task = await store.get_task("task-reclaim")

    assert task is not None
    assert task["status"] == "running"
    assert task["error"] is None


@pytest.mark.asyncio
async def test_chat_store_persists_waiting_interaction_and_answers_once(store):
    _, assistant_id = await store.prepare_task(
        conversation_id="conversation-interaction",
        task_id="task-interaction",
        message="510300",
    )
    interaction = await store.create_interaction(
        "task-interaction",
        "intent_clarification",
        "你希望做什么？",
        [{"id": "quote", "label": "查询行情"}],
        {"request": {"message": "510300"}},
    )
    assert await store.append_part(
        assistant_id,
        {"type": "interaction", "content": interaction},
        task_id="task-interaction",
    )
    await store.update_task("task-interaction", "waiting_user")
    task = await store.get_task("task-interaction")
    assert task is not None
    assert task["status"] == "waiting_user"

    answered = await store.answer_interaction(interaction["interaction_id"], "quote")
    assert answered["status"] == "answered"
    with pytest.raises(ValueError, match="已经处理"):
        await store.answer_interaction(interaction["interaction_id"], "quote")


@pytest.mark.asyncio
async def test_chat_task_pauses_and_resumes_after_interaction(store, monkeypatch):
    _, assistant_id = await store.prepare_task(
        conversation_id="conversation-pause",
        task_id="task-pause",
        message="510300",
    )

    class FakeAssetAgent:
        def prepare(self, **kwargs):
            return kwargs

        async def chat(self, request):
            del request
            yield {
                "type": "interaction_required",
                "kind": "intent_clarification",
                "question": "选择任务",
                "options": [{"id": "quote", "label": "行情"}],
                "resume": {"request": {"message": "510300"}},
            }

        async def resume_chat(self, interaction, option_id):
            assert interaction["selected_option"] == "quote"
            assert option_id == "quote"
            yield {"type": "text", "text": "已完成行情查询。"}

    monkeypatch.setattr(chat_service, "asset_agent", FakeAssetAgent())
    manager = ChatTaskManager(store)
    await manager.start(
        ChatTaskInput(
            task_id="task-pause",
            conversation_id="conversation-pause",
            message="510300",
            strategy=None,
            asset_type="etf",
            assistant_message_id=assistant_id,
        )
    )
    paused_events = [event async for event in manager.subscribe("task-pause")]
    interaction_events = [event for event in paused_events if event["event"] == "interaction_required"]
    assert interaction_events
    interaction = json.loads(interaction_events[-1]["data"])["interaction"]
    assert (await store.get_task("task-pause"))["status"] == "waiting_user"

    response = await manager.respond("task-pause", interaction["interaction_id"], "quote")
    resume_state = await store.get_task_state("task-pause")
    assert resume_state is not None
    assert resume_state["resume_interaction"]["selected_option"] == "quote"
    last_paused_event_id = max(int(event["id"]) for event in paused_events)
    assert response["last_event_id"] == last_paused_event_id
    resumed_events = [
        event
        async for event in manager.subscribe(
            "task-pause",
            after_sequence=response["last_event_id"],
        )
    ]
    assert all(int(event["id"]) > last_paused_event_id for event in resumed_events)
    assert not any(event["event"] == "interaction_required" for event in resumed_events)
    assert any(event["event"] == "done" for event in resumed_events)
    conversation = await store.get_conversation("conversation-pause")
    assert conversation is not None
    assert conversation["messages"][-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_chat_task_projects_plan_updates_to_stable_a2ui_surface(store, monkeypatch):
    _, assistant_id = await store.prepare_task(
        conversation_id="conversation-plan",
        task_id="task-plan",
        message="分析 600519",
    )

    class FakeAssetAgent:
        def prepare(self, **kwargs):
            return kwargs

        async def chat(self, request):
            del request
            yield {
                "type": "plan_update",
                "create": True,
                "plan": {
                    "plan_id": "plan-1",
                    "objective": "分析 600519",
                    "asset_type": "stock",
                    "tickers": ["600519"],
                    "as_of_date": "2026-08-22",
                    "depth": "standard",
                    "revision": 1,
                    "status": "completed",
                    "progress": 100,
                    "steps": [
                        {
                            "id": "market",
                            "kind": "market_snapshot",
                            "title": "获取行情",
                            "status": "completed",
                        }
                    ],
                },
            }
            yield {"type": "text", "text": "研究完成。"}

    monkeypatch.setattr(chat_service, "asset_agent", FakeAssetAgent())
    manager = ChatTaskManager(store)
    await manager.start(
        ChatTaskInput(
            task_id="task-plan",
            conversation_id="conversation-plan",
            message="分析 600519",
            strategy=None,
            asset_type="stock",
            assistant_message_id=assistant_id,
        )
    )
    events = [event async for event in manager.subscribe("task-plan")]

    a2ui_payloads = [json.loads(event["data"])["a2ui"] for event in events if event["event"] == "a2ui"]
    surfaces = [
        payload["createSurface"]["surfaceId"]
        for payload in a2ui_payloads
        if "createSurface" in payload
    ]
    assert "research-plan-task-plan" in surfaces
    state = await store.get_task_state("task-plan")
    assert state is not None
    assert state["graph_name"] == "market-research-plan"


@pytest.mark.asyncio
async def test_chat_worker_resumes_persisted_interaction_instead_of_restarting(store, monkeypatch):
    _, assistant_id = await store.prepare_task(
        conversation_id="conversation-worker-resume",
        task_id="task-worker-resume",
        message="510300",
    )
    interaction = await store.create_interaction(
        "task-worker-resume",
        "intent_clarification",
        "选择任务",
        [{"id": "quote", "label": "行情"}],
        {"request": {"message": "510300"}},
    )
    answered = await store.answer_interaction(interaction["interaction_id"], "quote")
    state = await store.get_task_state("task-worker-resume")
    assert state is not None
    state["resume_interaction"] = answered
    await store.set_task_state("task-worker-resume", state)

    calls: list[str] = []

    class FakeAssetAgent:
        def prepare(self, **kwargs):
            return kwargs

        async def chat(self, request):
            del request
            calls.append("chat")
            yield {"type": "text", "text": "不应重新执行"}

        async def resume_chat(self, resumed_interaction, option_id):
            calls.append("resume")
            assert resumed_interaction["interaction_id"] == interaction["interaction_id"]
            assert option_id == "quote"
            yield {"type": "text", "text": "已按选择恢复。"}

        @staticmethod
        def request_from_payload(payload):
            return payload

    monkeypatch.setattr(chat_service, "asset_agent", FakeAssetAgent())
    manager = ChatTaskManager(store)
    await manager.start_worker()
    try:
        events = [event async for event in manager.subscribe("task-worker-resume")]
    finally:
        await manager.stop_worker()

    assert calls == ["resume"]
    assert any(event["event"] == "done" for event in events)
    assert (await store.get_task("task-worker-resume"))["status"] == "completed"


@pytest.mark.asyncio
async def test_chat_store_keeps_persisted_history_by_default(store):
    await store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="第一问",
    )
    await store.update_task("task-1", "completed")
    await store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-2",
        message="修改后的第一问",
    )

    conversation = await store.get_conversation("conversation-1")
    assert conversation is not None
    assert [message["role"] for message in conversation["messages"]] == ["user", "assistant", "user", "assistant"]
    assert conversation["messages"][0]["parts"][0]["content"] == "第一问"
    assert conversation["messages"][2]["parts"][0]["content"] == "修改后的第一问"
    state = await store.get_task_state("task-2")
    assert state is not None
    assert state["history"][0]["content"] == "第一问"


@pytest.mark.asyncio
async def test_chat_store_edits_a_server_owned_message_branch(store):
    user_message_id, _ = await store.prepare_task(
        conversation_id="conversation-edit",
        task_id="task-edit-1",
        message="第一问",
    )
    await store.update_task("task-edit-1", "completed")
    await store.prepare_task(
        conversation_id="conversation-edit",
        task_id="task-edit-2",
        message="修改后的第一问",
        edit_message_id=user_message_id,
    )

    conversation = await store.get_conversation("conversation-edit")
    assert conversation is not None
    assert [message["role"] for message in conversation["messages"]] == ["user", "assistant"]
    assert conversation["messages"][0]["parts"][0]["content"] == "修改后的第一问"


@pytest.mark.asyncio
async def test_chat_store_branches_through_completed_assistant_reply(store):
    _, first_assistant_id = await store.prepare_task(
        conversation_id="conversation-source",
        task_id="task-source-1",
        message="第一问",
    )
    await store.append_part(first_assistant_id, {"type": "text", "content": "第一答"})
    await store.set_references(
        first_assistant_id,
        [{"title": "参考资料", "url": "https://example.com/source"}],
    )
    await store.update_task("task-source-1", "completed")

    await store.prepare_task(
        conversation_id="conversation-source",
        task_id="task-source-2",
        message="第二问",
    )
    await store.update_task("task-source-2", "completed")

    source_before = await store.get_conversation("conversation-source")
    branch = await store.branch_conversation("conversation-source", first_assistant_id)
    source_after = await store.get_conversation("conversation-source")

    assert source_before == source_after
    assert branch["conversation_id"] != "conversation-source"
    assert branch["title"].endswith("（分支）")
    assert [message["role"] for message in branch["messages"]] == ["user", "assistant"]
    assert [message["parts"][0]["content"] for message in branch["messages"]] == ["第一问", "第一答"]
    assert all(message["task_id"] is None for message in branch["messages"])
    assert {message["id"] for message in branch["messages"]}.isdisjoint(
        {message["id"] for message in source_before["messages"]}
    )
    assert branch["messages"][1]["references"] == [
        {"title": "参考资料", "url": "https://example.com/source"}
    ]
    assert branch["messages"][0]["created_at"] == source_before["messages"][0]["created_at"]

    await store.prepare_task(
        conversation_id=branch["conversation_id"],
        task_id="task-branch-follow-up",
        message="分支追问",
    )
    state = await store.get_task_state("task-branch-follow-up")
    assert state is not None
    assert [message["content"] for message in state["history"]] == ["第一问", "第一答"]

    matches = await store.list_conversations(query="第一答")
    assert {item["conversation_id"] for item in matches} == {
        "conversation-source",
        branch["conversation_id"],
    }


@pytest.mark.asyncio
async def test_chat_store_rejects_invalid_branch_target(store):
    user_message_id, assistant_message_id = await store.prepare_task(
        conversation_id="conversation-invalid-branch",
        task_id="task-invalid-branch",
        message="尚未完成的问题",
    )

    with pytest.raises(ValueError, match="助手回复"):
        await store.branch_conversation("conversation-invalid-branch", user_message_id)
    with pytest.raises(ValueError, match="已完成"):
        await store.branch_conversation("conversation-invalid-branch", assistant_message_id)


@pytest.mark.asyncio
async def test_chat_store_does_not_rebuild_populated_search_index_on_startup(tmp_path):
    db_path = tmp_path / "incremental-search.sqlite3"
    first = ChatStore(db_path)
    await first.init()
    await first.prepare_task(
        conversation_id="conversation-search",
        task_id="task-search",
        message="原始搜索内容",
    )
    await ChatMessageSearch.filter(conversation_id="conversation-search").update(content="index-sentinel")
    await first.close()

    second = ChatStore(db_path)
    await second.init()
    try:
        row = await ChatMessageSearch.filter(conversation_id="conversation-search").first()
        assert row is not None
        assert row.content == "index-sentinel"
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_chat_task_creation_is_idempotent_and_events_resume_from_cursor(store):
    _, assistant_id = await store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="查询 ETF 510300",
    )

    _, retry_assistant_id = await store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="查询 ETF 510300",
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
        )


@pytest.mark.asyncio
async def test_chat_conversation_streams_history_chart_surface(store, monkeypatch):
    _, assistant_id = await store.prepare_task(
        conversation_id="conversation-chart",
        task_id="task-chart",
        message="展示 ETF 510300 的历史走势",
    )

    class FakeAssetAgent:
        def prepare(self, **kwargs):
            return kwargs

        async def chat(self, request):
            del request
            yield {
                "type": "tool",
                "name": "get_historical_prices",
                "status": "completed",
                "result": json.dumps(
                    {
                        "ticker": "510300",
                        "asset_type": "etf",
                        "history": [
                            {"date": "2026-08-12", "close": 4.0, "volume": 100},
                            {"date": "2026-08-13", "close": 4.2, "volume": 120},
                        ],
                    },
                    ensure_ascii=False,
                ),
            }
            yield {"type": "text", "text": "近期走势温和向上。"}

    monkeypatch.setattr(chat_service, "asset_agent", FakeAssetAgent())
    manager = ChatTaskManager(store)
    await manager.start(
        ChatTaskInput(
            task_id="task-chart",
            conversation_id="conversation-chart",
            message="展示 ETF 510300 的历史走势",
            strategy=None,
            asset_type="etf",
            assistant_message_id=assistant_id,
        )
    )

    events = [event async for event in manager.subscribe("task-chart")]
    a2ui_messages = [
        json.loads(event["data"])["a2ui"]
        for event in events
        if event["event"] == "a2ui"
    ]
    chart_components = [
        component
        for message in a2ui_messages
        for component in message.get("updateComponents", {}).get("components", [])
        if component.get("component") == "LineChart"
    ]

    assert chart_components
    assert events[-1]["event"] == "done"
    conversation = await store.get_conversation("conversation-chart")
    assert conversation is not None
    assistant_parts = conversation["messages"][-1]["parts"]
    stored_a2ui_messages = [
        message
        for part in assistant_parts
        if part["type"] == "a2ui"
        for message in (part["content"] if isinstance(part["content"], list) else [part["content"]])
    ]
    assert any(
        message.get("updateComponents", {}).get("components")
        and any(
            component.get("component") == "LineChart"
            for component in message["updateComponents"]["components"]
        )
        for message in stored_a2ui_messages
    )


@pytest.mark.asyncio
async def test_chat_does_not_repeat_artifacts_already_rendered_in_a2ui_bundle(store, monkeypatch):
    _, assistant_id = await store.prepare_task(
        conversation_id="conversation-artifacts",
        task_id="task-artifacts",
        message="对比策略并生成回测产物",
    )
    artifact = {
        "artifact_id": "artifact-report",
        "name": "策略回测报告.html",
        "mime_type": "text/html",
        "size_bytes": 1024,
        "preview_url": "/api/artifacts/artifact-report/preview",
        "download_url": "/api/artifacts/artifact-report/download",
    }

    class FakeAssetAgent:
        def prepare(self, **kwargs):
            return kwargs

        async def chat(self, request):
            del request
            yield {
                "type": "tool",
                "name": "compare_strategy_backtests",
                "status": "completed",
                "result": json.dumps({"ticker": "510300", "artifacts": [artifact]}, ensure_ascii=False),
            }
            yield {"type": "text", "text": "策略对比已完成。"}

    monkeypatch.setattr(chat_service, "asset_agent", FakeAssetAgent())
    manager = ChatTaskManager(store)
    await manager.start(
        ChatTaskInput(
            task_id="task-artifacts",
            conversation_id="conversation-artifacts",
            message="对比策略并生成回测产物",
            strategy=None,
            asset_type="etf",
            assistant_message_id=assistant_id,
        )
    )

    events = [event async for event in manager.subscribe("task-artifacts")]
    assert not any(event["event"] == "artifact" for event in events)
    a2ui_messages = [
        json.loads(event["data"])["a2ui"]
        for event in events
        if event["event"] == "a2ui"
    ]
    assert any(
        component.get("component") == "ArtifactLink"
        for message in a2ui_messages
        for component in message.get("updateComponents", {}).get("components", [])
    )
    conversation = await store.get_conversation("conversation-artifacts")
    assert conversation is not None
    assistant_parts = conversation["messages"][-1]["parts"]
    assert not any(part["type"] == "artifact" for part in assistant_parts)


@pytest.mark.asyncio
async def test_cancelled_task_rejects_late_parts(store):
    _, assistant_id = await store.prepare_task(
        conversation_id="conversation-1",
        task_id="task-1",
        message="查询模拟盘",
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
        message="分析 ETF 510300 的短期趋势，测试标记 SHORT-2026",
    )
    await store.update_task("task-1", "completed")
    await store.prepare_task(
        conversation_id="conversation-2",
        task_id="task-2",
        message="查询模拟盘账户",
    )
    await store.update_task("task-2", "completed")
    results = await store.list_conversations(query="短期趋势")
    assert [item["conversation_id"] for item in results] == ["conversation-1"]
    results = await store.list_conversations(query="SHORT-2026")
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
