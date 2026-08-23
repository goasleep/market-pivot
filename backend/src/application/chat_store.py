"""Async Tortoise repository for conversations, messages, and chat tasks."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from tortoise import Tortoise
from tortoise.transactions import in_transaction

from application.chat_search import ChatSearchIndex, parts_text
from config import settings
from data.chat_models import (
    ChatConversation,
    ChatMessage,
    ChatMessageReference,
    ChatMessageSearch,
    ChatTask,
    ChatTaskEvent,
    ChatTaskInteraction,
    ChatTaskState,
)
from data.tortoise_db import close_database, init_database
from graph.checkpointing import checkpoint_manager


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _text_part(content: str) -> dict[str, str]:
    return {"type": "text", "content": content}


class ChatStore:
    """Async repository backed by SQLite or PostgreSQL through Tortoise."""

    _active_db_url: str | None = None

    def __init__(self, db_path: str | Path | None = None, db_url: str | None = None):
        if db_url:
            self.db_url = db_url
        elif db_path:
            self.db_url = f"sqlite://{Path(db_path).expanduser().resolve()}"
        else:
            self.db_url = settings.chat_database_url
        self._initialized = False
        self.search_index = ChatSearchIndex()

    async def init(self) -> None:
        if self._initialized:
            return
        await init_database(db_url=self.db_url)
        await self._ensure_legacy_tables()
        await self.search_index.initialize()
        await self._recover_interrupted_tasks()
        self._initialized = True

    async def close(self) -> None:
        await close_database()
        self._initialized = False
        self.search_index.reset()

    async def _ensure_ready(self) -> None:
        if not self._initialized:
            await self.init()

    async def _ensure_legacy_tables(self) -> None:
        connection = Tortoise.get_connection("default")
        await connection.execute_script(
            """
            CREATE TABLE IF NOT EXISTS chat_task_events (
                task_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (task_id, sequence)
            );
            CREATE INDEX IF NOT EXISTS idx_chat_task_events_task
                ON chat_task_events(task_id, sequence);
            CREATE TABLE IF NOT EXISTS chat_message_references (
                message_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                reference_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (message_id, position)
            );
            CREATE INDEX IF NOT EXISTS idx_chat_message_references_message
                ON chat_message_references(message_id, position);
            CREATE TABLE IF NOT EXISTS chat_task_states (
                task_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_task_interactions (
                interaction_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                question TEXT NOT NULL,
                options_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                selected_option TEXT,
                created_at TEXT NOT NULL,
                responded_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chat_task_interactions_task
                ON chat_task_interactions(task_id, status);
            """
        )
        for table in (
            "chat_task_events",
            "chat_message_references",
            "chat_task_states",
            "chat_task_interactions",
        ):
            try:
                await connection.execute_query(f"ALTER TABLE {table} ADD COLUMN id TEXT")
            except Exception:
                # The column already exists on newly generated schemas.
                pass

    async def _recover_interrupted_tasks(self) -> None:
        # Only reclaim tasks whose worker lease has gone stale. A node joining
        # a live cluster must not interrupt work owned by another node.
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        timestamp = _now()
        stale_running = await ChatTask.filter(status="running", updated_at__lt=cutoff).update(
            status="interrupted", updated_at=timestamp
        )
        await ChatTask.filter(status="cancel_requested", updated_at__lt=cutoff).update(
            status="cancelled", updated_at=timestamp
        )
        if stale_running:
            await ChatMessage.filter(status__in=["pending", "running"], updated_at__lt=cutoff).update(
                status="interrupted", updated_at=timestamp
            )

    async def recover_stale_tasks(self) -> None:
        await self._ensure_ready()
        await self._recover_interrupted_tasks()

    @staticmethod
    def _parts(item: dict[str, Any]) -> list[dict[str, Any]]:
        parts = item.get("parts")
        if isinstance(parts, list) and parts:
            return [
                part
                for part in parts
                if isinstance(part, dict)
                and part.get("type") in {"text", "a2ui", "widget", "artifact", "interaction"}
            ]
        return [_text_part(str(item.get("content", "")))]

    @staticmethod
    def _message_payload(row: dict[str, Any], references: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        status = row["status"]
        return {
            "id": row["message_id"],
            "role": row["role"],
            "parts": json.loads(row["parts_json"]),
            "created_at": row.get("created_at"),
            "status": status,
            "task_id": row["task_id"],
            "loading": status in {"pending", "running"},
            "references": references or [],
        }

    async def _sync_search(
        self,
        message_id: str,
        conversation_id: str,
        parts: list[dict[str, Any]],
        connection=None,
    ) -> None:
        await self.search_index.sync(message_id, conversation_id, parts, connection)

    async def prepare_task(
        self,
        conversation_id: str,
        task_id: str,
        message: str,
        strategy: str | None = None,
        asset_type: str | None = None,
        llm_profile_id: str | None = None,
        llm_model: str | None = None,
        llm_auto: bool = False,
        edit_message_id: str | None = None,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
    ) -> tuple[str, str]:
        """Append a task using server-owned history, optionally replacing one branch."""
        await self._ensure_ready()
        user_message_id = user_message_id or f"msg-{uuid4().hex}"
        assistant_message_id = assistant_message_id or f"msg-{uuid4().hex}"
        if user_message_id == assistant_message_id:
            raise ValueError("用户消息和助手消息必须使用不同 ID")
        timestamp = _now()
        title = message.strip()[:30] + ("…" if len(message.strip()) > 30 else "") or "新对话"
        pruned_task_ids: list[str] = []

        async with in_transaction() as connection:
            existing = await ChatTask.filter(task_id=task_id).using_db(connection).first()
            if existing:
                if existing.conversation_id != conversation_id:
                    raise ValueError("task_id 已被其他会话使用")
                return user_message_id, existing.message_id

            active = (
                await ChatTask.filter(
                    conversation_id=conversation_id,
                    status__in=["pending", "running", "cancel_requested", "waiting_user"],
                )
                .using_db(connection)
                .order_by("-created_at")
                .first()
            )
            if active:
                if active.status != "waiting_user":
                    raise ValueError(f"会话已有正在执行的任务: {active.task_id}")
                await ChatTask.filter(task_id=active.task_id).using_db(connection).update(
                    status="superseded",
                    updated_at=timestamp,
                )
                await ChatMessage.filter(message_id=active.message_id).using_db(connection).update(
                    status="superseded",
                    updated_at=timestamp,
                )
                await ChatTaskInteraction.filter(
                    task_id=active.task_id,
                    status="pending",
                ).using_db(connection).update(status="cancelled", responded_at=timestamp)

            conversation = (
                await ChatConversation.filter(conversation_id=conversation_id).using_db(connection).first()
            )
            if conversation is None:
                if edit_message_id:
                    raise ValueError("不能编辑尚未持久化的会话")
                await ChatConversation.create(
                    conversation_id=conversation_id,
                    title=title,
                    created_at=timestamp,
                    updated_at=timestamp,
                    using_db=connection,
                )
                persisted_messages = []
            else:
                await ChatConversation.filter(conversation_id=conversation_id).using_db(connection).update(
                    updated_at=timestamp
                )
                persisted_messages = await (
                    ChatMessage.filter(conversation_id=conversation_id)
                    .using_db(connection)
                    .order_by("position")
                    .values("message_id", "role", "parts_json", "status", "position", "task_id")
                )

            if edit_message_id:
                target = next(
                    (item for item in persisted_messages if item["message_id"] == edit_message_id),
                    None,
                )
                if target is None or target["role"] != "user":
                    raise ValueError("只能编辑当前会话中已持久化的用户消息")
                cutoff = int(target["position"])
                pruned_messages = [item for item in persisted_messages if int(item["position"]) >= cutoff]
                pruned_message_ids = [str(item["message_id"]) for item in pruned_messages]
                pruned_task_ids = list(
                    dict.fromkeys(str(item["task_id"]) for item in pruned_messages if item.get("task_id"))
                )
                if pruned_message_ids:
                    await ChatMessageReference.filter(message_id__in=pruned_message_ids).using_db(connection).delete()
                    await self.search_index.remove_messages(pruned_message_ids, connection)
                if pruned_task_ids:
                    await ChatTaskEvent.filter(task_id__in=pruned_task_ids).using_db(connection).delete()
                    await ChatTaskInteraction.filter(task_id__in=pruned_task_ids).using_db(connection).delete()
                    await ChatTaskState.filter(task_id__in=pruned_task_ids).using_db(connection).delete()
                    await ChatTask.filter(task_id__in=pruned_task_ids).using_db(connection).delete()
                await ChatMessage.filter(message_id__in=pruned_message_ids).using_db(connection).delete()
                persisted_messages = [item for item in persisted_messages if int(item["position"]) < cutoff]

            effective_history = []
            for item in persisted_messages:
                if item["status"] != "completed" or item["role"] not in {"user", "assistant"}:
                    continue
                parts = json.loads(item["parts_json"])
                effective_history.append(
                    {"role": item["role"], "content": parts_text(parts), "parts": parts}
                )
            user_position = (
                max(int(item["position"]) for item in persisted_messages) + 1
                if persisted_messages
                else 0
            )
            duplicate_message = await (
                ChatMessage.filter(message_id__in=[user_message_id, assistant_message_id])
                .using_db(connection)
                .exists()
            )
            if duplicate_message:
                raise ValueError("消息 ID 已被使用")

            user_parts = [_text_part(message)]
            await ChatMessage.create(
                message_id=user_message_id,
                conversation_id=conversation_id,
                role="user",
                parts_json=_json(user_parts),
                status="completed",
                task_id=None,
                position=user_position,
                created_at=timestamp,
                updated_at=timestamp,
                using_db=connection,
            )
            await self._sync_search(user_message_id, conversation_id, user_parts, connection)
            await ChatMessage.create(
                message_id=assistant_message_id,
                conversation_id=conversation_id,
                role="assistant",
                parts_json="[]",
                status="pending",
                task_id=task_id,
                position=user_position + 1,
                created_at=timestamp,
                updated_at=timestamp,
                using_db=connection,
            )
            await ChatTask.create(
                task_id=task_id,
                conversation_id=conversation_id,
                message_id=assistant_message_id,
                status="pending",
                error=None,
                created_at=timestamp,
                updated_at=timestamp,
                using_db=connection,
            )
            await ChatTaskState.create(
                task_id=task_id,
                state_json=_json(
                    {
                        "task_id": task_id,
                        "conversation_id": conversation_id,
                        "message": message,
                        "history": effective_history,
                        "strategy": strategy,
                        "asset_type": asset_type,
                        "llm_profile_id": llm_profile_id,
                        "llm_model": llm_model,
                        "llm_auto": llm_auto,
                        "assistant_message_id": assistant_message_id,
                        "execution_version": 2,
                        "graph_name": "asset-agent-chat",
                        "thread_id": task_id,
                    }
                ),
                updated_at=timestamp,
                using_db=connection,
            )

        if pruned_task_ids:
            for pruned_task_id in pruned_task_ids:
                await checkpoint_manager.delete_thread_family(pruned_task_id)
        return user_message_id, assistant_message_id

    async def append_part(self, message_id: str, part: dict[str, Any], task_id: str | None = None) -> bool:
        await self._ensure_ready()
        message = await ChatMessage.filter(message_id=message_id).first()
        if message is None or (task_id and message.task_id != task_id):
            return False
        if task_id:
            task = await ChatTask.filter(task_id=task_id).first()
            if task is None or task.status in {
                "cancel_requested",
                "cancelled",
                "completed",
                "failed",
                "interrupted",
                "superseded",
            }:
                return False
        parts = json.loads(message.parts_json)
        if part.get("type") == "text" and parts and parts[-1].get("type") == "text":
            parts[-1]["content"] += part.get("content", "")
        elif part.get("type") == "a2ui" and parts and parts[-1].get("type") == "a2ui":
            existing = parts[-1].setdefault("content", [])
            if not isinstance(existing, list):
                existing = [existing]
                parts[-1]["content"] = existing
            content = part.get("content")
            existing.extend(content if isinstance(content, list) else [content])
        else:
            parts.append(part)
        message.parts_json = _json(parts)
        message.status = "running"
        message.updated_at = _now()
        await message.save(update_fields=["parts_json", "status", "updated_at"])
        if task_id:
            await ChatTask.filter(task_id=task_id).update(updated_at=message.updated_at)
        await self._sync_search(message_id, message.conversation_id, parts)
        return True

    async def append_event(self, task_id: str, event: str, data: str) -> dict[str, Any]:
        await self._ensure_ready()
        last = await ChatTaskEvent.filter(task_id=task_id).order_by("-sequence").first()
        sequence = (last.sequence if last else 0) + 1
        await ChatTaskEvent.create(
            id=f"{task_id}:{sequence}",
            task_id=task_id,
            sequence=sequence,
            event=event,
            data=data,
            created_at=_now(),
        )
        await ChatTask.filter(task_id=task_id, status__in=["pending", "running"]).update(updated_at=_now())
        return {"id": str(sequence), "event": event, "data": data}

    async def list_events(self, task_id: str, after_sequence: int = 0) -> list[dict[str, Any]]:
        await self._ensure_ready()
        rows = await (
            ChatTaskEvent.filter(task_id=task_id, sequence__gt=max(0, after_sequence))
            .order_by("sequence")
            .values("sequence", "event", "data")
        )
        return [{"id": str(row["sequence"]), "event": row["event"], "data": row["data"]} for row in rows]

    async def latest_event_sequence(self, task_id: str) -> int:
        """Return the last durable event so a resumed stream can skip its history."""
        await self._ensure_ready()
        event = await ChatTaskEvent.filter(task_id=task_id).order_by("-sequence").first()
        return event.sequence if event else 0

    async def request_cancel(self, task_id: str) -> str | None:
        await self._ensure_ready()
        task = await ChatTask.filter(task_id=task_id).first()
        if task is None:
            return None
        if task.status not in {"pending", "running", "cancel_requested", "waiting_user"}:
            return task.status
        timestamp = _now()
        await ChatTask.filter(
            task_id=task_id,
            status__in=["pending", "running", "cancel_requested", "waiting_user"],
        ).update(
            status="cancel_requested", updated_at=timestamp
        )
        await ChatMessage.filter(message_id=task.message_id).update(status="cancel_requested", updated_at=timestamp)
        await ChatTaskInteraction.filter(task_id=task_id, status="pending").update(
            status="cancelled", responded_at=timestamp
        )
        return "cancel_requested"

    async def get_task_state(self, task_id: str) -> dict[str, Any] | None:
        await self._ensure_ready()
        row = await ChatTaskState.filter(task_id=task_id).first()
        return json.loads(row.state_json) if row else None

    async def set_task_state(self, task_id: str, state: dict[str, Any]) -> None:
        await self._ensure_ready()
        await ChatTaskState.update_or_create(
            task_id=task_id,
            defaults={"state_json": _json(state), "updated_at": _now()},
        )

    async def create_interaction(
        self,
        task_id: str,
        kind: str,
        question: str,
        options: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        await self._ensure_ready()
        interaction_id = f"interaction-{uuid4().hex}"
        timestamp = _now()
        await ChatTaskInteraction.create(
            interaction_id=interaction_id,
            task_id=task_id,
            kind=kind,
            question=question,
            options_json=_json(options),
            payload_json=_json(payload),
            status="pending",
            created_at=timestamp,
        )
        return {
            "interaction_id": interaction_id,
            "task_id": task_id,
            "kind": kind,
            "question": question,
            "options": options,
            "status": "pending",
        }

    async def get_interaction(self, interaction_id: str) -> dict[str, Any] | None:
        await self._ensure_ready()
        row = await ChatTaskInteraction.filter(interaction_id=interaction_id).first()
        if row is None:
            return None
        return {
            "interaction_id": row.interaction_id,
            "task_id": row.task_id,
            "kind": row.kind,
            "question": row.question,
            "options": json.loads(row.options_json),
            "payload": json.loads(row.payload_json),
            "status": row.status,
            "selected_option": row.selected_option,
            "created_at": row.created_at,
            "responded_at": row.responded_at,
        }

    async def answer_interaction(self, interaction_id: str, option_id: str) -> dict[str, Any]:
        await self._ensure_ready()
        row = await ChatTaskInteraction.filter(interaction_id=interaction_id).first()
        if row is None:
            raise KeyError("交互请求不存在")
        if row.status != "pending":
            raise ValueError("该交互请求已经处理")
        options = json.loads(row.options_json)
        if not any(str(option.get("id")) == option_id for option in options if isinstance(option, dict)):
            raise ValueError("无效的交互选项")
        updated = await ChatTaskInteraction.filter(
            interaction_id=interaction_id,
            status="pending",
        ).update(status="answered", selected_option=option_id, responded_at=_now())
        if not updated:
            raise ValueError("该交互请求已经处理")
        task = await ChatTask.filter(task_id=row.task_id).first()
        if task:
            message = await ChatMessage.filter(message_id=task.message_id).first()
            if message:
                parts = json.loads(message.parts_json)
                for part in parts:
                    if part.get("type") != "interaction":
                        continue
                    content = part.get("content")
                    if isinstance(content, dict) and content.get("interaction_id") == interaction_id:
                        content["status"] = "answered"
                        content["selected_option"] = option_id
                message.parts_json = _json(parts)
                message.updated_at = _now()
                await message.save(update_fields=["parts_json", "updated_at"])
        result = await self.get_interaction(interaction_id)
        if result is None:
            raise KeyError("交互请求不存在")
        return result

    async def update_task(self, task_id: str, status: str, error: str | None = None) -> None:
        await self._ensure_ready()
        timestamp = _now()
        await ChatTask.filter(task_id=task_id).update(status=status, error=error, updated_at=timestamp)
        task = await ChatTask.filter(task_id=task_id).first()
        if task:
            await ChatMessage.filter(message_id=task.message_id).update(status=status, updated_at=timestamp)

    async def touch_task(self, task_id: str) -> None:
        await self._ensure_ready()
        await ChatTask.filter(task_id=task_id, status="running").update(updated_at=_now())

    async def complete_task(self, task_id: str) -> bool:
        await self._ensure_ready()
        timestamp = _now()
        updated = await ChatTask.filter(task_id=task_id, status__in=["pending", "running"]).update(
            status="completed", error=None, updated_at=timestamp
        )
        if updated:
            task = await ChatTask.filter(task_id=task_id).first()
            if task:
                await ChatMessage.filter(message_id=task.message_id).update(status="completed", updated_at=timestamp)
        return bool(updated)

    async def begin_task(self, task_id: str) -> bool:
        await self._ensure_ready()
        timestamp = _now()
        updated = await ChatTask.filter(task_id=task_id, status__in=["pending", "interrupted"]).update(
            status="running", error=None, updated_at=timestamp
        )
        if updated:
            task = await ChatTask.filter(task_id=task_id).first()
            if task:
                await ChatMessage.filter(message_id=task.message_id).update(status="running", updated_at=timestamp)
        return bool(updated)

    async def list_runnable_tasks(self, limit: int = 50) -> list[str]:
        await self._ensure_ready()
        rows = await (
            ChatTask.filter(status__in=["pending", "interrupted"])
            .order_by("created_at")
            .limit(max(1, min(limit, 100)))
            .values("task_id")
        )
        return [str(row["task_id"]) for row in rows]

    async def mark_cancelled(self, task_id: str) -> bool:
        await self._ensure_ready()
        timestamp = _now()
        updated = await ChatTask.filter(task_id=task_id, status="cancel_requested").update(
            status="cancelled", updated_at=timestamp
        )
        if updated:
            task = await ChatTask.filter(task_id=task_id).first()
            if task:
                await ChatMessage.filter(message_id=task.message_id).update(status="cancelled", updated_at=timestamp)
        return bool(updated)

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        await self._ensure_ready()
        row = await ChatTask.filter(task_id=task_id).first()
        return row.__dict__.copy() if row else None

    async def set_references(self, message_id: str, references: list[dict[str, Any]]) -> None:
        await self._ensure_ready()
        await ChatMessageReference.filter(message_id=message_id).delete()
        timestamp = _now()
        await ChatMessageReference.bulk_create(
            [
                ChatMessageReference(
                    id=f"{message_id}:{position}",
                    message_id=message_id,
                    position=position,
                    reference_json=_json(reference),
                    created_at=timestamp,
                )
                for position, reference in enumerate(references)
            ]
        )

    async def _references_by_message(self, message_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not message_ids:
            return {}
        rows = await ChatMessageReference.filter(message_id__in=message_ids).order_by("message_id", "position").values(
            "message_id", "reference_json"
        )
        references: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            references.setdefault(row["message_id"], []).append(json.loads(row["reference_json"]))
        return references

    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        await self._ensure_ready()
        conversation = await ChatConversation.filter(conversation_id=conversation_id).first()
        if conversation is None:
            return None
        rows = await ChatMessage.filter(conversation_id=conversation_id).order_by("position").values(
            "message_id", "role", "parts_json", "status", "task_id", "created_at"
        )
        refs = await self._references_by_message([row["message_id"] for row in rows])
        return {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "messages": [self._message_payload(row, refs.get(row["message_id"], [])) for row in rows],
        }

    async def branch_conversation(
        self,
        source_conversation_id: str,
        through_message_id: str,
    ) -> dict[str, Any]:
        """Create an independent conversation snapshot through one completed assistant reply."""
        await self._ensure_ready()
        timestamp = _now()
        branch_conversation_id = f"conversation-{uuid4().hex}"

        async with in_transaction() as connection:
            source = (
                await ChatConversation.filter(conversation_id=source_conversation_id)
                .using_db(connection)
                .first()
            )
            if source is None:
                raise ValueError("源会话不存在")

            target = (
                await ChatMessage.filter(
                    message_id=through_message_id,
                    conversation_id=source_conversation_id,
                )
                .using_db(connection)
                .first()
            )
            if target is None:
                raise ValueError("分支消息不属于当前会话")
            if target.role != "assistant":
                raise ValueError("只能从助手回复创建分支")
            if target.status != "completed":
                raise ValueError("只能从已完成的助手回复创建分支")

            source_messages = await (
                ChatMessage.filter(
                    conversation_id=source_conversation_id,
                    position__lte=target.position,
                )
                .using_db(connection)
                .order_by("position")
                .values(
                    "message_id",
                    "role",
                    "parts_json",
                    "status",
                    "position",
                    "created_at",
                    "updated_at",
                )
            )
            source_message_ids = [str(message["message_id"]) for message in source_messages]
            source_references = await (
                ChatMessageReference.filter(message_id__in=source_message_ids)
                .using_db(connection)
                .order_by("message_id", "position")
                .values("message_id", "position", "reference_json", "created_at")
            )
            references_by_message: dict[str, list[dict[str, Any]]] = {}
            for reference in source_references:
                references_by_message.setdefault(str(reference["message_id"]), []).append(reference)

            source_title = source.title.strip() or "新对话"
            branch_title = f"{source_title}（分支）"
            await ChatConversation.create(
                conversation_id=branch_conversation_id,
                title=branch_title[:100],
                created_at=timestamp,
                updated_at=timestamp,
                using_db=connection,
            )

            for position, source_message in enumerate(source_messages):
                source_message_id = str(source_message["message_id"])
                branch_message_id = f"msg-{uuid4().hex}"
                parts = json.loads(source_message["parts_json"])
                await ChatMessage.create(
                    message_id=branch_message_id,
                    conversation_id=branch_conversation_id,
                    role=source_message["role"],
                    parts_json=source_message["parts_json"],
                    status=source_message["status"],
                    task_id=None,
                    position=position,
                    created_at=source_message["created_at"],
                    updated_at=source_message["updated_at"],
                    using_db=connection,
                )
                await self._sync_search(branch_message_id, branch_conversation_id, parts, connection)

                references = references_by_message.get(source_message_id, [])
                if references:
                    await ChatMessageReference.bulk_create(
                        [
                            ChatMessageReference(
                                id=f"{branch_message_id}:{reference_position}",
                                message_id=branch_message_id,
                                position=reference_position,
                                reference_json=reference["reference_json"],
                                created_at=reference["created_at"],
                            )
                            for reference_position, reference in enumerate(references)
                        ],
                        using_db=connection,
                    )

        conversation = await self.get_conversation(branch_conversation_id)
        if conversation is None:
            raise RuntimeError("分支会话创建后无法读取")
        return conversation

    async def list_conversations(self, query: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        await self._ensure_ready()
        conversations = ChatConversation.all()
        if query and query.strip():
            needle = query.strip()
            if self.search_index.sqlite_fts5_enabled and len(needle) >= 3:
                connection = Tortoise.get_connection("default")
                # Search input is user text, not an FTS expression. Quoting it
                # keeps punctuation such as hyphens from being interpreted as
                # column or operator syntax by SQLite FTS5.
                fts_query = '"' + needle.replace('"', '""') + '"'
                rows = await connection.execute_query_dict(
                    "SELECT DISTINCT conversation_id FROM chat_message_search_fts WHERE content MATCH ?",
                    [fts_query],
                )
                content_ids = [row["conversation_id"] for row in rows]
            else:
                content_ids = await (
                    ChatMessageSearch.filter(content__icontains=needle)
                    .distinct()
                    .values_list("conversation_id", flat=True)
                )
            title_ids = await ChatConversation.filter(title__icontains=needle).values_list(
                "conversation_id", flat=True
            )
            ids = list(set(title_ids) | set(content_ids))
            conversations = conversations.filter(conversation_id__in=ids)
        rows = await conversations.order_by("-updated_at").limit(max(1, min(limit, 200))).values(
            "conversation_id", "title", "created_at", "updated_at"
        )
        result = []
        for row in rows:
            messages = await ChatMessage.filter(conversation_id=row["conversation_id"]).order_by("position").values(
                "message_id", "role", "parts_json", "status", "task_id", "created_at"
            )
            refs = await self._references_by_message([message["message_id"] for message in messages])
            result.append(
                {
                    **row,
                    "messages": [
                        self._message_payload(message, refs.get(message["message_id"], [])) for message in messages
                    ],
                }
            )
        return result

    async def rename_conversation(self, conversation_id: str, title: str) -> bool:
        await self._ensure_ready()
        updated = await ChatConversation.filter(conversation_id=conversation_id).update(
            title=title.strip() or "新对话", updated_at=_now()
        )
        return bool(updated)

    async def delete_conversation(self, conversation_id: str) -> bool:
        await self._ensure_ready()
        active = await ChatTask.filter(
            conversation_id=conversation_id,
            status__in=["pending", "running", "cancel_requested"],
        ).exists()
        if active:
            return False
        task_ids = list(await ChatTask.filter(conversation_id=conversation_id).values_list("task_id", flat=True))
        message_ids = await ChatMessage.filter(conversation_id=conversation_id).values_list("message_id", flat=True)
        if message_ids:
            await ChatMessageReference.filter(message_id__in=list(message_ids)).delete()
            await self.search_index.remove_messages(list(message_ids))
        if task_ids:
            await ChatTaskEvent.filter(task_id__in=task_ids).delete()
            await ChatTaskInteraction.filter(task_id__in=task_ids).delete()
            await ChatTaskState.filter(task_id__in=task_ids).delete()
        await ChatMessage.filter(conversation_id=conversation_id).delete()
        await ChatTask.filter(conversation_id=conversation_id).delete()
        deleted = await ChatConversation.filter(conversation_id=conversation_id).delete()
        if deleted and task_ids:
            for task_id in task_ids:
                await checkpoint_manager.delete_thread_family(str(task_id))
        return bool(deleted)
