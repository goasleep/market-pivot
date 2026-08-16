"""Async Tortoise repository for conversations, messages, and chat tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger
from tortoise import Tortoise
from tortoise.transactions import in_transaction

from config import settings
from data.chat_models import (
    ChatConversation,
    ChatMessage,
    ChatMessageReference,
    ChatMessageSearch,
    ChatTask,
    ChatTaskEvent,
)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _text_part(content: str) -> dict[str, str]:
    return {"type": "text", "content": content}


def _parts_text(parts: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(part.get("content", ""))
        for part in parts
        if part.get("type") == "text" and part.get("content")
    ).strip()


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
        self._sqlite_fts5_enabled = False

    async def init(self) -> None:
        if self._initialized:
            return
        if Tortoise.is_inited():
            if self._active_db_url != self.db_url:
                raise RuntimeError("另一个数据库连接已经初始化，请先关闭当前连接")
            self._initialized = True
            return

        await Tortoise.init(
            db_url=self.db_url,
            modules={"models": ["data.chat_models"]},
            # FastAPI runs lifespan and request handlers in different asyncio
            # tasks; the fallback keeps the initialized connection visible to
            # both tasks while preserving Tortoise's normal context behavior.
            _enable_global_fallback=True,
        )
        await Tortoise.generate_schemas(safe=True)
        await self._ensure_legacy_tables()
        await self._ensure_search_indexes()
        await self._recover_interrupted_tasks()
        await self._rebuild_search_index()
        self._active_db_url = self.db_url
        self._initialized = True

    async def close(self) -> None:
        if Tortoise.is_inited():
            await Tortoise.close_connections()
        self._active_db_url = None
        self._initialized = False
        self._sqlite_fts5_enabled = False

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
            """
        )
        for table in ("chat_task_events", "chat_message_references"):
            try:
                await connection.execute_query(f"ALTER TABLE {table} ADD COLUMN id TEXT")
            except Exception:
                # The column already exists on newly generated schemas.
                pass

    async def _recover_interrupted_tasks(self) -> None:
        timestamp = _now()
        await ChatTask.filter(status__in=["pending", "running", "cancel_requested"]).update(
            status="interrupted", updated_at=timestamp
        )
        await ChatMessage.filter(status__in=["pending", "running"]).update(
            status="interrupted", updated_at=timestamp
        )

    async def _ensure_search_indexes(self) -> None:
        connection = Tortoise.get_connection("default")
        dialect = connection.capabilities.dialect
        if dialect == "sqlite":
            try:
                await connection.execute_script(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chat_message_search_fts
                    USING fts5(message_id UNINDEXED, conversation_id UNINDEXED, content, tokenize='trigram');
                    """
                )
                self._sqlite_fts5_enabled = True
            except Exception as exc:
                logger.warning("SQLite FTS5 is unavailable; falling back to ORM content search: {}", exc)
        elif dialect == "postgres":
            try:
                await connection.execute_script(
                    """
                    CREATE EXTENSION IF NOT EXISTS pg_trgm;
                    CREATE INDEX IF NOT EXISTS idx_chat_message_search_content_trgm
                        ON chat_message_search USING gin (content gin_trgm_ops);
                    CREATE INDEX IF NOT EXISTS idx_chat_conversations_title_trgm
                        ON chat_conversations USING gin (title gin_trgm_ops);
                    """
                )
            except Exception as exc:
                logger.warning(
                    "PostgreSQL trigram indexes are unavailable; falling back to ORM content search: {}", exc
                )

    async def _sync_sqlite_fts(self, message_id: str, conversation_id: str, content: str, connection=None) -> None:
        if not self._sqlite_fts5_enabled:
            return
        db = connection or Tortoise.get_connection("default")
        await db.execute_query("DELETE FROM chat_message_search_fts WHERE message_id = ?", [message_id])
        if content:
            await db.execute_query(
                "INSERT INTO chat_message_search_fts(message_id, conversation_id, content) VALUES (?, ?, ?)",
                [message_id, conversation_id, content],
            )

    async def _rebuild_search_index(self) -> None:
        await ChatMessageSearch.all().delete()
        messages = await ChatMessage.all().values("message_id", "conversation_id", "parts_json")
        entries = []
        for message in messages:
            parts = json.loads(message["parts_json"])
            content = _parts_text(parts)
            if content:
                entries.append(
                    ChatMessageSearch(
                        message_id=message["message_id"],
                        conversation_id=message["conversation_id"],
                        content=content,
                    )
                )
        if entries:
            await ChatMessageSearch.bulk_create(entries)
        if self._sqlite_fts5_enabled:
            connection = Tortoise.get_connection("default")
            await connection.execute_query("DELETE FROM chat_message_search_fts")
            if entries:
                await connection.execute_many(
                    "INSERT INTO chat_message_search_fts(message_id, conversation_id, content) VALUES (?, ?, ?)",
                    [(entry.message_id, entry.conversation_id, entry.content) for entry in entries],
                )

    @staticmethod
    def _parts(item: dict[str, Any]) -> list[dict[str, Any]]:
        parts = item.get("parts")
        if isinstance(parts, list) and parts:
            return [
                part
                for part in parts
                if isinstance(part, dict) and part.get("type") in {"text", "a2ui", "widget", "artifact"}
            ]
        return [_text_part(str(item.get("content", "")))]

    @staticmethod
    def _message_payload(row: dict[str, Any], references: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        status = row["status"]
        return {
            "id": row["message_id"],
            "role": row["role"],
            "parts": json.loads(row["parts_json"]),
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
        content = _parts_text(parts)
        query = ChatMessageSearch.filter(message_id=message_id)
        if connection is not None:
            query = query.using_db(connection)
        await query.delete()
        await self._sync_sqlite_fts(message_id, conversation_id, content, connection)
        if content:
            await ChatMessageSearch.create(
                message_id=message_id,
                conversation_id=conversation_id,
                content=content,
                using_db=connection,
            )

    async def prepare_task(
        self,
        conversation_id: str,
        task_id: str,
        message: str,
        history: list[dict[str, Any]],
    ) -> tuple[str, str]:
        await self._ensure_ready()
        user_message_id = f"msg-{uuid4().hex}"
        assistant_message_id = f"msg-{uuid4().hex}"
        timestamp = _now()
        title = next(
            (
                part.get("content", "").strip()
                for item in history + [{"role": "user", "parts": [_text_part(message)]}]
                if item.get("role") == "user"
                for part in self._parts(item)
                if part.get("type") == "text" and str(part.get("content", "")).strip()
            ),
            "新对话",
        )
        title = title[:30] + ("…" if len(title) > 30 else "")

        async with in_transaction() as connection:
            existing = await ChatTask.filter(task_id=task_id).using_db(connection).first()
            if existing:
                if existing.conversation_id != conversation_id:
                    raise ValueError("task_id 已被其他会话使用")
                return user_message_id, existing.message_id

            active = (
                await ChatTask.filter(
                    conversation_id=conversation_id,
                    status__in=["pending", "running", "cancel_requested"],
                )
                .using_db(connection)
                .order_by("-created_at")
                .first()
            )
            if active:
                raise ValueError(f"会话已有正在执行的任务: {active.task_id}")

            conversation = (
                await ChatConversation.filter(conversation_id=conversation_id).using_db(connection).first()
            )
            if conversation is None:
                await ChatConversation.create(
                    conversation_id=conversation_id,
                    title=title or "新对话",
                    created_at=timestamp,
                    updated_at=timestamp,
                    using_db=connection,
                )
            else:
                await ChatConversation.filter(conversation_id=conversation_id).using_db(connection).update(
                    updated_at=timestamp
                )
            old_messages = await (
                ChatMessage.filter(conversation_id=conversation_id)
                .using_db(connection)
                .values("message_id")
            )
            old_ids = [item["message_id"] for item in old_messages]
            if old_ids:
                await ChatMessageReference.filter(message_id__in=old_ids).using_db(connection).delete()
                await ChatMessageSearch.filter(message_id__in=old_ids).using_db(connection).delete()
            await ChatMessage.filter(conversation_id=conversation_id).using_db(connection).delete()

            for position, item in enumerate(history):
                role = item.get("role", "user")
                item_timestamp = str(item.get("created_at") or timestamp)
                message_id = f"msg-{uuid4().hex}"
                parts = self._parts(item)
                await ChatMessage.create(
                    message_id=message_id,
                    conversation_id=conversation_id,
                    role=role,
                    parts_json=_json(parts),
                    status="completed",
                    task_id=None,
                    position=position,
                    created_at=item_timestamp,
                    updated_at=item_timestamp,
                    using_db=connection,
                )
                await self._sync_search(message_id, conversation_id, parts, connection)
                references = item.get("references")
                if isinstance(references, list):
                    for reference_position, reference in enumerate(references):
                        if isinstance(reference, dict):
                            await ChatMessageReference.create(
                                id=f"{message_id}:{reference_position}",
                                message_id=message_id,
                                position=reference_position,
                                reference_json=_json(reference),
                                created_at=item_timestamp,
                                using_db=connection,
                            )

            user_position = len(history)
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
        return user_message_id, assistant_message_id

    async def append_part(self, message_id: str, part: dict[str, Any], task_id: str | None = None) -> bool:
        await self._ensure_ready()
        message = await ChatMessage.filter(message_id=message_id).first()
        if message is None or (task_id and message.task_id != task_id):
            return False
        if task_id:
            task = await ChatTask.filter(task_id=task_id).first()
            if task is None or task.status in {"cancel_requested", "cancelled", "completed", "failed", "interrupted"}:
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
        return {"id": str(sequence), "event": event, "data": data}

    async def list_events(self, task_id: str, after_sequence: int = 0) -> list[dict[str, Any]]:
        await self._ensure_ready()
        rows = await (
            ChatTaskEvent.filter(task_id=task_id, sequence__gt=max(0, after_sequence))
            .order_by("sequence")
            .values("sequence", "event", "data")
        )
        return [{"id": str(row["sequence"]), "event": row["event"], "data": row["data"]} for row in rows]

    async def request_cancel(self, task_id: str) -> str | None:
        await self._ensure_ready()
        task = await ChatTask.filter(task_id=task_id).first()
        if task is None:
            return None
        if task.status not in {"pending", "running", "cancel_requested"}:
            return task.status
        timestamp = _now()
        await ChatTask.filter(task_id=task_id, status__in=["pending", "running", "cancel_requested"]).update(
            status="cancel_requested", updated_at=timestamp
        )
        await ChatMessage.filter(message_id=task.message_id).update(status="cancel_requested", updated_at=timestamp)
        return "cancel_requested"

    async def update_task(self, task_id: str, status: str, error: str | None = None) -> None:
        await self._ensure_ready()
        timestamp = _now()
        await ChatTask.filter(task_id=task_id).update(status=status, error=error, updated_at=timestamp)
        task = await ChatTask.filter(task_id=task_id).first()
        if task:
            await ChatMessage.filter(message_id=task.message_id).update(status=status, updated_at=timestamp)

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
        updated = await ChatTask.filter(task_id=task_id, status="pending").update(
            status="running", updated_at=timestamp
        )
        if updated:
            task = await ChatTask.filter(task_id=task_id).first()
            if task:
                await ChatMessage.filter(message_id=task.message_id).update(status="running", updated_at=timestamp)
        return bool(updated)

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
            "message_id", "role", "parts_json", "status", "task_id"
        )
        refs = await self._references_by_message([row["message_id"] for row in rows])
        return {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "messages": [self._message_payload(row, refs.get(row["message_id"], [])) for row in rows],
        }

    async def list_conversations(self, query: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        await self._ensure_ready()
        conversations = ChatConversation.all()
        if query and query.strip():
            needle = query.strip()
            if self._sqlite_fts5_enabled and len(needle) >= 3:
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
                "message_id", "role", "parts_json", "status", "task_id"
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
        message_ids = await ChatMessage.filter(conversation_id=conversation_id).values_list("message_id", flat=True)
        if message_ids:
            await ChatMessageReference.filter(message_id__in=list(message_ids)).delete()
            await ChatMessageSearch.filter(message_id__in=list(message_ids)).delete()
        await ChatMessage.filter(conversation_id=conversation_id).delete()
        await ChatTask.filter(conversation_id=conversation_id).delete()
        deleted = await ChatConversation.filter(conversation_id=conversation_id).delete()
        return bool(deleted)
