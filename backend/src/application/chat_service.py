"""Durable chat history and server-owned cancellable chat tasks."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from loguru import logger

from agents.stock_agent import stock_agent
from config import settings
from widgets.a2ui import (
    render_activity,
    render_markdown,
    render_tool_result,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _text_part(content: str) -> dict[str, str]:
    return {"type": "text", "content": content}


class ChatStore:
    """SQLite repository for conversations, messages, and task state."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or settings.database_file_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _init_schema(self) -> None:
        with self._thread_lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
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

                CREATE TABLE IF NOT EXISTS chat_tasks (
                    task_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation
                    ON chat_messages(conversation_id, position ASC);
                CREATE INDEX IF NOT EXISTS idx_chat_tasks_conversation
                    ON chat_tasks(conversation_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS chat_task_events (
                    task_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, sequence),
                    FOREIGN KEY(task_id) REFERENCES chat_tasks(task_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chat_task_events_task
                    ON chat_task_events(task_id, sequence ASC);
                """
            )
            connection.execute(
                """
                UPDATE chat_tasks
                   SET status = 'interrupted', updated_at = ?
                 WHERE status IN ('pending', 'running', 'cancel_requested')
                """,
                (_now(),),
            )
            connection.execute(
                """
                UPDATE chat_messages
                   SET status = 'interrupted', updated_at = ?
                 WHERE status IN ('pending', 'running')
                """,
                (_now(),),
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

    def _ensure_conversation(self, connection: sqlite3.Connection, conversation_id: str, title: str) -> None:
        timestamp = _now()
        connection.execute(
            """
            INSERT INTO chat_conversations (conversation_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (conversation_id, title or "新对话", timestamp, timestamp),
        )

    def prepare_task(
        self,
        conversation_id: str,
        task_id: str,
        message: str,
        history: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Create a task exactly once, while preserving edit-history semantics.

        ``task_id`` is the idempotency key. A retried POST returns the original
        assistant message instead of deleting and recreating the conversation.
        """
        user_message_id = f"msg-{uuid4().hex}"
        assistant_message_id = f"msg-{uuid4().hex}"
        timestamp = _now()
        title = next(
            (
                part.get("content", "").strip()
                for item in history + [{"role": "user", "parts": [_text_part(message)]}]
                if item.get("role") == "user"
                for part in self._parts(item)
                if part.get("type") == "text" and part.get("content", "").strip()
            ),
            "新对话",
        )
        title = title[:30] + ("…" if len(title) > 30 else "")
        with self._thread_lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT conversation_id, message_id FROM chat_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if existing:
                if existing["conversation_id"] != conversation_id:
                    raise ValueError("task_id 已被其他会话使用")
                return user_message_id, existing["message_id"]

            active = connection.execute(
                """
                SELECT task_id FROM chat_tasks
                 WHERE conversation_id = ?
                   AND status IN ('pending', 'running', 'cancel_requested')
                 ORDER BY created_at DESC LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            if active:
                raise ValueError(f"会话已有正在执行的任务: {active['task_id']}")

            self._ensure_conversation(connection, conversation_id, title)
            connection.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
            for position, item in enumerate(history):
                role = item.get("role", "user")
                connection.execute(
                    """
                    INSERT INTO chat_messages
                      (message_id, conversation_id, role, parts_json, status, task_id, position, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'completed', NULL, ?, ?, ?)
                    """,
                    (
                        f"msg-{uuid4().hex}",
                        conversation_id,
                        role,
                        _json(self._parts(item)),
                        position,
                        timestamp,
                        timestamp,
                    ),
                )
            user_position = len(history)
            connection.execute(
                """
                INSERT INTO chat_messages
                  (message_id, conversation_id, role, parts_json, status, task_id, position, created_at, updated_at)
                VALUES (?, ?, 'user', ?, 'completed', NULL, ?, ?, ?)
                """,
                (user_message_id, conversation_id, _json([_text_part(message)]), user_position, timestamp, timestamp),
            )
            connection.execute(
                """
                INSERT INTO chat_messages
                  (message_id, conversation_id, role, parts_json, status, task_id, position, created_at, updated_at)
                VALUES (?, ?, 'assistant', '[]', 'pending', ?, ?, ?, ?)
                """,
                (
                    assistant_message_id,
                    conversation_id,
                    task_id,
                    user_position + 1,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO chat_tasks
                  (task_id, conversation_id, message_id, status, error, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', NULL, ?, ?)
                """,
                (task_id, conversation_id, assistant_message_id, timestamp, timestamp),
            )
            connection.commit()
        return user_message_id, assistant_message_id

    def append_part(self, message_id: str, part: dict[str, Any], task_id: str | None = None) -> bool:
        with self._thread_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT parts_json, task_id FROM chat_messages WHERE message_id = ?", (message_id,)
            ).fetchone()
            if row is None:
                return False
            if task_id and row["task_id"] != task_id:
                return False
            if task_id:
                task = connection.execute(
                    "SELECT status FROM chat_tasks WHERE task_id = ?", (task_id,)
                ).fetchone()
                if task is None or task["status"] in {
                    "cancel_requested", "cancelled", "completed", "failed", "interrupted"
                }:
                    return False
            parts = json.loads(row["parts_json"])
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
            connection.execute(
                "UPDATE chat_messages SET parts_json = ?, status = 'running', updated_at = ? WHERE message_id = ?",
                (_json(parts), _now(), message_id),
            )
            connection.commit()
        return True

    def append_event(self, task_id: str, event: str, data: str) -> dict[str, Any]:
        """Append one durable SSE event and assign its monotonic task sequence."""
        with self._thread_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM chat_task_events WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            sequence = int(row["next_sequence"])
            connection.execute(
                """
                INSERT INTO chat_task_events(task_id, sequence, event, data, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task_id, sequence, event, data, _now()),
            )
            connection.commit()
        return {"id": str(sequence), "event": event, "data": data}

    def list_events(self, task_id: str, after_sequence: int = 0) -> list[dict[str, Any]]:
        with self._thread_lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event, data FROM chat_task_events
                 WHERE task_id = ? AND sequence > ? ORDER BY sequence ASC
                """,
                (task_id, max(0, after_sequence)),
            ).fetchall()
        return [
            {"id": str(row["sequence"]), "event": row["event"], "data": row["data"]}
            for row in rows
        ]

    def request_cancel(self, task_id: str) -> str | None:
        """Atomically move an active task into cancellation requested state."""
        with self._thread_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM chat_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            status = row["status"]
            if status not in {"pending", "running", "cancel_requested"}:
                return status
            connection.execute(
                """
                UPDATE chat_tasks SET status = 'cancel_requested', updated_at = ?
                 WHERE task_id = ? AND status IN ('pending', 'running', 'cancel_requested')
                """,
                (_now(), task_id),
            )
            connection.execute(
                """
                UPDATE chat_messages SET status = 'cancel_requested', updated_at = ?
                 WHERE message_id = (SELECT message_id FROM chat_tasks WHERE task_id = ?)
                """,
                (_now(), task_id),
            )
            connection.commit()
        return "cancel_requested"

    def update_task(self, task_id: str, status: str, error: str | None = None) -> None:
        with self._thread_lock, self._connect() as connection:
            timestamp = _now()
            connection.execute(
                "UPDATE chat_tasks SET status = ?, error = ?, updated_at = ? WHERE task_id = ?",
                (status, error, timestamp, task_id),
            )
            connection.execute(
                """
                UPDATE chat_messages
                   SET status = ?, updated_at = ?
                 WHERE message_id = (SELECT message_id FROM chat_tasks WHERE task_id = ?)
                """,
                (status, timestamp, task_id),
            )
            connection.commit()

    def complete_task(self, task_id: str) -> bool:
        """Complete only if cancellation has not won the state transition race."""
        with self._thread_lock, self._connect() as connection:
            timestamp = _now()
            cursor = connection.execute(
                """
                UPDATE chat_tasks SET status = 'completed', error = NULL, updated_at = ?
                 WHERE task_id = ? AND status IN ('pending', 'running')
                """,
                (timestamp, task_id),
            )
            if cursor.rowcount:
                connection.execute(
                    """
                    UPDATE chat_messages SET status = 'completed', updated_at = ?
                     WHERE message_id = (SELECT message_id FROM chat_tasks WHERE task_id = ?)
                    """,
                    (timestamp, task_id),
                )
            connection.commit()
        return cursor.rowcount > 0

    def begin_task(self, task_id: str) -> bool:
        """Claim a pending task without overriding an already requested cancel."""
        with self._thread_lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE chat_tasks SET status = 'running', updated_at = ? WHERE task_id = ? AND status = 'pending'",
                (_now(), task_id),
            )
            if cursor.rowcount:
                connection.execute(
                    """
                    UPDATE chat_messages SET status = 'running', updated_at = ?
                     WHERE message_id = (SELECT message_id FROM chat_tasks WHERE task_id = ?)
                    """,
                    (_now(), task_id),
                )
            connection.commit()
        return cursor.rowcount > 0

    def mark_cancelled(self, task_id: str) -> bool:
        """Finalize cancellation only for a task that requested cancellation."""
        with self._thread_lock, self._connect() as connection:
            timestamp = _now()
            cursor = connection.execute(
                """
                UPDATE chat_tasks SET status = 'cancelled', updated_at = ?
                 WHERE task_id = ? AND status = 'cancel_requested'
                """,
                (timestamp, task_id),
            )
            if cursor.rowcount:
                connection.execute(
                    """
                    UPDATE chat_messages SET status = 'cancelled', updated_at = ?
                     WHERE message_id = (SELECT message_id FROM chat_tasks WHERE task_id = ?)
                    """,
                    (timestamp, task_id),
                )
            connection.commit()
        return cursor.rowcount > 0

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._thread_lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM chat_tasks WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _message_payload(row: sqlite3.Row) -> dict[str, Any]:
        status = row["status"]
        return {
            "id": row["message_id"],
            "role": row["role"],
            "parts": json.loads(row["parts_json"]),
            "status": status,
            "task_id": row["task_id"],
            "loading": status in {"pending", "running"},
        }

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._thread_lock, self._connect() as connection:
            conversation = connection.execute(
                "SELECT * FROM chat_conversations WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
            if conversation is None:
                return None
            messages = connection.execute(
                "SELECT * FROM chat_messages WHERE conversation_id = ? ORDER BY position ASC",
                (conversation_id,),
            ).fetchall()
        return {
            "conversation_id": conversation["conversation_id"],
            "title": conversation["title"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "messages": [self._message_payload(row) for row in messages],
        }

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._thread_lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chat_conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
            messages_by_conversation: dict[str, list[sqlite3.Row]] = {}
            for row in rows:
                messages_by_conversation[row["conversation_id"]] = connection.execute(
                    "SELECT * FROM chat_messages WHERE conversation_id = ? ORDER BY position ASC",
                    (row["conversation_id"],),
                ).fetchall()
        return [
            {
                "conversation_id": row["conversation_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "messages": [
                    self._message_payload(message) for message in messages_by_conversation[row["conversation_id"]]
                ],
            }
            for row in rows
        ]

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        with self._thread_lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE chat_conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
                (title.strip() or "新对话", _now(), conversation_id),
            )
            connection.commit()
        return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._thread_lock, self._connect() as connection:
            active = connection.execute(
                """
                SELECT 1 FROM chat_tasks
                 WHERE conversation_id = ?
                   AND status IN ('pending', 'running', 'cancel_requested')
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            if active:
                return False
            cursor = connection.execute("DELETE FROM chat_conversations WHERE conversation_id = ?", (conversation_id,))
            connection.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
            connection.execute("DELETE FROM chat_tasks WHERE conversation_id = ?", (conversation_id,))
            connection.commit()
        return cursor.rowcount > 0


@dataclass(frozen=True)
class ChatTaskInput:
    task_id: str
    conversation_id: str
    message: str
    history: list[dict[str, Any]]
    strategy: str | None
    asset_type: Any
    assistant_message_id: str


class ChatTaskManager:
    """Run Agent tasks independently from any one SSE subscriber."""

    def __init__(self, store: ChatStore):
        self.store = store
        self._tasks: dict[str, asyncio.Task] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict | None]]] = {}
        self._task_locks: dict[str, asyncio.Lock] = {}

    async def start(self, task_input: ChatTaskInput) -> None:
        if task_input.task_id in self._tasks:
            return
        record = self.store.get_task(task_input.task_id)
        if record and record["status"] == "cancel_requested":
            self.store.update_task(task_input.task_id, "cancelled")
            return
        if record and record["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            return
        self._subscribers.setdefault(task_input.task_id, set())
        self._task_locks.setdefault(task_input.task_id, asyncio.Lock())
        self._tasks[task_input.task_id] = asyncio.create_task(
            self._run(task_input),
            name=f"chat-agent-{task_input.task_id}",
        )

    async def subscribe(self, task_id: str, after_sequence: int = 0) -> AsyncIterator[dict]:
        """Replay missed events and subscribe atomically to future events."""
        lock = self._task_locks.setdefault(task_id, asyncio.Lock())
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        async with lock:
            replay = self.store.list_events(task_id, after_sequence)
            record = self.store.get_task(task_id)
            terminal = bool(record and record["status"] in {"completed", "failed", "cancelled", "interrupted"})
            self._subscribers.setdefault(task_id, set()).add(queue)
        try:
            for event in replay:
                yield event
            if terminal and not replay:
                yield {"event": "done", "data": "{}"}
            if terminal or any(event.get("event") == "done" for event in replay):
                return
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            self._subscribers.get(task_id, set()).discard(queue)

    async def cancel(self, task_id: str) -> dict[str, Any]:
        task = self._tasks.get(task_id)
        status = self.store.request_cancel(task_id)
        if status is None:
            return {"task_id": task_id, "status": "not_found"}
        if status != "cancel_requested":
            return {"task_id": task_id, "status": status}
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        else:
            self.store.mark_cancelled(task_id)
        final_record = self.store.get_task(task_id)
        return {"task_id": task_id, "status": final_record["status"] if final_record else "not_found"}

    async def _broadcast(self, task_id: str, event: dict | None) -> None:
        if event is None:
            for queue in tuple(self._subscribers.get(task_id, set())):
                queue.put_nowait(event)
            return
        lock = self._task_locks.setdefault(task_id, asyncio.Lock())
        async with lock:
            record = self.store.get_task(task_id)
            if event.get("event") != "done" and record and record["status"] in {
                "cancel_requested", "cancelled", "completed", "failed", "interrupted"
            }:
                return
            persisted = self.store.append_event(task_id, event["event"], event["data"])
            for queue in tuple(self._subscribers.get(task_id, set())):
                queue.put_nowait(persisted)

    async def _emit_text(self, task_input: ChatTaskInput, text: str) -> None:
        """Emit all assistant prose as a native A2UI Markdown surface."""
        await self._emit_a2ui(task_input, render_markdown(text))

    async def _emit_a2ui(self, task_input: ChatTaskInput, messages: list[dict[str, Any]]) -> None:
        """Persist and stream one or more protocol-compliant A2UI messages."""
        for message in messages:
            if not self.store.append_part(
                task_input.assistant_message_id,
                {"type": "a2ui", "content": message},
                task_input.task_id,
            ):
                raise asyncio.CancelledError
            await self._broadcast(
                task_input.task_id,
                {"event": "a2ui", "data": _json({"a2ui": message})},
            )

    async def _emit_artifact(self, task_input: ChatTaskInput, artifact: dict[str, Any]) -> None:
        """Persist and stream one generated file artifact."""
        if not self.store.append_part(
            task_input.assistant_message_id,
            {"type": "artifact", "content": artifact},
            task_input.task_id,
        ):
            raise asyncio.CancelledError
        await self._broadcast(
            task_input.task_id,
            {"event": "artifact", "data": _json({"artifact": artifact})},
        )

    async def _emit_tool_event(self, task_input: ChatTaskInput, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Render tool progress and, when possible, its structured result."""
        name = str(event.get("name", "unknown"))
        status = str(event.get("status", "completed"))
        result = event.get("result")
        error_message = ""
        if status == "failed" and result:
            try:
                payload = json.loads(str(result))
                error = payload.get("error", {}) if isinstance(payload, dict) else {}
                if isinstance(error, dict):
                    code = str(error.get("code", "tool_error"))
                    message = str(error.get("message", "工具执行失败"))
                    error_message = f"{code}: {message}"
                else:
                    error_message = str(error)
            except (TypeError, json.JSONDecodeError):
                error_message = str(result)
        await self._emit_a2ui(
            task_input,
            render_activity(name, status, error=error_message),
        )
        artifacts: list[dict[str, Any]] = []
        if result:
            try:
                tool_payload = json.loads(str(result))
            except (TypeError, json.JSONDecodeError):
                tool_payload = {}
            for artifact in tool_payload.get("artifacts", []) if isinstance(tool_payload, dict) else []:
                if isinstance(artifact, dict):
                    artifacts.append(artifact)
            surface = render_tool_result(name, str(result))
            if surface:
                await self._emit_a2ui(task_input, surface)
        return artifacts

    @staticmethod
    def _queue_unique_artifacts(target: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> None:
        """Keep one visible report per instrument in a single Agent Loop task."""
        existing = {
            (item.get("name"), item.get("ticker"), item.get("asset_type"))
            for item in target
        }
        for artifact in artifacts:
            key = (artifact.get("name"), artifact.get("ticker"), artifact.get("asset_type"))
            if key in existing and any(key):
                continue
            target.append(artifact)
            existing.add(key)



    async def _run(self, task_input: ChatTaskInput) -> None:
        task_id = task_input.task_id
        try:
            if not self.store.begin_task(task_id):
                raise asyncio.CancelledError
            await self._emit_text(task_input, "A-Share Agent：正在让模型判断任务并选择数据工具。")
            orchestration_surface = f"orchestration-{task_id}"
            await self._emit_a2ui(
                task_input,
                render_activity("Agent 正在规划任务并选择数据工具", "running", orchestration_surface),
            )
            request = stock_agent.prepare(
                message=task_input.message,
                history=task_input.history,
                strategy=task_input.strategy,
                conversation_id=task_input.conversation_id,
                asset_type=task_input.asset_type,
            )
            pending_artifacts: list[dict[str, Any]] = []
            seen_tool_events: set[str] = set()
            async for event in stock_agent.chat(request):
                if event.get("type") == "tool":
                    event_key = f"{event.get('name', 'unknown')}:{event.get('result', '')}"
                    if event_key not in seen_tool_events:
                        seen_tool_events.add(event_key)
                        self._queue_unique_artifacts(
                            pending_artifacts,
                            await self._emit_tool_event(task_input, event),
                        )
                else:
                    text = event.get("text", "")
                    if text:
                        prefix = "分析摘要：" if event.get("type") == "reasoning" else ""
                        await self._emit_text(task_input, f"{prefix}{text}")
            await self._emit_a2ui(
                task_input,
                render_activity(
                    "Agent 正在规划任务并选择数据工具",
                    "completed",
                    orchestration_surface,
                    include_create=False,
                ),
            )
            for artifact in pending_artifacts:
                await self._emit_artifact(task_input, artifact)
            record = self.store.get_task(task_id)
            if record is None or record["status"] == "cancel_requested":
                raise asyncio.CancelledError
            if not self.store.complete_task(task_id):
                raise asyncio.CancelledError
            await self._broadcast(task_id, {"event": "done", "data": "{}"})
        except asyncio.CancelledError:
            logger.info("[StockAgent] Chat task cancelled: {}", task_id)
            if self.store.mark_cancelled(task_id):
                await self._broadcast(task_id, {"event": "done", "data": "{}"})
        except Exception as exc:
            logger.exception("[StockAgent] Chat task failed: {}", exc)
            await self._emit_text(task_input, f"请求失败：{exc}")
            self.store.update_task(task_id, "failed", str(exc))
            await self._broadcast(task_id, {"event": "done", "data": "{}"})
        finally:
            await self._broadcast(task_id, None)
            self._tasks.pop(task_id, None)
            self._subscribers.pop(task_id, None)
            self._task_locks.pop(task_id, None)


chat_store = ChatStore()
chat_task_manager = ChatTaskManager(chat_store)
