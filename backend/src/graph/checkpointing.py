"""Lifecycle for the official LangGraph SQLite/PostgreSQL checkpointers."""

from __future__ import annotations

from contextlib import AsyncExitStack
from datetime import datetime, timedelta, timezone
from typing import Any

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from loguru import logger

from config import settings

CHECKPOINT_ALLOWED_MSGPACK_TYPES = (
    ("models.schemas", "AssetType"),
    ("models.schemas", "MarketContext"),
    ("models.schemas", "Decision"),
    ("models.schemas", "AgentReport"),
    ("models.schemas", "SignalType"),
    ("models.schemas", "TradeDecision"),
)


def _checkpoint_serializer() -> JsonPlusSerializer:
    """Allow only the application state types intentionally persisted by LangGraph."""
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_ALLOWED_MSGPACK_TYPES)


def _psycopg_url(value: str) -> str:
    """Normalize ORM-style PostgreSQL URLs for psycopg."""
    normalized = value.replace("postgresql+asyncpg://", "postgresql://", 1)
    return normalized.replace("postgres://", "postgresql://", 1)


class CheckpointManager:
    """Own one process-wide saver while FastAPI and its workers are alive."""

    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self.saver: BaseCheckpointSaver[str] | None = None

    async def start(self) -> BaseCheckpointSaver[str]:
        if self.saver is not None:
            return self.saver
        stack = AsyncExitStack()
        database_url = settings.resolved_checkpoint_database_url
        serializer = _checkpoint_serializer()
        if database_url:
            saver = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(_psycopg_url(database_url), serde=serializer)
            )
            backend = "postgres"
        else:
            connection = await stack.enter_async_context(
                aiosqlite.connect(str(settings.checkpoint_file_path))
            )
            saver = AsyncSqliteSaver(connection, serde=serializer)
            backend = "sqlite"
        await saver.setup()
        self._stack = stack
        self.saver = saver
        logger.info("LangGraph checkpoint backend ready: {}", backend)
        return saver

    async def stop(self) -> None:
        stack, self._stack = self._stack, None
        self.saver = None
        if stack is not None:
            await stack.aclose()

    async def delete_thread(self, thread_id: str) -> None:
        if self.saver is not None and thread_id:
            await self.saver.adelete_thread(thread_id)

    async def delete_thread_family(self, thread_id: str) -> None:
        """Delete a task thread plus separately checkpointed research children."""
        if self.saver is None or not thread_id:
            return
        matches = {thread_id}
        async for item in self.saver.alist(None):
            candidate = str(item.config.get("configurable", {}).get("thread_id", ""))
            if candidate.startswith(f"{thread_id}:"):
                matches.add(candidate)
        for candidate in matches:
            await self.saver.adelete_thread(candidate)

    async def prune_stale_threads(self, retention_days: int | None = None) -> int:
        """Delete terminal-age checkpoint threads using their newest snapshot."""
        if self.saver is None:
            return 0
        days = retention_days if retention_days is not None else settings.checkpoint_retention_days
        if days <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        newest: dict[str, datetime] = {}
        async for item in self.saver.alist(None):
            configurable = item.config.get("configurable", {})
            thread_id = str(configurable.get("thread_id", ""))
            timestamp = item.checkpoint.get("ts")
            if not thread_id or not isinstance(timestamp, str):
                continue
            try:
                created_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                continue
            newest[thread_id] = max(newest.get(thread_id, created_at), created_at)
        stale = [thread_id for thread_id, created_at in newest.items() if created_at < cutoff]
        if stale:
            from data.chat_models import ChatTask

            base_ids = {thread_id.split(":research:", 1)[0] for thread_id in stale}
            terminal_ids = set(
                await ChatTask.filter(
                    task_id__in=base_ids,
                    status__in=["completed", "failed", "cancelled", "superseded"],
                ).values_list("task_id", flat=True)
            )
            stale = [thread_id for thread_id in stale if thread_id.split(":research:", 1)[0] in terminal_ids]
        for thread_id in stale:
            await self.saver.adelete_thread(thread_id)
        return len(stale)

    @staticmethod
    def graph_config(thread_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(config or {})
        configurable = dict(merged.get("configurable") or {})
        configurable["thread_id"] = thread_id
        merged["configurable"] = configurable
        merged["recursion_limit"] = 120
        return merged


checkpoint_manager = CheckpointManager()
