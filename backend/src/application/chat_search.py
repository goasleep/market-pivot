"""Derived full-text index maintenance for durable chat messages."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from tortoise import Tortoise

from data.chat_models import ChatMessage, ChatMessageSearch


def parts_text(parts: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(part.get("content", "")) for part in parts if part.get("type") == "text" and part.get("content")
    ).strip()


class ChatSearchIndex:
    """Maintain the derived search table and optional SQLite FTS mirror."""

    def __init__(self) -> None:
        self.sqlite_fts5_enabled = False

    async def initialize(self) -> None:
        await self._ensure_native_index()
        await self._backfill_if_empty()
        await self._sync_sqlite_fts_from_search_table()

    def reset(self) -> None:
        self.sqlite_fts5_enabled = False

    async def _ensure_native_index(self) -> None:
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
                self.sqlite_fts5_enabled = True
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

    async def _backfill_if_empty(self) -> None:
        """Migrate legacy messages once; normal writes maintain the index incrementally."""
        if await ChatMessageSearch.exists() or not await ChatMessage.exists():
            return
        messages = await ChatMessage.all().values("message_id", "conversation_id", "parts_json")
        entries = []
        for message in messages:
            content = parts_text(json.loads(message["parts_json"]))
            if content:
                entries.append(
                    ChatMessageSearch(
                        message_id=message["message_id"],
                        conversation_id=message["conversation_id"],
                        content=content,
                    )
                )
        if entries:
            await ChatMessageSearch.bulk_create(entries, ignore_conflicts=True)

    async def _sync_sqlite_fts_from_search_table(self) -> None:
        if not self.sqlite_fts5_enabled:
            return
        connection = Tortoise.get_connection("default")
        await connection.execute_script(
            """
            DELETE FROM chat_message_search_fts
            WHERE message_id NOT IN (SELECT message_id FROM chat_message_search);
            INSERT INTO chat_message_search_fts(message_id, conversation_id, content)
            SELECT source.message_id, source.conversation_id, source.content
            FROM chat_message_search AS source
            WHERE NOT EXISTS (
                SELECT 1 FROM chat_message_search_fts AS target
                WHERE target.message_id = source.message_id
            );
            """
        )

    async def sync(
        self,
        message_id: str,
        conversation_id: str,
        parts: list[dict[str, Any]],
        connection=None,
    ) -> None:
        content = parts_text(parts)
        query = ChatMessageSearch.filter(message_id=message_id)
        if connection is not None:
            query = query.using_db(connection)
        await query.delete()
        await self._sync_sqlite_message(message_id, conversation_id, content, connection)
        if content:
            await ChatMessageSearch.create(
                message_id=message_id,
                conversation_id=conversation_id,
                content=content,
                using_db=connection,
            )

    async def remove_messages(self, message_ids: list[str], connection=None) -> None:
        if not message_ids:
            return
        query = ChatMessageSearch.filter(message_id__in=message_ids)
        if connection is not None:
            query = query.using_db(connection)
        await query.delete()
        if not self.sqlite_fts5_enabled:
            return
        db = connection or Tortoise.get_connection("default")
        for message_id in message_ids:
            await db.execute_query("DELETE FROM chat_message_search_fts WHERE message_id = ?", [message_id])

    async def _sync_sqlite_message(
        self,
        message_id: str,
        conversation_id: str,
        content: str,
        connection=None,
    ) -> None:
        if not self.sqlite_fts5_enabled:
            return
        db = connection or Tortoise.get_connection("default")
        await db.execute_query("DELETE FROM chat_message_search_fts WHERE message_id = ?", [message_id])
        if content:
            await db.execute_query(
                "INSERT INTO chat_message_search_fts(message_id, conversation_id, content) VALUES (?, ?, ?)",
                [message_id, conversation_id, content],
            )
