"""SQLAlchemy ORM checkpoint saver for both SQLite and PostgreSQL."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from sqlalchemy import delete, select

from config import settings
from data.orm import (
    LangGraphCheckpointBlobRecord,
    LangGraphCheckpointRecord,
    LangGraphCheckpointWriteRecord,
    build_database,
)


class OrmCheckpointSaver(BaseCheckpointSaver[str]):
    """Persist LangGraph checkpoints with dialect-neutral ORM operations."""

    def __init__(self, db_path: str | Path | None = None, database_url: str | None = None):
        super().__init__()
        self.database = build_database(
            database_url=database_url,
            db_path=db_path or settings.database_file_path,
        )

    async def start(self) -> None:
        await asyncio.to_thread(self.database.ensure_schema)

    async def close(self) -> None:
        await asyncio.to_thread(self.database.dispose)

    @staticmethod
    def _key(config: RunnableConfig) -> tuple[str, str]:
        configurable = config["configurable"]
        return str(configurable["thread_id"]), str(configurable.get("checkpoint_ns", ""))

    def _load_tuple(self, row: LangGraphCheckpointRecord, config: RunnableConfig) -> CheckpointTuple:
        thread_id = row.thread_id
        checkpoint_ns = row.checkpoint_ns
        checkpoint_id = row.checkpoint_id
        checkpoint = self.serde.loads_typed((row.checkpoint_type, row.checkpoint_blob))
        with self.database.session() as session:
            blob_rows = session.scalars(
                select(LangGraphCheckpointBlobRecord).where(
                    LangGraphCheckpointBlobRecord.thread_id == thread_id,
                    LangGraphCheckpointBlobRecord.checkpoint_ns == checkpoint_ns,
                )
            ).all()
            write_rows = session.scalars(
                select(LangGraphCheckpointWriteRecord)
                .where(
                    LangGraphCheckpointWriteRecord.thread_id == thread_id,
                    LangGraphCheckpointWriteRecord.checkpoint_ns == checkpoint_ns,
                    LangGraphCheckpointWriteRecord.checkpoint_id == checkpoint_id,
                )
                .order_by(LangGraphCheckpointWriteRecord.task_id, LangGraphCheckpointWriteRecord.write_index)
            ).all()

        channel_values: dict[str, Any] = {}
        versions = checkpoint.get("channel_versions", {})
        wanted = {(str(channel), str(version)) for channel, version in versions.items()}
        for blob in blob_rows:
            if (blob.channel, blob.version) not in wanted:
                continue
            if blob.value_type != "empty":
                channel_values[blob.channel] = self.serde.loads_typed((blob.value_type, blob.value_blob))
        checkpoint = {**checkpoint, "channel_values": channel_values}
        pending_writes = [
            (write.task_id, write.channel, self.serde.loads_typed((write.value_type, write.value_blob)))
            for write in write_rows
        ]
        parent_config = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": row.parent_checkpoint_id,
                }
            }
            if row.parent_checkpoint_id
            else None
        )
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=self.serde.loads_typed((row.metadata_type, row.metadata_blob)),
            pending_writes=pending_writes,
            parent_config=parent_config,
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id, checkpoint_ns = self._key(config)
        checkpoint_id = get_checkpoint_id(config)
        with self.database.session() as session:
            statement = select(LangGraphCheckpointRecord).where(
                LangGraphCheckpointRecord.thread_id == thread_id,
                LangGraphCheckpointRecord.checkpoint_ns == checkpoint_ns,
            )
            if checkpoint_id:
                statement = statement.where(LangGraphCheckpointRecord.checkpoint_id == checkpoint_id)
            else:
                statement = statement.order_by(LangGraphCheckpointRecord.checkpoint_id.desc()).limit(1)
            row = session.scalar(statement)
        return self._load_tuple(row, config) if row else None

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        statement = select(LangGraphCheckpointRecord)
        if config:
            thread_id, checkpoint_ns = self._key(config)
            statement = statement.where(
                LangGraphCheckpointRecord.thread_id == thread_id,
                LangGraphCheckpointRecord.checkpoint_ns == checkpoint_ns,
            )
        if before and get_checkpoint_id(before):
            statement = statement.where(LangGraphCheckpointRecord.checkpoint_id < get_checkpoint_id(before))
        statement = statement.order_by(LangGraphCheckpointRecord.checkpoint_id.desc())
        if limit is not None:
            statement = statement.limit(max(0, limit))
        with self.database.session() as session:
            rows = session.scalars(statement).all()
        for row in rows:
            item = self._load_tuple(row, row_to_config(row))
            if filter and not all(item.metadata.get(key) == value for key, value in filter.items()):
                continue
            yield item

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id, checkpoint_ns = self._key(config)
        checkpoint_id = str(checkpoint["id"])
        values = dict(checkpoint.get("channel_values", {}))
        checkpoint_without_values = {key: value for key, value in checkpoint.items() if key != "channel_values"}
        checkpoint_type, checkpoint_blob = self.serde.dumps_typed(checkpoint_without_values)
        metadata_type, metadata_blob = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        with self.database.session() as session:
            session.merge(
                LangGraphCheckpointRecord(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=config["configurable"].get("checkpoint_id"),
                    checkpoint_type=checkpoint_type,
                    checkpoint_blob=checkpoint_blob,
                    metadata_type=metadata_type,
                    metadata_blob=metadata_blob,
                )
            )
            for channel, version in new_versions.items():
                value_type, value_blob = (
                    self.serde.dumps_typed(values[channel]) if channel in values else ("empty", b"")
                )
                session.merge(
                    LangGraphCheckpointBlobRecord(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        channel=channel,
                        version=str(version),
                        value_type=value_type,
                        value_blob=value_blob,
                    )
                )
            session.commit()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, checkpoint_ns = self._key(config)
        checkpoint_id = str(config["configurable"]["checkpoint_id"])
        with self.database.session() as session:
            for index, (channel, value) in enumerate(writes):
                write_index = WRITES_IDX_MAP.get(channel, index)
                existing = session.get(
                    LangGraphCheckpointWriteRecord,
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, write_index),
                )
                if existing is not None:
                    continue
                value_type, value_blob = self.serde.dumps_typed(value)
                session.add(
                    LangGraphCheckpointWriteRecord(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        task_id=task_id,
                        write_index=write_index,
                        channel=channel,
                        value_type=value_type,
                        value_blob=value_blob,
                        task_path=task_path,
                    )
                )
            session.commit()

    def delete_thread(self, thread_id: str) -> None:
        with self.database.session() as session:
            for model in (
                LangGraphCheckpointRecord,
                LangGraphCheckpointBlobRecord,
                LangGraphCheckpointWriteRecord,
            ):
                session.execute(delete(model).where(model.thread_id == thread_id))
            session.commit()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in await asyncio.to_thread(lambda: list(self.list(config, filter=filter, before=before, limit=limit))):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)


class SQLiteCheckpointSaver(OrmCheckpointSaver):
    """Backward-compatible name for the ORM checkpoint saver."""


def row_to_config(row: LangGraphCheckpointRecord) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": row.thread_id,
            "checkpoint_ns": row.checkpoint_ns,
            "checkpoint_id": row.checkpoint_id,
        }
    }
