"""Database-selected async facade for durable runtime state."""

from __future__ import annotations

import asyncio

from config import settings
from data.sqlite_coordination import SQLiteCoordination


def _is_postgres_url(url: str | None) -> bool:
    return bool(url and url.startswith(("postgres://", "postgresql://")))


class SQLiteRuntimeStore:
    """ORM runtime store for local single-node SQLite mode."""

    backend = "sqlite"

    def __init__(self, store: SQLiteCoordination | None = None):
        self.store = store or SQLiteCoordination(db_path=settings.database_file_path)

    async def start(self) -> None:
        await asyncio.to_thread(self.store.database.ensure_schema)

    async def close(self) -> None:
        await asyncio.to_thread(self.store.database.dispose)

    async def acquire_lease(self, *args, **kwargs):
        return await asyncio.to_thread(self.store.acquire_lease, *args, **kwargs)

    async def renew_lease(self, *args, **kwargs):
        return await asyncio.to_thread(self.store.renew_lease, *args, **kwargs)

    async def release_lease(self, *args, **kwargs):
        return await asyncio.to_thread(self.store.release_lease, *args, **kwargs)

    async def create_job(self, *args, **kwargs):
        return await asyncio.to_thread(self.store.create_job, *args, **kwargs)

    async def get_job(self, *args, **kwargs):
        return await asyncio.to_thread(self.store.get_job, *args, **kwargs)

    async def list_jobs(self, *args, **kwargs):
        return await asyncio.to_thread(self.store.list_jobs, *args, **kwargs)

    async def claim_job(self, *args, **kwargs):
        return await asyncio.to_thread(self.store.claim_job, *args, **kwargs)

    async def update_job(self, *args, **kwargs):
        return await asyncio.to_thread(self.store.update_job, *args, **kwargs)

    async def heartbeat_job(self, *args, **kwargs):
        return await asyncio.to_thread(self.store.heartbeat_job, *args, **kwargs)

    async def append_event(self, *args, **kwargs):
        return await asyncio.to_thread(self.store.append_event, *args, **kwargs)

    async def list_events(self, *args, **kwargs):
        return await asyncio.to_thread(self.store.list_events, *args, **kwargs)


class PostgresRuntimeStore(SQLiteRuntimeStore):
    """The same ORM runtime repository backed by shared PostgreSQL."""

    backend = "postgres"

    def __init__(self, database_url: str):
        super().__init__(SQLiteCoordination(database_url=database_url))


def build_runtime_store():
    if _is_postgres_url(settings.database_url):
        return PostgresRuntimeStore(settings.database_url or "")
    return SQLiteRuntimeStore()


runtime_store = build_runtime_store()
