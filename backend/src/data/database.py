"""ORM-backed cache and application settings storage."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sqlalchemy import delete

from data.orm import AppSetting, CacheEntry, OrmDatabase, build_database


class SQLiteDatabase:
    """Backward-compatible name for the dialect-neutral ORM database.

    Existing callers use this small cache/settings API. The implementation now
    uses SQLAlchemy models and selects PostgreSQL whenever ``database_url`` is
    supplied, while retaining SQLite as the local default.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        database_url: str | None = None,
        orm_database: OrmDatabase | None = None,
    ):
        self.db_path = str(db_path) if db_path is not None else None
        self.database_url = database_url
        self.orm_database = orm_database or build_database(database_url=database_url, db_path=db_path)

    def get_cache(self, key: str) -> tuple[str, float] | None:
        with self.orm_database.session() as session:
            entry = session.get(CacheEntry, key)
            return (entry.value, entry.timestamp) if entry else None

    def set_cache(self, key: str, value: Any) -> None:
        with self.orm_database.session() as session:
            entry = session.get(CacheEntry, key)
            payload = json.dumps(value, ensure_ascii=False, default=str)
            if entry is None:
                session.add(CacheEntry(key=key, value=payload, timestamp=time.time()))
            else:
                entry.value = payload
                entry.timestamp = time.time()
            session.commit()

    def clear_cache(self) -> None:
        with self.orm_database.session() as session:
            session.execute(delete(CacheEntry))
            session.commit()

    def get_setting(self, key: str) -> Any | None:
        with self.orm_database.session() as session:
            entry = session.get(AppSetting, key)
            return json.loads(entry.value) if entry else None

    def set_setting(self, key: str, value: Any) -> None:
        with self.orm_database.session() as session:
            entry = session.get(AppSetting, key)
            payload = json.dumps(value, ensure_ascii=False, default=str)
            if entry is None:
                session.add(AppSetting(key=key, value=payload, updated_at=time.time()))
            else:
                entry.value = payload
                entry.updated_at = time.time()
            session.commit()
