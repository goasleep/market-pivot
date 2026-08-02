"""Shared SQLite storage for cache and application settings."""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class SQLiteDatabase:
    """Small thread-safe SQLite store shared by all persistent backend data."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _init_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            connection.commit()

    def get_cache(self, key: str) -> tuple[str, float] | None:
        with self._lock, self._connect() as connection:
            return connection.execute(
                "SELECT value, timestamp FROM cache WHERE key = ?",
                (key,),
            ).fetchone()

    def set_cache(self, key: str, value: Any) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO cache (key, value, timestamp) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False, default=str), time.time()),
            )
            connection.commit()

    def clear_cache(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM cache")
            connection.commit()

    def get_setting(self, key: str) -> Any | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_settings WHERE key = ?",
                (key,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def set_setting(self, key: str, value: Any) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False, default=str), time.time()),
            )
            connection.commit()
