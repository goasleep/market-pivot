"""ORM-backed runtime leases, jobs, and events.

The historical module name is retained for compatibility. The repository now
selects SQLite or PostgreSQL through SQLAlchemy, so the same transaction and
row-locking code is used in both deployment modes.
"""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from config import settings
from data.orm import RuntimeEventRecord, RuntimeJobRecord, RuntimeLeaseRecord, build_database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


class SQLiteCoordination:
    """Dialect-neutral runtime repository; SQLite remains the local default."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        owner_id: str | None = None,
        database_url: str | None = None,
    ):
        self.owner_id = owner_id or f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:12]}"
        self.database = build_database(
            database_url=database_url,
            db_path=db_path or settings.database_file_path,
        )

    def acquire_lease(self, lease_name: str, ttl_seconds: int = 120) -> bool:
        now = time.time()
        expiry = now + max(1, ttl_seconds)
        with self.database.session() as session:
            row = session.scalar(
                select(RuntimeLeaseRecord).where(RuntimeLeaseRecord.lease_name == lease_name).with_for_update()
            )
            if row and row.owner_id != self.owner_id and row.expires_at > now:
                return False
            if row is None:
                session.add(
                    RuntimeLeaseRecord(
                        lease_name=lease_name,
                        owner_id=self.owner_id,
                        expires_at=expiry,
                        updated_at=now,
                    )
                )
            else:
                row.owner_id = self.owner_id
                row.expires_at = expiry
                row.updated_at = now
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return False
        return True

    def renew_lease(self, lease_name: str, ttl_seconds: int = 120) -> bool:
        with self.database.session() as session:
            row = session.scalar(
                select(RuntimeLeaseRecord).where(
                    RuntimeLeaseRecord.lease_name == lease_name,
                    RuntimeLeaseRecord.owner_id == self.owner_id,
                )
            )
            if row is None:
                return False
            row.expires_at = time.time() + max(1, ttl_seconds)
            row.updated_at = time.time()
            session.commit()
        return True

    def release_lease(self, lease_name: str) -> None:
        with self.database.session() as session:
            row = session.scalar(
                select(RuntimeLeaseRecord).where(
                    RuntimeLeaseRecord.lease_name == lease_name,
                    RuntimeLeaseRecord.owner_id == self.owner_id,
                )
            )
            if row is not None:
                session.delete(row)
                session.commit()

    def create_job(self, job_id: str, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        timestamp = _now()
        with self.database.session() as session:
            session.add(
                RuntimeJobRecord(
                    job_id=job_id,
                    kind=kind,
                    params_json=params,
                    status="queued",
                    progress_json=[],
                    result_json=None,
                    error=None,
                    owner_id=None,
                    lease_expires_at=None,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
            session.commit()
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.database.session() as session:
            row = session.get(RuntimeJobRecord, job_id)
            if row is None:
                raise KeyError(f"任务不存在: {job_id}")
            return self._job_from_row(row)

    def list_jobs(self, kind: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.database.session() as session:
            rows = session.scalars(
                select(RuntimeJobRecord)
                .where(RuntimeJobRecord.kind == kind)
                .order_by(RuntimeJobRecord.created_at.asc())
                .limit(max(1, min(limit, 500)))
            ).all()
            return [self._job_from_row(row) for row in rows]

    def claim_job(self, job_id: str, ttl_seconds: int = 600) -> dict[str, Any] | None:
        now = time.time()
        with self.database.session() as session:
            row = session.scalar(select(RuntimeJobRecord).where(RuntimeJobRecord.job_id == job_id).with_for_update())
            if row is None:
                raise KeyError(f"任务不存在: {job_id}")
            if row.status in {"completed", "failed", "cancelled"}:
                return None
            if row.status == "running" and row.lease_expires_at and row.lease_expires_at > now:
                return None
            row.status = "running"
            row.owner_id = self.owner_id
            row.lease_expires_at = now + max(1, ttl_seconds)
            row.updated_at = _now()
            session.commit()
            return self._job_from_row(row)

    def update_job(self, job_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {"status", "progress_json", "result_json", "error", "owner_id", "lease_expires_at"}
        with self.database.session() as session:
            row = session.get(RuntimeJobRecord, job_id)
            if row is None:
                raise KeyError(f"任务不存在: {job_id}")
            for key, value in changes.items():
                if key in allowed:
                    setattr(row, key, _json_value(value) if key in {"progress_json", "result_json"} else value)
            row.updated_at = _now()
            session.commit()
            return self._job_from_row(row)

    def heartbeat_job(self, job_id: str, ttl_seconds: int = 600) -> bool:
        with self.database.session() as session:
            row = session.scalar(
                select(RuntimeJobRecord).where(
                    RuntimeJobRecord.job_id == job_id,
                    RuntimeJobRecord.owner_id == self.owner_id,
                    RuntimeJobRecord.status == "running",
                )
            )
            if row is None:
                return False
            row.lease_expires_at = time.time() + max(1, ttl_seconds)
            row.updated_at = _now()
            session.commit()
        return True

    def append_event(self, stream: str, stream_key: str, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        timestamp = _now()
        with self.database.session() as session:
            row = RuntimeEventRecord(
                stream=stream,
                stream_key=stream_key,
                event_type=event_type,
                data_json=data,
                created_at=timestamp,
            )
            session.add(row)
            session.commit()
            event_id = row.event_id
        return {
            "event_id": event_id,
            "type": event_type,
            "stream": stream,
            "stream_key": stream_key,
            "timestamp": timestamp,
            "data": data,
        }

    def list_events(self, stream: str, stream_key: str, after_id: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        with self.database.session() as session:
            rows = session.scalars(
                select(RuntimeEventRecord)
                .where(
                    RuntimeEventRecord.stream == stream,
                    RuntimeEventRecord.stream_key == stream_key,
                    RuntimeEventRecord.event_id > max(0, after_id),
                )
                .order_by(RuntimeEventRecord.event_id.asc())
                .limit(max(1, min(limit, 500)))
            ).all()
        return [
            {
                "event_id": row.event_id,
                "type": row.event_type,
                "stream": stream,
                "stream_key": stream_key,
                "timestamp": row.created_at,
                "data": _json_value(row.data_json),
            }
            for row in rows
        ]

    @staticmethod
    def _job_from_row(row: RuntimeJobRecord) -> dict[str, Any]:
        return {
            "job_id": row.job_id,
            "kind": row.kind,
            "params": _json_value(row.params_json),
            "status": row.status,
            "progress": _json_value(row.progress_json),
            "result": _json_value(row.result_json) if row.result_json is not None else None,
            "error": row.error,
            "owner_id": row.owner_id,
            "lease_expires_at": row.lease_expires_at,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }


sqlite_coordination = SQLiteCoordination()
