"""Shared SQLAlchemy ORM models for SQLite and PostgreSQL persistence.

The application deliberately keeps chat models in Tortoise ORM because that
module already uses Tortoise's async lifecycle.  The remaining repositories
are synchronous and use these models so their public service APIs can stay
stable while the database dialect is selected by ``DATABASE_URL``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Float, Index, Integer, LargeBinary, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class OrmBase(DeclarativeBase):
    pass


class CacheEntry(OrmBase):
    __tablename__ = "cache"

    key: Mapped[str] = mapped_column(String(512), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)


class AppSetting(OrmBase):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)


class SimulationAccountRecord(OrmBase):
    __tablename__ = "simulation_accounts"

    account_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    current_date: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    portfolio_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class SimulationOrderRecord(OrmBase):
    __tablename__ = "simulation_orders"

    order_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    order_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class SimulationSnapshotRecord(OrmBase):
    __tablename__ = "simulation_snapshots"

    account_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot_date: Mapped[str] = mapped_column(String(32), primary_key=True)
    snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)


class AutomationTaskRecord(OrmBase):
    __tablename__ = "automation_tasks"

    account_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="idle")
    last_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_run_date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class AgentRunRecord(OrmBase):
    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    run_date: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class AgentDecisionRecord(OrmBase):
    __tablename__ = "agent_decisions"

    decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    account_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    decision_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)


class AutomationEventRecord(OrmBase):
    __tablename__ = "automation_events"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)


class BacktestExperimentRecord(OrmBase):
    __tablename__ = "backtest_experiments"

    experiment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class ArtifactRecord(OrmBase):
    __tablename__ = "artifacts"

    artifact_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    ticker: Mapped[str | None] = mapped_column(String(32), nullable=True)
    asset_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    conversation_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)


class RuntimeLeaseRecord(OrmBase):
    __tablename__ = "runtime_leases"

    lease_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)


class RuntimeJobRecord(OrmBase):
    __tablename__ = "runtime_jobs"

    job_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    params_json: Mapped[Any] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    progress_json: Mapped[Any] = mapped_column(JSON, nullable=False)
    result_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lease_expires_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class RuntimeEventRecord(OrmBase):
    __tablename__ = "runtime_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stream: Mapped[str] = mapped_column(String(64), nullable=False)
    stream_key: Mapped[str] = mapped_column(String(255), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    data_json: Mapped[Any] = mapped_column(JSON, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)


class LangGraphCheckpointRecord(OrmBase):
    __tablename__ = "langgraph_checkpoints"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(String(255), primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    parent_checkpoint_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    checkpoint_type: Mapped[str] = mapped_column(String(64), nullable=False)
    checkpoint_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    metadata_type: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class LangGraphCheckpointBlobRecord(OrmBase):
    __tablename__ = "langgraph_checkpoint_blobs"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(String(255), primary_key=True)
    channel: Mapped[str] = mapped_column(String(255), primary_key=True)
    version: Mapped[str] = mapped_column(String(255), primary_key=True)
    value_type: Mapped[str] = mapped_column(String(64), nullable=False)
    value_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class LangGraphCheckpointWriteRecord(OrmBase):
    __tablename__ = "langgraph_checkpoint_writes"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(String(255), primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    write_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel: Mapped[str] = mapped_column(String(255), nullable=False)
    value_type: Mapped[str] = mapped_column(String(64), nullable=False)
    value_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    task_path: Mapped[str] = mapped_column(String(1024), nullable=False)


Index(
    "idx_runtime_jobs_claim",
    RuntimeJobRecord.kind,
    RuntimeJobRecord.status,
    RuntimeJobRecord.lease_expires_at,
    RuntimeJobRecord.created_at,
)
Index(
    "idx_runtime_events_stream",
    RuntimeEventRecord.stream,
    RuntimeEventRecord.stream_key,
    RuntimeEventRecord.event_id,
)
Index("idx_artifacts_created_at", ArtifactRecord.created_at)
Index("idx_agent_runs_account_date", AgentRunRecord.account_id, AgentRunRecord.run_date)
Index("idx_agent_decisions_run", AgentDecisionRecord.run_id, AgentDecisionRecord.created_at)
Index("idx_automation_events_account", AutomationEventRecord.account_id, AutomationEventRecord.created_at)


class OrmDatabase:
    """A dialect-neutral SQLAlchemy engine and session factory."""

    def __init__(self, database_url: str | None = None, db_path: str | Path | None = None):
        if database_url:
            self.url = database_url.replace("postgres://", "postgresql+psycopg://", 1)
            self.url = self.url.replace("postgresql://", "postgresql+psycopg://", 1)
        else:
            path = Path(db_path or "./data/cache.db").expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.url = f"sqlite:///{path}"
        connect_args = {"check_same_thread": False} if self.url.startswith("sqlite:") else {}
        self.engine = create_engine(self.url, connect_args=connect_args, pool_pre_ping=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if not self._schema_ready:
                OrmBase.metadata.create_all(self.engine)
                self._schema_ready = True

    def session(self):
        self.ensure_schema()
        return self.sessions()

    def dispose(self) -> None:
        self.engine.dispose()


def build_database(database_url: str | None = None, db_path: str | Path | None = None) -> OrmDatabase:
    return OrmDatabase(database_url=database_url, db_path=db_path)
