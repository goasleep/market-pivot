"""Durable state for unattended Agent simulation tasks.

The simulation account database owns cash, positions, orders, and snapshots.
This store owns the automation control plane: task configuration, run
idempotency, decision audit records, and a small event log.  Keeping these
concerns separate makes account reset safe while preserving the operational
history needed to explain what an Agent did.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from config import settings
from models.schemas import AgentDecisionAudit, AgentRunSummary, AutomationTaskConfig, TradeDecision


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False)


class AutomationStore:
    """Thread-safe SQLite repository for the automation control plane."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or settings.database_file_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _init_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS automation_tasks (
                    account_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    last_run_id TEXT,
                    last_run_date TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    run_date TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_decisions (
                    decision_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS automation_events (
                    event_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    run_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_runs_account_date
                    ON agent_runs(account_id, run_date DESC);
                CREATE INDEX IF NOT EXISTS idx_agent_decisions_run
                    ON agent_decisions(run_id, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_automation_events_account
                    ON automation_events(account_id, created_at DESC);
                """
            )

    def get_task(self, account_id: str) -> dict:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM automation_tasks WHERE account_id = ?", (account_id,)
            ).fetchone()
        if row is None:
            task = {
                "account_id": account_id,
                "config": AutomationTaskConfig(),
                "status": "idle",
                "last_run_id": None,
                "last_run_date": None,
                "last_error": None,
                "created_at": None,
                "updated_at": None,
            }
            self._save_task(task)
            return task
        return self._task_from_row(row)

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> dict:
        return {
            "account_id": row["account_id"],
            "config": AutomationTaskConfig.model_validate(json.loads(row["config_json"])),
            "status": row["status"],
            "last_run_id": row["last_run_id"],
            "last_run_date": row["last_run_date"],
            "last_error": row["last_error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _save_task(self, task: dict) -> dict:
        timestamp = _now()
        created_at = task.get("created_at") or timestamp
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO automation_tasks
                   (account_id, config_json, status, last_run_id, last_run_date,
                    last_error, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(account_id) DO UPDATE SET
                     config_json = excluded.config_json,
                     status = excluded.status,
                     last_run_id = excluded.last_run_id,
                     last_run_date = excluded.last_run_date,
                     last_error = excluded.last_error,
                     updated_at = excluded.updated_at""",
                (
                    task["account_id"],
                    _json(task["config"]),
                    task.get("status", "idle"),
                    task.get("last_run_id"),
                    task.get("last_run_date"),
                    task.get("last_error"),
                    created_at,
                    timestamp,
                ),
            )
        task["created_at"] = created_at
        task["updated_at"] = timestamp
        return task

    def update_task(self, account_id: str, **changes) -> dict:
        task = self.get_task(account_id)
        if "config" in changes:
            task["config"] = AutomationTaskConfig.model_validate(changes.pop("config"))
        elif changes:
            config_changes = {
                key: value
                for key, value in changes.items()
                if key in AutomationTaskConfig.model_fields
            }
            if config_changes:
                task["config"] = task["config"].model_copy(update=config_changes)
        for key, value in changes.items():
            if key not in AutomationTaskConfig.model_fields:
                task[key] = value
        return self._save_task(task)

    def create_run(
        self,
        account_id: str,
        run_date: str,
        trigger: str,
        config: AutomationTaskConfig,
        idempotency_key: str,
    ) -> AgentRunSummary:
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT summary_json FROM agent_runs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                return AgentRunSummary.model_validate(json.loads(existing["summary_json"]))
            summary = AgentRunSummary(
                run_id=f"run-{uuid4().hex[:16]}",
                account_id=account_id,
                run_date=run_date,
                trigger=trigger,
                status="queued",
                mode=config.mode,
                strategy_name=config.strategy_name,
                idempotency_key=idempotency_key,
            )
            timestamp = _now()
            connection.execute(
                """INSERT INTO agent_runs
                   (run_id, account_id, run_date, trigger, status, summary_json,
                    idempotency_key, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    summary.run_id,
                    account_id,
                    run_date,
                    trigger,
                    summary.status,
                    _json(summary),
                    idempotency_key,
                    timestamp,
                    timestamp,
                ),
            )
        self.update_task(account_id, last_run_id=summary.run_id, last_run_date=run_date, last_error=None)
        return summary

    def update_run(self, run_id: str, **changes) -> AgentRunSummary:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT summary_json FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Agent run 不存在: {run_id}")
            summary = AgentRunSummary.model_validate(json.loads(row["summary_json"]))
            summary = summary.model_copy(update=changes)
            connection.execute(
                "UPDATE agent_runs SET status = ?, summary_json = ?, updated_at = ? WHERE run_id = ?",
                (summary.status, _json(summary), _now(), run_id),
            )
        if summary.status in {"completed", "failed", "cancelled", "skipped"}:
            self.update_task(
                summary.account_id,
                status="idle" if summary.status != "failed" else "failed",
                last_error=summary.error,
            )
        return summary

    def claim_run(self, run_id: str, **changes) -> AgentRunSummary | None:
        """Atomically claim a queued run; return None for another worker's run."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT summary_json FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Agent run 不存在: {run_id}")
            summary = AgentRunSummary.model_validate(json.loads(row["summary_json"]))
            if summary.status != "queued":
                return None
            summary = summary.model_copy(update={"status": "running", **changes})
            cursor = connection.execute(
                """UPDATE agent_runs SET status = 'running', summary_json = ?, updated_at = ?
                   WHERE run_id = ? AND status = 'queued'""",
                (_json(summary), _now(), run_id),
            )
            if cursor.rowcount != 1:
                return None
        return summary

    def recover_stale_runs(self, max_age_minutes: int = 30) -> int:
        """Mark runs left in ``running`` after a process restart as failed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
        stale_accounts: list[tuple[str, str]] = []
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, summary_json FROM agent_runs WHERE status = 'running' AND updated_at < ?",
                (cutoff,),
            ).fetchall()
            for row in rows:
                summary = AgentRunSummary.model_validate(json.loads(row["summary_json"])).model_copy(
                    update={
                        "status": "failed",
                        "completed_at": _now(),
                        "error": "服务重启后检测到未完成的 Agent run",
                    }
                )
                connection.execute(
                    "UPDATE agent_runs SET status = 'failed', summary_json = ?, updated_at = ? WHERE run_id = ?",
                    (_json(summary), _now(), row["run_id"]),
                )
                stale_accounts.append((summary.account_id, summary.error or ""))
        for account_id, error in stale_accounts:
            self.update_task(account_id, status="failed", last_error=error)
        return len(rows)

    def get_run(self, run_id: str) -> AgentRunSummary:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT summary_json FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Agent run 不存在: {run_id}")
        return AgentRunSummary.model_validate(json.loads(row["summary_json"]))

    def list_runs(self, account_id: str, limit: int = 50) -> list[AgentRunSummary]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT summary_json FROM agent_runs
                   WHERE account_id = ? ORDER BY created_at DESC LIMIT ?""",
                (account_id, max(1, min(limit, 500))),
            ).fetchall()
        return [AgentRunSummary.model_validate(json.loads(row["summary_json"])) for row in rows]

    def add_decision(
        self,
        run_id: str,
        account_id: str,
        decision: TradeDecision,
        current_price: float,
        risk_status: str = "pending",
        risk_reason: str | None = None,
        order_id: str | None = None,
    ) -> AgentDecisionAudit:
        audit = AgentDecisionAudit(
            decision_id=f"decision-{uuid4().hex[:16]}",
            run_id=run_id,
            account_id=account_id,
            ticker=decision.ticker,
            decision=decision,
            current_price=current_price,
            risk_status=risk_status,
            risk_reason=risk_reason,
            order_id=order_id,
            created_at=_now(),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO agent_decisions
                   (decision_id, run_id, account_id, ticker, decision_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    audit.decision_id,
                    run_id,
                    account_id,
                    decision.ticker,
                    _json(audit),
                    audit.created_at,
                ),
            )
        return audit

    def list_decisions(self, account_id: str, run_id: str | None = None, limit: int = 100) -> list[AgentDecisionAudit]:
        query = "SELECT decision_json FROM agent_decisions WHERE account_id = ?"
        params: list[object] = [account_id]
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(limit, 1000)))
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [AgentDecisionAudit.model_validate(json.loads(row["decision_json"])) for row in rows]

    def get_decision(self, decision_id: str) -> AgentDecisionAudit:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT decision_json FROM agent_decisions WHERE decision_id = ?", (decision_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Agent decision 不存在: {decision_id}")
        return AgentDecisionAudit.model_validate(json.loads(row["decision_json"]))

    def update_decision(self, decision_id: str, **changes) -> AgentDecisionAudit:
        audit = self.get_decision(decision_id).model_copy(update=changes)
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE agent_decisions SET decision_json = ? WHERE decision_id = ?",
                (_json(audit), decision_id),
            )
        return audit

    def add_event(self, account_id: str, event_type: str, payload: dict, run_id: str | None = None) -> dict:
        event = {
            "event_id": f"event-{uuid4().hex[:16]}",
            "account_id": account_id,
            "run_id": run_id,
            "event_type": event_type,
            "payload": payload,
            "created_at": _now(),
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO automation_events
                   (event_id, account_id, run_id, event_type, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event["event_id"],
                    account_id,
                    run_id,
                    event_type,
                    _json(payload),
                    event["created_at"],
                ),
            )
        return event

    def list_events(self, account_id: str, limit: int = 100) -> list[dict]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT event_id, account_id, run_id, event_type, payload_json, created_at
                   FROM automation_events WHERE account_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (account_id, max(1, min(limit, 1000))),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "account_id": row["account_id"],
                "run_id": row["run_id"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]


automation_store = AutomationStore()
