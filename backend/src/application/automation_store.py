"""Durable state for unattended Agent simulation tasks.

The simulation account database owns cash, positions, orders, and snapshots.
This store owns the automation control plane: task configuration, run
idempotency, decision audit records, and a small event log.  Keeping these
concerns separate makes account reset safe while preserving the operational
history needed to explain what an Agent did.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, update

from config import settings
from data.orm import (
    AgentDecisionRecord,
    AgentRunRecord,
    AutomationEventRecord,
    AutomationTaskRecord,
    build_database,
)
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
        self.database = build_database(
            database_url=settings.database_url if db_path is None else None,
            db_path=db_path or settings.database_file_path,
        )

    def _session(self):
        return self.database.session()

    def get_task(self, account_id: str) -> dict:
        with self._session() as session:
            row = session.get(AutomationTaskRecord, account_id)
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
    def _task_from_row(row: AutomationTaskRecord) -> dict:
        return {
            "account_id": row.account_id,
            "config": AutomationTaskConfig.model_validate(json.loads(row.config_json)),
            "status": row.status,
            "last_run_id": row.last_run_id,
            "last_run_date": row.last_run_date,
            "last_error": row.last_error,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    def _save_task(self, task: dict, session=None) -> dict:
        timestamp = _now()
        created_at = task.get("created_at") or timestamp
        owns_session = session is None
        session = session or self._session()
        row = session.get(AutomationTaskRecord, task["account_id"])
        if row is None:
            row = AutomationTaskRecord(
                account_id=task["account_id"],
                config_json=_json(task["config"]),
                status=task.get("status", "idle"),
                last_run_id=task.get("last_run_id"),
                last_run_date=task.get("last_run_date"),
                last_error=task.get("last_error"),
                created_at=created_at,
                updated_at=timestamp,
            )
            session.add(row)
        else:
            row.config_json = _json(task["config"])
            row.status = task.get("status", "idle")
            row.last_run_id = task.get("last_run_id")
            row.last_run_date = task.get("last_run_date")
            row.last_error = task.get("last_error")
            row.updated_at = timestamp
        if owns_session:
            session.commit()
            session.close()
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
        with self._session() as session:
            existing = session.scalar(
                select(AgentRunRecord).where(AgentRunRecord.idempotency_key == idempotency_key)
            )
            if existing:
                return AgentRunSummary.model_validate(json.loads(existing.summary_json))
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
            session.add(
                AgentRunRecord(
                    run_id=summary.run_id,
                    account_id=account_id,
                    run_date=run_date,
                    trigger=trigger,
                    status=summary.status,
                    summary_json=_json(summary),
                    idempotency_key=idempotency_key,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
            session.commit()
        self.update_task(account_id, last_run_id=summary.run_id, last_run_date=run_date, last_error=None)
        return summary

    def update_run(self, run_id: str, **changes) -> AgentRunSummary:
        with self._session() as session:
            row = session.get(AgentRunRecord, run_id)
            if row is None:
                raise KeyError(f"Agent run 不存在: {run_id}")
            summary = AgentRunSummary.model_validate(json.loads(row.summary_json))
            summary = summary.model_copy(update=changes)
            row.status = summary.status
            row.summary_json = _json(summary)
            row.updated_at = _now()
            session.commit()
        if summary.status in {"completed", "failed", "cancelled", "skipped"}:
            self.update_task(
                summary.account_id,
                status="idle" if summary.status != "failed" else "failed",
                last_error=summary.error,
            )
        return summary

    def claim_run(self, run_id: str, **changes) -> AgentRunSummary | None:
        """Atomically claim a queued run; return None for another worker's run."""
        with self._session() as session:
            row = session.get(AgentRunRecord, run_id)
            if row is None:
                raise KeyError(f"Agent run 不存在: {run_id}")
            summary = AgentRunSummary.model_validate(json.loads(row.summary_json))
            if summary.status != "queued":
                return None
            summary = summary.model_copy(update={"status": "running", **changes})
            result = session.execute(
                update(AgentRunRecord)
                .where(AgentRunRecord.run_id == run_id, AgentRunRecord.status == "queued")
                .values(status="running", summary_json=_json(summary), updated_at=_now())
            )
            session.commit()
            if result.rowcount != 1:
                return None
        return summary

    def recover_stale_runs(self, max_age_minutes: int = 30) -> int:
        """Mark runs left in ``running`` after a process restart as failed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
        stale_accounts: list[tuple[str, str]] = []
        with self._session() as session:
            rows = session.scalars(
                select(AgentRunRecord).where(
                    AgentRunRecord.status == "running", AgentRunRecord.updated_at < cutoff
                )
            ).all()
            for row in rows:
                summary = AgentRunSummary.model_validate(json.loads(row.summary_json)).model_copy(
                    update={
                        "status": "failed",
                        "completed_at": _now(),
                        "error": "服务重启后检测到未完成的 Agent run",
                    }
                )
                row.status = "failed"
                row.summary_json = _json(summary)
                row.updated_at = _now()
                stale_accounts.append((summary.account_id, summary.error or ""))
            session.commit()
        for account_id, error in stale_accounts:
            self.update_task(account_id, status="failed", last_error=error)
        return len(rows)

    def get_run(self, run_id: str) -> AgentRunSummary:
        with self._session() as session:
            row = session.get(AgentRunRecord, run_id)
        if row is None:
            raise KeyError(f"Agent run 不存在: {run_id}")
        return AgentRunSummary.model_validate(json.loads(row.summary_json))

    def list_runs(self, account_id: str, limit: int = 50) -> list[AgentRunSummary]:
        with self._session() as session:
            rows = session.scalars(
                select(AgentRunRecord)
                .where(AgentRunRecord.account_id == account_id)
                .order_by(AgentRunRecord.created_at.desc())
                .limit(max(1, min(limit, 500)))
            ).all()
        return [AgentRunSummary.model_validate(json.loads(row.summary_json)) for row in rows]

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
        with self._session() as session:
            session.add(
                AgentDecisionRecord(
                    decision_id=audit.decision_id,
                    run_id=run_id,
                    account_id=account_id,
                    ticker=decision.ticker,
                    decision_json=_json(audit),
                    created_at=audit.created_at,
                )
            )
            session.commit()
        return audit

    def list_decisions(self, account_id: str, run_id: str | None = None, limit: int = 100) -> list[AgentDecisionAudit]:
        statement = select(AgentDecisionRecord).where(AgentDecisionRecord.account_id == account_id)
        if run_id:
            statement = statement.where(AgentDecisionRecord.run_id == run_id)
        statement = statement.order_by(AgentDecisionRecord.created_at.desc()).limit(max(1, min(limit, 1000)))
        with self._session() as session:
            rows = session.scalars(statement).all()
        return [AgentDecisionAudit.model_validate(json.loads(row.decision_json)) for row in rows]

    def get_decision(self, decision_id: str) -> AgentDecisionAudit:
        with self._session() as session:
            row = session.get(AgentDecisionRecord, decision_id)
        if row is None:
            raise KeyError(f"Agent decision 不存在: {decision_id}")
        return AgentDecisionAudit.model_validate(json.loads(row.decision_json))

    def update_decision(self, decision_id: str, **changes) -> AgentDecisionAudit:
        audit = self.get_decision(decision_id).model_copy(update=changes)
        with self._session() as session:
            row = session.get(AgentDecisionRecord, decision_id)
            if row is None:
                raise KeyError(f"Agent decision 不存在: {decision_id}")
            row.decision_json = _json(audit)
            session.commit()
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
        with self._session() as session:
            session.add(
                AutomationEventRecord(
                    event_id=event["event_id"],
                    account_id=account_id,
                    run_id=run_id,
                    event_type=event_type,
                    payload_json=_json(payload),
                    created_at=event["created_at"],
                )
            )
            session.commit()
        return event

    def list_events(self, account_id: str, limit: int = 100) -> list[dict]:
        with self._session() as session:
            rows = session.scalars(
                select(AutomationEventRecord)
                .where(AutomationEventRecord.account_id == account_id)
                .order_by(AutomationEventRecord.created_at.desc())
                .limit(max(1, min(limit, 1000)))
            ).all()
        return [
            {
                "event_id": row.event_id,
                "account_id": row.account_id,
                "run_id": row.run_id,
                "event_type": row.event_type,
                "payload": json.loads(row.payload_json),
                "created_at": row.created_at,
            }
            for row in rows
        ]


automation_store = AutomationStore()
