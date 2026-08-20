"""Tortoise-backed automation control-plane repository."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from config import settings
from data.db_models import (
    AgentDecisionRecord,
    AgentRunRecord,
    AutomationEventRecord,
    AutomationTaskRecord,
)
from data.tortoise_db import init_database
from models.schemas import AgentDecisionAudit, AgentRunSummary, AutomationTaskConfig, TradeDecision


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False)


class AutomationStore:
    """Async repository for automation configuration, runs, decisions, and events."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or settings.database_file_path)
        self.db_url = settings.database_url if db_path is None else None

    async def _ready(self) -> None:
        await init_database(db_path=None if self.db_url else self.db_path, db_url=self.db_url)

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

    async def get_task(self, account_id: str) -> dict:
        await self._ready()
        row = await AutomationTaskRecord.get_or_none(account_id=account_id)
        if row is None:
            timestamp = _now()
            row = await AutomationTaskRecord.create(
                account_id=account_id,
                config_json=_json(AutomationTaskConfig()),
                status="idle",
                created_at=timestamp,
                updated_at=timestamp,
            )
        return self._task_from_row(row)

    async def update_task(self, account_id: str, **changes) -> dict:
        task = await self.get_task(account_id)
        if "config" in changes:
            task["config"] = AutomationTaskConfig.model_validate(changes.pop("config"))
        else:
            config_changes = {
                key: value for key, value in changes.items() if key in AutomationTaskConfig.model_fields
            }
            if config_changes:
                task["config"] = task["config"].model_copy(update=config_changes)
        for key, value in changes.items():
            if key not in AutomationTaskConfig.model_fields:
                task[key] = value

        timestamp = _now()
        await AutomationTaskRecord.filter(account_id=account_id).update(
            config_json=_json(task["config"]),
            status=task.get("status", "idle"),
            last_run_id=task.get("last_run_id"),
            last_run_date=task.get("last_run_date"),
            last_error=task.get("last_error"),
            updated_at=timestamp,
        )
        task["updated_at"] = timestamp
        return task

    async def create_run(
        self,
        account_id: str,
        run_date: str,
        trigger: str,
        config: AutomationTaskConfig,
        idempotency_key: str,
    ) -> AgentRunSummary:
        await self._ready()
        existing = await AgentRunRecord.get_or_none(idempotency_key=idempotency_key)
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
        await AgentRunRecord.create(
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
        await self.update_task(account_id, last_run_id=summary.run_id, last_run_date=run_date, last_error=None)
        return summary

    async def update_run(self, run_id: str, **changes) -> AgentRunSummary:
        await self._ready()
        row = await AgentRunRecord.get_or_none(run_id=run_id)
        if row is None:
            raise KeyError(f"Agent run 不存在: {run_id}")
        summary = AgentRunSummary.model_validate(json.loads(row.summary_json)).model_copy(update=changes)
        timestamp = _now()
        await AgentRunRecord.filter(run_id=run_id).update(
            status=summary.status,
            summary_json=_json(summary),
            updated_at=timestamp,
        )
        if summary.status in {"completed", "failed", "cancelled", "skipped"}:
            await self.update_task(
                summary.account_id,
                status="idle" if summary.status != "failed" else "failed",
                last_error=summary.error,
            )
        return summary

    async def claim_run(self, run_id: str, **changes) -> AgentRunSummary | None:
        """Atomically claim a queued run in one conditional update."""
        await self._ready()
        row = await AgentRunRecord.get_or_none(run_id=run_id)
        if row is None:
            raise KeyError(f"Agent run 不存在: {run_id}")
        summary = AgentRunSummary.model_validate(json.loads(row.summary_json))
        if summary.status != "queued":
            return None
        summary = summary.model_copy(update={"status": "running", **changes})
        updated = await AgentRunRecord.filter(run_id=run_id, status="queued").update(
            status="running",
            summary_json=_json(summary),
            updated_at=_now(),
        )
        return summary if updated == 1 else None

    async def recover_stale_runs(self, max_age_minutes: int = 30) -> int:
        await self._ready()
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
        rows = await AgentRunRecord.filter(status="running", updated_at__lt=cutoff).all()
        for row in rows:
            summary = AgentRunSummary.model_validate(json.loads(row.summary_json)).model_copy(
                update={
                    "status": "failed",
                    "completed_at": _now(),
                    "error": "服务重启后检测到未完成的 Agent run",
                }
            )
            await AgentRunRecord.filter(run_id=row.run_id).update(
                status="failed",
                summary_json=_json(summary),
                updated_at=_now(),
            )
            await self.update_task(summary.account_id, status="failed", last_error=summary.error)
        return len(rows)

    async def get_run(self, run_id: str) -> AgentRunSummary:
        await self._ready()
        row = await AgentRunRecord.get_or_none(run_id=run_id)
        if row is None:
            raise KeyError(f"Agent run 不存在: {run_id}")
        return AgentRunSummary.model_validate(json.loads(row.summary_json))

    async def list_runs(self, account_id: str, limit: int = 50) -> list[AgentRunSummary]:
        await self._ready()
        rows = await AgentRunRecord.filter(account_id=account_id).order_by("-created_at").limit(max(1, min(limit, 500)))
        return [AgentRunSummary.model_validate(json.loads(row.summary_json)) for row in rows]

    async def add_decision(
        self,
        run_id: str,
        account_id: str,
        decision: TradeDecision,
        current_price: float,
        risk_status: str = "pending",
        risk_reason: str | None = None,
        order_id: str | None = None,
    ) -> AgentDecisionAudit:
        await self._ready()
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
        await AgentDecisionRecord.create(
            decision_id=audit.decision_id,
            run_id=run_id,
            account_id=account_id,
            ticker=decision.ticker,
            decision_json=_json(audit),
            created_at=audit.created_at,
        )
        return audit

    async def list_decisions(
        self, account_id: str, run_id: str | None = None, limit: int = 100
    ) -> list[AgentDecisionAudit]:
        await self._ready()
        query = AgentDecisionRecord.filter(account_id=account_id)
        if run_id:
            query = query.filter(run_id=run_id)
        rows = await query.order_by("-created_at").limit(max(1, min(limit, 1000)))
        return [AgentDecisionAudit.model_validate(json.loads(row.decision_json)) for row in rows]

    async def get_decision(self, decision_id: str) -> AgentDecisionAudit:
        await self._ready()
        row = await AgentDecisionRecord.get_or_none(decision_id=decision_id)
        if row is None:
            raise KeyError(f"Agent decision 不存在: {decision_id}")
        return AgentDecisionAudit.model_validate(json.loads(row.decision_json))

    async def update_decision(self, decision_id: str, **changes) -> AgentDecisionAudit:
        audit = (await self.get_decision(decision_id)).model_copy(update=changes)
        await AgentDecisionRecord.filter(decision_id=decision_id).update(decision_json=_json(audit))
        return audit

    async def add_event(self, account_id: str, event_type: str, payload: dict, run_id: str | None = None) -> dict:
        await self._ready()
        event = {
            "event_id": f"event-{uuid4().hex[:16]}",
            "account_id": account_id,
            "run_id": run_id,
            "event_type": event_type,
            "payload": payload,
            "created_at": _now(),
        }
        await AutomationEventRecord.create(
            event_id=event["event_id"],
            account_id=account_id,
            run_id=run_id,
            event_type=event_type,
            payload_json=_json(payload),
            created_at=event["created_at"],
        )
        return event

    async def list_events(self, account_id: str, limit: int = 100) -> list[dict]:
        await self._ready()
        rows = await AutomationEventRecord.filter(account_id=account_id).order_by("-created_at").limit(
            max(1, min(limit, 1000))
        )
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
