"""In-process background jobs for long-running demo backtests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import uuid4

from application.backtest_service import run_backtest, run_pool_backtest


@dataclass
class BacktestJob:
    job_id: str
    params: dict[str, Any]
    status: str = "queued"
    progress: list[dict[str, str]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task: asyncio.Task | None = field(default=None, repr=False)


class BacktestJobManager:
    """Keep jobs local to the process; durable experiment results use Tortoise."""

    def __init__(self):
        self._jobs: dict[str, BacktestJob] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        tasks = tuple(self._tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def submit(self, params: dict[str, Any]) -> BacktestJob:
        job_id = f"bt-{uuid4().hex[:16]}"
        job = BacktestJob(job_id=job_id, params=dict(params))
        self._jobs[job_id] = job
        self._events[job_id] = []
        task = asyncio.create_task(self._run(job), name=f"backtest-{job_id}")
        self._tasks[job_id] = task
        job.task = task
        return job

    async def get(self, job_id: str) -> BacktestJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"回测任务不存在: {job_id}") from exc

    async def _publish(self, job: BacktestJob, event_type: str, data: dict[str, Any]) -> None:
        events = self._events[job.job_id]
        events.append({"event_id": len(events) + 1, "event": event_type, "data": data})

    async def _run(self, job: BacktestJob) -> None:
        job.status = "running"

        async def callback(stage: str, message: str):
            event = {"stage": stage, "message": message}
            job.progress.append(event)
            await self._publish(job, "progress", event)

        try:
            params = dict(job.params)
            params.pop("mode", None)
            runner = run_pool_backtest if params.get("tickers") or params.get("portfolio_spec") else run_backtest
            job.result = await runner(**params, progress_callback=callback)
            job.status = "completed"
            await self._publish(job, "complete", job.result)
        except asyncio.CancelledError:
            job.status = "cancelled"
            await self._publish(job, "cancelled", {})
            raise
        except Exception as exc:  # pragma: no cover - exercised through API failures
            job.status = "failed"
            job.error = str(exc)
            await self._publish(job, "error", {"error": str(exc)})
        finally:
            self._tasks.pop(job.job_id, None)

    @staticmethod
    def serialise(job: BacktestJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "params": job.params,
            "progress": job.progress,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
        }

    async def stream(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        await self.get(job_id)
        cursor = 0
        while True:
            job = await self.get(job_id)
            for event in self._events[job_id][cursor:]:
                cursor = event["event_id"]
                yield {"event": event["event"], "data": event["data"]}
            if job.status in {"completed", "failed", "cancelled"} and cursor >= len(self._events[job_id]):
                break
            await asyncio.sleep(0.1)


backtest_jobs = BacktestJobManager()
