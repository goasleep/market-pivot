"""In-process background jobs for long-running backtests."""

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
    queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue, repr=False)


class BacktestJobManager:
    def __init__(self):
        self._jobs: dict[str, BacktestJob] = {}
        self._lock = asyncio.Lock()

    async def submit(self, params: dict[str, Any]) -> BacktestJob:
        job = BacktestJob(job_id=f"bt-{uuid4().hex[:16]}", params=params)
        async with self._lock:
            self._jobs[job.job_id] = job
        job.task = asyncio.create_task(self._run(job))
        return job

    async def get(self, job_id: str) -> BacktestJob:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"回测任务不存在: {job_id}")
        return job

    async def _run(self, job: BacktestJob) -> None:
        job.status = "running"

        async def callback(stage: str, message: str):
            event = {"stage": stage, "message": message}
            job.progress.append(event)
            await job.queue.put({"event": "progress", "data": event})

        try:
            params = dict(job.params)
            params.pop("mode", None)
            runner = run_pool_backtest if params.get("tickers") or params.get("portfolio_spec") else run_backtest
            job.result = await runner(**params, progress_callback=callback)
            job.status = "completed"
            await job.queue.put({"event": "complete", "data": job.result})
        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        except Exception as exc:  # pragma: no cover - exercised through API failures
            job.status = "failed"
            job.error = str(exc)
            await job.queue.put({"event": "error", "data": {"error": str(exc)}})

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
        job = await self.get(job_id)
        while True:
            if job.task and job.task.done() and job.queue.empty():
                break
            yield await job.queue.get()


backtest_jobs = BacktestJobManager()
