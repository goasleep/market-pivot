"""SQLite-backed background jobs for long-running backtests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import uuid4

from application.backtest_service import run_backtest, run_pool_backtest
from data.runtime_store import runtime_store


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
    def __init__(self, store=None):
        self.store = store or runtime_store
        self._tasks: dict[str, asyncio.Task] = {}
        self._objects: dict[str, BacktestJob] = {}
        self._lock = asyncio.Lock()
        self._worker: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._worker and not self._worker.done():
            return
        self._stopping = asyncio.Event()
        self._worker = asyncio.create_task(self._poll(), name="backtest-job-worker")

    async def stop(self) -> None:
        self._stopping.set()
        if self._worker:
            await self._worker
        self._worker = None
        for task in tuple(self._tasks.values()):
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    async def submit(self, params: dict[str, Any]) -> BacktestJob:
        job_id = f"bt-{uuid4().hex[:16]}"
        await self.store.create_job(job_id, "backtest", params)
        job = self._from_record(await self.store.get_job(job_id))
        self._objects[job_id] = job
        await self._start_job(job_id)
        job.task = self._tasks.get(job_id)
        return job

    async def get(self, job_id: str) -> BacktestJob:
        return self._from_record(await self.store.get_job(job_id))

    async def _start_job(self, job_id: str) -> None:
        async with self._lock:
            existing = self._tasks.get(job_id)
            if existing and not existing.done():
                return
            claimed = await self.store.claim_job(job_id)
            if claimed is None:
                return
            task = asyncio.create_task(self._run(job_id), name=f"backtest-{job_id}")
            self._tasks[job_id] = task

    async def _poll(self) -> None:
        while not self._stopping.is_set():
            try:
                for record in await self.store.list_jobs("backtest", limit=100):
                    if record["status"] in {"queued", "running"}:
                        await self._start_job(record["job_id"])
            except Exception:
                # A temporary SQLite lock must not terminate the worker loop.
                pass
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=2)
            except asyncio.TimeoutError:
                continue

    async def _run(self, job_id: str) -> None:
        record = await self.store.get_job(job_id)
        params = dict(record["params"])
        progress = list(record["progress"])

        heartbeat = asyncio.create_task(self._heartbeat(job_id), name=f"backtest-heartbeat-{job_id}")

        async def callback(stage: str, message: str):
            event = {"stage": stage, "message": message}
            progress.append(event)
            local_job = self._objects.get(job_id)
            if local_job:
                local_job.status = "running"
                local_job.progress.append(event)
            await self.store.update_job(job_id, progress_json=json.dumps(progress, ensure_ascii=False))
            await self.store.append_event("backtest", job_id, "progress", event)

        try:
            params.pop("mode", None)
            runner = run_pool_backtest if params.get("tickers") or params.get("portfolio_spec") else run_backtest
            result = await runner(**params, progress_callback=callback)
            await self.store.update_job(
                job_id,
                status="completed",
                result_json=json.dumps(result, ensure_ascii=False, default=str),
                owner_id=None,
                lease_expires_at=None,
            )
            local_job = self._objects.get(job_id)
            if local_job:
                local_job.status = "completed"
                local_job.result = result
            await self.store.append_event("backtest", job_id, "complete", result)
        except asyncio.CancelledError:
            # Process shutdown must release the job for another node instead
            # of turning a still-valid backtest into a terminal cancellation.
            await self.store.update_job(job_id, status="queued", owner_id=None, lease_expires_at=None)
            local_job = self._objects.get(job_id)
            if local_job:
                local_job.status = "queued"
            await self.store.append_event("backtest", job_id, "requeued", {})
            raise
        except Exception as exc:  # pragma: no cover - exercised through API failures
            await self.store.update_job(
                job_id,
                status="failed",
                error=str(exc),
                owner_id=None,
                lease_expires_at=None,
            )
            local_job = self._objects.get(job_id)
            if local_job:
                local_job.status = "failed"
                local_job.error = str(exc)
            await self.store.append_event("backtest", job_id, "error", {"error": str(exc)})
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            self._tasks.pop(job_id, None)

    async def _heartbeat(self, job_id: str) -> None:
        while True:
            await asyncio.sleep(30)
            if not await self.store.heartbeat_job(job_id):
                return

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

    @staticmethod
    def _from_record(record: dict[str, Any]) -> BacktestJob:
        return BacktestJob(
            job_id=record["job_id"],
            params=record["params"],
            status=record["status"],
            progress=record["progress"],
            result=record["result"],
            error=record["error"],
            created_at=record["created_at"],
        )

    async def stream(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        await self.store.get_job(job_id)
        cursor = 0
        while True:
            events = await self.store.list_events("backtest", job_id, after_id=cursor)
            for event in events:
                cursor = int(event["event_id"])
                yield {"event": event["type"], "data": event["data"]}
            job = await self.store.get_job(job_id)
            if job["status"] in {"completed", "failed", "cancelled"} and not events:
                break
            await asyncio.sleep(0.25)


backtest_jobs = BacktestJobManager()
