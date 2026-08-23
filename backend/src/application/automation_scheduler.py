"""Polling scheduler for persistent automation tasks.

The scheduler owns process lifecycle and calendar polling only.  Trading and
research behavior stays behind the injected automation service.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time
from typing import Protocol
from zoneinfo import ZoneInfo

from loguru import logger

from application.automation_store import automation_store
from config import settings
from data.trading_calendar import is_trading_day
from engine.simulation_account import simulation_accounts
from models.schemas import AutomationTaskConfig

SHANGHAI = ZoneInfo("Asia/Shanghai")


class AutomationRunner(Protocol):
    async def settle_account(self, account_id: str, settlement_date: str | None = None) -> dict: ...

    async def run_account(
        self,
        account_id: str,
        trigger: str = "manual",
        run_date: str | None = None,
    ): ...


class AutomationScheduler:
    """Poll enabled accounts and delegate due runs to an automation service."""

    def __init__(self, service: AutomationRunner):
        self.service = service
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        await automation_store.recover_stale_runs()
        self._stopping = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="agent-automation-scheduler")

    async def stop(self) -> None:
        self._stopping.set()
        task, self._task = self._task, None
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.tick()
            except Exception:
                logger.exception("Automation scheduler tick failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=30)
            except asyncio.TimeoutError:
                continue

    async def tick(self, now: datetime | None = None) -> None:
        current = now or datetime.now(SHANGHAI)
        eligible: list[str] = []
        for account in await simulation_accounts.list_accounts():
            if account.status != "active":
                continue
            task = await automation_store.get_task(account.account_id)
            config: AutomationTaskConfig = task["config"]
            if not config.enabled or current.weekday() not in config.weekdays:
                continue
            if not await asyncio.to_thread(is_trading_day, current.date()):
                continue
            try:
                scheduled = time.fromisoformat(config.schedule_time)
            except ValueError:
                logger.warning("Invalid automation schedule for {}: {}", account.account_id, config.schedule_time)
                continue
            scheduled_at = datetime.combine(current.date(), scheduled, tzinfo=current.tzinfo)
            if current < scheduled_at or task["last_run_date"] == current.date().isoformat():
                continue
            eligible.append(account.account_id)

        semaphore = asyncio.Semaphore(settings.automation_max_concurrency)

        async def run_one(account_id: str) -> None:
            async with semaphore:
                try:
                    await self.service.settle_account(account_id, current.date().isoformat())
                    await self.service.run_account(
                        account_id,
                        trigger="schedule",
                        run_date=current.date().isoformat(),
                    )
                except Exception:
                    logger.exception("Automation scheduler account cycle failed for {}", account_id)

        await asyncio.gather(*(run_one(account_id) for account_id in eligible))
