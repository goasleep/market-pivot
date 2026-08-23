"""Persistence for deterministic strategy runtime memory."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from data.db_models import StrategyRuntimeStateRecord
from data.tortoise_db import init_database
from models.schemas import StrategyRuntimeState


class StrategyRuntimeStateStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = db_path

    async def _ready(self) -> None:
        await init_database(db_path=self.db_path)

    async def get(self, deployment_id: str, ticker: str) -> StrategyRuntimeState:
        await self._ready()
        row = await StrategyRuntimeStateRecord.get_or_none(
            deployment_id=deployment_id,
            ticker=ticker,
        )
        return StrategyRuntimeState.model_validate_json(row.state_json) if row else StrategyRuntimeState()

    async def save(
        self,
        deployment_id: str,
        ticker: str,
        state: StrategyRuntimeState,
    ) -> StrategyRuntimeState:
        await self._ready()
        await StrategyRuntimeStateRecord.update_or_create(
            id=f"{deployment_id}:{ticker}",
            defaults={
                "deployment_id": deployment_id,
                "ticker": ticker,
                "state_json": state.model_dump_json(),
                "last_evaluated_date": state.last_evaluated_date,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return state


strategy_runtime_states = StrategyRuntimeStateStore()
