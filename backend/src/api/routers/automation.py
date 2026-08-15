"""Control and audit API for unattended Agent simulation runs."""

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from application.automation import automation_service
from application.automation_store import automation_store
from models.schemas import AutomationTaskConfig

router = APIRouter()


class RunRequest(BaseModel):
    run_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class SettlementRequest(BaseModel):
    settlement_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    prices: dict[str, float] = Field(default_factory=dict)
    open_prices: dict[str, float] = Field(default_factory=dict)


@router.get("/accounts/{account_id}")
async def get_automation(account_id: str):
    try:
        return await asyncio.to_thread(automation_service.get_task_payload, account_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/accounts/{account_id}")
async def update_automation(account_id: str, config: AutomationTaskConfig):
    try:
        return await automation_service.update_task(account_id, config)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/accounts/{account_id}/run")
async def run_automation(account_id: str, req: RunRequest = RunRequest()):
    try:
        summary = await automation_service.run_account(account_id, trigger="manual", run_date=req.run_date)
        return summary.model_dump(mode="json")
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/accounts/{account_id}/settle")
async def settle_automation(account_id: str, req: SettlementRequest = SettlementRequest()):
    try:
        return await automation_service.settle_account(
            account_id,
            settlement_date=req.settlement_date,
            prices=req.prices,
            open_prices=req.open_prices,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/accounts/{account_id}/runs")
async def list_runs(account_id: str, limit: int = 50):
    try:
        runs = await asyncio.to_thread(automation_store.list_runs, account_id, limit)
        return {"runs": [run.model_dump(mode="json") for run in runs]}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/accounts/{account_id}/runs/{run_id}")
async def get_run(account_id: str, run_id: str):
    try:
        run = await asyncio.to_thread(automation_store.get_run, run_id)
        if run.account_id != account_id:
            raise HTTPException(status_code=404, detail="Agent run 不属于该模拟账户")
        decisions = await asyncio.to_thread(automation_store.list_decisions, account_id, run_id, 1000)
        return {
            "run": run.model_dump(mode="json"),
            "decisions": [decision.model_dump(mode="json") for decision in decisions],
        }
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/accounts/{account_id}/decisions")
async def list_decisions(account_id: str, run_id: str | None = None, limit: int = 100):
    decisions = await asyncio.to_thread(automation_store.list_decisions, account_id, run_id, limit)
    return {"decisions": [decision.model_dump(mode="json") for decision in decisions]}


@router.post("/accounts/{account_id}/decisions/{decision_id}/confirm")
async def confirm_decision(account_id: str, decision_id: str, price: float | None = None):
    try:
        audit = await automation_service.confirm_decision(account_id, decision_id, price)
        return audit.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/accounts/{account_id}/events")
async def list_events(account_id: str, limit: int = 100):
    return {"events": await asyncio.to_thread(automation_store.list_events, account_id, limit)}
