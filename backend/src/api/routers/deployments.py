"""Backtest-to-paper deployment API."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from application.deployments import deployment_service
from application.strategy_candidates import strategy_candidates
from models.schemas import AssetType

router = APIRouter()


class DeployExperimentRequest(BaseModel):
    account_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    create_account: bool = True
    account_name: str | None = Field(default=None, max_length=120)
    initial_cash: float | None = Field(default=None, gt=0)
    enabled: bool = True
    mode: str = Field(default="confirm", pattern="^(observe|confirm|auto)$")
    schedule_time: str = Field(default="15:10", pattern=r"^\d{2}:\d{2}$")
    weekdays: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    execution_key: str | None = Field(default=None, max_length=255)


class GenerateCandidateRequest(BaseModel):
    objective: str = Field(..., min_length=3, max_length=4000)
    ticker: str = Field(..., min_length=6, max_length=16)
    asset_type: AssetType = AssetType.ETF
    start_date: str
    end_date: str
    initial_capital: float = Field(default=1_000_000, gt=0)


class ReviewCandidateRequest(BaseModel):
    approved: bool
    reviewed_by: str = Field(..., min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)


class DeployCandidateRequest(BaseModel):
    account_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    execution_mode: str = Field(default="paper", pattern="^(paper|live)$")


@router.get("")
async def list_deployments(
    account_id: str | None = None,
    experiment_id: str | None = None,
    include_archived: bool = False,
):
    items = await deployment_service.list(
        account_id=account_id,
        experiment_id=experiment_id,
        include_archived=include_archived,
    )
    return {"deployments": [item.model_dump(mode="json") for item in items]}


@router.get("/candidates")
async def list_strategy_candidates(status: str | None = None):
    return {"candidates": [item.model_dump(mode="json") for item in await strategy_candidates.list(status=status)]}


@router.post("/candidates/generate")
async def generate_strategy_candidate(req: GenerateCandidateRequest):
    try:
        candidate = await strategy_candidates.generate(**req.model_dump())
        return candidate.model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/candidates/{candidate_id}")
async def get_strategy_candidate(candidate_id: str):
    try:
        return (await strategy_candidates.get(candidate_id)).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/candidates/{candidate_id}/review")
async def review_strategy_candidate(candidate_id: str, req: ReviewCandidateRequest):
    try:
        candidate = await strategy_candidates.review(candidate_id, **req.model_dump())
        return candidate.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/candidates/{candidate_id}/deploy")
async def deploy_strategy_candidate(candidate_id: str, req: DeployCandidateRequest):
    try:
        deployment = await strategy_candidates.deploy_to_paper(candidate_id, **req.model_dump())
        return deployment.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{deployment_id}")
async def get_deployment(deployment_id: str):
    try:
        return (await deployment_service.get(deployment_id)).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}")
async def deploy_experiment(experiment_id: str, req: DeployExperimentRequest):
    try:
        deployment = await deployment_service.create_from_experiment(
            experiment_id,
            **req.model_dump(),
        )
        return deployment.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{deployment_id}/activate")
async def activate_deployment(deployment_id: str):
    return await _set_status(deployment_id, "active")


@router.post("/{deployment_id}/pause")
async def pause_deployment(deployment_id: str):
    return await _set_status(deployment_id, "paused")


@router.post("/{deployment_id}/archive")
async def archive_deployment(deployment_id: str):
    return await _set_status(deployment_id, "archived")


async def _set_status(deployment_id: str, status: str):
    try:
        return (await deployment_service.set_status(deployment_id, status)).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
