"""Backtest router - single-symbol, pool, and portfolio research runs."""

import asyncio
import json

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from application.backtest_jobs import backtest_jobs
from application.backtest_service import run_backtest, run_pool_backtest
from models.schemas import AssetType, PortfolioSpec
from strategies.compiler import available_indicators
from strategies.skill_manager import get_strategy_spec

router = APIRouter()


@router.get("/indicators")
async def list_backtest_indicators():
    return {"indicators": available_indicators()}


class BacktestRequest(BaseModel):
    mode: str = Field(default="auto", pattern="^(auto|single|pool|portfolio)$")
    ticker: str | None = None
    tickers: list[str] = Field(default_factory=list)
    asset_type: AssetType = AssetType.STOCK
    start_date: str
    end_date: str
    initial_capital: float = Field(default=1_000_000, gt=0)
    initial_position: int = Field(default=0, ge=0)
    decision_interval: int = Field(default=1, ge=1)
    fill_time: str = Field(default="next_open", pattern="^(next_open|same_close)$")
    strategy: str = "bull_trend"
    strategy_spec: dict | None = None
    portfolio_spec: PortfolioSpec | None = None


def _symbols(req: BacktestRequest) -> list[str]:
    return req.tickers or ([req.ticker] if req.ticker else [])


def _validate_mode(mode: str, symbols: list[str]) -> None:
    if not symbols:
        raise ValueError("ticker 或 tickers 至少提供一个标的")
    if mode == "single" and len(symbols) != 1:
        raise ValueError("single 模式只能提供一个标的")
    if mode in {"pool", "portfolio"} and len(symbols) < 2:
        raise ValueError(f"{mode} 模式至少需要两个标的")


def _strategy_payload(
    strategy_name: str,
    strategy_spec: dict | None,
    asset_type: AssetType,
) -> dict:
    if strategy_spec is not None:
        return strategy_spec
    spec = get_strategy_spec(strategy_name)
    if spec is None:
        raise ValueError(f"策略不存在或不可执行: {strategy_name}")
    if asset_type not in spec.asset_types:
        raise ValueError(f"策略 {strategy_name} 不支持资产类型 {asset_type.value}")
    return spec.model_dump(mode="json")


@router.post("/run")
async def run_backtest_api(req: BacktestRequest):
    symbols = _symbols(req)
    _validate_mode(req.mode, symbols)
    strategy_spec = _strategy_payload(req.strategy, req.strategy_spec, req.asset_type)
    logger.info(f"Backtest {req.mode}: {symbols} {req.start_date} -> {req.end_date}")
    portfolio_spec = (
        req.portfolio_spec.model_dump(mode="json")
        if req.portfolio_spec
        else PortfolioSpec().model_dump(mode="json")
        if req.mode == "portfolio"
        else None
    )
    if len(symbols) > 1 or req.mode in {"pool", "portfolio"}:
        return await run_pool_backtest(
            tickers=symbols,
            start_date=req.start_date,
            end_date=req.end_date,
            initial_capital=req.initial_capital,
            decision_interval=req.decision_interval,
            fill_time=req.fill_time,
            strategy_name=req.strategy,
            asset_type=req.asset_type,
            strategy_spec=strategy_spec,
            portfolio_spec=portfolio_spec if req.mode == "portfolio" else None,
        )
    return await run_backtest(
        ticker=symbols[0], start_date=req.start_date, end_date=req.end_date,
        initial_capital=req.initial_capital, initial_position=req.initial_position,
        decision_interval=req.decision_interval, fill_time=req.fill_time,
        strategy_name=req.strategy, asset_type=req.asset_type, strategy_spec=strategy_spec,
    )


@router.get("/stream")
async def stream_backtest(
    ticker: str, start_date: str, end_date: str, initial_capital: float = 1_000_000,
    initial_position: int = 0, decision_interval: int = 1, fill_time: str = "next_open",
    strategy: str = "bull_trend", asset_type: AssetType = AssetType.STOCK,
    strategy_spec: dict | None = None,
):
    async def event_generator():
        queue: asyncio.Queue[dict] = asyncio.Queue()
        executable_spec = _strategy_payload(strategy, strategy_spec, asset_type)

        async def progress(stage: str, message: str):
            await queue.put({"event": "progress", "stage": stage, "message": message})

        task = asyncio.create_task(run_backtest(
            ticker=ticker, start_date=start_date, end_date=end_date,
            initial_capital=initial_capital, initial_position=initial_position,
            decision_interval=decision_interval, fill_time=fill_time, strategy_name=strategy,
            asset_type=asset_type, progress_callback=progress, strategy_spec=executable_spec,
        ))
        try:
            while True:
                if task.done() and queue.empty():
                    yield {"event": "complete", "data": json.dumps(task.result(), ensure_ascii=False)}
                    break
                event = await queue.get()
                yield {"event": event.pop("event"), "data": json.dumps(event, ensure_ascii=False)}
        except Exception as exc:
            if not task.done():
                task.cancel()
            yield {"event": "error", "data": json.dumps({"error": str(exc)}, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


@router.post("/jobs")
async def create_backtest_job(req: BacktestRequest):
    symbols = _symbols(req)
    _validate_mode(req.mode, symbols)
    params = req.model_dump(exclude_none=True)
    params.pop("mode", None)
    params["strategy_name"] = params.pop("strategy", None)
    params["strategy_spec"] = _strategy_payload(
        params["strategy_name"],
        params.get("strategy_spec"),
        req.asset_type,
    )
    if len(symbols) > 1 or req.mode in {"pool", "portfolio"}:
        params.pop("ticker", None)
        if req.mode == "portfolio" and not params.get("portfolio_spec"):
            params["portfolio_spec"] = PortfolioSpec().model_dump(mode="json")
    else:
        params["ticker"] = symbols[0]
        params.pop("tickers", None)
        params.pop("portfolio_spec", None)
    job = await backtest_jobs.submit(params)
    return {"job_id": job.job_id, "status": job.status}


@router.get("/jobs/{job_id}")
async def get_backtest_job(job_id: str):
    job = await backtest_jobs.get(job_id)
    return backtest_jobs.serialise(job)


@router.get("/jobs/{job_id}/stream")
async def stream_backtest_job(job_id: str):
    async def event_generator():
        async for event in backtest_jobs.stream(job_id):
            yield {"event": event["event"], "data": json.dumps(event["data"], ensure_ascii=False)}

    return EventSourceResponse(event_generator())
