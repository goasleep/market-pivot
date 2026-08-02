"""Backtest router - run historical strategy backtest."""

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from application.backtest_jobs import backtest_jobs
from engine.backtester import run_backtest

router = APIRouter()


class BacktestRequest(BaseModel):
    ticker: str = Field(..., description="A-share stock code")
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    initial_capital: float = Field(default=1_000_000, description="Initial capital in CNY")
    initial_position: int = Field(default=0, description="Initial shares held")
    decision_interval: int = Field(default=1, description="Run agent every N trading days")
    fill_time: str = Field(default="next_open", description="next_open or same_close")
    strategy: str | None = Field(default=None, description="Optional strategy name")


@router.post("/run")
async def run_backtest_api(req: BacktestRequest):
    """Run backtest with historical data."""
    logger.info(f"Backtest request: {req.ticker} {req.start_date} -> {req.end_date}")

    result = await run_backtest(
        ticker=req.ticker,
        start_date=req.start_date,
        end_date=req.end_date,
        initial_capital=req.initial_capital,
        initial_position=req.initial_position,
        decision_interval=req.decision_interval,
        fill_time=req.fill_time,
        strategy_name=req.strategy,
    )
    return result


@router.get("/stream")
async def stream_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 1_000_000,
    initial_position: int = 0,
    decision_interval: int = 1,
    fill_time: str = "next_open",
    strategy: str | None = None,
):
    """Run a backtest with real progress callbacks over SSE."""
    import asyncio
    import json

    async def event_generator():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        async def progress(stage: str, message: str):
            await queue.put({"event": "progress", "stage": stage, "message": message})

        task = asyncio.create_task(
            run_backtest(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                initial_position=initial_position,
                decision_interval=decision_interval,
                fill_time=fill_time,
                strategy_name=strategy,
                progress_callback=progress,
            )
        )
        try:
            while True:
                if task.done() and queue.empty():
                    result = task.result()
                    yield {
                        "event": "complete",
                        "data": json.dumps(result, ensure_ascii=False),
                    }
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
    """Queue a long-running backtest and return immediately."""
    params = req.model_dump(exclude_none=True)
    params["strategy_name"] = params.pop("strategy", None)
    job = await backtest_jobs.submit(params)
    return {"job_id": job.job_id, "status": job.status}


@router.get("/jobs/{job_id}")
async def get_backtest_job(job_id: str):
    job = await backtest_jobs.get(job_id)
    return backtest_jobs.serialise(job)


@router.get("/jobs/{job_id}/stream")
async def stream_backtest_job(job_id: str):
    import json

    async def event_generator():
        async for event in backtest_jobs.stream(job_id):
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"], ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
