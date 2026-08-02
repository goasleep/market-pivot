"""Backtest router - run historical strategy backtest."""

import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from loguru import logger

from engine.backtester import run_backtest

router = APIRouter()


class BacktestRequest(BaseModel):
    ticker: str = Field(..., description="A-share stock code")
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    initial_capital: float = Field(default=1_000_000, description="Initial capital in CNY")
    initial_position: int = Field(default=0, description="Initial shares held")
    decision_interval: int = Field(default=1, description="Run agent every N trading days")


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
    )
    return result
