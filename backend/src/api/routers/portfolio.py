"""Portfolio router - manage virtual portfolio."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class Position(BaseModel):
    ticker: str
    shares: int
    avg_cost: float
    current_price: float
    market_value: float
    pnl: float
    pnl_pct: float


class Portfolio(BaseModel):
    cash: float
    total_value: float
    positions: list[Position]
    daily_pnl: float


@router.get("/")
async def get_portfolio():
    """Get current portfolio state."""
    return Portfolio(
        cash=1_000_000,
        total_value=1_000_000,
        positions=[],
        daily_pnl=0.0,
    )


@router.post("/reset")
async def reset_portfolio():
    """Reset portfolio to initial state."""
    return {"status": "ok", "message": "Portfolio reset to 1,000,000 CNY"}
