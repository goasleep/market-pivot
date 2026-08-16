"""Lightweight market quote endpoints for watchlists and UI previews."""

from fastapi import APIRouter
from pydantic import BaseModel

from data.fund_provider import async_get_fund_realtime
from data.stock_provider import async_get_stock_realtime
from models.schemas import AssetType

router = APIRouter()


class QuoteResponse(BaseModel):
    ticker: str
    asset_type: AssetType
    quote: dict
    available: bool


@router.get("/quote", response_model=QuoteResponse)
async def get_quote(ticker: str, asset_type: AssetType = AssetType.STOCK):
    if asset_type == AssetType.STOCK:
        quote = await async_get_stock_realtime(ticker)
    else:
        quote = await async_get_fund_realtime(ticker, asset_type=asset_type.value)
    return QuoteResponse(
        ticker=ticker,
        asset_type=asset_type,
        quote=quote,
        available=bool(quote.get("price")),
    )
