"""Lightweight market quote endpoints for watchlists and UI previews."""

from fastapi import APIRouter
from pydantic import BaseModel

from data.exchange_fund_provider import async_get_exchange_fund_quote
from data.stock_provider import async_get_stock_realtime
from models.schemas import AssetType

router = APIRouter()


class QuoteResponse(BaseModel):
    ticker: str
    asset_type: AssetType
    quote: dict
    available: bool
    status: str = "available"
    message: str | None = None


@router.get("/quote", response_model=QuoteResponse)
async def get_quote(ticker: str, asset_type: AssetType = AssetType.STOCK):
    if asset_type == AssetType.OPEN_FUND:
        return QuoteResponse(
            ticker=ticker,
            asset_type=asset_type,
            quote={},
            available=False,
            status="not_applicable",
            message="场外开放式基金没有实时市场价格，请使用 NAV 快照",
        )
    if asset_type == AssetType.STOCK:
        quote = await async_get_stock_realtime(ticker)
    else:
        quote = await async_get_exchange_fund_quote(ticker, asset_type=asset_type.value)
    return QuoteResponse(
        ticker=ticker,
        asset_type=asset_type,
        quote=quote,
        available=bool(quote.get("price")),
        status="available" if quote.get("price") else "data_unavailable",
    )
