"""Stock, ETF and LOF market tools."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from data.akshare_provider import async_get_asset_spot
from data.fund_provider import (
    async_get_fund_history,
    async_get_fund_nav_history,
    async_get_fund_realtime,
)
from data.source_registry import provenance
from data.stock_provider import async_get_financial_data, async_get_stock_history, async_get_stock_realtime
from models.schemas import AssetType
from screening.fund_screener import FundScreener


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _as_of(records: list[dict[str, Any]]) -> str | None:
    if not records:
        return None
    latest = records[-1]
    return str(latest.get("date") or latest.get("日期") or latest.get("trade_date") or "") or None


@tool
async def get_realtime_quote(ticker: str, asset_type: str = "stock") -> str:
    """获取结构化市场数据：股票、ETF或LOF的当前价格和成交信息。"""
    kind = AssetType(asset_type)
    quote = (
        await async_get_stock_realtime(ticker)
        if kind == AssetType.STOCK
        else await async_get_fund_realtime(ticker, asset_type=kind.value)
    )
    return _dump(
        {
            "data_type": "market_data",
            "ticker": ticker,
            "asset_type": kind.value,
            "quote": quote,
            "provenance": provenance(
                "akshare",
                as_of=str(quote.get("data_date") or quote.get("date") or "") or None,
                freshness="realtime",
            ),
        }
    )


@tool
async def get_historical_prices(ticker: str, asset_type: str = "stock", limit: int = 60) -> str:
    """获取结构化市场数据：股票、ETF或LOF的历史日线价格，用于走势和技术分析。"""
    kind = AssetType(asset_type)
    history = (
        await async_get_stock_history(ticker)
        if kind == AssetType.STOCK
        else await async_get_fund_history(ticker, asset_type=kind.value)
    )
    records = history.tail(max(1, min(limit, 250))).to_dict("records")
    return _dump(
        {
            "data_type": "market_data",
            "ticker": ticker,
            "asset_type": kind.value,
            "history": records,
            "provenance": provenance("akshare", as_of=_as_of(records), freshness="historical"),
        }
    )


@tool
async def get_fund_nav_history(ticker: str, asset_type: str = "etf", limit: int = 60) -> str:
    """获取 ETF/LOF 的历史单位净值和累计净值。"""
    kind = AssetType(asset_type)
    if kind not in {AssetType.ETF, AssetType.LOF}:
        raise ValueError("asset_type 必须是 etf 或 lof")
    frame = await async_get_fund_nav_history(ticker, asset_type=kind.value)
    records = frame.tail(max(1, min(int(limit), 250))).to_dict("records")
    return _dump(
        {
            "data_type": "fund_nav",
            "ticker": ticker,
            "asset_type": kind.value,
            "history": records,
            "provenance": provenance("akshare", as_of=_as_of(records), freshness="historical"),
        }
    )


@tool
async def get_fundamentals(ticker: str, asset_type: str = "stock") -> str:
    """获取股票财务数据，或 ETF/LOF 的基金基础数据；不生成主观结论。"""
    kind = AssetType(asset_type)
    if kind == AssetType.STOCK:
        data = await async_get_financial_data(ticker)
        return _dump(
            {
                "data_type": "fundamentals",
                "ticker": ticker,
                "asset_type": kind.value,
                "data": data,
                "provenance": provenance("akshare", freshness="latest_available"),
            }
        )
    realtime = await async_get_fund_realtime(ticker, asset_type=kind.value)
    nav = await async_get_fund_nav_history(ticker, asset_type=kind.value)
    latest_nav = nav.tail(1).to_dict("records") if not nav.empty else []
    return _dump(
        {
            "data_type": "fundamentals",
            "ticker": ticker,
            "asset_type": kind.value,
            "data": {"realtime": realtime, "latest_nav": latest_nav},
            "provenance": provenance("akshare", freshness="latest_available"),
        }
    )


@tool
async def compare_quotes(tickers: list[str], asset_type: str = "stock") -> str:
    """获取多个股票、ETF或LOF的实时行情，用于行情层面对比。"""
    kind = AssetType(asset_type)
    requested = tickers[:10]
    snapshot = await async_get_asset_spot(kind.value, limit=5000)
    by_ticker = {str(item.get("ticker", "")).zfill(6): item for item in snapshot}
    quotes = []
    for ticker in requested:
        normalized = str(ticker).strip().lower()
        if normalized.startswith(("sh", "sz")):
            normalized = normalized[2:]
        normalized = normalized.zfill(6)
        quotes.append({"ticker": ticker, "quote": by_ticker.get(normalized, {})})
    return _dump(
        {
            "data_type": "market_data",
            "asset_type": kind.value,
            "quotes": quotes,
            "provenance": provenance("akshare", freshness="realtime"),
        }
    )


@tool
async def screen_assets(
    asset_type: str = "etf",
    min_pct_chg: float | None = None,
    max_pct_chg: float | None = None,
    min_amount: float | None = None,
    min_turnover: float | None = None,
    keyword: str | None = None,
    sort_by: str = "screen_score",
    limit: int = 20,
) -> str:
    """按结构化市场数据筛选股票/ETF/LOF候选标的。"""
    kind = AssetType(asset_type)
    records = await async_get_asset_spot(kind.value, limit=5000)
    if kind in {AssetType.ETF, AssetType.LOF}:
        screened = FundScreener().screen_snapshot(
            records,
            asset_type=kind.value,
            min_pct_chg=min_pct_chg,
            max_pct_chg=max_pct_chg,
            min_amount=min_amount,
            min_turnover=min_turnover,
            keyword=keyword,
            sort_by=sort_by,
            limit=limit,
        )
        return _dump(
            {
                "asset_type": kind.value,
                "screen_type": "polars_fund_snapshot",
                "count": len(screened),
                "results": screened,
                "provenance": provenance("akshare", freshness="realtime"),
            }
        )

    query = (keyword or "").strip().lower()
    if query:
        records = [item for item in records if query in str(item.get("name", "")).lower()]
    if min_pct_chg is not None:
        records = [item for item in records if item.get("pct_chg", 0) >= min_pct_chg]
    if max_pct_chg is not None:
        records = [item for item in records if item.get("pct_chg", 0) <= max_pct_chg]
    if min_amount is not None:
        records = [item for item in records if item.get("amount", 0) >= min_amount]
    if min_turnover is not None:
        records = [item for item in records if item.get("turnover", 0) >= min_turnover]
    sort_key = sort_by if sort_by in {"pct_chg", "amount", "turnover", "total_mv"} else "amount"
    records.sort(key=lambda item: item.get(sort_key, 0), reverse=True)
    return _dump(
        {
            "asset_type": kind.value,
            "screen_type": "realtime_snapshot",
            "count": len(records),
            "results": records[: max(1, min(limit, 50))],
            "provenance": provenance("akshare", freshness="realtime"),
        }
    )


TOOLS = [
    get_realtime_quote,
    get_historical_prices,
    get_fund_nav_history,
    get_fundamentals,
    compare_quotes,
    screen_assets,
]
