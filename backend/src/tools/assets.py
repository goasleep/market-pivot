"""Stock, ETF and LOF market tools."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

from langchain_core.tools import tool
from loguru import logger

from data.akshare_provider import async_get_asset_spot
from data.fund_provider import (
    async_get_fund_history,
    async_get_fund_nav_history,
    async_get_fund_realtime,
)
from data.source_registry import provenance, utc_now
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


def _positive_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _quote_from_history(ticker: str, kind: AssetType, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    latest = records[-1]
    price = _positive_number(latest.get("close") or latest.get("收盘"))
    if price is None:
        return {}
    previous = records[-2] if len(records) > 1 else {}
    previous_close = _positive_number(previous.get("close") or previous.get("收盘"))
    pct_chg = latest.get("pct_chg", latest.get("涨跌幅"))
    if pct_chg is None and previous_close:
        pct_chg = (price / previous_close - 1) * 100
    return {
        "ticker": ticker,
        "asset_type": kind.value,
        "price": price,
        "pct_chg": pct_chg,
        "open": latest.get("open", latest.get("开盘")),
        "high": latest.get("high", latest.get("最高")),
        "low": latest.get("low", latest.get("最低")),
        "prev_close": previous_close,
        "volume": latest.get("volume", latest.get("成交量")),
        "amount": latest.get("amount", latest.get("成交额")),
        "turnover": latest.get("turnover", latest.get("换手率")),
        "data_date": _as_of(records),
        "data_status": "historical_fallback",
        "data_note": "实时行情暂不可用，已降级为最近交易日历史收盘数据",
    }


async def _historical_quote(ticker: str, kind: AssetType) -> dict[str, Any]:
    try:
        history = (
            await async_get_stock_history(ticker)
            if kind == AssetType.STOCK
            else await async_get_fund_history(ticker, asset_type=kind.value)
        )
    except Exception as exc:
        logger.warning("Historical quote fallback failed for {}:{}: {}", kind.value, ticker, exc)
        return {}
    return _quote_from_history(ticker, kind, history.tail(2).to_dict("records"))


async def _realtime_quote_with_fallback(ticker: str, kind: AssetType) -> tuple[dict[str, Any], str, str]:
    try:
        quote = (
            await async_get_stock_realtime(ticker)
            if kind == AssetType.STOCK
            else await async_get_fund_realtime(ticker, asset_type=kind.value)
        )
    except Exception as exc:
        logger.warning("Realtime quote failed for {}:{}: {}", kind.value, ticker, exc)
        quote = {}
    if _positive_number(quote.get("price")) is not None:
        quote = dict(quote)
        quote.setdefault("data_status", "realtime")
        return quote, "available", "realtime"
    fallback = await _historical_quote(ticker, kind)
    if fallback:
        return fallback, "degraded", "historical_fallback"
    return {}, "unavailable", "unavailable"


@tool
async def get_realtime_quote(ticker: str, asset_type: str = "stock") -> str:
    """获取结构化市场数据：股票、ETF或LOF的当前价格和成交信息。"""
    kind = AssetType(asset_type)
    quote, status, freshness = await _realtime_quote_with_fallback(ticker, kind)
    as_of = str(quote.get("data_date") or quote.get("date") or "") or None
    if quote and freshness == "realtime" and as_of is None:
        as_of = utc_now()
        quote["updated_at"] = as_of
    message = None
    error = None
    if status == "degraded":
        message = str(quote.get("data_note"))
    elif status == "unavailable":
        message = "实时行情与历史数据均不可用"
        error = {"code": "market_data_unavailable", "message": message}
    return _dump(
        {
            "data_type": "market_data",
            "ticker": ticker,
            "asset_type": kind.value,
            "available": bool(quote),
            "data_status": status,
            "message": message,
            "error": error,
            "quote": quote,
            "provenance": provenance(
                "akshare",
                as_of=as_of,
                freshness=freshness,
                status=status,
            ),
        }
    )


@tool
async def get_historical_prices(ticker: str, asset_type: str = "stock", limit: int = 60) -> str:
    """获取结构化市场数据：股票、ETF或LOF的历史日线价格，用于走势和技术分析。"""
    kind = AssetType(asset_type)
    try:
        history = (
            await async_get_stock_history(ticker)
            if kind == AssetType.STOCK
            else await async_get_fund_history(ticker, asset_type=kind.value)
        )
        records = history.tail(max(1, min(limit, 250))).to_dict("records")
    except Exception as exc:
        logger.warning("Historical prices failed for {}:{}: {}", kind.value, ticker, exc)
        records = []
    status = "available" if records else "unavailable"
    return _dump(
        {
            "data_type": "market_data",
            "ticker": ticker,
            "asset_type": kind.value,
            "available": bool(records),
            "error": None if records else {"code": "market_history_unavailable", "message": "历史价格数据不可用"},
            "history": records,
            "provenance": provenance("akshare", as_of=_as_of(records), freshness="historical", status=status),
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
    try:
        snapshot = await async_get_asset_spot(kind.value, limit=5000)
    except Exception as exc:
        logger.warning("Market snapshot failed for {} comparison: {}", kind.value, exc)
        snapshot = []
    by_ticker = {str(item.get("ticker", "")).zfill(6): item for item in snapshot}
    normalized_tickers = []
    for ticker in requested:
        normalized = str(ticker).strip().lower()
        if normalized.startswith(("sh", "sz")):
            normalized = normalized[2:]
        normalized = normalized.zfill(6)
        normalized_tickers.append((ticker, normalized))
    missing = [normalized for _, normalized in normalized_tickers if not by_ticker.get(normalized)]
    fallback_quotes = await asyncio.gather(*(_historical_quote(ticker, kind) for ticker in missing))
    fallbacks = dict(zip(missing, fallback_quotes, strict=True))
    quotes = [
        {"ticker": ticker, "quote": by_ticker.get(normalized) or fallbacks.get(normalized, {})}
        for ticker, normalized in normalized_tickers
    ]
    available_count = sum(bool(item["quote"]) for item in quotes)
    used_fallback = any(item["quote"].get("data_status") == "historical_fallback" for item in quotes)
    if available_count == len(quotes) and not used_fallback:
        status = "available"
    else:
        status = "degraded" if available_count else "unavailable"
    freshness = "historical_fallback" if used_fallback else "realtime" if available_count else "unavailable"
    as_of_dates = [str(item["quote"].get("data_date") or "") for item in quotes if item["quote"]]
    as_of = max(as_of_dates) if as_of_dates else utc_now() if available_count else None
    return _dump(
        {
            "data_type": "market_data",
            "asset_type": kind.value,
            "available": available_count > 0,
            "data_status": status,
            "message": (
                f"{len(quotes) - available_count} 个标的缺少行情；其余结果可能使用历史收盘数据"
                if status != "available"
                else None
            ),
            "error": (
                {"code": "market_data_unavailable", "message": "所有候选标的均缺少可用行情"}
                if status == "unavailable"
                else None
            ),
            "quotes": quotes,
            "provenance": provenance(
                "akshare",
                as_of=as_of,
                freshness=freshness,
                status=status,
            ),
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
