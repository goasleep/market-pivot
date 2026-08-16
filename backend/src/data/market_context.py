"""Build a single, time-bounded market snapshot for analysis workflows."""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from data.akshare_provider import (
    async_get_financial_data,
    async_get_fund_history,
    async_get_fund_realtime,
    async_get_stock_history,
    async_get_stock_news,
    async_get_stock_realtime,
)
from data.serper_provider import async_search_web_parallel
from data.web_content import async_enrich_web_results
from models.schemas import AssetType, MarketContext


def _normalise_history(df: pd.DataFrame, as_of_date: str | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    result = df.copy()
    result["date"] = result["date"].astype(str).str[:10]
    if as_of_date:
        result = result[result["date"] <= as_of_date]
    return result.to_dict(orient="records")


def _quote_from_history(ticker: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"ticker": ticker}
    latest = records[-1]
    return {
        "ticker": ticker,
        "date": latest.get("date", ""),
        "price": float(latest.get("close", 0) or 0),
        "open": float(latest.get("open", 0) or 0),
        "high": float(latest.get("high", 0) or 0),
        "low": float(latest.get("low", 0) or 0),
        "prev_close": float(latest.get("close", 0) or 0),
        "volume": float(latest.get("volume", 0) or 0),
        "amount": float(latest.get("amount", 0) or 0),
        "pct_chg": float(latest.get("pct_chg", 0) or 0),
        "turnover": float(latest.get("turnover", 0) or 0),
    }


def _detect_regime(records: list[dict[str, Any]]) -> str:
    if len(records) < 25:
        return "unknown"
    closes = pd.Series([float(row.get("close", 0) or 0) for row in records])
    ma20 = closes.rolling(20).mean()
    if pd.isna(ma20.iloc[-1]) or pd.isna(ma20.iloc[-6]):
        return "unknown"
    close = closes.iloc[-1]
    current_ma = ma20.iloc[-1]
    previous_ma = ma20.iloc[-6]
    if close > current_ma and current_ma > previous_ma:
        return "trending_up"
    if close < current_ma and current_ma < previous_ma:
        return "trending_down"
    return "sideways"


async def build_market_context(
    ticker: str,
    *,
    asset_type: AssetType | str = AssetType.STOCK,
    as_of_date: str | None = None,
    current_price: float | None = None,
    history_df: pd.DataFrame | None = None,
    include_live_enrichment: bool = True,
) -> MarketContext:
    """Build a consistent snapshot for live analysis or historical simulation."""

    asset_type = AssetType(asset_type)
    is_backtest = as_of_date is not None
    if history_df is None:
        if asset_type == AssetType.STOCK:
            history_df = await async_get_stock_history(ticker, start_date="", end_date=as_of_date or "")
        else:
            history_df = await async_get_fund_history(
                ticker,
                asset_type=asset_type.value,
                start_date="",
                end_date=as_of_date or "",
            )
    records = _normalise_history(history_df, as_of_date)

    # Historical simulation must not request current information. Daily OHLCV
    # is the only source that is reliably available as-of a past trading day.
    if is_backtest or not include_live_enrichment:
        realtime = _quote_from_history(ticker, records)
        if current_price is not None:
            realtime["price"] = current_price
        return MarketContext(
            ticker=ticker,
            asset_type=asset_type,
            as_of_date=as_of_date,
            current_price=float(current_price if current_price is not None else realtime.get("price", 0.0)),
            realtime=realtime,
            history=records,
            market_regime=_detect_regime(records),
            is_backtest=is_backtest,
            data_status={
                "history": bool(records),
                "realtime": bool(realtime.get("price")),
                "financial": False,
                "news": False,
                "latest_history_date": records[-1].get("date", "") if records else "",
            },
        )

    if asset_type == AssetType.STOCK:
        realtime_task = async_get_stock_realtime(ticker)
        financial_task = async_get_financial_data(ticker)
        news_task = async_get_stock_news(ticker, limit=10)
        realtime, financial, news = await asyncio.gather(realtime_task, financial_task, news_task)
    else:
        realtime = await async_get_fund_realtime(ticker, asset_type=asset_type.value)
        financial = {"ticker": ticker, "not_applicable": "场内基金不适用个股财务指标"}
        news = []
    asset_label = "股票" if asset_type == AssetType.STOCK else f"{asset_type.value.upper()} 场内基金"
    web_search = await async_search_web_parallel(
        f"{ticker} {asset_label} 最新公告 新闻 走势",
        num_results=8,
        tbs="qdr:m",
    )
    web_results = await async_enrich_web_results(web_search.get("results", []))
    full_text_count = sum(item.get("content_status") == "full_text" for item in web_results)
    live_price = current_price if current_price and current_price > 0 else None
    price = float(live_price if live_price is not None else realtime.get("price", 0.0) or 0.0)
    return MarketContext(
        ticker=ticker,
        asset_type=asset_type,
        current_price=price,
        realtime=realtime,
        history=records,
        financial=financial,
        news=news,
        web_results=web_results,
        market_regime=_detect_regime(records),
        data_status={
            "history": bool(records),
            "realtime": bool(realtime.get("price")),
            "financial": asset_type == AssetType.STOCK and bool(financial),
            "news": bool(news),
            "web_search": bool(web_results),
            "web_full_text": full_text_count,
            "web_search_source": web_search.get("source", "") if web_results else "",
            "latest_history_date": records[-1].get("date", "") if records else "",
            "source": "AkShare / 东方财富" if asset_type != AssetType.STOCK else "AkShare",
        },
    )
