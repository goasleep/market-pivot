"""Data and application capabilities exposed to the conversational LLM."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool, tool

from data.akshare_provider import (
    async_get_fund_history,
    async_get_fund_realtime,
    async_get_stock_history,
    async_get_stock_news,
    async_get_stock_realtime,
)
from models.schemas import AssetType
from strategies.skill_manager import list_strategies


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


@tool
async def get_realtime_quote(ticker: str, asset_type: str = "stock") -> str:
    """获取股票、ETF或LOF的实时行情。ticker必须是六位代码。"""
    kind = AssetType(asset_type)
    quote = (
        await async_get_stock_realtime(ticker)
        if kind == AssetType.STOCK
        else await async_get_fund_realtime(ticker, asset_type=kind.value)
    )
    return _dump({"ticker": ticker, "asset_type": kind.value, "quote": quote})


@tool
async def get_historical_prices(ticker: str, asset_type: str = "stock", limit: int = 60) -> str:
    """获取股票、ETF或LOF的历史日线价格，用于走势和技术分析。"""
    kind = AssetType(asset_type)
    history = (
        await async_get_stock_history(ticker)
        if kind == AssetType.STOCK
        else await async_get_fund_history(ticker, asset_type=kind.value)
    )
    records = history.tail(max(1, min(limit, 250))).to_dict("records")
    return _dump({"ticker": ticker, "asset_type": kind.value, "history": records})


@tool
async def get_latest_news(ticker: str) -> str:
    """获取股票的最新新闻和舆情。ETF/LOF没有个股新闻时返回空结果。"""
    return _dump({"ticker": ticker, "news": await async_get_stock_news(ticker, limit=10)})


@tool
async def compare_quotes(tickers: list[str], asset_type: str = "stock") -> str:
    """获取多个股票、ETF或LOF的实时行情，用于行情层面对比。"""
    kind = AssetType(asset_type)
    quotes = []
    for ticker in tickers[:10]:
        quote = (
            await async_get_stock_realtime(ticker)
            if kind == AssetType.STOCK
            else await async_get_fund_realtime(ticker, asset_type=kind.value)
        )
        quotes.append({"ticker": ticker, "quote": quote})
    return _dump({"asset_type": kind.value, "quotes": quotes})


@tool
async def list_trading_strategies() -> str:
    """列出系统支持的研究和交易策略。"""
    return _dump(await __import__("asyncio").to_thread(list_strategies))


def build_chat_tools(analysis_tool: StructuredTool) -> list[StructuredTool]:
    """Return the complete tool surface available to the chat model."""
    return [
        get_realtime_quote,
        get_historical_prices,
        get_latest_news,
        compare_quotes,
        list_trading_strategies,
        analysis_tool,
    ]
