"""Data and application capabilities exposed to the conversational LLM."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import StructuredTool, tool

from data.akshare_provider import (
    async_get_asset_spot,
    async_get_fund_history,
    async_get_fund_realtime,
    async_get_stock_history,
    async_get_stock_news,
    async_get_stock_realtime,
)
from data.ddgs_provider import async_search_web_ddgs
from data.serper_provider import async_search_web_parallel
from engine.simulation_account import simulation_accounts
from models.schemas import AssetType, Decision
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
async def search_web(query: str, num_results: int = 8, freshness: str | None = None) -> str:
    """并行使用 Serper 和 DDGS 搜索最新网页资讯，并合并去重结果。"""
    allowed_freshness = {None, "qdr:h", "qdr:d", "qdr:w", "qdr:m", "qdr:y"}
    if freshness not in allowed_freshness:
        freshness = None
    return _dump(await async_search_web_parallel(query, num_results=num_results, tbs=freshness))


@tool
async def search_web_ddgs(query: str, num_results: int = 8, freshness: str | None = None) -> str:
    """明确使用 DDGS 免费元搜索，返回标题、摘要、来源和链接。"""
    allowed_freshness = {None, "qdr:h", "qdr:d", "qdr:w", "qdr:m", "qdr:y"}
    if freshness not in allowed_freshness:
        freshness = None
    timelimit = {"qdr:h": "h", "qdr:d": "d", "qdr:w": "w", "qdr:m": "m", "qdr:y": "y"}.get(freshness or "")
    return _dump(await async_search_web_ddgs(query, num_results=num_results, timelimit=timelimit))


@tool
async def compare_quotes(tickers: list[str], asset_type: str = "stock") -> str:
    """获取多个股票、ETF或LOF的实时行情，用于行情层面对比。"""
    kind = AssetType(asset_type)
    requested = tickers[:10]
    # Both the stock and fund realtime adapters fetch a whole-market snapshot
    # for one code. Fetch that snapshot once for comparison instead of making
    # the same expensive upstream request once per ticker.
    snapshot = await async_get_asset_spot(kind.value, limit=5000)
    by_ticker = {str(item.get("ticker", "")).zfill(6): item for item in snapshot}
    quotes = []
    for ticker in requested:
        normalized = str(ticker).strip().lower()
        if normalized.startswith(("sh", "sz")):
            normalized = normalized[2:]
        normalized = normalized.zfill(6)
        quotes.append({"ticker": ticker, "quote": by_ticker.get(normalized, {})})
    return _dump({"asset_type": kind.value, "quotes": quotes})


@tool
async def list_trading_strategies() -> str:
    """列出系统支持的研究和交易策略。"""
    return _dump(await __import__("asyncio").to_thread(list_strategies))


@tool
async def screen_assets(
    asset_type: str = "etf",
    min_pct_chg: float | None = None,
    max_pct_chg: float | None = None,
    min_amount: float | None = None,
    min_turnover: float | None = None,
    keyword: str | None = None,
    sort_by: str = "amount",
    limit: int = 20,
) -> str:
    """按实时涨跌幅、成交额、换手率和名称筛选股票/ETF/LOF候选标的。

    金额单位为元，涨跌幅和换手率单位为百分比。结果是实时初筛，
    需要交易决策时还应继续调用历史数据和综合分析工具。
    """
    kind = AssetType(asset_type)
    records = await async_get_asset_spot(kind.value, limit=5000)
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
        }
    )


def _simulation_summary(account: Any) -> dict[str, Any]:
    portfolio = account.portfolio
    return {
        "account_id": account.account_id,
        "status": account.status,
        "current_date": account.current_date,
        "asset_type": account.config.asset_type.value,
        "cash": portfolio.cash,
        "total_value": portfolio.total_value,
        "total_pnl": portfolio.total_pnl,
        "total_return_pct": portfolio.total_return_pct,
        "positions": [position.model_dump(mode="json") for position in portfolio.positions],
        "trades": [trade.model_dump(mode="json") for trade in portfolio.trades[-20:]],
    }


@tool
async def get_simulation_portfolio(account_id: str = "default") -> str:
    """查询纸面交易模拟盘账户、现金、持仓和收益；绝不连接实盘。"""
    account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
    return _dump({"ok": True, "paper_trading": True, "portfolio": _simulation_summary(account)})


@tool
async def get_simulation_orders(account_id: str = "default") -> str:
    """查询纸面交易模拟盘的订单状态；绝不连接实盘。"""
    orders = await asyncio.to_thread(simulation_accounts.list_orders, account_id)
    return _dump(
        {
            "ok": True,
            "paper_trading": True,
            "account_id": account_id,
            "orders": [order.model_dump(mode="json") for order in orders[:50]],
        }
    )


@tool
async def submit_simulation_order(
    ticker: str,
    side: str,
    shares: int,
    account_id: str = "default",
    asset_type: str = "stock",
    order_type: str = "market",
    limit_price: float | None = None,
) -> str:
    """创建一笔待成交的纸面交易订单。只有用户明确要求下单时调用，绝不进行实盘交易。"""
    order = await asyncio.to_thread(
        simulation_accounts.create_order,
        account_id,
        ticker,
        Decision(side),
        shares,
        order_type,
        limit_price,
        None,
        "agent",
        None,
        "manual",
        AssetType(asset_type),
    )
    return _dump(
        {
            "ok": True,
            "paper_trading": True,
            "message": "已创建待成交模拟盘订单",
            "order": order.model_dump(mode="json"),
        }
    )


@tool
async def cancel_simulation_order(order_id: str) -> str:
    """取消一笔尚未成交的纸面交易订单。只有用户明确要求取消时调用，绝不操作实盘。"""
    order = await asyncio.to_thread(simulation_accounts.cancel_order, order_id)
    return _dump({"ok": True, "paper_trading": True, "order": order.model_dump(mode="json")})


def build_chat_tools(analysis_tool: StructuredTool) -> list[StructuredTool]:
    """Return the complete tool surface available to the chat model."""
    return [
        get_realtime_quote,
        get_historical_prices,
        get_latest_news,
        search_web,
        search_web_ddgs,
        compare_quotes,
        list_trading_strategies,
        screen_assets,
        get_simulation_portfolio,
        get_simulation_orders,
        submit_simulation_order,
        cancel_simulation_order,
        analysis_tool,
    ]
