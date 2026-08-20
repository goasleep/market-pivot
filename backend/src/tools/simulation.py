"""Paper-trading simulation tools."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from data.source_registry import provenance
from engine.simulation_account import simulation_accounts
from models.schemas import AssetType, Decision


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


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
    account = await simulation_accounts.get_account(account_id)
    return _dump(
        {
            "ok": True,
            "paper_trading": True,
            "portfolio": _simulation_summary(account),
            "provenance": provenance("paper_trading_db", freshness="local_state"),
        }
    )


@tool
async def get_simulation_orders(account_id: str = "default") -> str:
    """查询纸面交易模拟盘的订单状态；绝不连接实盘。"""
    orders = await simulation_accounts.list_orders(account_id)
    return _dump(
        {
            "ok": True,
            "paper_trading": True,
            "account_id": account_id,
            "orders": [order.model_dump(mode="json") for order in orders[:50]],
            "provenance": provenance("paper_trading_db", freshness="local_state"),
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
    """创建一笔待成交的纸面交易订单；只有用户明确要求下单时调用。"""
    order = await simulation_accounts.create_order(
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
            "provenance": provenance("paper_trading_db", freshness="local_state"),
        }
    )


@tool
async def cancel_simulation_order(order_id: str) -> str:
    """取消一笔尚未成交的纸面交易订单；绝不操作实盘。"""
    order = await simulation_accounts.cancel_order(order_id)
    return _dump(
        {
            "ok": True,
            "paper_trading": True,
            "order": order.model_dump(mode="json"),
            "provenance": provenance("paper_trading_db", freshness="local_state"),
        }
    )


TOOLS = [get_simulation_portfolio, get_simulation_orders, submit_simulation_order, cancel_simulation_order]
