"""Paper-trading simulation tools."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.tools import tool

from application.deployments import deployment_service
from data.source_registry import provenance
from engine.simulation_account import simulation_accounts
from models.schemas import AssetType, Decision, SimulationAccountConfig


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
async def list_simulation_accounts() -> str:
    """列出所有纸面交易模拟账户，便于选择账户；绝不连接实盘。"""
    accounts = await simulation_accounts.list_accounts()
    return _dump(
        {
            "ok": True,
            "paper_trading": True,
            "accounts": [_simulation_summary(account) for account in accounts],
            "provenance": provenance("paper_trading_db", freshness="local_state"),
        }
    )


@tool
async def list_strategy_deployments(account_id: str | None = None) -> str:
    """查询回测实验部署到模拟盘的不可变策略版本和运行状态。"""
    deployments = await deployment_service.list(account_id=account_id)
    return _dump(
        {
            "ok": True,
            "paper_trading": True,
            "deployments": [item.model_dump(mode="json") for item in deployments],
            "provenance": provenance("paper_trading_db", freshness="local_state"),
        }
    )


@tool
async def create_simulation_account(
    account_id: str,
    name: str,
    initial_cash: float = 1_000_000,
    asset_type: str = "stock",
    execution_key: str | None = None,
) -> str:
    """创建一个空的内部模拟账户；只有用户明确要求创建时调用。"""
    if execution_key:
        try:
            existing = await simulation_accounts.get_account(account_id)
            return _dump(
                {
                    "ok": True,
                    "paper_trading": True,
                    "idempotent_replay": True,
                    "account": _simulation_summary(existing),
                }
            )
        except KeyError:
            pass
    account = await simulation_accounts.create_account(
        account_id,
        SimulationAccountConfig(
            name=name,
            initial_cash=initial_cash,
            asset_type=AssetType(asset_type),
        ),
    )
    return _dump({"ok": True, "paper_trading": True, "account": _simulation_summary(account)})


@tool
async def deploy_backtest_experiment(
    experiment_id: str,
    account_id: str,
    account_name: str | None = None,
    initial_cash: float | None = None,
    mode: str = "confirm",
    enabled: bool = True,
    execution_key: str | None = None,
) -> str:
    """把已完成回测的不可变策略快照部署到新的内部模拟账户；必须先由用户确认。"""
    deployment = await deployment_service.create_from_experiment(
        experiment_id,
        account_id=account_id,
        create_account=True,
        account_name=account_name,
        initial_cash=initial_cash,
        mode=mode,
        enabled=enabled,
        execution_key=execution_key,
    )
    return _dump(
        {
            "ok": True,
            "paper_trading": True,
            "deployment": deployment.model_dump(mode="json"),
        }
    )


@tool
async def set_strategy_deployment_status(deployment_id: str, status: str) -> str:
    """启用、暂停或归档一个模拟盘策略部署；必须先由用户确认。"""
    deployment = await deployment_service.set_status(deployment_id, status)
    return _dump({"ok": True, "paper_trading": True, "deployment": deployment.model_dump(mode="json")})


@tool
async def submit_simulation_order(
    ticker: str,
    side: str,
    shares: int,
    account_id: str = "default",
    asset_type: str = "stock",
    order_type: str = "market",
    limit_price: float | None = None,
    execution_key: str | None = None,
) -> str:
    """创建一笔待成交的纸面交易订单；只有用户明确要求下单时调用。"""
    deterministic_order_id = f"sim-{hashlib.sha256(execution_key.encode()).hexdigest()[:24]}" if execution_key else None
    order = await simulation_accounts.create_order(
        account_id=account_id,
        ticker=ticker,
        side=Decision(side),
        shares=shares,
        order_type=order_type,
        limit_price=limit_price,
        source="agent",
        fill_policy="manual",
        asset_type=AssetType(asset_type),
        order_id=deterministic_order_id,
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


TOOLS = [
    get_simulation_portfolio,
    get_simulation_orders,
    list_simulation_accounts,
    list_strategy_deployments,
    create_simulation_account,
    deploy_backtest_experiment,
    set_strategy_deployment_status,
    submit_simulation_order,
    cancel_simulation_order,
]
