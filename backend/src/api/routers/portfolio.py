"""Async API for persistent paper-trading accounts."""

import asyncio
from datetime import date

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from data.fund_provider import async_get_fund_realtime
from data.stock_provider import async_get_stock_realtime
from engine.broker_adapters import (
    SimulationBrokerUnavailableError,
    broker_status,
    get_simulation_broker,
    live_broker_status,
)
from engine.simulation_account import simulation_accounts
from engine.simulation_events import simulation_events
from models.schemas import (
    AssetType,
    Decision,
    ExternalSimulationConfig,
    LiveTradingConfig,
    SimulationAccountConfig,
)

router = APIRouter()


async def _quote(ticker: str, asset_type: AssetType):
    if asset_type == AssetType.STOCK:
        return await async_get_stock_realtime(ticker)
    return await async_get_fund_realtime(ticker, asset_type=asset_type.value)


class ConfigRequest(BaseModel):
    config: SimulationAccountConfig


class CreateAccountRequest(BaseModel):
    account_id: str = Field(..., min_length=1, max_length=64)
    config: SimulationAccountConfig = Field(default_factory=SimulationAccountConfig)


class StatusRequest(BaseModel):
    status: str = Field(..., pattern="^(active|paused)$")


class OrderRequest(BaseModel):
    ticker: str = Field(..., min_length=6, max_length=8)
    side: Decision
    shares: int = Field(..., gt=0)
    order_type: str = "market"
    limit_price: float | None = Field(default=None, gt=0)
    price: float | None = Field(default=None, gt=0)
    fill_immediately: bool = True
    trade_date: str | None = None
    asset_type: AssetType | None = None


class MarkRequest(BaseModel):
    prices: dict[str, float] = Field(default_factory=dict)
    trade_date: str | None = None


class FillRequest(BaseModel):
    price: float = Field(..., gt=0)
    trade_date: str = Field(..., min_length=10, max_length=10)


def _payload(account, orders=None, daily_pnl: float = 0.0) -> dict:
    portfolio = account.portfolio
    external = account.config.external.model_dump(mode="json", exclude={"token"})
    external["token_set"] = bool(account.config.external.token)
    external["token_masked"] = (
        account.config.external.token[:4] + "***" + account.config.external.token[-4:]
        if len(account.config.external.token) > 8
        else ("***" if account.config.external.token else "")
    )
    live = account.config.live.model_dump(mode="json", exclude={"token"})
    live["token_set"] = bool(account.config.live.token)
    live["token_masked"] = (
        account.config.live.token[:4] + "***" + account.config.live.token[-4:]
        if len(account.config.live.token) > 8
        else ("***" if account.config.live.token else "")
    )
    config = account.config.model_dump(mode="json", exclude={"external"})
    config["external"] = external
    config["live"] = live
    return {
        "account_id": account.account_id,
        "name": account.config.name,
        "status": account.status,
        "current_date": account.current_date,
        "initial_capital": portfolio.initial_capital,
        "total_pnl": portfolio.total_pnl,
        "total_return_pct": portfolio.total_return_pct,
        "cash": portfolio.cash,
        "total_value": portfolio.total_value,
        "positions": [
            {
                **position.model_dump(mode="json"),
                "market_value": position.market_value,
                "pnl": position.pnl,
                "pnl_pct": position.pnl_pct,
            }
            for position in portfolio.positions
        ],
        "trades": [trade.model_dump(mode="json") for trade in portfolio.trades],
        "orders": [order.model_dump(mode="json") for order in (orders or [])],
        "config": config,
        "daily_pnl": daily_pnl,
        "broker": broker_status(account.config.external),
        "live_broker": live_broker_status(account.config.live),
    }


async def _account_payload(account_id: str = "default", refresh_quotes: bool = False) -> dict:
    account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
    if refresh_quotes and account.portfolio.positions:
        quotes = await asyncio.gather(
            *(_quote(position.ticker, position.asset_type) for position in account.portfolio.positions)
        )
        prices = {
            position.ticker: float(quote.get("price", 0) or position.current_price)
            for position, quote in zip(account.portfolio.positions, quotes)
            if quote.get("price") or position.current_price
        }
        if prices:
            account = await asyncio.to_thread(
                simulation_accounts.mark_to_market,
                account_id,
                prices,
                date.today().isoformat(),
            )
    orders = await asyncio.to_thread(simulation_accounts.list_orders, account_id)
    daily_pnl = await asyncio.to_thread(simulation_accounts.daily_pnl, account_id)
    return _payload(account, orders, daily_pnl)


async def _publish_account_update(
    account_id: str,
    event_type: str = "account.updated",
    data: dict | None = None,
) -> None:
    """Publish a lightweight event; REST remains the source of truth."""
    await simulation_events.publish(account_id, event_type, data or {})


@router.get("/")
async def get_portfolio(account_id: str = "default", refresh_quotes: bool = False):
    return await _account_payload(account_id, refresh_quotes=refresh_quotes)


@router.post("/accounts")
async def create_account(req: CreateAccountRequest):
    try:
        account = await asyncio.to_thread(simulation_accounts.create_account, req.account_id, req.config)
        return _payload(account, [])
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/accounts")
async def list_accounts():
    accounts = await asyncio.to_thread(simulation_accounts.list_accounts)
    daily_pnls = await asyncio.gather(
        *(asyncio.to_thread(simulation_accounts.daily_pnl, account.account_id) for account in accounts)
    )
    return {"accounts": [_payload(account, daily_pnl=daily_pnl) for account, daily_pnl in zip(accounts, daily_pnls)]}


@router.get("/accounts/{account_id}")
async def get_account(account_id: str, refresh_quotes: bool = False):
    return await _account_payload(account_id, refresh_quotes=refresh_quotes)


@router.websocket("/accounts/{account_id}/stream")
async def stream_account(account_id: str, websocket: WebSocket):
    """Push simulation account events to the web frontend."""
    try:
        await asyncio.to_thread(simulation_accounts.get_account, account_id)
    except KeyError:
        await websocket.close(code=4404, reason="模拟账户不存在")
        return

    await websocket.accept()
    queue = await simulation_events.subscribe(account_id)
    try:
        initial = await _account_payload(account_id)
        await websocket.send_json(
            {
                "type": "connected",
                "account_id": account_id,
                "data": {"broker": initial["broker"], "total_value": initial["total_value"]},
            }
        )
        while True:
            try:
                result = await asyncio.wait_for(queue.get(), timeout=30)
                await websocket.send_json(result)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "account_id": account_id})
    except WebSocketDisconnect:
        pass
    finally:
        await simulation_events.unsubscribe(account_id, queue)


@router.put("/accounts/{account_id}/config")
async def update_config(account_id: str, req: ConfigRequest):
    try:
        account = await asyncio.to_thread(simulation_accounts.update_config, account_id, req.config)
        payload = await _account_payload(account.account_id)
        await _publish_account_update(account.account_id, data={"reason": "config_updated"})
        return payload
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/accounts/{account_id}/external")
async def update_external_config(account_id: str, req: ExternalSimulationConfig):
    try:
        account = await asyncio.to_thread(simulation_accounts.update_external_config, account_id, req)
        payload = await _account_payload(account.account_id)
        await _publish_account_update(
            account.account_id,
            event_type="broker.updated",
            data={"broker": payload["broker"]},
        )
        return payload
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/accounts/{account_id}/live")
async def update_live_config(account_id: str, req: LiveTradingConfig):
    """Update live broker configuration; token is retained when omitted."""
    try:
        account = await asyncio.to_thread(simulation_accounts.update_live_config, account_id, req)
        payload = await _account_payload(account.account_id)
        await _publish_account_update(
            account.account_id,
            event_type="live-broker.updated",
            data={"live_broker": payload["live_broker"]},
        )
        return payload
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/accounts/{account_id}/broker")
async def get_broker_status(account_id: str):
    try:
        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        return broker_status(account.config.external)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/accounts/{account_id}/broker/validate")
async def validate_broker(account_id: str):
    """Validate the configured external broker without opening a connection."""
    try:
        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        return broker_status(account.config.external)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/accounts/{account_id}/broker/sync")
async def sync_broker(account_id: str):
    """Read the configured external simulation output files into the local mirror."""
    try:
        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        broker = get_simulation_broker(account.config.external)
        snapshot = await asyncio.to_thread(broker.sync)
        await asyncio.to_thread(simulation_accounts.apply_external_snapshot, account_id, snapshot)
        payload = await _account_payload(account_id)
        await _publish_account_update(
            account_id,
            event_type="broker.synced",
            data={"broker": payload["broker"], "as_of": snapshot.get("as_of")},
        )
        return payload
    except SimulationBrokerUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/accounts/{account_id}/status")
async def update_status(account_id: str, req: StatusRequest):
    try:
        account = await asyncio.to_thread(simulation_accounts.set_status, account_id, req.status)
        payload = await _account_payload(account.account_id)
        await _publish_account_update(account.account_id, data={"status": account.status})
        return payload
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/accounts/{account_id}/reset")
async def reset_account(account_id: str):
    try:
        account = await asyncio.to_thread(simulation_accounts.reset_account, account_id)
        payload = await _account_payload(account.account_id)
        await _publish_account_update(account.account_id, event_type="account.reset", data={})
        return payload
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reset")
async def reset_default_account(account_id: str = "default"):
    return await reset_account(account_id)


@router.post("/accounts/{account_id}/orders")
async def create_order(account_id: str, req: OrderRequest):
    try:
        submitted_date = req.trade_date or date.today().isoformat()
        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        order = await asyncio.to_thread(
            simulation_accounts.create_order,
            account_id,
            req.ticker,
            req.side,
            req.shares,
            req.order_type,
            req.limit_price,
            submitted_date,
            asset_type=req.asset_type,
        )
        if account.config.external.enabled and account.config.external.provider != "internal":
            try:
                broker = get_simulation_broker(account.config.external)
                await asyncio.to_thread(broker.submit_order, order)
            except SimulationBrokerUnavailableError as exc:
                await asyncio.to_thread(simulation_accounts.cancel_order, order.order_id)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            await _publish_account_update(
                account_id,
                event_type="order.updated",
                data={"order": order.model_dump(mode="json")},
            )
            return order.model_dump(mode="json")
        if req.fill_immediately and req.order_type == "market":
            fill_price = req.price
            if fill_price is None:
                quote = await _quote(order.ticker, order.asset_type)
                fill_price = float(quote.get("price", 0) or 0)
            if fill_price <= 0:
                await asyncio.to_thread(simulation_accounts.cancel_order, order.order_id)
                raise HTTPException(status_code=400, detail="无法取得可执行价格")
            order = await asyncio.to_thread(simulation_accounts.fill_order, order.order_id, fill_price, submitted_date)
        await _publish_account_update(
            account_id,
            event_type="order.updated",
            data={"order": order.model_dump(mode="json")},
        )
        return order.model_dump(mode="json")
    except HTTPException:
        raise
    except SimulationBrokerUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/accounts/{account_id}/orders")
async def list_orders(account_id: str):
    orders = await asyncio.to_thread(simulation_accounts.list_orders, account_id)
    return {"orders": [order.model_dump(mode="json") for order in orders]}


@router.post("/accounts/{account_id}/orders/{order_id}/cancel")
async def cancel_order(account_id: str, order_id: str):
    try:
        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        orders = await asyncio.to_thread(simulation_accounts.list_orders, account_id)
        if not any(item.order_id == order_id for item in orders):
            raise HTTPException(status_code=404, detail="订单不属于该模拟账户")
        if account.config.external.enabled and account.config.external.provider != "internal":
            broker = get_simulation_broker(account.config.external)
            await asyncio.to_thread(broker.cancel_order, order_id)
        order = await asyncio.to_thread(simulation_accounts.cancel_order, order_id)
        await _publish_account_update(
            account_id,
            event_type="order.updated",
            data={"order": order.model_dump(mode="json")},
        )
        return order.model_dump(mode="json")
    except HTTPException:
        raise
    except SimulationBrokerUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/accounts/{account_id}/orders/{order_id}/fill")
async def fill_order(account_id: str, order_id: str, req: FillRequest):
    try:
        orders = await asyncio.to_thread(simulation_accounts.list_orders, account_id)
        if not any(item.order_id == order_id for item in orders):
            raise HTTPException(status_code=404, detail="订单不属于该模拟账户")
        order = await asyncio.to_thread(
            simulation_accounts.fill_order,
            order_id,
            req.price,
            req.trade_date,
        )
        await _publish_account_update(
            account_id,
            event_type="order.updated",
            data={"order": order.model_dump(mode="json")},
        )
        return order.model_dump(mode="json")
    except HTTPException:
        raise
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/accounts/{account_id}/mark")
async def mark_account(account_id: str, req: MarkRequest):
    prices = dict(req.prices)
    if not prices:
        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        quotes = await asyncio.gather(
            *(_quote(position.ticker, position.asset_type) for position in account.portfolio.positions)
        )
        prices = {
            position.ticker: float(quote.get("price", 0) or 0)
            for position, quote in zip(account.portfolio.positions, quotes)
            if quote.get("price")
        }
    snapshot_date = req.trade_date or date.today().isoformat()
    account = await asyncio.to_thread(simulation_accounts.mark_to_market, account_id, prices, snapshot_date)
    filled_orders = []
    for order in await asyncio.to_thread(simulation_accounts.list_orders, account_id):
        if order.status != "pending" or order.ticker not in prices:
            continue
        price = prices[order.ticker]
        should_fill = order.order_type == "market" or (
            order.side == Decision.BUY and price <= (order.limit_price or 0)
        ) or (order.side == Decision.SELL and price >= (order.limit_price or 0))
        if should_fill:
            filled_orders.append(
                await asyncio.to_thread(simulation_accounts.fill_order, order.order_id, price, snapshot_date)
            )
    payload = await _account_payload(account_id)
    for order in filled_orders:
        await _publish_account_update(
            account_id,
            event_type="order.updated",
            data={"order": order.model_dump(mode="json")},
        )
    await _publish_account_update(
        account_id,
        event_type="account.updated",
        data={"total_value": payload["total_value"], "current_date": payload["current_date"]},
    )
    return payload


@router.get("/accounts/{account_id}/snapshots")
async def list_snapshots(account_id: str, limit: int = 100):
    snapshots = await asyncio.to_thread(
        simulation_accounts.list_snapshots,
        account_id,
        max(1, min(limit, 5000)),
    )
    return {"snapshots": [snapshot.model_dump(mode="json") for snapshot in snapshots]}
