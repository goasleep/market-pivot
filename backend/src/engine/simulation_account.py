"""Persistent, configurable simulation accounts.

The service is intentionally independent from the Agent workflow. Agents
produce trade intents; this module owns account state, order lifecycle, A-share
rules, and persistence. An external paper-trading provider can be added behind
the ``ExternalSimulationConfig`` without changing the account API.
"""

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from loguru import logger

from config import settings
from engine.trading_engine import TradingEngine
from models.schemas import (
    AssetType,
    Decision,
    ExternalSimulationConfig,
    LiveTradingConfig,
    PortfolioState,
    Position,
    SimulationAccount,
    SimulationAccountConfig,
    SimulationOrder,
    SimulationSnapshot,
    TradeRecord,
)

ACCOUNT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(model) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False)


class SimulationAccountService:
    """Thread-safe SQLite-backed account and order service."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or settings.database_file_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()
        self._ensure_default_account()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _init_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS simulation_accounts (
                    account_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    current_date TEXT NOT NULL DEFAULT '',
                    config_json TEXT NOT NULL,
                    portfolio_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS simulation_orders (
                    order_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    order_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(account_id) REFERENCES simulation_accounts(account_id)
                );

                CREATE TABLE IF NOT EXISTS simulation_snapshots (
                    account_id TEXT NOT NULL,
                    snapshot_date TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(account_id, snapshot_date),
                    FOREIGN KEY(account_id) REFERENCES simulation_accounts(account_id)
                );
                """
            )

    def _ensure_default_account(self) -> None:
        if self._get_row("default") is None:
            self.create_account("default", SimulationAccountConfig())

    def _get_row(self, account_id: str) -> sqlite3.Row | None:
        with self._lock, self._connect() as connection:
            return connection.execute(
                "SELECT * FROM simulation_accounts WHERE account_id = ?",
                (account_id,),
            ).fetchone()

    def _account_from_row(self, row: sqlite3.Row) -> SimulationAccount:
        return SimulationAccount(
            account_id=row["account_id"],
            status=row["status"],
            current_date=row["current_date"],
            config=SimulationAccountConfig.model_validate(json.loads(row["config_json"])),
            portfolio=PortfolioState.model_validate(json.loads(row["portfolio_json"])),
        )

    def _save_account(self, account: SimulationAccount, connection: sqlite3.Connection) -> None:
        timestamp = _now()
        connection.execute(
            """UPDATE simulation_accounts
               SET status = ?, current_date = ?, config_json = ?, portfolio_json = ?, updated_at = ?
               WHERE account_id = ?""",
            (
                account.status,
                account.current_date,
                _json(account.config),
                _json(account.portfolio),
                timestamp,
                account.account_id,
            ),
        )

    def create_account(self, account_id: str, config: SimulationAccountConfig | None = None) -> SimulationAccount:
        if not ACCOUNT_ID_PATTERN.fullmatch(account_id):
            raise ValueError("account_id 只能包含字母、数字、下划线和连字符，长度 1-64")
        account_config = config or SimulationAccountConfig()
        account = SimulationAccount(
            account_id=account_id,
            config=account_config,
            portfolio=PortfolioState(
                cash=account_config.initial_cash,
                initial_capital=account_config.initial_cash,
            ),
        )
        with self._lock, self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM simulation_accounts WHERE account_id = ?", (account_id,)
            ).fetchone():
                raise ValueError(f"模拟账户已存在: {account_id}")
            timestamp = _now()
            connection.execute(
                """INSERT INTO simulation_accounts
                   (account_id, status, current_date, config_json, portfolio_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    account.account_id,
                    account.status,
                    account.current_date,
                    _json(account.config),
                    _json(account.portfolio),
                    timestamp,
                    timestamp,
                ),
            )
        logger.info(f"Created simulation account: {account_id}")
        return account

    def list_accounts(self) -> list[SimulationAccount]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT * FROM simulation_accounts ORDER BY created_at").fetchall()
        return [self._account_from_row(row) for row in rows]

    def get_account(self, account_id: str = "default") -> SimulationAccount:
        row = self._get_row(account_id)
        if row is None:
            raise KeyError(f"模拟账户不存在: {account_id}")
        return self._account_from_row(row)

    def update_config(self, account_id: str, config: SimulationAccountConfig) -> SimulationAccount:
        account = self.get_account(account_id)
        if account.portfolio.trades and config.initial_cash != account.config.initial_cash:
            raise ValueError("已有成交记录后不能修改 initial_cash，请先重置账户")
        if not account.portfolio.trades and config.initial_cash != account.config.initial_cash:
            account.portfolio.cash = config.initial_cash
            account.portfolio.initial_capital = config.initial_cash
        if not config.external.token:
            config = config.model_copy(update={"external": account.config.external})
        if not config.live.token:
            config = config.model_copy(update={"live": account.config.live})
        account.config = config
        with self._lock, self._connect() as connection:
            self._save_account(account, connection)
        return account

    def update_external_config(
        self, account_id: str, update: ExternalSimulationConfig
    ) -> SimulationAccount:
        account = self.get_account(account_id)
        current = account.config.external
        if not update.token:
            update = update.model_copy(update={"token": current.token})
        account.config = account.config.model_copy(update={"external": update})
        with self._lock, self._connect() as connection:
            self._save_account(account, connection)
        return account

    def update_live_config(self, account_id: str, update: LiveTradingConfig) -> SimulationAccount:
        """Persist live broker settings without exposing or losing the token."""
        account = self.get_account(account_id)
        current = account.config.live
        if not update.token:
            update = update.model_copy(update={"token": current.token})
        account.config = account.config.model_copy(update={"live": update})
        with self._lock, self._connect() as connection:
            self._save_account(account, connection)
        return account

    def apply_live_snapshot(
        self,
        account_id: str,
        cash: float,
        positions: list[dict],
        snapshot_date: str | None = None,
    ) -> SimulationAccount:
        """Mirror broker cash/positions locally for sizing and audit only."""
        if cash < 0:
            raise ValueError("实盘同步返回的 cash 不能为负数")
        account = self.get_account(account_id)
        account.portfolio.cash = cash
        account.portfolio.positions = [Position.model_validate(item) for item in positions]
        if snapshot_date:
            account.current_date = snapshot_date
        with self._lock, self._connect() as connection:
            self._save_account(account, connection)
        return account

    def apply_external_snapshot(self, account_id: str, snapshot: dict) -> SimulationAccount:
        """Mirror an external paper account and reconcile its local order IDs."""
        if not isinstance(snapshot, dict) or snapshot.get("cash") is None:
            raise ValueError("外部模拟同步缺少 cash")
        cash = float(snapshot["cash"])
        if cash < 0:
            raise ValueError("外部模拟同步返回的 cash 不能为负数")

        account = self.get_account(account_id)
        previous_positions = {item.ticker: item for item in account.portfolio.positions}
        positions = []
        for item in snapshot.get("positions", []):
            ticker = (
                str(item.get("ticker", ""))
                .strip()
                .lower()
                .removeprefix("sh")
                .removeprefix("sz")
                .removeprefix("bj")
                .zfill(6)
            )
            shares = int(item.get("shares", 0) or 0)
            if not ticker or shares <= 0:
                continue
            previous = previous_positions.get(ticker)
            positions.append(
                Position(
                    ticker=ticker,
                    asset_type=account.config.asset_type,
                    shares=shares,
                    avg_cost=float(item.get("avg_cost", 0) or 0),
                    current_price=float(item.get("current_price", 0) or 0),
                    available_shares=max(0, min(shares, int(item.get("available_shares", shares) or 0))),
                    frozen_shares=max(0, min(shares, int(item.get("frozen_shares", 0) or 0))),
                    stop_loss=previous.stop_loss if previous else None,
                    take_profit=previous.take_profit if previous else None,
                )
            )

        existing_external_ids = {trade.external_id for trade in account.portfolio.trades if trade.external_id}
        for item in snapshot.get("trades", []):
            external_id = str(item.get("external_id", "")).strip()
            shares = int(item.get("shares", 0) or 0)
            price = float(item.get("price", 0) or 0)
            if not external_id or external_id in existing_external_ids or shares <= 0 or price <= 0:
                continue
            account.portfolio.trades.append(
                TradeRecord(
                    date=str(item.get("date") or snapshot.get("as_of") or account.current_date),
                    action=Decision(str(item.get("action", "buy"))),
                    ticker=(
                        str(item.get("ticker", ""))
                        .strip()
                        .lower()
                        .removeprefix("sh")
                        .removeprefix("sz")
                        .removeprefix("bj")
                        .zfill(6)
                    ),
                    asset_type=account.config.asset_type,
                    shares=shares,
                    price=price,
                    amount=float(item.get("amount", 0) or shares * price),
                    external_id=external_id,
                )
            )
            existing_external_ids.add(external_id)

        account.portfolio.cash = cash
        account.portfolio.positions = positions
        snapshot_date = str(snapshot.get("as_of") or "")[:10]
        if snapshot_date:
            account.current_date = snapshot_date

        with self._lock, self._connect() as connection:
            self._reconcile_external_orders(snapshot.get("orders", []), connection)
            self._save_account(account, connection)
            if snapshot_date:
                synced_snapshot = SimulationSnapshot(
                    account_id=account_id,
                    date=snapshot_date,
                    cash=account.portfolio.cash,
                    total_value=account.portfolio.total_value,
                    total_pnl=account.portfolio.total_pnl,
                    total_return_pct=account.portfolio.total_return_pct,
                    positions=account.portfolio.positions,
                )
                connection.execute(
                    "INSERT OR REPLACE INTO simulation_snapshots "
                    "(account_id, snapshot_date, snapshot_json, created_at) VALUES (?, ?, ?, ?)",
                    (account_id, snapshot_date, _json(synced_snapshot), _now()),
                )
        return account

    @staticmethod
    def _reconcile_external_orders(orders: list[dict], connection: sqlite3.Connection) -> None:
        """Apply Eastmoney status rows to local orders identified by ``sid``."""
        for remote in orders:
            sid = str(remote.get("sid", "")).strip()
            if not sid:
                continue
            row = connection.execute(
                "SELECT * FROM simulation_orders WHERE order_id = ?",
                (sid,),
            ).fetchone()
            if row is None:
                continue
            order = SimulationOrder.model_validate(json.loads(row["order_json"]))
            status = remote.get("status")
            if status in {"pending", "cancelled", "rejected", "filled"}:
                order.status = status
            if status == "filled":
                order.fill_price = float(remote.get("fill_price") or 0) or order.fill_price
                order.fill_date = str(remote.get("fill_date") or "")[:10] or order.fill_date
            if remote.get("reject_reason"):
                order.reject_reason = str(remote["reject_reason"])
            connection.execute(
                "UPDATE simulation_orders SET status = ?, order_json = ?, updated_at = ? WHERE order_id = ?",
                (order.status, _json(order), _now(), sid),
            )

    def reset_account(self, account_id: str = "default") -> SimulationAccount:
        account = self.get_account(account_id)
        account.status = "active"
        account.current_date = ""
        account.portfolio = PortfolioState(
            cash=account.config.initial_cash,
            initial_capital=account.config.initial_cash,
        )
        with self._lock, self._connect() as connection:
            self._save_account(account, connection)
            connection.execute("DELETE FROM simulation_orders WHERE account_id = ?", (account_id,))
            connection.execute("DELETE FROM simulation_snapshots WHERE account_id = ?", (account_id,))
        logger.info(f"Reset simulation account: {account_id}")
        return account

    def set_status(self, account_id: str, status: str) -> SimulationAccount:
        if status not in {"active", "paused"}:
            raise ValueError("status 必须是 active 或 paused")
        account = self.get_account(account_id)
        account.status = status
        with self._lock, self._connect() as connection:
            self._save_account(account, connection)
        return account

    def advance_date(self, account_id: str, current_date: str) -> SimulationAccount:
        account = self.get_account(account_id)
        engine = TradingEngine(
            account.portfolio.initial_capital,
            account.config,
            current_date=account.current_date,
        )
        engine.portfolio = account.portfolio
        engine.set_date(current_date)
        account.portfolio = engine.portfolio
        account.current_date = current_date
        with self._lock, self._connect() as connection:
            self._save_account(account, connection)
        return account

    def mark_to_market(self, account_id: str, prices: dict[str, float], snapshot_date: str) -> SimulationAccount:
        account = self.get_account(account_id)
        engine = TradingEngine(
            account.portfolio.initial_capital,
            account.config,
            current_date=account.current_date,
        )
        engine.portfolio = account.portfolio
        engine.set_date(snapshot_date)
        engine.update_prices({ticker: float(price) for ticker, price in prices.items()})
        account.portfolio = engine.portfolio
        account.current_date = snapshot_date
        snapshot = SimulationSnapshot(
            account_id=account_id,
            date=snapshot_date,
            cash=account.portfolio.cash,
            total_value=account.portfolio.total_value,
            total_pnl=account.portfolio.total_pnl,
            total_return_pct=account.portfolio.total_return_pct,
            positions=account.portfolio.positions,
        )
        with self._lock, self._connect() as connection:
            self._save_account(account, connection)
            connection.execute(
                """INSERT OR REPLACE INTO simulation_snapshots
                   (account_id, snapshot_date, snapshot_json, created_at) VALUES (?, ?, ?, ?)""",
                (account_id, snapshot_date, _json(snapshot), _now()),
            )
        return account

    def create_order(
        self,
        account_id: str,
        ticker: str,
        side: Decision,
        shares: int,
        order_type: str = "market",
        limit_price: float | None = None,
        submitted_date: str = "",
        source: str = "manual",
        run_id: str | None = None,
        fill_policy: str | None = None,
        asset_type: AssetType | str | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> SimulationOrder:
        account = self.get_account(account_id)
        if account.status != "active":
            raise ValueError("模拟账户已暂停")
        ticker = ticker.strip().lower().removeprefix("sh").removeprefix("sz").removeprefix("bj").zfill(6)
        if side == Decision.HOLD:
            raise ValueError("订单 side 只能是 buy 或 sell")
        if shares <= 0:
            raise ValueError("shares 必须大于 0")
        if order_type not in {"market", "limit"}:
            raise ValueError("order_type 必须是 market 或 limit")
        if source not in {"manual", "agent", "backtest", "system"}:
            raise ValueError("source 必须是 manual、agent、backtest 或 system")
        effective_fill_policy = fill_policy or ("manual" if source == "manual" else "next_open")
        if effective_fill_policy not in {"next_open", "same_close", "manual"}:
            raise ValueError("fill_policy 必须是 next_open、same_close 或 manual")
        if order_type == "limit" and (limit_price is None or limit_price <= 0):
            raise ValueError("限价单必须提供正数 limit_price")
        order = SimulationOrder(
            order_id=f"sim-{uuid4().hex[:16]}",
            account_id=account_id,
            ticker=ticker,
            asset_type=AssetType(asset_type or account.config.asset_type),
            side=side,
            shares=shares,
            order_type=order_type,
            limit_price=limit_price,
            submitted_date=submitted_date or account.current_date,
            source=source,
            run_id=run_id,
            fill_policy=effective_fill_policy,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        with self._lock, self._connect() as connection:
            timestamp = _now()
            connection.execute(
                """INSERT INTO simulation_orders
                   (order_id, account_id, status, order_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (order.order_id, account_id, order.status, _json(order), timestamp, timestamp),
            )
        return order

    def fill_order(self, order_id: str, fill_price: float, fill_date: str) -> SimulationOrder:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM simulation_orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"订单不存在: {order_id}")
            order = SimulationOrder.model_validate(json.loads(row["order_json"]))
            if order.status != "pending":
                raise ValueError(f"订单当前状态为 {order.status}，不能成交")
            if fill_price <= 0:
                raise ValueError("fill_price 必须大于 0")
            if order.order_type == "limit":
                if order.side == Decision.BUY and fill_price > (order.limit_price or 0):
                    raise ValueError("买入限价单成交价高于限价")
                if order.side == Decision.SELL and fill_price < (order.limit_price or 0):
                    raise ValueError("卖出限价单成交价低于限价")

            account = self.get_account(order.account_id)
            risk_reason = self._risk_rejection_reason(account, order, fill_price)
            if risk_reason:
                order.status = "rejected"
                order.reject_reason = risk_reason
                connection.execute(
                    "UPDATE simulation_orders SET status = ?, order_json = ?, updated_at = ? WHERE order_id = ?",
                    (order.status, _json(order), _now(), order_id),
                )
                return order

            execution_price = fill_price
            if order.order_type == "market" and account.config.slippage_bps:
                slippage = account.config.slippage_bps / 10_000
                execution_price = fill_price * (1 + slippage if order.side == Decision.BUY else 1 - slippage)
            # TradingEngine also supports slippage for standalone backtests;
            # this service applies it above so limit validation and stored fill
            # prices remain explicit and it is not applied twice.
            execution_rules = account.config.model_copy(
                update={"slippage_bps": 0.0, "asset_type": order.asset_type}
            )
            engine = TradingEngine(
                account.portfolio.initial_capital,
                execution_rules,
                current_date=account.current_date,
            )
            engine.portfolio = account.portfolio
            engine.set_date(fill_date)
            engine.update_prices({order.ticker: execution_price})
            if order.side == Decision.BUY:
                trade = engine.buy(
                    order.ticker,
                    order.shares,
                    execution_price,
                    fill_date,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                )
            else:
                trade = engine.sell(order.ticker, order.shares, execution_price, fill_date)

            if trade is None:
                order.status = "rejected"
                order.reject_reason = "账户资金、持仓或交易规则不满足"
            else:
                order.status = "filled"
                order.fill_date = fill_date
                order.fill_price = trade.price
                account.portfolio = engine.portfolio
                account.current_date = fill_date
                self._save_account(account, connection)

            order_json = _json(order)
            connection.execute(
                "UPDATE simulation_orders SET status = ?, order_json = ?, updated_at = ? WHERE order_id = ?",
                (order.status, order_json, _now(), order_id),
            )
        return order

    @staticmethod
    def _risk_rejection_reason(
        account: SimulationAccount,
        order: SimulationOrder,
        price: float,
    ) -> str | None:
        """Apply account-level position limits before the engine mutates state."""
        if order.side != Decision.BUY:
            return None
        portfolio = account.portfolio
        total_value = portfolio.total_value or portfolio.cash
        if total_value <= 0:
            return "账户总资产必须大于 0"
        position = next((item for item in portfolio.positions if item.ticker == order.ticker), None)
        existing_value = 0.0
        if position:
            reference_price = position.current_price or position.avg_cost
            existing_value = position.shares * reference_price
        proposed_value = existing_value + order.shares * price
        if proposed_value > total_value * account.config.max_single_position_pct:
            return "超过单股最大仓位限制"
        invested_value = sum(
            item.shares * (item.current_price or item.avg_cost) for item in portfolio.positions
        )
        if invested_value + order.shares * price > total_value * account.config.max_total_position_pct:
            return "超过组合最大仓位限制"
        return None

    def cancel_order(self, order_id: str) -> SimulationOrder:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM simulation_orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"订单不存在: {order_id}")
            order = SimulationOrder.model_validate(json.loads(row["order_json"]))
            if order.status == "pending":
                order.status = "cancelled"
                connection.execute(
                    "UPDATE simulation_orders SET status = ?, order_json = ?, updated_at = ? WHERE order_id = ?",
                    (order.status, _json(order), _now(), order_id),
                )
        return order

    def list_orders(self, account_id: str = "default") -> list[SimulationOrder]:
        self.get_account(account_id)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT order_json FROM simulation_orders WHERE account_id = ? ORDER BY created_at DESC",
                (account_id,),
            ).fetchall()
        return [SimulationOrder.model_validate(json.loads(row["order_json"])) for row in rows]

    def list_snapshots(self, account_id: str = "default", limit: int = 100) -> list[SimulationSnapshot]:
        self.get_account(account_id)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT snapshot_json FROM simulation_snapshots
                   WHERE account_id = ? ORDER BY snapshot_date DESC LIMIT ?""",
                (account_id, limit),
            ).fetchall()
        return [SimulationSnapshot.model_validate(json.loads(row["snapshot_json"])) for row in rows]

    def daily_pnl(self, account_id: str = "default") -> float:
        """Return change from the previous stored mark-to-market snapshot."""
        account = self.get_account(account_id)
        snapshots = self.list_snapshots(account_id, limit=2)
        if len(snapshots) < 2:
            return account.portfolio.total_pnl
        return snapshots[0].total_value - snapshots[1].total_value


simulation_accounts = SimulationAccountService()
