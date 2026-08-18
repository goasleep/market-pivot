"""Unattended daily Agent orchestration for simulation accounts."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time
from uuid import uuid4
from zoneinfo import ZoneInfo

from loguru import logger

from application.automation_store import automation_store
from application.research import research_service
from config import settings
from data.stock_provider import async_get_stock_history, async_get_stock_realtime
from data.trading_calendar import is_trading_day
from engine.broker_adapters import (
    LiveBrokerUnavailableError,
    SimulationBrokerUnavailableError,
    get_live_broker,
    get_simulation_broker,
)
from engine.simulation_account import simulation_accounts
from engine.simulation_events import simulation_events
from engine.trading_engine import decision_shares
from models.schemas import AgentRunSummary, AutomationTaskConfig, Decision, LiveOrderIntent, TradeDecision

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _normalise_ticker(ticker: str) -> str:
    return ticker.strip().lower().removeprefix("sh").removeprefix("sz").removeprefix("bj").zfill(6)


def _today() -> str:
    return datetime.now(SHANGHAI).date().isoformat()


class AutomationService:
    """Run Agent decisions and route them into paper or live adapters."""

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, account_id: str) -> asyncio.Lock:
        if account_id not in self._locks:
            self._locks[account_id] = asyncio.Lock()
        return self._locks[account_id]

    def get_task_payload(self, account_id: str) -> dict:
        simulation_accounts.get_account(account_id)
        task = automation_store.get_task(account_id)
        return {
            "account_id": account_id,
            "config": task["config"].model_dump(mode="json"),
            "status": task["status"],
            "last_run_id": task["last_run_id"],
            "last_run_date": task["last_run_date"],
            "last_error": task["last_error"],
            "created_at": task["created_at"],
            "updated_at": task["updated_at"],
        }

    async def update_task(self, account_id: str, config: AutomationTaskConfig) -> dict:
        simulation_accounts.get_account(account_id)
        await asyncio.to_thread(automation_store.update_task, account_id, config=config)
        await simulation_events.publish(account_id, "automation.updated", self.get_task_payload(account_id))
        return self.get_task_payload(account_id)

    async def run_account(
        self,
        account_id: str,
        trigger: str = "manual",
        run_date: str | None = None,
    ) -> AgentRunSummary:
        """Run one idempotent daily cycle for an account.

        A scheduled date can only produce one run.  Repeated scheduler ticks
        therefore return the existing run instead of re-running Agents or
        creating duplicate orders.
        """

        async with self._lock_for(account_id):
            account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
            task = await asyncio.to_thread(automation_store.get_task, account_id)
            config: AutomationTaskConfig = task["config"]
            effective_date = run_date or _today()
            if account.status != "active":
                return await self._skipped_run(account_id, effective_date, trigger, config, "模拟账户已暂停")
            universe = [_normalise_ticker(ticker) for ticker in (config.universe or account.config.universe)]
            universe = list(dict.fromkeys(ticker for ticker in universe if ticker.isdigit()))[
                : config.max_symbols_per_run
            ]
            if not universe:
                return await self._skipped_run(account_id, effective_date, trigger, config, "股票池为空")

            idempotency_key = f"{account_id}:{effective_date}:{'schedule' if trigger == 'schedule' else trigger}"
            summary = await asyncio.to_thread(
                automation_store.create_run,
                account_id,
                effective_date,
                trigger,
                config,
                idempotency_key,
            )
            if summary.status not in {"queued"}:
                return summary
            claimed = await asyncio.to_thread(
                automation_store.claim_run,
                summary.run_id,
                symbols_total=len(universe),
                started_at=datetime.now(SHANGHAI).isoformat(),
            )
            if claimed is None:
                return await asyncio.to_thread(automation_store.get_run, summary.run_id)
            await asyncio.to_thread(automation_store.update_task, account_id, status="running")
            await simulation_events.publish(
                account_id,
                "agent.run.started",
                {"run_id": summary.run_id, "run_date": effective_date, "symbols": universe},
            )

            decisions_count = 0
            orders_count = 0
            failures: list[str] = []
            for index, ticker in enumerate(universe, start=1):
                try:
                    if config.asset_type.value == "stock":
                        result = await research_service.run(ticker, config.strategy_name)
                    else:
                        result = await research_service.run(
                            ticker,
                            config.strategy_name,
                            asset_type=config.asset_type,
                        )
                    decision = result.get("final_decision")
                    if not isinstance(decision, TradeDecision):
                        raise ValueError("Agent 未返回有效 TradeDecision")
                    current_price = float(result.get("current_price") or 0.0)
                    if current_price <= 0:
                        context = result.get("market_context")
                        current_price = float(getattr(context, "current_price", 0.0) or 0.0)
                    freshness_error = self._data_freshness_error(
                        result.get("market_context"), effective_date, config.data_max_age_seconds
                    )
                    if freshness_error:
                        raise ValueError(freshness_error)
                    decisions_count += 1

                    risk_status, risk_reason, order_id = await self._execute_decision(
                        account_id,
                        summary.run_id,
                        decision,
                        current_price,
                        effective_date,
                        config,
                        orders_count,
                        False,
                    )
                    orders_count += int(order_id is not None)
                    audit = await asyncio.to_thread(
                        automation_store.add_decision,
                        summary.run_id,
                        account_id,
                        decision,
                        current_price,
                        risk_status,
                        risk_reason,
                        order_id,
                    )
                    await asyncio.to_thread(
                        automation_store.add_event,
                        account_id,
                        "agent.decision",
                        audit.model_dump(mode="json"),
                        summary.run_id,
                    )
                    await simulation_events.publish(
                        account_id,
                        "agent.decision",
                        audit.model_dump(mode="json"),
                    )
                except Exception as exc:  # one bad symbol must not stop the pool
                    failures.append(f"{ticker}: {exc}")
                    logger.exception("Agent cycle failed for {}", ticker)
                await asyncio.to_thread(
                    automation_store.update_run,
                    summary.run_id,
                    symbols_processed=index,
                    decisions_count=decisions_count,
                    orders_count=orders_count,
                )

            error = "; ".join(failures) if failures else None
            final = await asyncio.to_thread(
                automation_store.update_run,
                summary.run_id,
                status="completed" if decisions_count or not failures else "failed",
                symbols_processed=len(universe),
                decisions_count=decisions_count,
                orders_count=orders_count,
                completed_at=datetime.now(SHANGHAI).isoformat(),
                error=error,
            )
            await simulation_events.publish(
                account_id,
                "agent.run.completed" if final.status == "completed" else "agent.run.failed",
                final.model_dump(mode="json"),
            )
            return final

    @staticmethod
    def _data_freshness_error(context, effective_date: str, max_age_seconds: int) -> str | None:
        if context is None or not max_age_seconds:
            return None
        history = getattr(context, "history", []) or []
        if not history:
            return "没有可用的历史行情数据"
        dates = [str(item.get("date", ""))[:10] for item in history if item.get("date")]
        if not dates:
            return "行情数据缺少日期，无法进行新鲜度校验"
        try:
            latest = max(date.fromisoformat(value) for value in dates)
            age_seconds = max(0, (date.fromisoformat(effective_date) - latest).days) * 86400
        except ValueError:
            return "行情日期格式无效，无法进行新鲜度校验"
        if age_seconds > max_age_seconds:
            return f"行情数据已过期（{age_seconds // 86400} 天），超过允许的 {max_age_seconds} 秒"
        return None

    async def _skipped_run(
        self,
        account_id: str,
        run_date: str,
        trigger: str,
        config: AutomationTaskConfig,
        reason: str,
    ) -> AgentRunSummary:
        key = f"{account_id}:{run_date}:{trigger}:skipped:{reason}"
        summary = await asyncio.to_thread(
            automation_store.create_run,
            account_id,
            run_date,
            trigger,
            config,
            key,
        )
        if summary.status == "queued":
            summary = await asyncio.to_thread(
                automation_store.update_run,
                summary.run_id,
                status="skipped",
                completed_at=datetime.now(SHANGHAI).isoformat(),
                error=reason,
            )
        return summary

    async def _execute_decision(
        self,
        account_id: str,
        run_id: str,
        decision: TradeDecision,
        price: float,
        trade_date: str,
        config: AutomationTaskConfig,
        orders_count: int,
        force: bool = False,
    ) -> tuple[str, str | None, str | None]:
        """Apply mode/risk gates and route to paper or live execution."""

        if config.execution_mode == "live":
            return await self._execute_live_decision(
                account_id,
                run_id,
                decision,
                price,
                trade_date,
                config,
                orders_count,
                force,
            )

        if config.mode != "auto" and not force:
            return "approved", "仅记录决策，自动下单未启用", None
        if orders_count >= config.max_orders_per_run:
            return "rejected", "达到本次 Agent 运行的最大订单数", None
        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        if config.daily_loss_limit_pct:
            daily_pnl = await asyncio.to_thread(simulation_accounts.daily_pnl, account_id)
            if daily_pnl <= -account.portfolio.initial_capital * config.daily_loss_limit_pct:
                return "rejected", "触发单日亏损限额", None
        if decision.decision == Decision.HOLD:
            return "approved", "Agent 建议持有，不生成订单", None
        shares = decision_shares(account.portfolio, account.config, decision, price)
        if shares <= 0:
            return "rejected", "按仓位和 A 股最小交易单位计算后无可交易数量", None
        order = await asyncio.to_thread(
            simulation_accounts.create_order,
            account_id,
            decision.ticker,
            decision.decision,
            shares,
            "market",
            None,
            trade_date,
            "agent",
            run_id,
            config.fill_time,
            asset_type=decision.asset_type,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
        )
        if account.config.external.enabled and account.config.external.provider != "internal":
            try:
                broker = get_simulation_broker(account.config.external)
                await asyncio.to_thread(broker.submit_order, order)
            except SimulationBrokerUnavailableError as exc:
                await asyncio.to_thread(simulation_accounts.cancel_order, order.order_id)
                return "rejected", str(exc), order.order_id
            return "approved", "订单已写入东方财富文件单，等待终端回报", order.order_id
        if config.fill_time == "same_close":
            order = await asyncio.to_thread(simulation_accounts.fill_order, order.order_id, price, trade_date)
        return (
            "approved" if order.status in {"pending", "filled"} else "rejected",
            order.reject_reason,
            order.order_id,
        )

    async def _execute_live_decision(
        self,
        account_id: str,
        run_id: str,
        decision: TradeDecision,
        price: float,
        trade_date: str,
        config: AutomationTaskConfig,
        orders_count: int,
        force: bool,
    ) -> tuple[str, str | None, str | None]:
        """Submit a validated intent to a live broker sidecar.

        The local simulation account is never mutated by a live submission.
        The service flag, task arming, account configuration, and (by default)
        explicit confirmation are all required before a request leaves the
        process.
        """
        if config.mode != "auto" and not force:
            return "approved", "实盘决策已记录，等待人工确认", None
        if not settings.live_trading_enabled:
            return "rejected", "服务端 LIVE_TRADING_ENABLED 未开启", None
        if not config.live_armed:
            return "rejected", "该自动化任务尚未 armed 实盘执行", None
        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        live_config = account.config.live
        if not live_config.enabled:
            return "rejected", "账户未启用实盘 Adapter", None
        if live_config.require_manual_approval and not force:
            return "rejected", "该实盘账户要求人工确认", None
        if orders_count >= config.max_orders_per_run:
            return "rejected", "达到本次 Agent 运行的最大订单数", None
        if config.daily_loss_limit_pct:
            daily_pnl = await asyncio.to_thread(simulation_accounts.daily_pnl, account_id)
            if daily_pnl <= -account.portfolio.initial_capital * config.daily_loss_limit_pct:
                return "rejected", "触发单日亏损限额", None
        if decision.decision == Decision.HOLD:
            return "approved", "Agent 建议持有，不生成实盘订单", None
        if price <= 0:
            return "rejected", "实盘订单缺少有效参考价格", None

        shares = decision_shares(account.portfolio, account.config, decision, price)
        if shares <= 0:
            return "rejected", "按仓位和 A 股最小交易单位计算后无可交易数量", None
        order_value = shares * price
        if live_config.max_order_value and order_value > live_config.max_order_value:
            return "rejected", "超过实盘账户单笔金额上限", None

        client_order_id = f"live-{uuid4().hex[:16]}"
        intent = LiveOrderIntent(
            client_order_id=client_order_id,
            account_id=live_config.account_id or account_id,
            ticker=decision.ticker,
            asset_type=decision.asset_type,
            side=decision.decision,
            shares=shares,
            order_type="market",
            submitted_date=trade_date,
            fill_policy=config.fill_time,
        )
        try:
            broker = get_live_broker(live_config)
            result = await asyncio.to_thread(broker.submit_order, intent)
        except LiveBrokerUnavailableError as exc:
            return "rejected", str(exc), None
        except Exception as exc:  # A network failure may occur after broker acceptance.
            logger.exception("Live broker submission outcome is unknown for {}", client_order_id)
            return "pending", f"实盘提交结果未知，需要对账：{exc}", client_order_id

        order_id = result.broker_order_id or result.client_order_id
        if result.status == "rejected":
            return "rejected", result.message or "券商拒绝订单", order_id
        if result.status == "unknown":
            return "pending", result.message or "券商返回未知状态，需要对账", order_id
        try:
            snapshot = await asyncio.to_thread(broker.sync)
            self._apply_live_snapshot(account_id, snapshot)
        except Exception:
            logger.warning("Live broker accepted {} but snapshot sync is unavailable", order_id)
        return "approved", result.message or f"实盘订单已提交：{result.status}", order_id

    @staticmethod
    def _apply_live_snapshot(account_id: str, payload: dict) -> bool:
        """Apply only an explicit cash/positions snapshot from the broker."""
        if not isinstance(payload, dict) or "cash" not in payload or "positions" not in payload:
            return False
        simulation_accounts.apply_live_snapshot(
            account_id,
            float(payload["cash"]),
            list(payload["positions"]),
            payload.get("as_of") or payload.get("date"),
        )
        return True

    async def confirm_decision(
        self,
        account_id: str,
        decision_id: str,
        price: float | None = None,
    ):
        audit = await asyncio.to_thread(automation_store.get_decision, decision_id)
        if audit.account_id != account_id:
            raise KeyError("Agent decision 不属于该模拟账户")
        if audit.risk_status == "rejected":
            raise ValueError("该 Agent 决策已被风控拦截，不能确认")
        if audit.order_id:
            return audit
        task = await asyncio.to_thread(automation_store.get_task, account_id)
        config: AutomationTaskConfig = task["config"]
        fill_price = price or audit.current_price
        if fill_price <= 0:
            quote = await async_get_stock_realtime(audit.ticker)
            fill_price = float(quote.get("price", 0) or 0)
        if fill_price <= 0:
            raise ValueError("无法取得可执行价格")
        risk_status, risk_reason, order_id = await self._execute_decision(
            account_id,
            audit.run_id,
            audit.decision,
            fill_price,
            _today(),
            config,
            0,
            True,
        )
        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        uses_external = account.config.external.enabled and account.config.external.provider != "internal"
        if order_id and config.execution_mode != "live" and not uses_external and config.fill_time == "manual":
            filled_order = await asyncio.to_thread(
                simulation_accounts.fill_order,
                order_id,
                fill_price,
                _today(),
            )
            if filled_order.status != "filled":
                risk_status = "rejected"
                risk_reason = filled_order.reject_reason or "手动确认订单未成交"
        updated = await asyncio.to_thread(
            automation_store.update_decision,
            decision_id,
            current_price=fill_price,
            risk_status=risk_status,
            risk_reason=risk_reason,
            order_id=order_id,
        )
        await simulation_events.publish(account_id, "agent.decision.confirmed", updated.model_dump(mode="json"))
        return updated

    async def settle_account(
        self,
        account_id: str,
        settlement_date: str | None = None,
        prices: dict[str, float] | None = None,
        open_prices: dict[str, float] | None = None,
    ) -> dict:
        task = await asyncio.to_thread(automation_store.get_task, account_id)
        if task["config"].execution_mode == "live":
            return await self.sync_live_account(account_id)
        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        if account.config.external.enabled and account.config.external.provider != "internal":
            return await self.sync_external_account(account_id)
        async with self._lock_for(account_id):
            return await self._settle_account(account_id, settlement_date, prices, open_prices)

    async def sync_external_account(self, account_id: str) -> dict:
        """Read an external paper account and mirror it into the local UI model."""
        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        try:
            broker = get_simulation_broker(account.config.external)
            snapshot = await asyncio.to_thread(broker.sync)
            await asyncio.to_thread(simulation_accounts.apply_external_snapshot, account_id, snapshot)
        except SimulationBrokerUnavailableError as exc:
            raise ValueError(str(exc)) from exc
        payload = {
            "account_id": account_id,
            "mode": "external_simulation",
            "provider": account.config.external.provider,
            "as_of": snapshot.get("as_of"),
            "positions": len(snapshot.get("positions", [])),
            "orders": len(snapshot.get("orders", [])),
            "trades": len(snapshot.get("trades", [])),
        }
        await asyncio.to_thread(automation_store.add_event, account_id, "external.sync", payload)
        await simulation_events.publish(account_id, "external.sync", payload)
        return {**payload, "sync": snapshot}

    async def sync_live_account(self, account_id: str) -> dict:
        """Reconcile live broker state without applying paper fills locally."""
        task = await asyncio.to_thread(automation_store.get_task, account_id)
        config: AutomationTaskConfig = task["config"]
        if config.execution_mode != "live":
            raise ValueError("该账户当前不是实盘执行模式")
        if not settings.live_trading_enabled:
            raise ValueError("服务端 LIVE_TRADING_ENABLED 未开启")
        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        try:
            broker = get_live_broker(account.config.live)
            payload = await asyncio.to_thread(broker.sync)
        except LiveBrokerUnavailableError as exc:
            raise ValueError(str(exc)) from exc
        mirrored = await asyncio.to_thread(self._apply_live_snapshot, account_id, payload)
        payload = {**payload, "portfolio_mirrored": mirrored}
        await asyncio.to_thread(automation_store.add_event, account_id, "live.sync", payload)
        await simulation_events.publish(account_id, "live.sync", payload)
        return {"account_id": account_id, "mode": "live", "sync": payload}

    async def _settle_account(
        self,
        account_id: str,
        settlement_date: str | None = None,
        prices: dict[str, float] | None = None,
        open_prices: dict[str, float] | None = None,
    ) -> dict:
        """Fill next-open orders and persist a daily mark-to-market snapshot."""

        account = await asyncio.to_thread(simulation_accounts.get_account, account_id)
        target_date = settlement_date or _today()
        price_map = {key: float(value) for key, value in (prices or {}).items() if float(value) > 0}
        symbols = {position.ticker for position in account.portfolio.positions}
        pending_orders = await asyncio.to_thread(simulation_accounts.list_orders, account_id)
        symbols.update(order.ticker for order in pending_orders if order.status == "pending")
        open_price_map: dict[str, float] = dict(open_prices or {})
        missing_symbols = sorted(symbols - price_map.keys())
        if missing_symbols:
            daily_rows = await asyncio.gather(
                *(async_get_stock_history(symbol, end_date=target_date) for symbol in missing_symbols)
            )
            for symbol, frame in zip(missing_symbols, daily_rows):
                if frame.empty:
                    continue
                row = frame.iloc[-1]
                close = float(row.get("close", 0) or 0)
                opening = float(row.get("open", 0) or 0)
                if close > 0:
                    price_map[symbol] = close
                if opening > 0:
                    open_price_map.setdefault(symbol, opening)
        missing_symbols = sorted(symbols - price_map.keys())
        if missing_symbols:
            quotes = await asyncio.gather(*(async_get_stock_realtime(symbol) for symbol in missing_symbols))
            price_map.update(
                {
                    symbol: float(quote.get("price", 0) or 0)
                    for symbol, quote in zip(missing_symbols, quotes)
                    if quote.get("price")
                }
            )
        await asyncio.to_thread(simulation_accounts.advance_date, account_id, target_date)
        fill_map = {**price_map, **open_price_map}
        filled = []
        for order in pending_orders:
            if order.status != "pending" or order.fill_policy != "next_open" or order.ticker not in fill_map:
                continue
            filled.append(
                await asyncio.to_thread(
                    simulation_accounts.fill_order,
                    order.order_id,
                    float(fill_map[order.ticker]),
                    target_date,
                )
            )
        account = await asyncio.to_thread(simulation_accounts.mark_to_market, account_id, price_map, target_date)
        payload = {
            "account_id": account_id,
            "date": target_date,
            "filled_orders": [order.model_dump(mode="json") for order in filled],
            "total_value": account.portfolio.total_value,
            "daily_pnl": await asyncio.to_thread(simulation_accounts.daily_pnl, account_id),
        }
        await asyncio.to_thread(automation_store.add_event, account_id, "daily.settled", payload)
        await simulation_events.publish(account_id, "daily.settled", payload)
        return payload


automation_service = AutomationService()


class AutomationScheduler:
    """Small persistent-aware polling scheduler for daily tasks."""

    def __init__(self, service: AutomationService):
        self.service = service
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        await asyncio.to_thread(automation_store.recover_stale_runs)
        self._stopping = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="agent-automation-scheduler")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task:
            await self._task
        self._task = None

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.tick()
            except Exception:
                logger.exception("Automation scheduler tick failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=30)
            except asyncio.TimeoutError:
                continue

    async def tick(self, now: datetime | None = None) -> None:
        current = now or datetime.now(SHANGHAI)
        for account in await asyncio.to_thread(simulation_accounts.list_accounts):
            if account.status != "active":
                continue
            task = await asyncio.to_thread(automation_store.get_task, account.account_id)
            config: AutomationTaskConfig = task["config"]
            if not config.enabled or current.weekday() not in config.weekdays:
                continue
            if not await asyncio.to_thread(is_trading_day, current.date()):
                continue
            try:
                scheduled = time.fromisoformat(config.schedule_time)
            except ValueError:
                logger.warning("Invalid automation schedule for {}: {}", account.account_id, config.schedule_time)
                continue
            scheduled_at = datetime.combine(current.date(), scheduled, tzinfo=SHANGHAI)
            if current < scheduled_at:
                continue
            if task["last_run_date"] == current.date().isoformat():
                continue
            await self.service.settle_account(account.account_id, current.date().isoformat())
            await self.service.run_account(account.account_id, trigger="schedule", run_date=current.date().isoformat())


automation_scheduler = AutomationScheduler(automation_service)
