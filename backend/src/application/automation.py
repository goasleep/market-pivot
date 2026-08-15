"""Unattended daily Agent orchestration for simulation accounts."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from loguru import logger

from application.automation_store import automation_store
from application.research import research_service
from data.akshare_provider import async_get_stock_history, async_get_stock_realtime
from data.trading_calendar import is_trading_day
from engine.simulation_account import simulation_accounts
from engine.simulation_events import simulation_events
from engine.trading_engine import decision_shares
from models.schemas import AgentRunSummary, AutomationTaskConfig, Decision, TradeDecision

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _normalise_ticker(ticker: str) -> str:
    return ticker.strip().lower().removeprefix("sh").removeprefix("sz").removeprefix("bj").zfill(6)


def _today() -> str:
    return datetime.now(SHANGHAI).date().isoformat()


class AutomationService:
    """Run Agent decisions and route approved intents into paper orders."""

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
        """Apply mode/risk gates and optionally create a simulation order."""

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
        )
        if config.fill_time == "same_close":
            order = await asyncio.to_thread(simulation_accounts.fill_order, order.order_id, price, trade_date)
        return (
            "approved" if order.status in {"pending", "filled"} else "rejected",
            order.reject_reason,
            order.order_id,
        )

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
        if order_id and config.fill_time == "manual":
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
        async with self._lock_for(account_id):
            return await self._settle_account(account_id, settlement_date, prices, open_prices)

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
