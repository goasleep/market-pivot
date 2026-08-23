"""Unattended daily Agent orchestration for simulation accounts."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import date, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from loguru import logger

from application.automation_scheduler import AutomationScheduler
from application.automation_store import automation_store
from application.deployments import deployment_service
from application.research import research_service
from application.strategy_state import strategy_runtime_states
from config import get_llm_config, settings
from data.backtest_data import prepare_backtest_data
from data.fund_provider import async_get_fund_history
from data.market_context import build_market_context
from data.stock_provider import async_get_stock_history, async_get_stock_realtime
from engine.broker_adapters import (
    LiveBrokerUnavailableError,
    SimulationBrokerUnavailableError,
    get_live_broker,
    get_simulation_broker,
)
from engine.simulation_account import simulation_accounts
from engine.simulation_events import simulation_events
from engine.strategy_runtime import (
    decision_from_intent,
    evaluate_strategy_intent,
    normalize_target_exposures,
    plan_rebalance,
)
from engine.trading_engine import decision_shares
from models.schemas import AgentRunSummary, AutomationTaskConfig, Decision, LiveOrderIntent, TradeDecision
from strategies.plugin_registry import strategy_plugins_manifest

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _normalise_ticker(ticker: str) -> str:
    return ticker.strip().lower().removeprefix("sh").removeprefix("sz").removeprefix("bj").zfill(6)


def _today() -> str:
    return datetime.now(SHANGHAI).date().isoformat()


class AutomationService:
    """Run Agent decisions and route them into paper or live adapters."""

    def __init__(self, *, strategy_states=None):
        self._locks: dict[str, asyncio.Lock] = {}
        self.strategy_states = strategy_states or strategy_runtime_states

    def _lock_for(self, account_id: str) -> asyncio.Lock:
        if account_id not in self._locks:
            self._locks[account_id] = asyncio.Lock()
        return self._locks[account_id]

    async def get_task_payload(self, account_id: str) -> dict:
        await simulation_accounts.get_account(account_id)
        task = await automation_store.get_task(account_id)
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
        await simulation_accounts.get_account(account_id)
        current = (await automation_store.get_task(account_id))["config"]
        if current.deployment_id:
            protected = (
                "deployment_id",
                "universe",
                "asset_type",
                "strategy_name",
                "fill_time",
                "execution_mode",
                "simulation_only",
            )
            if any(getattr(config, field) != getattr(current, field) for field in protected):
                raise ValueError("已部署策略的版本、标的池和执行参数不可在自动化配置中修改")
        await automation_store.update_task(account_id, config=config)
        payload = await self.get_task_payload(account_id)
        await simulation_events.publish(account_id, "automation.updated", payload)
        return payload

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
            account = await simulation_accounts.get_account(account_id)
            task = await automation_store.get_task(account_id)
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
            summary = await automation_store.create_run(account_id, effective_date, trigger, config, idempotency_key)
            if summary.status not in {"queued"}:
                return summary
            claimed = await automation_store.claim_run(
                summary.run_id,
                symbols_total=len(universe),
                started_at=datetime.now(SHANGHAI).isoformat(),
            )
            if claimed is None:
                return await automation_store.get_run(summary.run_id)
            await automation_store.update_task(account_id, status="running")
            await simulation_events.publish(
                account_id,
                "agent.run.started",
                {"run_id": summary.run_id, "run_date": effective_date, "symbols": universe},
            )

            if config.deployment_id:
                return await self._run_deployed_account(
                    account,
                    config,
                    claimed,
                    universe,
                    effective_date,
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
                    audit = await automation_store.add_decision(
                        summary.run_id, account_id, decision, current_price, risk_status, risk_reason, order_id
                    )
                    await automation_store.add_event(
                        account_id, "agent.decision", audit.model_dump(mode="json"), summary.run_id
                    )
                    await simulation_events.publish(
                        account_id,
                        "agent.decision",
                        audit.model_dump(mode="json"),
                    )
                except Exception as exc:  # one bad symbol must not stop the pool
                    failures.append(f"{ticker}: {exc}")
                    logger.exception("Agent cycle failed for {}", ticker)
                await automation_store.update_run(
                    summary.run_id,
                    symbols_processed=index,
                    decisions_count=decisions_count,
                    orders_count=orders_count,
                )

            error = "; ".join(failures) if failures else None
            final = await automation_store.update_run(
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

    async def _run_deployed_account(
        self,
        account,
        config: AutomationTaskConfig,
        summary: AgentRunSummary,
        universe: list[str],
        effective_date: str,
    ) -> AgentRunSummary:
        deployment = await deployment_service.get(config.deployment_id or "")
        if deployment.account_id != account.account_id or deployment.status != "active":
            return await automation_store.update_run(
                summary.run_id,
                status="failed",
                completed_at=datetime.now(SHANGHAI).isoformat(),
                error="策略部署与模拟账户不匹配或未启用",
            )
        if deployment.strategy_spec.source == "sandbox" and config.execution_mode != "paper":
            return await automation_store.update_run(
                summary.run_id,
                status="failed",
                completed_at=datetime.now(SHANGHAI).isoformat(),
                error="LLM/沙盒研究策略禁止进入实盘执行链路",
            )
        if deployment.execution.get("strategy_plugins") != strategy_plugins_manifest(
            deployment.strategy_spec.components
        ):
            return await automation_store.update_run(
                summary.run_id,
                status="failed",
                completed_at=datetime.now(SHANGHAI).isoformat(),
                error="混合策略插件代码或版本已变化，拒绝执行不可复现的部署",
            )
        llm = get_llm_config()
        await automation_store.update_run(
            summary.run_id,
            deployment_id=deployment.deployment_id,
            strategy_sha256=deployment.strategy_sha256,
            llm_runtime={
                "provider": llm.get("type"),
                "model": llm.get("model"),
                "temperature": llm.get("temperature"),
                "max_tokens": llm.get("max_tokens"),
                "workflow_version": "deployed-strategy-agent-gate-v1",
            },
        )
        start_date = (date.fromisoformat(effective_date) - timedelta(days=1095)).isoformat()
        analyses: dict[str, dict] = {}
        failures: list[str] = []
        positions = {position.ticker: position for position in account.portfolio.positions}
        for index, ticker in enumerate(universe, start=1):
            try:
                if deployment.asset_type.value == "stock":
                    frame = await async_get_stock_history(ticker, start_date=start_date, end_date=effective_date)
                else:
                    frame = await async_get_fund_history(
                        ticker,
                        asset_type=deployment.asset_type.value,
                        start_date=start_date,
                        end_date=effective_date,
                    )
                normalized, _ = prepare_backtest_data(
                    frame,
                    ticker=ticker,
                    asset_type=deployment.asset_type.value,
                    start_date=start_date,
                    end_date=effective_date,
                    adjustment="qfq" if deployment.asset_type.value == "stock" else "provider_default",
                )
                normalized = normalized[normalized["date"] <= effective_date]
                if normalized.empty:
                    raise ValueError("没有截至运行日的有效历史行情")
                current_price = float(normalized.iloc[-1]["close"])
                context = await build_market_context(
                    ticker,
                    asset_type=deployment.asset_type,
                    as_of_date=effective_date,
                    current_price=current_price,
                    history_df=normalized,
                    include_live_enrichment=False,
                )
                freshness_error = self._data_freshness_error(context, effective_date, config.data_max_age_seconds)
                if freshness_error:
                    raise ValueError(freshness_error)
                runtime_state = await self.strategy_states.get(deployment.deployment_id, ticker)
                held = positions.get(ticker)
                marked_total = account.portfolio.cash + sum(
                    position.shares * (current_price if position.ticker == ticker else position.current_price)
                    for position in account.portfolio.positions
                )
                current_exposure = (
                    held.shares * current_price / marked_total
                    if held is not None and marked_total
                    else 0.0
                )
                if held is not None and runtime_state.entry_price is None:
                    runtime_state.entry_price = held.avg_cost
                    runtime_state.lifecycle = "active"
                    runtime_state.target_exposure = current_exposure
                intent, runtime_state, evaluation = evaluate_strategy_intent(
                    deployment.strategy_spec,
                    normalized,
                    asset_type=deployment.asset_type,
                    current_exposure=current_exposure,
                    state=runtime_state,
                )
                await self.strategy_states.save(deployment.deployment_id, ticker, runtime_state)
                target_exposure = intent.target_exposure
                base_decision = decision_from_intent(
                    deployment.strategy_spec,
                    intent,
                    ticker=ticker,
                    asset_type=deployment.asset_type,
                    current_price=current_price,
                )
                gate = {"approved": True, "reason": "策略为 HOLD，无需 Agent 审核"}
                if base_decision.decision != Decision.HOLD:
                    agent_result = await research_service.run(
                        ticker,
                        strategy=deployment.strategy_name,
                        asset_type=deployment.asset_type,
                        market_context=context,
                        current_price=current_price,
                        as_of_date=effective_date,
                    )
                    agent_decision = agent_result.get("final_decision")
                    approved = isinstance(agent_decision, TradeDecision) and (
                        agent_decision.decision == base_decision.decision
                    )
                    gate = {
                        "approved": approved,
                        "reason": ("Agent 与确定性策略方向一致" if approved else "Agent 未批准该策略交易方向"),
                        "agent_decision": (
                            agent_decision.model_dump(mode="json")
                            if isinstance(agent_decision, TradeDecision)
                            else None
                        ),
                    }
                analyses[ticker] = {
                    "decision": base_decision,
                    "evaluation": evaluation,
                    "gate": gate,
                    "price": current_price,
                    "target_exposure": target_exposure,
                }
            except Exception as exc:
                failures.append(f"{ticker}: {exc}")
                logger.exception("Deployed strategy cycle failed for {}", ticker)
            await automation_store.update_run(summary.run_id, symbols_processed=index)

        approved_decisions = {
            ticker: item["decision"] for ticker, item in analyses.items() if item["gate"].get("approved")
        }
        prices = {ticker: item["price"] for ticker, item in analyses.items()}
        proposals: dict[str, dict] = {}
        policy = deployment.strategy_spec.position_policy
        marked_total = account.portfolio.cash + sum(
            position.shares * float(prices.get(position.ticker, position.current_price))
            for position in account.portfolio.positions
        )
        raw_targets = {}
        for ticker, item in analyses.items():
            held = positions.get(ticker)
            current_weight = (
                held.shares * float(item["price"]) / marked_total
                if held is not None and marked_total
                else 0.0
            )
            raw_targets[ticker] = (
                float(item["target_exposure"])
                if item["gate"].get("approved") and item["target_exposure"] is not None
                else current_weight
            )
        for ticker, held in positions.items():
            raw_targets.setdefault(
                ticker,
                held.shares * float(prices.get(ticker, held.current_price)) / marked_total if marked_total else 0.0,
            )
        portfolio_spec = deployment.portfolio_spec
        weights = normalize_target_exposures(
            raw_targets,
            max_position_weight=(portfolio_spec.max_position_weight if portfolio_spec else policy.max_exposure),
            max_positions=(portfolio_spec.max_positions if portfolio_spec else len(raw_targets)),
            cash_reserve=(portfolio_spec.cash_reserve if portfolio_spec else 1 - policy.max_exposure),
        )
        planned = plan_rebalance(account.portfolio, account.config, weights, prices, approved_decisions)
        proposals = {
            proposal["ticker"]: proposal
            for proposal in planned
            if proposal["ticker"] in approved_decisions
            and approved_decisions[proposal["ticker"]].decision.value == proposal["side"]
        }

        orders_count = 0
        ordered_tickers = sorted(
            analyses,
            key=lambda ticker: (0 if (proposals.get(ticker) or {}).get("side") == "sell" else 1, ticker),
        )
        for ticker in ordered_tickers:
            item = analyses[ticker]
            proposal = proposals.get(ticker)
            gate_approved = bool(item["gate"].get("approved"))
            risk_status = "approved" if gate_approved else "rejected"
            risk_reason = item["gate"].get("reason")
            confirmation = "pending" if proposal and config.mode == "confirm" else "none"
            audit = await automation_store.add_decision(
                summary.run_id,
                account.account_id,
                item["decision"],
                item["price"],
                risk_status,
                risk_reason,
                signal_source="deployed_strategy",
                strategy_evaluation=item["evaluation"],
                agent_gate=item["gate"],
                proposed_order=proposal,
                confirmation_status=confirmation,
            )
            if proposal and config.mode == "auto" and orders_count < config.max_orders_per_run:
                try:
                    order = await self._submit_proposal(
                        account.account_id,
                        summary.run_id,
                        audit.decision_id,
                        deployment.deployment_id,
                        proposal,
                        effective_date,
                        config,
                    )
                    orders_count += 1
                    audit = await automation_store.update_decision(
                        audit.decision_id,
                        order_id=order.order_id,
                        confirmation_status="confirmed",
                        risk_status="approved" if order.status in {"pending", "filled"} else "rejected",
                        risk_reason=order.reject_reason,
                    )
                except ValueError as exc:
                    audit = await automation_store.update_decision(
                        audit.decision_id,
                        risk_status="rejected",
                        risk_reason=str(exc),
                    )
            elif proposal and config.mode == "auto":
                audit = await automation_store.update_decision(
                    audit.decision_id,
                    risk_status="rejected",
                    risk_reason="达到本次 Agent 运行的最大订单数",
                )
            await automation_store.add_event(
                account.account_id, "agent.decision", audit.model_dump(mode="json"), summary.run_id
            )
            await simulation_events.publish(account.account_id, "agent.decision", audit.model_dump(mode="json"))

        final = await automation_store.update_run(
            summary.run_id,
            status="completed" if analyses or not failures else "failed",
            symbols_processed=len(universe),
            decisions_count=len(analyses),
            orders_count=orders_count,
            completed_at=datetime.now(SHANGHAI).isoformat(),
            error="; ".join(failures) if failures else None,
        )
        await simulation_events.publish(
            account.account_id,
            "agent.run.completed" if final.status == "completed" else "agent.run.failed",
            final.model_dump(mode="json"),
        )
        return final

    async def _submit_proposal(
        self,
        account_id: str,
        run_id: str,
        decision_id: str,
        deployment_id: str,
        proposal: dict,
        trade_date: str,
        config: AutomationTaskConfig,
    ):
        account = await simulation_accounts.get_account(account_id)
        if config.daily_loss_limit_pct:
            daily_pnl = await simulation_accounts.daily_pnl(account_id)
            if daily_pnl <= -account.portfolio.initial_capital * config.daily_loss_limit_pct:
                raise ValueError("触发单日亏损限额")
        order_id = f"sim-{hashlib.sha256(f'{deployment_id}:{run_id}:{decision_id}'.encode()).hexdigest()[:24]}"
        order = await simulation_accounts.create_order(
            account_id=account_id,
            ticker=proposal["ticker"],
            side=Decision(proposal["side"]),
            shares=int(proposal["shares"]),
            order_type="market",
            submitted_date=trade_date,
            source="agent",
            run_id=run_id,
            fill_policy=config.fill_time,
            asset_type=config.asset_type,
            stop_loss=proposal.get("stop_loss"),
            take_profit=proposal.get("take_profit"),
            order_id=order_id,
            deployment_id=deployment_id,
            decision_id=decision_id,
        )
        if config.fill_time == "same_close" and order.status == "pending":
            order = await simulation_accounts.fill_order(order.order_id, float(proposal["price"]), trade_date)
        return order

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
        summary = await automation_store.create_run(account_id, run_date, trigger, config, key)
        if summary.status == "queued":
            summary = await automation_store.update_run(
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
        account = await simulation_accounts.get_account(account_id)
        if config.daily_loss_limit_pct:
            daily_pnl = await simulation_accounts.daily_pnl(account_id)
            if daily_pnl <= -account.portfolio.initial_capital * config.daily_loss_limit_pct:
                return "rejected", "触发单日亏损限额", None
        if decision.decision == Decision.HOLD:
            return "approved", "Agent 建议持有，不生成订单", None
        shares = decision_shares(account.portfolio, account.config, decision, price)
        if shares <= 0:
            return "rejected", "按仓位和 A 股最小交易单位计算后无可交易数量", None
        order = await simulation_accounts.create_order(
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
                await simulation_accounts.cancel_order(order.order_id)
                return "rejected", str(exc), order.order_id
            return "approved", "订单已写入东方财富文件单，等待终端回报", order.order_id
        if config.fill_time == "same_close":
            order = await simulation_accounts.fill_order(order.order_id, price, trade_date)
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
        account = await simulation_accounts.get_account(account_id)
        live_config = account.config.live
        if not live_config.enabled:
            return "rejected", "账户未启用实盘 Adapter", None
        if live_config.require_manual_approval and not force:
            return "rejected", "该实盘账户要求人工确认", None
        if orders_count >= config.max_orders_per_run:
            return "rejected", "达到本次 Agent 运行的最大订单数", None
        if config.daily_loss_limit_pct:
            daily_pnl = await simulation_accounts.daily_pnl(account_id)
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
        except LiveBrokerUnavailableError as exc:
            return "rejected", str(exc), None

        try:
            try:
                result = await broker.submit_order(intent)
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
                snapshot = await broker.sync()
                await self._apply_live_snapshot(account_id, snapshot)
            except Exception:
                logger.warning("Live broker accepted {} but snapshot sync is unavailable", order_id)
            return "approved", result.message or f"实盘订单已提交：{result.status}", order_id
        finally:
            await broker.close()

    @staticmethod
    async def _apply_live_snapshot(account_id: str, payload: dict) -> bool:
        """Apply only an explicit cash/positions snapshot from the broker."""
        if not isinstance(payload, dict) or "cash" not in payload or "positions" not in payload:
            return False
        await simulation_accounts.apply_live_snapshot(
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
        audit = await automation_store.get_decision(decision_id)
        if audit.account_id != account_id:
            raise KeyError("Agent decision 不属于该模拟账户")
        if audit.confirmation_status == "rejected":
            raise ValueError("该 Agent 决策已被用户拒绝，不能确认")
        if audit.risk_status == "rejected":
            raise ValueError("该 Agent 决策已被风控拦截，不能确认")
        if audit.order_id:
            return audit
        task = await automation_store.get_task(account_id)
        config: AutomationTaskConfig = task["config"]
        if audit.proposed_order and config.deployment_id:
            run = await automation_store.get_run(audit.run_id)
            if config.fill_time == "same_close" and run.run_date != _today():
                updated = await automation_store.update_decision(
                    decision_id,
                    confirmation_status="expired",
                    risk_status="rejected",
                    risk_reason="同日收盘执行提案已过期，请重新运行策略",
                )
                raise ValueError(updated.risk_reason)
            proposal = dict(audit.proposed_order)
            if price is not None:
                proposal["price"] = price
            order = await self._submit_proposal(
                account_id,
                audit.run_id,
                audit.decision_id,
                config.deployment_id,
                proposal,
                run.run_date,
                config,
            )
            updated = await automation_store.update_decision(
                decision_id,
                current_price=float(proposal["price"]),
                risk_status="approved" if order.status in {"pending", "filled"} else "rejected",
                risk_reason=order.reject_reason,
                order_id=order.order_id,
                confirmation_status="confirmed",
            )
            await simulation_events.publish(account_id, "agent.decision.confirmed", updated.model_dump(mode="json"))
            return updated
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
        account = await simulation_accounts.get_account(account_id)
        uses_external = account.config.external.enabled and account.config.external.provider != "internal"
        if order_id and config.execution_mode != "live" and not uses_external and config.fill_time == "manual":
            filled_order = await simulation_accounts.fill_order(order_id, fill_price, _today())
            if filled_order.status != "filled":
                risk_status = "rejected"
                risk_reason = filled_order.reject_reason or "手动确认订单未成交"
        updated = await automation_store.update_decision(
            decision_id,
            current_price=fill_price,
            risk_status=risk_status,
            risk_reason=risk_reason,
            order_id=order_id,
        )
        await simulation_events.publish(account_id, "agent.decision.confirmed", updated.model_dump(mode="json"))
        return updated

    async def reject_decision(self, account_id: str, decision_id: str):
        """Record an explicit user rejection without changing risk evaluation."""
        audit = await automation_store.get_decision(decision_id)
        if audit.account_id != account_id:
            raise KeyError("Agent decision 不属于该模拟账户")
        if audit.order_id:
            raise ValueError("该 Agent 决策已经生成订单，不能拒绝")
        if audit.confirmation_status == "rejected":
            return audit
        if audit.confirmation_status != "pending":
            raise ValueError("该 Agent 决策当前不需要确认")
        updated = await automation_store.update_decision(
            decision_id,
            confirmation_status="rejected",
        )
        await simulation_events.publish(account_id, "agent.decision.rejected", updated.model_dump(mode="json"))
        return updated

    async def confirm_run(self, account_id: str, run_id: str) -> dict:
        run = await automation_store.get_run(run_id)
        if run.account_id != account_id:
            raise KeyError("Agent run 不属于该模拟账户")
        decisions = await automation_store.list_decisions(account_id, run_id, 1000)
        task = await automation_store.get_task(account_id)
        config: AutomationTaskConfig = task["config"]
        pending = [
            item
            for item in decisions
            if item.proposed_order and item.confirmation_status == "pending" and not item.order_id
        ]
        pending.sort(
            key=lambda item: (
                0 if (item.proposed_order or {}).get("side") == "sell" else 1,
                item.ticker,
            )
        )
        pending = pending[: config.max_orders_per_run]
        confirmed = []
        failures: list[dict[str, str]] = []
        for audit in pending:
            try:
                confirmed.append(await self.confirm_decision(account_id, audit.decision_id))
            except ValueError as exc:
                failures.append({"decision_id": audit.decision_id, "error": str(exc)})
        refreshed = await automation_store.get_run(run_id)
        await automation_store.update_run(
            run_id,
            orders_count=max(
                refreshed.orders_count, len([item for item in decisions if item.order_id]) + len(confirmed)
            ),
        )
        return {
            "run_id": run_id,
            "confirmed": [item.model_dump(mode="json") for item in confirmed],
            "failures": failures,
        }

    async def settle_account(
        self,
        account_id: str,
        settlement_date: str | None = None,
        prices: dict[str, float] | None = None,
        open_prices: dict[str, float] | None = None,
    ) -> dict:
        task = await automation_store.get_task(account_id)
        if task["config"].execution_mode == "live":
            return await self.sync_live_account(account_id)
        account = await simulation_accounts.get_account(account_id)
        if account.config.external.enabled and account.config.external.provider != "internal":
            return await self.sync_external_account(account_id)
        async with self._lock_for(account_id):
            return await self._settle_account(account_id, settlement_date, prices, open_prices)

    async def sync_external_account(self, account_id: str) -> dict:
        """Read an external paper account and mirror it into the local UI model."""
        account = await simulation_accounts.get_account(account_id)
        try:
            broker = get_simulation_broker(account.config.external)
            snapshot = await asyncio.to_thread(broker.sync)
            await simulation_accounts.apply_external_snapshot(account_id, snapshot)
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
        await automation_store.add_event(account_id, "external.sync", payload)
        await simulation_events.publish(account_id, "external.sync", payload)
        return {**payload, "sync": snapshot}

    async def sync_live_account(self, account_id: str) -> dict:
        """Reconcile live broker state without applying paper fills locally."""
        task = await automation_store.get_task(account_id)
        config: AutomationTaskConfig = task["config"]
        if config.execution_mode != "live":
            raise ValueError("该账户当前不是实盘执行模式")
        if not settings.live_trading_enabled:
            raise ValueError("服务端 LIVE_TRADING_ENABLED 未开启")
        account = await simulation_accounts.get_account(account_id)
        try:
            broker = get_live_broker(account.config.live)
        except LiveBrokerUnavailableError as exc:
            raise ValueError(str(exc)) from exc
        try:
            payload = await broker.sync()
        finally:
            await broker.close()
        mirrored = await self._apply_live_snapshot(account_id, payload)
        payload = {**payload, "portfolio_mirrored": mirrored}
        await automation_store.add_event(account_id, "live.sync", payload)
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

        account = await simulation_accounts.get_account(account_id)
        target_date = settlement_date or _today()
        price_map = {key: float(value) for key, value in (prices or {}).items() if float(value) > 0}
        symbols = {position.ticker for position in account.portfolio.positions}
        pending_orders = await simulation_accounts.list_orders(account_id)
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
        await simulation_accounts.advance_date(account_id, target_date)
        fill_map = {**price_map, **open_price_map}
        filled = []
        for order in pending_orders:
            if order.status != "pending" or order.fill_policy != "next_open" or order.ticker not in fill_map:
                continue
            filled.append(
                await simulation_accounts.fill_order(order.order_id, float(fill_map[order.ticker]), target_date)
            )
        task = await automation_store.get_task(account_id)
        account = await simulation_accounts.mark_to_market(
            account_id,
            price_map,
            target_date,
            trigger_exits=not bool(task["config"].deployment_id),
        )
        payload = {
            "account_id": account_id,
            "date": target_date,
            "filled_orders": [order.model_dump(mode="json") for order in filled],
            "total_value": account.portfolio.total_value,
            "daily_pnl": await simulation_accounts.daily_pnl(account_id),
        }
        await automation_store.add_event(account_id, "daily.settled", payload)
        await simulation_events.publish(account_id, "daily.settled", payload)
        return payload


automation_service = AutomationService()
automation_scheduler = AutomationScheduler(automation_service)
