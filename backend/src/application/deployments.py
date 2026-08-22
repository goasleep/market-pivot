"""Immutable promotion of completed backtests into paper accounts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from application.automation_store import AutomationStore, automation_store
from application.backtest_experiment import BacktestExperimentStore, backtest_experiments
from data.db_models import StrategyDeploymentRecord
from data.tortoise_db import init_database
from engine.simulation_account import SimulationAccountService, simulation_accounts
from models.schemas import (
    AssetTradingRules,
    AssetType,
    AutomationTaskConfig,
    PortfolioSpec,
    SimulationAccountConfig,
    StrategyDeployment,
    StrategySpec,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class DeploymentService:
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        accounts: SimulationAccountService | None = None,
        automations: AutomationStore | None = None,
        experiments: BacktestExperimentStore | None = None,
    ) -> None:
        self.db_path = db_path
        self.accounts = accounts or simulation_accounts
        self.automations = automations or automation_store
        self.experiments = experiments or backtest_experiments

    async def _ready(self) -> None:
        await init_database(db_path=self.db_path)

    @staticmethod
    def _from_row(row: StrategyDeploymentRecord) -> StrategyDeployment:
        return StrategyDeployment(
            deployment_id=row.deployment_id,
            experiment_id=row.experiment_id,
            account_id=row.account_id,
            status=row.status,
            strategy_name=row.strategy_name,
            strategy_version=row.strategy_version,
            strategy_sha256=row.strategy_sha256,
            strategy_spec=StrategySpec.model_validate(json.loads(row.strategy_spec_json)),
            portfolio_spec=(
                PortfolioSpec.model_validate(json.loads(row.portfolio_spec_json)) if row.portfolio_spec_json else None
            ),
            universe=json.loads(row.universe_json),
            asset_type=AssetType(row.asset_type),
            execution=json.loads(row.execution_json),
            created_at=row.created_at,
            updated_at=row.updated_at,
            activated_at=row.activated_at,
            archived_at=row.archived_at,
        )

    async def get(self, deployment_id: str) -> StrategyDeployment:
        await self._ready()
        row = await StrategyDeploymentRecord.get_or_none(deployment_id=deployment_id)
        if row is None:
            raise KeyError(f"策略部署不存在: {deployment_id}")
        return self._from_row(row)

    async def active_for_account(self, account_id: str) -> StrategyDeployment | None:
        await self._ready()
        row = await StrategyDeploymentRecord.get_or_none(account_id=account_id, status="active")
        return self._from_row(row) if row else None

    async def list(
        self,
        *,
        account_id: str | None = None,
        experiment_id: str | None = None,
        include_archived: bool = False,
    ) -> list[StrategyDeployment]:
        await self._ready()
        query = StrategyDeploymentRecord.all()
        if account_id:
            query = query.filter(account_id=account_id)
        if experiment_id:
            query = query.filter(experiment_id=experiment_id)
        if not include_archived:
            query = query.exclude(status="archived")
        rows = await query.order_by("-created_at")
        return [self._from_row(row) for row in rows]

    async def create_from_experiment(
        self,
        experiment_id: str,
        *,
        account_id: str,
        create_account: bool = True,
        account_name: str | None = None,
        initial_cash: float | None = None,
        enabled: bool = True,
        mode: str = "confirm",
        schedule_time: str = "15:10",
        weekdays: list[int] | None = None,
        execution_key: str | None = None,
    ) -> StrategyDeployment:
        await self._ready()
        deploy_key = execution_key or f"{experiment_id}:{account_id}"
        existing = await StrategyDeploymentRecord.get_or_none(deployment_key=deploy_key)
        if existing:
            return self._from_row(existing)

        experiment = await self.experiments.get(experiment_id)
        if not experiment or experiment.get("status") != "completed":
            raise ValueError("只能部署已完成的回测实验")
        if not experiment.get("strategy_spec"):
            raise ValueError("回测实验缺少可执行 StrategySpec")
        strategy = StrategySpec.model_validate(experiment["strategy_spec"])
        if strategy.source == "sandbox":
            from application.strategy_candidates import strategy_candidates

            candidate_id = str(experiment.get("candidate_id") or "")
            if not candidate_id:
                raise ValueError("沙盒策略实验缺少已审批候选引用")
            candidate = await strategy_candidates.get(candidate_id)
            if candidate.status != "approved" or not candidate.result.get("promotion_eligible"):
                raise ValueError("沙盒策略必须通过等价验证和人工审批后才能部署")
            expected = hashlib.sha256(_canonical(strategy).encode()).hexdigest()
            candidate_digest = hashlib.sha256(_canonical(candidate.strategy_spec).encode()).hexdigest()
            if expected != candidate_digest or experiment.get("source_sha256") != candidate.source_sha256:
                raise ValueError("沙盒候选冻结哈希与回测实验不一致")
        portfolio_spec = (
            PortfolioSpec.model_validate(experiment["portfolio_spec"]) if experiment.get("portfolio_spec") else None
        )
        result = dict(experiment.get("result") or {})
        asset_type = AssetType(result.get("asset_type") or strategy.asset_types[0])
        universe = list(dict.fromkeys(result.get("tickers") or [result.get("ticker")]))
        universe = [str(item) for item in universe if item and item not in {"pool", "portfolio"}]
        if not universe:
            raise ValueError("回测实验没有可部署的有效标的")
        if asset_type not in strategy.asset_types:
            raise ValueError("策略资产类型与回测结果不一致")

        execution = dict(result.get("execution") or {})
        capital = float(initial_cash or result.get("initial_capital") or 1_000_000)
        max_single = portfolio_spec.max_position_weight if portfolio_spec else strategy.position_size_pct
        max_total = 1 - portfolio_spec.cash_reserve if portfolio_spec else 0.95
        trading_rules = AssetTradingRules.defaults_for(asset_type).model_copy(
            update={
                "min_lot": int(execution.get("min_lot", 100)),
                "slippage_bps": float(execution.get("slippage_bps", 5.0)),
                "buy_commission_rate": float(execution.get("buy_commission_rate", 0.0003)),
                "sell_commission_rate": float(execution.get("sell_commission_rate", 0.0003)),
                "minimum_commission": float(execution.get("minimum_commission", 5.0)),
                "stamp_tax_rate": float(
                    execution.get("stamp_tax_rate", 0.0 if asset_type != AssetType.STOCK else 0.001)
                ),
                "transfer_fee_rate": float(
                    execution.get("transfer_fee_rate", 0.0 if asset_type != AssetType.STOCK else 0.00002)
                ),
                "auto_exit_levels": False,
                "max_single_position_pct": max_single,
                "max_total_position_pct": max_total,
            }
        )
        account_config = SimulationAccountConfig(
            name=account_name or f"{strategy.name} 模拟盘",
            initial_cash=capital,
            fill_time=execution.get("fill_time", "next_open"),
            slippage_bps=trading_rules.slippage_bps,
            buy_commission_rate=trading_rules.buy_commission_rate,
            sell_commission_rate=trading_rules.sell_commission_rate,
            minimum_commission=trading_rules.minimum_commission,
            stamp_tax_rate=trading_rules.stamp_tax_rate,
            transfer_fee_rate=trading_rules.transfer_fee_rate,
            min_lot=trading_rules.min_lot,
            max_single_position_pct=max_single,
            max_total_position_pct=max_total,
            default_stop_loss_pct=strategy.stop_loss_pct or 0.08,
            asset_type=asset_type,
            universe=universe,
            trading_rules=trading_rules,
        )
        if create_account:
            await self.accounts.create_account(account_id, account_config)
        else:
            account = await self.accounts.get_account(account_id)
            orders = await self.accounts.list_orders(account_id)
            if (
                account.portfolio.positions
                or account.portfolio.trades
                or any(order.status == "pending" for order in orders)
            ):
                raise ValueError("只能部署到空仓、无成交且无待处理订单的模拟账户")
            if await self.active_for_account(account_id):
                raise ValueError("该模拟账户已有活动策略部署")
            await self.accounts.update_config(account_id, account_config)

        strategy_json = _canonical(strategy)
        digest = hashlib.sha256(strategy_json.encode()).hexdigest()
        deployment_id = f"deploy-{hashlib.sha256(deploy_key.encode()).hexdigest()[:20]}"
        timestamp = _now()
        row = await StrategyDeploymentRecord.create(
            deployment_id=deployment_id,
            deployment_key=deploy_key,
            experiment_id=experiment_id,
            account_id=account_id,
            status="active" if enabled else "paused",
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            strategy_sha256=digest,
            strategy_spec_json=strategy_json,
            portfolio_spec_json=_canonical(portfolio_spec) if portfolio_spec else None,
            universe_json=_canonical(universe),
            asset_type=asset_type.value,
            execution_json=_canonical(execution),
            created_at=timestamp,
            updated_at=timestamp,
            activated_at=timestamp if enabled else None,
        )
        task_config = AutomationTaskConfig(
            enabled=enabled,
            mode=mode,
            execution_mode="paper",
            schedule_time=schedule_time,
            weekdays=weekdays or [0, 1, 2, 3, 4],
            universe=universe,
            asset_type=asset_type,
            strategy_name=strategy.name,
            deployment_id=deployment_id,
            fill_time=account_config.fill_time,
            simulation_only=True,
        )
        await self.automations.update_task(account_id, config=task_config)
        return self._from_row(row)

    async def set_status(self, deployment_id: str, status: str) -> StrategyDeployment:
        if status not in {"active", "paused", "archived"}:
            raise ValueError("部署状态必须是 active、paused 或 archived")
        deployment = await self.get(deployment_id)
        if status == "active":
            other = await StrategyDeploymentRecord.get_or_none(account_id=deployment.account_id, status="active")
            if other and other.deployment_id != deployment_id:
                raise ValueError("该模拟账户已有活动策略部署")
        timestamp = _now()
        updates: dict[str, Any] = {"status": status, "updated_at": timestamp}
        if status == "active":
            updates["activated_at"] = timestamp
            updates["archived_at"] = None
        elif status == "archived":
            updates["archived_at"] = timestamp
        await StrategyDeploymentRecord.filter(deployment_id=deployment_id).update(**updates)
        task = await self.automations.get_task(deployment.account_id)
        config = task["config"].model_copy(update={"enabled": status == "active"})
        await self.automations.update_task(deployment.account_id, config=config)
        await self.accounts.set_status(deployment.account_id, status)
        return await self.get(deployment_id)


deployment_service = DeploymentService()
