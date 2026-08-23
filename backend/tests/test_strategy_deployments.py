from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from application import automation as automation_module
from application import automation_scheduler as automation_scheduler_module
from application.automation import AutomationService
from application.automation_store import AutomationStore
from application.backtest_experiment import BacktestExperimentStore
from application.deployments import DeploymentService
from engine.simulation_account import SimulationAccountService
from engine.strategy_runtime import decision_from_strategy, plan_rebalance
from models.schemas import (
    AssetType,
    AutomationTaskConfig,
    Decision,
    PortfolioState,
    Position,
    SimulationAccountConfig,
    StrategySpec,
    TradeDecision,
)


def _history(end: str = "2026-08-03") -> pd.DataFrame:
    dates = pd.date_range(end=end, periods=8, freq="D").strftime("%Y-%m-%d")
    close = [10 + index * 0.1 for index in range(8)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": [value + 0.1 for value in close],
            "low": [value - 0.1 for value in close],
            "close": close,
            "volume": [1000] * 8,
        }
    )


async def _deployment_service(tmp_path):
    db_path = tmp_path / "deployments.sqlite3"
    accounts = SimulationAccountService(db_path)
    automations = AutomationStore(db_path)
    experiments = BacktestExperimentStore(db_path)
    await experiments.save(
        "exp-1",
        "completed",
        {
            "strategy_spec": {
                "name": "deployed_trend",
                "version": "1.0.0",
                "asset_types": ["etf"],
                "entry_conditions": [
                    {"indicator": "return_pct", "operator": "gt", "value": -1, "window": 1}
                ],
                "exit_conditions": [
                    {"indicator": "return_pct", "operator": "lt", "value": -5, "window": 1}
                ],
                "position_size_pct": 0.2,
            },
            "portfolio_spec": None,
            "result": {
                "ticker": "510300",
                "tickers": ["510300"],
                "asset_type": "etf",
                "initial_capital": 100_000,
                "execution": {"fill_time": "next_open", "min_lot": 100},
            },
        },
    )
    return DeploymentService(
        db_path,
        accounts=accounts,
        automations=automations,
        experiments=experiments,
    ), accounts, automations


@pytest.mark.asyncio
async def test_completed_experiment_deploys_idempotently_to_an_empty_account(tmp_path):
    service, accounts, automations = await _deployment_service(tmp_path)
    deployment = await service.create_from_experiment(
        "exp-1",
        account_id="paper_one",
        account_name="趋势模拟盘",
        execution_key="task:call",
    )
    repeated = await service.create_from_experiment(
        "exp-1",
        account_id="paper_one",
        execution_key="task:call",
    )

    assert repeated.deployment_id == deployment.deployment_id
    assert len(deployment.strategy_sha256) == 64
    account = await accounts.get_account("paper_one")
    assert account.portfolio.positions == []
    assert account.portfolio.trades == []
    assert await accounts.list_orders("paper_one") == []
    task = await automations.get_task("paper_one")
    assert task["config"].deployment_id == deployment.deployment_id
    assert task["config"].mode == "confirm"

    paused = await service.set_status(deployment.deployment_id, "paused")
    assert paused.status == "paused"
    assert (await accounts.get_account("paper_one")).status == "paused"


@pytest.mark.asyncio
async def test_deployment_api_creates_lists_and_pauses_an_account(monkeypatch, tmp_path):
    import api.routers.deployments as deployments_router

    service, accounts, _automations = await _deployment_service(tmp_path)
    monkeypatch.setattr(deployments_router, "deployment_service", service)
    app = FastAPI()
    app.include_router(deployments_router.router, prefix="/api/deployments")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/deployments/experiments/exp-1",
            json={
                "account_id": "paper_api",
                "account_name": "API 模拟盘",
                "mode": "confirm",
                "execution_key": "api:test:deploy",
            },
        )
        assert created.status_code == 200
        deployment = created.json()
        assert deployment["account_id"] == "paper_api"
        assert deployment["status"] == "active"

        listed = await client.get("/api/deployments", params={"account_id": "paper_api"})
        assert listed.status_code == 200
        assert [item["deployment_id"] for item in listed.json()["deployments"]] == [
            deployment["deployment_id"]
        ]

        paused = await client.post(f"/api/deployments/{deployment['deployment_id']}/pause")
        assert paused.status_code == 200
        assert paused.json()["status"] == "paused"
        assert (await accounts.get_account("paper_api")).status == "paused"


def test_shared_strategy_runtime_applies_stop_exit_and_sell_first_rebalance():
    spec = StrategySpec(
        name="runtime",
        asset_types=[AssetType.ETF],
        entry_conditions=[{"indicator": "return_pct", "operator": "gt", "value": -1, "window": 1}],
    )
    position = Position(
        ticker="510300",
        asset_type=AssetType.ETF,
        shares=1000,
        available_shares=1000,
        avg_cost=10,
        current_price=9,
        stop_loss=9.5,
    )
    decision, evaluation = decision_from_strategy(
        spec,
        _history(),
        asset_type=AssetType.ETF,
        ticker="510300",
        current_price=9,
        position=position,
    )
    assert decision.decision == Decision.SELL
    assert evaluation["exit_reason"] == "stop_loss_triggered"

    portfolio = PortfolioState(cash=50_000, initial_capital=100_000, positions=[position])
    proposals = plan_rebalance(
        portfolio,
        SimulationAccountConfig(asset_type=AssetType.ETF),
        {"159915": 0.4},
        {"510300": 9, "159915": 2},
    )
    assert [item["side"] for item in proposals] == ["sell", "buy"]


@pytest.mark.asyncio
async def test_deployed_confirm_mode_waits_for_confirmation_and_is_idempotent(monkeypatch, tmp_path):
    deployments, accounts, store = await _deployment_service(tmp_path)
    deployment = await deployments.create_from_experiment("exp-1", account_id="paper_confirm")
    monkeypatch.setattr(automation_module, "simulation_accounts", accounts)
    monkeypatch.setattr(automation_module, "automation_store", store)
    monkeypatch.setattr(automation_module, "deployment_service", deployments)
    async def fake_history(*_args, **_kwargs):
        return _history()

    monkeypatch.setattr(automation_module, "async_get_fund_history", fake_history)

    async def fake_context(*_args, **_kwargs):
        return SimpleNamespace(
            current_price=10.7,
            history=[{"date": "2026-08-03", "close": 10.7}],
        )

    async def fake_agent(ticker, **_kwargs):
        return {
            "final_decision": TradeDecision(
                ticker=ticker,
                asset_type=AssetType.ETF,
                decision=Decision.BUY,
                confidence=0.8,
            )
        }

    monkeypatch.setattr(automation_module, "build_market_context", fake_context)
    monkeypatch.setattr(automation_module.research_service, "run", fake_agent)
    service = AutomationService()
    locked_config = (await store.get_task("paper_confirm"))["config"]
    with pytest.raises(ValueError, match="不可在自动化配置中修改"):
        await service.update_task(
            "paper_confirm",
            locked_config.model_copy(update={"universe": ["159915"]}),
        )
    summary = await service.run_account("paper_confirm", run_date="2026-08-03")
    decisions = await store.list_decisions("paper_confirm", summary.run_id)

    assert summary.orders_count == 0
    assert decisions[0].confirmation_status == "pending"
    assert await accounts.list_orders("paper_confirm") == []

    first = await service.confirm_run("paper_confirm", summary.run_id)
    second = await service.confirm_run("paper_confirm", summary.run_id)
    assert len(first["confirmed"]) == 1
    assert second["confirmed"] == []
    orders = await accounts.list_orders("paper_confirm")
    assert len(orders) == 1
    assert orders[0].deployment_id == deployment.deployment_id


@pytest.mark.asyncio
async def test_agent_can_veto_a_deployed_stop_loss_sell(monkeypatch, tmp_path):
    deployments, accounts, store = await _deployment_service(tmp_path)
    await deployments.create_from_experiment("exp-1", account_id="paper_veto")
    seeded = await accounts.create_order(
        "paper_veto",
        "510300",
        Decision.BUY,
        100,
        submitted_date="2026-08-02",
        asset_type=AssetType.ETF,
        stop_loss=9.5,
    )
    await accounts.fill_order(seeded.order_id, 10, "2026-08-02")
    history = _history()
    history.loc[history.index[-1], ["open", "high", "low", "close"]] = [9, 9.1, 8.9, 9]

    async def fake_history(*_args, **_kwargs):
        return history.copy()

    async def fake_context(*_args, **_kwargs):
        return SimpleNamespace(current_price=9, history=[{"date": "2026-08-03", "close": 9}])

    async def veto_agent(ticker, **_kwargs):
        return {
            "final_decision": TradeDecision(
                ticker=ticker,
                asset_type=AssetType.ETF,
                decision=Decision.HOLD,
            )
        }

    monkeypatch.setattr(automation_module, "simulation_accounts", accounts)
    monkeypatch.setattr(automation_module, "automation_store", store)
    monkeypatch.setattr(automation_module, "deployment_service", deployments)
    monkeypatch.setattr(automation_module, "async_get_fund_history", fake_history)
    monkeypatch.setattr(automation_module, "build_market_context", fake_context)
    monkeypatch.setattr(automation_module.research_service, "run", veto_agent)

    summary = await AutomationService().run_account("paper_veto", run_date="2026-08-03")
    audit = (await store.list_decisions("paper_veto", summary.run_id))[0]
    assert audit.decision.decision == Decision.SELL
    assert audit.strategy_evaluation["exit_reason"] == "stop_loss_triggered"
    assert audit.agent_gate["approved"] is False
    assert audit.proposed_order is None
    assert len(await accounts.list_orders("paper_veto")) == 1


@pytest.mark.asyncio
async def test_agent_can_veto_a_deployed_buy(monkeypatch, tmp_path):
    deployments, accounts, store = await _deployment_service(tmp_path)
    await deployments.create_from_experiment("exp-1", account_id="paper_buy_veto")

    async def fake_history(*_args, **_kwargs):
        return _history()

    async def fake_context(*_args, **_kwargs):
        return SimpleNamespace(current_price=10.7, history=[{"date": "2026-08-03", "close": 10.7}])

    async def veto_agent(ticker, **_kwargs):
        return {
            "final_decision": TradeDecision(
                ticker=ticker,
                asset_type=AssetType.ETF,
                decision=Decision.HOLD,
            )
        }

    monkeypatch.setattr(automation_module, "simulation_accounts", accounts)
    monkeypatch.setattr(automation_module, "automation_store", store)
    monkeypatch.setattr(automation_module, "deployment_service", deployments)
    monkeypatch.setattr(automation_module, "async_get_fund_history", fake_history)
    monkeypatch.setattr(automation_module, "build_market_context", fake_context)
    monkeypatch.setattr(automation_module.research_service, "run", veto_agent)

    summary = await AutomationService().run_account("paper_buy_veto", run_date="2026-08-03")
    audit = (await store.list_decisions("paper_buy_veto", summary.run_id))[0]
    assert audit.decision.decision == Decision.BUY
    assert audit.agent_gate["approved"] is False
    assert audit.proposed_order is None
    assert await accounts.list_orders("paper_buy_veto") == []


@pytest.mark.asyncio
async def test_scheduler_isolates_accounts_and_uses_bounded_parallelism(monkeypatch, tmp_path):
    db_path = tmp_path / "scheduler-many.sqlite3"
    accounts = SimulationAccountService(db_path)
    store = AutomationStore(db_path)
    for account_id in ("first", "second"):
        await accounts.create_account(account_id)
        await store.update_task(
            account_id,
            config=AutomationTaskConfig(enabled=True, universe=["000001"], schedule_time="15:10"),
        )
    monkeypatch.setattr(automation_scheduler_module, "simulation_accounts", accounts)
    monkeypatch.setattr(automation_scheduler_module, "automation_store", store)
    monkeypatch.setattr(automation_scheduler_module, "is_trading_day", lambda _target: True)
    monkeypatch.setattr(automation_scheduler_module.settings, "automation_max_concurrency", 2)
    completed: list[str] = []

    class FakeService:
        async def settle_account(self, account_id, *_args):
            if account_id == "first":
                raise RuntimeError("isolated failure")

        async def run_account(self, account_id, **_kwargs):
            completed.append(account_id)

    scheduler = automation_scheduler_module.AutomationScheduler(FakeService())
    await scheduler.tick(automation_module.datetime(2026, 8, 3, 15, 10, tzinfo=automation_module.SHANGHAI))
    assert completed == ["second"]
