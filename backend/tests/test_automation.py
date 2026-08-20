import pytest

from application import automation as automation_module
from application.automation import AutomationService
from application.automation_store import AutomationStore
from engine.simulation_account import SimulationAccountService
from models.schemas import AutomationTaskConfig, Decision, LiveTradingConfig, SimulationAccountConfig, TradeDecision


async def _fake_run(_ticker: str, _strategy: str | None = None):
    return {
        "current_price": 10.0,
        "final_decision": TradeDecision(
            ticker="000001",
            decision=Decision.BUY,
            confidence=0.9,
            position_size=0.1,
            reasoning="test decision",
        ),
    }


@pytest.mark.asyncio
async def test_agent_run_is_idempotent_and_settles_next_open(monkeypatch, tmp_path):
    db_path = tmp_path / "automation.sqlite3"
    accounts = SimulationAccountService(db_path)
    store = AutomationStore(db_path)
    await accounts.update_config(
        "default",
        SimulationAccountConfig(
            initial_cash=100_000,
            max_single_position_pct=0.5,
            universe=["000001"],
        ),
    )
    monkeypatch.setattr(automation_module, "simulation_accounts", accounts)
    monkeypatch.setattr(automation_module, "automation_store", store)
    monkeypatch.setattr(automation_module.research_service, "run", _fake_run)

    config = AutomationTaskConfig(
        enabled=True,
        mode="auto",
        universe=["000001"],
        fill_time="next_open",
    )
    await store.update_task("default", config=config)
    service = AutomationService()

    first = await service.run_account("default", run_date="2026-08-03")
    second = await service.run_account("default", run_date="2026-08-03")
    assert first.run_id == second.run_id
    assert first.status == "completed"
    assert first.orders_count == 1
    assert len(await accounts.list_orders("default")) == 1
    assert (await accounts.list_orders("default"))[0].status == "pending"

    await service.settle_account(
        "default",
        settlement_date="2026-08-04",
        prices={"000001": 11.0},
        open_prices={"000001": 10.5},
    )
    account = await accounts.get_account("default")
    assert (await accounts.list_orders("default"))[0].status == "filled"
    assert account.current_date == "2026-08-04"
    assert account.portfolio.positions[0].shares == 1000
    assert (await store.list_decisions("default", first.run_id))[0].order_id


@pytest.mark.asyncio
async def test_scheduler_skips_non_trading_day(monkeypatch, tmp_path):
    db_path = tmp_path / "scheduler.sqlite3"
    accounts = SimulationAccountService(db_path)
    store = AutomationStore(db_path)
    await store.update_task(
        "default",
        config=AutomationTaskConfig(enabled=True, universe=["000001"], schedule_time="15:10"),
    )
    monkeypatch.setattr(automation_module, "simulation_accounts", accounts)
    monkeypatch.setattr(automation_module, "automation_store", store)
    monkeypatch.setattr(automation_module, "is_trading_day", lambda _target: False)

    called = []

    class FakeService:
        async def settle_account(self, *args, **kwargs):
            called.append("settle")

        async def run_account(self, *args, **kwargs):
            called.append("run")

    scheduler = automation_module.AutomationScheduler(FakeService())
    await scheduler.tick(automation_module.datetime(2026, 8, 3, 15, 10, tzinfo=automation_module.SHANGHAI))
    assert called == []


@pytest.mark.asyncio
async def test_store_claim_run_is_single_winner(tmp_path):
    store = AutomationStore(tmp_path / "claim.sqlite3")
    config = AutomationTaskConfig(universe=["000001"])
    created = await store.create_run("account", "2026-08-03", "schedule", config, "account:2026-08-03:schedule")
    first = await store.claim_run(created.run_id, symbols_total=1)
    second = await store.claim_run(created.run_id, symbols_total=1)
    assert first is not None
    assert first.status == "running"
    assert second is None


@pytest.mark.asyncio
async def test_store_recovers_stale_run_without_locking_database(tmp_path):
    store = AutomationStore(tmp_path / "recover.sqlite3")
    config = AutomationTaskConfig(universe=["000001"])
    created = await store.create_run("account", "2026-08-03", "schedule", config, "account:recover")
    await store.claim_run(created.run_id, symbols_total=1)
    assert await store.recover_stale_runs(max_age_minutes=0) == 1
    assert (await store.get_run(created.run_id)).status == "failed"


def test_live_mode_requires_explicit_non_simulation_config():
    with pytest.raises(ValueError, match="simulation_only=false"):
        AutomationTaskConfig(execution_mode="live")


@pytest.mark.asyncio
async def test_live_mode_fails_closed_when_service_gate_is_disabled(monkeypatch, tmp_path):
    db_path = tmp_path / "live-gate.sqlite3"
    accounts = SimulationAccountService(db_path)
    store = AutomationStore(db_path)
    await accounts.update_config(
        "default",
        SimulationAccountConfig(
            initial_cash=100_000,
            max_single_position_pct=0.5,
            universe=["000001"],
            live=LiveTradingConfig(
                provider="custom_http",
                enabled=True,
                endpoint="http://broker-sidecar",
                account_id="live-1",
            ),
        ),
    )
    monkeypatch.setattr(automation_module, "simulation_accounts", accounts)
    monkeypatch.setattr(automation_module, "automation_store", store)
    monkeypatch.setattr(automation_module.settings, "live_trading_enabled", False)
    monkeypatch.setattr(automation_module.research_service, "run", _fake_run)

    config = AutomationTaskConfig(
        enabled=True,
        mode="auto",
        execution_mode="live",
        live_armed=True,
        simulation_only=False,
        universe=["000001"],
    )
    await store.update_task("default", config=config)
    service = AutomationService()

    summary = await service.run_account("default", run_date="2026-08-03")
    assert summary.status == "completed"
    assert summary.orders_count == 0
    decision = (await store.list_decisions("default", summary.run_id))[0]
    assert decision.risk_status == "rejected"
    assert "LIVE_TRADING_ENABLED" in (decision.risk_reason or "")
    assert await accounts.list_orders("default") == []
