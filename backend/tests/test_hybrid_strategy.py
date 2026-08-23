from types import SimpleNamespace

import pandas as pd
import pytest

from application import automation as automation_module
from application.automation import AutomationService
from application.automation_store import AutomationStore
from application.backtest_experiment import BacktestExperimentStore
from application.deployments import DeploymentService
from application.strategy_state import StrategyRuntimeStateStore
from engine.backtester import run_backtest
from engine.simulation_account import SimulationAccountService
from engine.strategy_runtime import evaluate_strategy_intent
from models.schemas import AssetType, Decision, StrategyExpression, TradeDecision
from strategies.compiler import evaluate_expression, strategy_from_mapping
from strategies.plugin_registry import strategy_plugins_manifest


def _history(periods: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=periods).strftime("%Y-%m-%d")
    close = [10 + index * 0.08 for index in range(periods)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": [value + 0.1 for value in close],
            "low": [value - 0.1 for value in close],
            "close": close,
            "volume": [1_000 + index * 10 for index in range(periods)],
        }
    )


def _hybrid_mapping() -> dict:
    return {
        "name": "continuous_hybrid",
        "asset_types": ["etf"],
        "components": [
            {
                "id": "trend",
                "type": "python",
                "plugin": "core.trend_score",
                "plugin_version": "1.0.0",
                "weight": 0.6,
                "params": {"fast_window": 3, "slow_window": 8, "sensitivity": 30},
            },
            {
                "id": "confirmation",
                "type": "dsl",
                "weight": 0.4,
                "expression": {
                    "type": "all",
                    "children": [
                        {
                            "type": "compare",
                            "left": {"type": "indicator", "indicator": "close"},
                            "operator": "gt",
                            "right": {"type": "constant", "value": 10},
                        },
                        {
                            "type": "sustained",
                            "bars": 2,
                            "expression": {
                                "type": "compare",
                                "left": {"type": "indicator", "indicator": "return_pct", "window": 1},
                                "operator": "gt",
                                "right": {"type": "constant", "value": 0},
                            },
                        },
                    ],
                },
            },
        ],
        "fusion": {"type": "weighted_score", "entry_threshold": 0.1, "exit_threshold": 0.0},
        "position_policy": {
            "mode": "continuous",
            "min_exposure": 0,
            "max_exposure": 0.83,
            "minimum_change": 0.01,
            "max_increase_per_rebalance": 0.11,
            "max_decrease_per_rebalance": 0.25,
            "rebalance_frequency": "daily",
        },
        "state_policy": {"enabled": True, "cooldown_bars_after_exit": 2},
    }


def test_nested_expression_supports_indicator_cross_and_temporal_conditions():
    spec = strategy_from_mapping(_hybrid_mapping(), source="user")
    expression = spec.components[1].expression
    result = evaluate_expression(expression, _history(), indicator_specs=spec.indicator_specs)

    assert result["matched"] is True
    assert result["children"][1]["matched_count"] == 2

    cross = StrategyExpression.model_validate(
        {
            "type": "crosses_above",
            "left": {"type": "indicator", "indicator": "ma", "window": 2},
            "right": {"type": "indicator", "indicator": "ma", "window": 3},
        }
    )
    crossed = evaluate_expression(cross, pd.DataFrame({"close": [10, 9, 8, 11]}))
    assert crossed["matched"] is True


def test_hybrid_runtime_returns_continuous_exposure_and_is_idempotent_per_bar():
    spec = strategy_from_mapping(_hybrid_mapping(), source="user")
    intent, state, trace = evaluate_strategy_intent(
        spec,
        _history(),
        asset_type=AssetType.ETF,
        current_exposure=0,
    )

    assert intent.decision.value == "buy"
    assert intent.target_exposure == 0.11
    assert 0 < intent.score < 1
    assert trace["fusion"]["type"] == "weighted_score"
    repeated, repeated_state, _ = evaluate_strategy_intent(
        spec,
        _history(),
        asset_type=AssetType.ETF,
        current_exposure=0,
        state=state,
    )
    assert repeated == intent
    assert repeated_state.bars_in_state == state.bars_in_state


def test_hybrid_runtime_applies_fixed_risk_exit_before_fusion_target():
    mapping = _hybrid_mapping()
    mapping["stop_loss_pct"] = 0.05
    spec = strategy_from_mapping(mapping, source="user")
    history = _history()
    entered, state, _ = evaluate_strategy_intent(
        spec,
        history,
        asset_type="etf",
        current_exposure=0,
    )
    next_bar = history.iloc[-1].copy()
    next_bar["date"] = "2026-03-02"
    next_bar[["open", "high", "low", "close"]] = [12.0, 12.1, 11.9, 12.0]
    falling = pd.concat([history, pd.DataFrame([next_bar])], ignore_index=True)

    exited, _state, trace = evaluate_strategy_intent(
        spec,
        falling,
        asset_type="etf",
        current_exposure=entered.target_exposure,
        state=state,
    )

    assert exited.decision == Decision.SELL
    assert exited.target_exposure == 0
    assert trace["exit_reason"] == "stop_loss_triggered"


def test_hybrid_strategy_freezes_registered_python_plugin_hashes():
    spec = strategy_from_mapping(_hybrid_mapping(), source="user")
    manifest = strategy_plugins_manifest(spec.components)

    assert manifest[0]["name"] == "core.trend_score"
    assert len(manifest[0]["sha256"]) == 64

    invalid = _hybrid_mapping()
    invalid["components"][0]["plugin"] = "user.inline_code"
    with pytest.raises(ValueError, match="未注册的策略插件"):
        strategy_from_mapping(invalid, source="user")


@pytest.mark.asyncio
async def test_hybrid_backtest_uses_continuous_target_exposure():
    frame = _history()
    result = await run_backtest(
        ticker="510300",
        start_date=str(frame.iloc[0]["date"]),
        end_date=str(frame.iloc[-1]["date"]),
        asset_type="etf",
        strategy_spec=_hybrid_mapping(),
        prepared_data=(
            frame,
            {
                "sha256": "h" * 64,
                "actual_start_date": str(frame.iloc[0]["date"]),
                "actual_end_date": str(frame.iloc[-1]["date"]),
            },
        ),
    )

    targets = [point["target_exposure"] for point in result["signal_curve"]]
    assert result["strategy_spec"]["position_policy"]["mode"] == "continuous"
    assert result["execution"]["strategy_plugins"][0]["name"] == "core.trend_score"
    assert 0.11 in targets
    assert any(0 < target < 0.83 for target in targets)


@pytest.mark.asyncio
async def test_runtime_state_persists_custom_variables(tmp_path):
    store = StrategyRuntimeStateStore(tmp_path / "hybrid-state.sqlite3")
    spec = strategy_from_mapping(_hybrid_mapping(), source="user")
    _intent, state, _trace = evaluate_strategy_intent(
        spec,
        _history(),
        asset_type="etf",
        current_exposure=0,
    )
    state.variables["confirmation_count"] = 2

    await store.save("deploy-hybrid", "510300", state)
    restored = await store.get("deploy-hybrid", "510300")

    assert restored.target_exposure == state.target_exposure
    assert restored.variables["confirmation_count"] == 2
    assert restored.last_evaluated_date == state.last_evaluated_date


@pytest.mark.asyncio
async def test_deployed_hybrid_strategy_uses_target_weight_and_persists_state(monkeypatch, tmp_path):
    db_path = tmp_path / "hybrid-deployment.sqlite3"
    accounts = SimulationAccountService(db_path)
    automations = AutomationStore(db_path)
    experiments = BacktestExperimentStore(db_path)
    states = StrategyRuntimeStateStore(db_path)
    strategy = strategy_from_mapping(_hybrid_mapping(), source="user")
    manifest = strategy_plugins_manifest(strategy.components)
    frame = _history()
    effective_date = str(frame.iloc[-1]["date"])
    await experiments.save(
        "hybrid-exp",
        "completed",
        {
            "strategy_spec": strategy.model_dump(mode="json"),
            "portfolio_spec": None,
            "result": {
                "ticker": "510300",
                "asset_type": "etf",
                "initial_capital": 100_000,
                "execution": {
                    "fill_time": "next_open",
                    "min_lot": 100,
                    "strategy_plugins": manifest,
                },
            },
        },
    )
    deployments = DeploymentService(
        db_path,
        accounts=accounts,
        automations=automations,
        experiments=experiments,
    )
    deployment = await deployments.create_from_experiment(
        "hybrid-exp",
        account_id="hybrid-paper",
    )
    monkeypatch.setattr(automation_module, "simulation_accounts", accounts)
    monkeypatch.setattr(automation_module, "automation_store", automations)
    monkeypatch.setattr(automation_module, "deployment_service", deployments)

    async def fake_history(*_args, **_kwargs):
        return frame.copy()

    async def fake_context(*_args, **_kwargs):
        return SimpleNamespace(history=[{"date": effective_date, "close": float(frame.iloc[-1]["close"])}])

    async def approving_agent(ticker, **_kwargs):
        return {
            "final_decision": TradeDecision(
                ticker=ticker,
                asset_type=AssetType.ETF,
                decision=Decision.BUY,
                confidence=0.8,
            )
        }

    monkeypatch.setattr(automation_module, "async_get_fund_history", fake_history)
    monkeypatch.setattr(automation_module, "build_market_context", fake_context)
    monkeypatch.setattr(automation_module.research_service, "run", approving_agent)

    summary = await AutomationService(strategy_states=states).run_account(
        "hybrid-paper",
        run_date=effective_date,
    )
    decisions = await automations.list_decisions("hybrid-paper", summary.run_id)
    restored = await states.get(deployment.deployment_id, "510300")

    assert summary.status == "completed"
    assert decisions[0].proposed_order["side"] == "buy"
    assert decisions[0].decision.plan.position_size == 0.11
    assert restored.target_exposure == 0.11
