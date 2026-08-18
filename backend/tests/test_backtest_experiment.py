import pandas as pd
import pytest

from data.backtest_data import BacktestDataError, prepare_backtest_data
from models.schemas import AssetType
from strategies.compiler import available_indicators, evaluate_strategy, strategy_from_mapping


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            "open": [10, 10.1, 10.2, 10.3, 10.4],
            "high": [10.2, 10.3, 10.4, 10.5, 10.6],
            "low": [9.8, 9.9, 10.0, 10.1, 10.2],
            "close": [10, 10.2, 10.4, 10.5, 10.6],
            "volume": [100, 110, 120, 130, 140],
        }
    )


def test_indicator_contract_and_extended_indicators_are_deterministic():
    names = {item["name"] for item in available_indicators()}
    assert {"price_vs_ma_pct", "rsi", "atr", "volatility"}.issubset(names)

    spec = strategy_from_mapping(
        {
            "name": "controlled_trend",
            "asset_types": ["etf"],
            "indicators": ["price_vs_ma_pct", "rsi"],
            "entry_conditions": [
                {"indicator": "price_vs_ma_pct", "operator": "gt", "value": -5, "window": 3}
            ],
        },
        source="llm",
    )
    result = evaluate_strategy(spec, _history(), asset_type=AssetType.ETF)
    assert result["matched"] is True
    assert result["conditions"][0]["value"] is not None


def test_unknown_indicator_is_rejected_before_backtest():
    try:
        strategy_from_mapping(
            {
                "name": "unsafe",
                "asset_types": ["etf"],
                "entry_conditions": [{"indicator": "future_magic", "operator": "gt", "value": 0}],
            },
            source="llm",
        )
    except ValueError as exc:
        assert "不受支持的指标" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unknown indicators must be rejected")


def test_llm_strategy_mapping_normalizes_common_shapes_and_percentages():
    spec = strategy_from_mapping(
        {
            "name": "llm_shape_variants",
            "asset_types": "etf",
            "indicators": {"rsi": {}, "ma": {}},
            "indicator_specs": {
                "rsi": {"type": "rsi", "period": 14, "source": "close", "role": "momentum"},
                "ma": {"length": 20, "source": "close", "role": "trend"},
            },
            "entry_conditions": {
                "indicator": "rsi",
                "op": "lt",
                "threshold": 35,
                "period": 14,
            },
            "stop_loss_pct": 5,
            "take_profit_pct": "12%",
            "position_size_pct": 20,
        },
        source="llm",
    )
    assert spec.asset_types == [AssetType.ETF]
    assert {item.name for item in spec.indicator_specs} == {"rsi", "ma"}
    assert {item.role for item in spec.indicator_specs} == {"confirmation", "filter"}
    assert spec.entry_conditions[0].operator == "lt"
    assert spec.entry_conditions[0].window == 14
    assert spec.stop_loss_pct == 0.05
    assert spec.take_profit_pct == 0.12
    assert spec.position_size_pct == 0.2


@pytest.mark.asyncio
async def test_agent_strategy_design_is_converted_to_a_validated_spec(monkeypatch):
    import application.backtest_experiment as experiment_module

    class FakeLLM:
        async def chat_json(self, prompt, system):
            assert "available_indicators" in prompt
            assert "受控指标" in system
            return {
                "name": "agent_rsi_strategy",
                "description": "Momentum with a bounded oscillator.",
                "asset_types": ["etf"],
                "indicators": ["rsi"],
                "entry_conditions": [{"indicator": "rsi", "operator": "lt", "value": 35, "window": 14}],
                "exit_conditions": [{"indicator": "rsi", "operator": "gt", "value": 70, "window": 14}],
            }

    monkeypatch.setattr(experiment_module, "get_llm_service", lambda: FakeLLM())
    monkeypatch.setattr(experiment_module, "register_strategy_spec", lambda spec: spec)
    spec = await experiment_module.design_strategy(
        objective="设计一个 ETF 动量策略",
        asset_type=AssetType.ETF,
        ticker="510300",
    )
    assert spec.source == "llm"
    assert spec.entry_conditions[0].indicator == "rsi"


@pytest.mark.asyncio
async def test_agent_portfolio_design_is_converted_to_a_validated_spec(monkeypatch):
    import application.backtest_experiment as experiment_module

    class FakeLLM:
        async def chat_json(self, prompt, system):
            assert "组合配置 Agent" in system
            assert "510300" in prompt
            return {
                "allocation_method": "equal_weight",
                "rebalance_frequency": "weekly",
                "max_position_weight": 0.4,
                "max_positions": 3,
                "cash_reserve": 0.1,
            }

    monkeypatch.setattr(experiment_module, "get_llm_service", lambda: FakeLLM())
    spec = await experiment_module.design_portfolio(
        objective="控制 ETF 组合回撤",
        asset_type=AssetType.ETF,
        tickers=["510300", "159915"],
    )
    assert spec.allocation_method == "equal_weight"
    assert spec.rebalance_frequency == "weekly"
    assert spec.max_positions == 3


def test_backtest_data_manifest_is_content_addressed_and_rejects_bad_ohlc():
    frame, manifest = prepare_backtest_data(
        _history(),
        ticker="510300",
        asset_type="etf",
        start_date="2026-01-01",
        end_date="2026-01-05",
    )
    assert len(frame) == 5
    assert len(manifest["sha256"]) == 64
    assert manifest["quality"]["status"] == "valid"

    invalid = _history()
    invalid.loc[0, "high"] = 9
    try:
        prepare_backtest_data(
            invalid,
            ticker="510300",
            asset_type="etf",
            start_date="2026-01-01",
            end_date="2026-01-05",
        )
    except BacktestDataError as exc:
        assert "OHLC" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("invalid OHLC must be rejected")


@pytest.mark.asyncio
async def test_experiment_persists_a_replayable_payload_and_report_artifacts(monkeypatch, tmp_path):
    import application.backtest_experiment as experiment_module

    class FakeArtifactService:
        def create_user_artifacts(self, artifacts, **kwargs):
            assert kwargs["source"] == "backtest"
            assert kwargs["task_id"].startswith("bt-exp-")
            return [{"artifact_id": f"artifact-{index}", "name": item["name"]} for index, item in enumerate(artifacts)]

    async def fake_run_backtest(**kwargs):
        return {
            "ticker": kwargs["ticker"],
            "asset_type": "etf",
            "start_date": "2026-01-01",
            "end_date": "2026-01-05",
            "initial_capital": 100000,
            "final_value": 101000,
            "total_return": 0.01,
            "max_drawdown": 0.02,
            "sharpe_ratio": 0.5,
            "win_rate": 0.5,
            "realized_pnl": 500,
            "total_fees": 20,
            "total_trades": 1,
            "equity_curve": [{"date": "2026-01-05", "value": 101000}],
            "trades": [],
            "data_snapshot": {"ticker": "510300", "sha256": "a" * 64, "quality": {"status": "valid"}},
            "_data_snapshot_rows": [
                {"date": "2026-01-01", "open": 4.0, "high": 4.1, "low": 3.9, "close": 4.05, "volume": 100}
            ],
        }

    monkeypatch.setattr(experiment_module, "artifact_service", FakeArtifactService())
    experiment_store = experiment_module.BacktestExperimentStore(tmp_path / "exp.db")
    monkeypatch.setattr(experiment_module, "backtest_experiments", experiment_store)
    monkeypatch.setattr(experiment_module, "run_backtest", fake_run_backtest)

    payload = await experiment_module.run_backtest_experiment(
        objective="测试 ETF 趋势策略",
        ticker="510300",
        asset_type="etf",
        start_date="2026-01-01",
        end_date="2026-01-05",
        initial_capital=100000,
        strategy_spec={
            "name": "test_trend",
            "asset_types": ["etf"],
            "entry_conditions": [{"indicator": "return_pct", "operator": "gt", "value": 0, "window": 1}],
        },
    )

    assert payload["status"] == "completed"
    assert len(payload["artifacts"]) == 5
    assert any("历史数据快照" in item["name"] for item in payload["artifacts"])
    saved = experiment_module.backtest_experiments.get(payload["experiment_id"])
    assert saved is not None
    assert saved["result"]["final_value"] == 101000


@pytest.mark.asyncio
async def test_portfolio_experiment_persists_portfolio_artifacts(monkeypatch, tmp_path):
    import application.backtest_experiment as experiment_module

    class FakeArtifactService:
        def create_user_artifacts(self, artifacts, **_kwargs):
            return [{"artifact_id": f"artifact-{index}", "name": item["name"]} for index, item in enumerate(artifacts)]

    async def fake_run_pool(**_kwargs):
        return {
            "ticker": "pool",
            "mode": "portfolio",
            "tickers": ["510300", "159915"],
            "asset_type": "etf",
            "start_date": "2026-01-01",
            "end_date": "2026-01-05",
            "initial_capital": 100000,
            "final_value": 101500,
            "total_return": 0.015,
            "max_drawdown": 0.02,
            "sharpe_ratio": 0.5,
            "win_rate": 0.5,
            "realized_pnl": 500,
            "total_fees": 20,
            "total_trades": 2,
            "equity_curve": [{"date": "2026-01-05", "value": 101500}],
            "trades": [],
            "portfolio_spec": {
                "allocation_method": "equal_weight",
                "rebalance_frequency": "weekly",
                "max_position_weight": 0.4,
                "max_positions": 2,
                "cash_reserve": 0.1,
            },
            "portfolio_history": [{"date": "2026-01-05", "cash": 10000, "total_value": 101500, "positions": []}],
            "target_weights_history": [{"date": "2026-01-03", "weights": {"510300": 0.4}}],
            "symbol_metrics": [{"ticker": "510300", "trade_count": 1}],
            "data_snapshots": [],
            "_data_snapshot_rows": {"510300": [{"date": "2026-01-01", "close": 4.0}]},
        }

    monkeypatch.setattr(experiment_module, "artifact_service", FakeArtifactService())
    monkeypatch.setattr(experiment_module, "run_pool_backtest", fake_run_pool)
    experiment_store = experiment_module.BacktestExperimentStore(tmp_path / "portfolio.db")
    monkeypatch.setattr(experiment_module, "backtest_experiments", experiment_store)

    payload = await experiment_module.run_backtest_experiment(
        objective="测试 ETF 组合策略",
        tickers=["510300", "159915"],
        mode="portfolio",
        asset_type="etf",
        start_date="2026-01-01",
        end_date="2026-01-05",
        initial_capital=100000,
        strategy_spec={
            "name": "test_portfolio",
            "asset_types": ["etf"],
            "entry_conditions": [{"indicator": "return_pct", "operator": "gt", "value": 0, "window": 1}],
        },
        portfolio_spec={
            "allocation_method": "equal_weight",
            "rebalance_frequency": "weekly",
            "max_position_weight": 0.4,
            "max_positions": 2,
            "cash_reserve": 0.1,
        },
    )

    assert payload["mode"] == "portfolio"
    assert payload["portfolio_spec"]["allocation_method"] == "equal_weight"
    assert len(payload["artifacts"]) == 8
    assert any("持仓快照" in item["name"] for item in payload["artifacts"])
    assert any("目标权重" in item["name"] for item in payload["artifacts"])
    assert any("标的归因" in item["name"] for item in payload["artifacts"])
