import pandas as pd
import pytest
from strategy_helpers import compare_expression, strategy_mapping

from data.backtest_data import BacktestDataError, prepare_backtest_data
from engine.strategy_runtime import evaluate_strategy_intent
from models.schemas import AssetType, Decision
from strategies.compiler import available_indicators, strategy_from_mapping


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


def test_experiment_html_report_embeds_echarts_for_equity_and_drawdown():
    from application.backtest_experiment import _render_report_html

    report = _render_report_html(
        "# 实验报告\n\n## 二、回测结果\n\n- 总收益率：10%",
        "实验报告",
        result={
            "equity_curve": [
                {"date": "2026-01-01", "value": 100000},
                {"date": "2026-01-02", "value": 105000},
                {"date": "2026-01-03", "value": 102000},
            ],
            "trades": [{"date": "2026-01-02", "action": Decision.BUY}],
        },
    )

    assert "echarts.min.js" in report
    assert 'id="experiment-equity-chart"' in report
    assert 'id="experiment-drawdown-chart"' in report
    assert '"markPoint"' in report
    assert '"value":"买"' in report
    assert "echarts.init" in report


def test_indicator_contract_and_extended_indicators_are_deterministic():
    names = {item["name"] for item in available_indicators()}
    assert {"price_vs_ma_pct", "rsi", "atr", "volatility"}.issubset(names)

    spec = strategy_from_mapping(
        strategy_mapping(
            "controlled_trend",
            entry=compare_expression("price_vs_ma_pct", "gt", -5, 3),
        ),
        source="llm",
    )
    intent, _state, trace = evaluate_strategy_intent(spec, _history(), asset_type=AssetType.ETF)
    assert intent.decision == Decision.BUY
    assert trace["expression_traces"]["entry"]["left"] is not None


def test_unknown_indicator_is_rejected_before_backtest():
    try:
        strategy_from_mapping(
            {
                "name": "unsafe",
                "asset_types": ["etf"],
                "components": [
                    {
                        "id": "unsafe",
                        "expression": compare_expression("future_magic", "gt", 0),
                    }
                ],
            },
            source="llm",
        )
    except ValueError as exc:
        assert "不受支持的指标" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unknown indicators must be rejected")


def test_llm_strategy_mapping_normalizes_indicator_shapes_and_percentages():
    spec = strategy_from_mapping(
        strategy_mapping(
            "llm_shape_variants",
            indicator_specs={
                "rsi": {"type": "rsi", "period": 14, "source": "close", "role": "momentum"},
                "ma": {"length": 20, "source": "close", "role": "trend"},
            },
            entry=compare_expression("rsi", "lt", 35, 14),
            stop_loss_pct=5,
            take_profit_pct="12%",
        ),
        source="llm",
    )
    assert spec.asset_types == [AssetType.ETF]
    assert {item.name for item in spec.indicator_specs} == {"rsi", "ma"}
    assert {item.role for item in spec.indicator_specs} == {"confirmation", "filter"}
    assert spec.components[0].expression.operator == "lt"
    assert spec.components[0].expression.left.window == 14
    assert spec.stop_loss_pct == 0.05
    assert spec.take_profit_pct == 0.12


@pytest.mark.asyncio
async def test_agent_strategy_design_is_converted_to_a_validated_spec(monkeypatch):
    import application.backtest_experiment as experiment_module

    class FakeLLM:
        async def chat_json(self, prompt, system):
            assert "available_indicators" in prompt
            assert "受控指标" in system
            return strategy_mapping(
                "agent_rsi_strategy",
                entry=compare_expression("rsi", "lt", 35, 14),
                exit=compare_expression("rsi", "gt", 70, 14),
                description="Momentum with a bounded oscillator.",
            )

    monkeypatch.setattr(experiment_module, "get_llm_service", lambda: FakeLLM())
    monkeypatch.setattr(experiment_module, "register_strategy_spec", lambda spec: spec)
    spec = await experiment_module.design_strategy(
        objective="设计一个 ETF 动量策略",
        asset_type=AssetType.ETF,
        ticker="510300",
    )
    assert spec.source == "llm"
    assert spec.components[0].expression.left.indicator == "rsi"


@pytest.mark.asyncio
async def test_agent_strategy_design_repairs_invented_moving_average_aliases(monkeypatch):
    import application.backtest_experiment as experiment_module

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat_json(self, prompt, system):
            self.calls += 1
            if self.calls == 1:
                assert "不能自行发明 fast_ma、slow_ma" in system
                return {
                    "name": "broken_ma_cross",
                    "asset_types": ["etf"],
                    "components": [
                        {"id": "broken", "expression": compare_expression("fast_ma", "gt", 0)}
                    ],
                }
            assert "validation_error" in prompt
            assert "fast_ma" in prompt
            assert "修复整个策略 JSON" in system
            return strategy_mapping(
                "repaired_ma_cross",
                indicator_specs=[
                    {
                        "name": "ma_spread_pct",
                        "alias": "ma_spread_5_20",
                        "source": "close",
                        "role": "entry",
                        "params": {"fast_window": 5, "slow_window": 20},
                    }
                ],
                entry=compare_expression("ma_spread_5_20", "gt", 0),
                exit=compare_expression("ma_spread_5_20", "lte", 0),
            )

    fake_llm = FakeLLM()
    monkeypatch.setattr(experiment_module, "get_llm_service", lambda: fake_llm)
    monkeypatch.setattr(experiment_module, "register_strategy_spec", lambda spec: spec)

    spec = await experiment_module.design_strategy(
        objective="设计 MA5/20 均线交叉策略",
        asset_type=AssetType.ETF,
        ticker="510300",
    )

    assert fake_llm.calls == 2
    assert spec.indicator_specs[0].name == "ma_spread_pct"
    assert spec.indicator_specs[0].params == {"fast_window": 5, "slow_window": 20}
    assert spec.components[0].expression.left.indicator == "ma_spread_5_20"


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

    _, sourced_manifest = prepare_backtest_data(
        _history(),
        ticker="510300",
        asset_type="etf",
        start_date="2026-01-01",
        end_date="2026-01-05",
        source="sina",
        source_metadata={
            "source_id": "sina",
            "source_name": "新浪财经",
            "fallback": True,
        },
    )
    assert sourced_manifest["source"] == "sina"
    assert sourced_manifest["source_metadata"]["fallback"] is True

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
        async def create_user_artifacts(self, artifacts, **kwargs):
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
        strategy_spec=strategy_mapping(
            "test_trend",
            entry=compare_expression("return_pct", "gt", 0, 1),
        ),
    )

    assert payload["status"] == "completed"
    assert len(payload["artifacts"]) == 5
    assert any("历史数据快照" in item["name"] for item in payload["artifacts"])
    saved = await experiment_module.backtest_experiments.get(payload["experiment_id"])
    assert saved is not None
    assert saved["result"]["final_value"] == 101000


@pytest.mark.asyncio
async def test_portfolio_experiment_persists_portfolio_artifacts(monkeypatch, tmp_path):
    import application.backtest_experiment as experiment_module

    class FakeArtifactService:
        async def create_user_artifacts(self, artifacts, **_kwargs):
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
        strategy_spec=strategy_mapping(
            "test_portfolio",
            entry=compare_expression("return_pct", "gt", 0, 1),
        ),
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
