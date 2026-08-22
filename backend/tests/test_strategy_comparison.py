import pandas as pd
import pytest

import application.strategy_comparison as comparison_module
from application.strategy_comparison import build_comparison_spec, compare_strategies, standard_strategy_suite


def test_standard_suite_has_distinct_benchmark_trend_momentum_and_reversal_strategies():
    suite = standard_strategy_suite("etf")
    names = {item.name for item in suite}

    assert len(suite) == 8
    assert {
        "buy_hold",
        "ma_5_20",
        "ma_20_60",
        "momentum_20",
        "rsi_reversal",
        "bollinger_reversal",
        "breakout_20",
        "trend_pullback",
    } == names
    assert all(item.asset_types == ["etf"] or item.asset_types[0].value == "etf" for item in suite)


@pytest.mark.asyncio
async def test_formal_comparison_uses_one_snapshot_and_satisfies_full_contract(monkeypatch):
    dates = pd.date_range("2016-01-01", "2026-01-01", periods=40).strftime("%Y-%m-%d")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [10 + index * 0.1 for index in range(40)],
            "high": [10.2 + index * 0.1 for index in range(40)],
            "low": [9.8 + index * 0.1 for index in range(40)],
            "close": [10 + index * 0.1 for index in range(40)],
            "volume": [1_000_000] * 40,
        }
    )
    prepared = (
        frame,
        {
            "sha256": "b" * 64,
            "actual_start_date": dates[0],
            "actual_end_date": dates[-1],
        },
    )
    fetches = 0
    prepared_ids = set()

    async def fake_prepare(**_kwargs):
        nonlocal fetches
        fetches += 1
        return prepared

    async def fake_run(**kwargs):
        prepared_ids.add(id(kwargs["prepared_data"]))
        bump = (sum(ord(char) for char in kwargs["strategy_name"]) % 8) / 100
        curve = [
            {
                "date": day,
                "value": 1_000_000 * (1 + bump * index / 39),
                "exposure": 0.8,
            }
            for index, day in enumerate(dates)
        ]
        return {
            "total_return": bump,
            "annualized_return": bump / 10,
            "annualized_volatility": 0.1,
            "max_drawdown": 0.03,
            "sharpe_ratio": 0.8,
            "sortino_ratio": 1.0,
            "calmar_ratio": 0.5,
            "win_rate": 0.5,
            "profit_factor": 1.2,
            "exposure": 0.8,
            "turnover": 1.1,
            "total_fees": 50,
            "final_value": curve[-1]["value"],
            "total_trades": 4,
            "equity_curve": curve,
            "signal_curve": [
                {"date": day, "target_position": int(index >= 3)} for index, day in enumerate(dates)
            ],
            "trades": [],
        }

    monkeypatch.setattr(comparison_module, "prepare_single_backtest_data", fake_prepare)
    monkeypatch.setattr(comparison_module, "run_backtest", fake_run)
    spec = build_comparison_spec(
        ticker="510300",
        start_date=dates[0],
        end_date=dates[-1],
        asset_type="etf",
    )

    result = await compare_strategies(spec)

    assert fetches == 1
    assert prepared_ids == {id(prepared)}
    assert result["strategy_count"] == 8
    assert result["acceptance"]["satisfied"] is True
    assert result["acceptance"]["missing"] == []
    assert set(result["cost_scenarios"]) == {"low", "base", "stress"}
    assert all(row["diagnostics"]["out_of_sample"] for row in result["comparisons"])
    assert result["data_snapshot"]["sha256"] == "b" * 64
    expected_ranking = sorted(
        result["comparisons"],
        key=lambda row: row["total_return"],
        reverse=True,
    )
    assert result["ranking"] == [row["strategy_name"] for row in expected_ranking]
    assert result["ranking_metric"] == "total_return"
    assert result["cost_consistency"] == {"passed": True, "mismatches": []}
    comparison_by_name = {row["strategy_name"]: row for row in result["comparisons"]}
    for row in result["cost_scenarios"]["base"]:
        comparison = comparison_by_name[row["strategy_name"]]
        assert row["total_return"] == comparison["total_return"]
        assert row["total_trades"] == comparison["total_trades"]


def test_comparison_ranking_metric_follows_explicit_user_goal():
    profit = build_comparison_spec(
        ticker="510300",
        start_date="2020-01-01",
        end_date="2026-01-01",
        asset_type="etf",
        objective="对比不同策略的盈利情况",
    )
    risk_adjusted = build_comparison_spec(
        ticker="510300",
        start_date="2020-01-01",
        end_date="2026-01-01",
        asset_type="etf",
        objective="按风险调整后的夏普比率比较",
    )

    assert profit.ranking_metric == "total_return"
    assert risk_adjusted.ranking_metric == "sharpe_ratio"
