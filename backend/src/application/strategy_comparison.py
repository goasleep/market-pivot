"""Reproducible multi-strategy comparison and robustness diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from engine.backtester import prepare_single_backtest_data, run_backtest
from models.schemas import AssetType, IndicatorSpec, SimulationAccountConfig, StrategyCondition, StrategySpec
from models.strategy_research import (
    CostScenario,
    StrategyComparisonSpec,
    TaskAcceptance,
    strategy_comparison_contract,
)


def standard_strategy_suite(asset_type: AssetType | str) -> tuple[StrategySpec, ...]:
    """Return transparent benchmark strategies with materially different signals."""
    kind = AssetType(asset_type)
    assets = [kind]
    common = {"asset_types": assets, "position_size_pct": 0.95, "source": "yaml"}
    return (
        StrategySpec(
            name="buy_hold",
            description="首个可交易日买入并持有，作为含成本基准",
            indicators=["close"],
            entry_conditions=[StrategyCondition(indicator="close", operator="gt", value=0)],
            **common,
        ),
        _ma_strategy("ma_5_20", "MA5/20 趋势", 5, 20, common),
        _ma_strategy("ma_20_60", "MA20/60 趋势", 20, 60, common),
        StrategySpec(
            name="momentum_20",
            description="20 日动量为正时持有，动量转负时退出",
            indicators=["return_pct"],
            entry_conditions=[StrategyCondition(indicator="return_pct", operator="gt", value=0, window=20)],
            exit_conditions=[StrategyCondition(indicator="return_pct", operator="lte", value=0, window=20)],
            **common,
        ),
        StrategySpec(
            name="rsi_reversal",
            description="RSI 低于 30 时买入，恢复至 55 时退出",
            indicators=["rsi"],
            entry_conditions=[StrategyCondition(indicator="rsi", operator="lt", value=30, window=14)],
            exit_conditions=[StrategyCondition(indicator="rsi", operator="gte", value=55, window=14)],
            stop_loss_pct=0.08,
            **common,
        ),
        StrategySpec(
            name="bollinger_reversal",
            description="价格低于布林中轨两个标准差时买入，回归中轨时退出",
            indicators=["bollinger_zscore"],
            entry_conditions=[
                StrategyCondition(indicator="bollinger_zscore", operator="lte", value=-2, window=20)
            ],
            exit_conditions=[
                StrategyCondition(indicator="bollinger_zscore", operator="gte", value=0, window=20)
            ],
            stop_loss_pct=0.08,
            **common,
        ),
        StrategySpec(
            name="breakout_20",
            description="收盘价突破此前 20 日高点时买入，跌破 MA20 时退出",
            indicators=["rolling_breakout_pct", "price_vs_ma_pct"],
            entry_conditions=[
                StrategyCondition(indicator="rolling_breakout_pct", operator="gt", value=0, window=20)
            ],
            exit_conditions=[
                StrategyCondition(indicator="price_vs_ma_pct", operator="lt", value=0, window=20)
            ],
            **common,
        ),
        StrategySpec(
            name="trend_pullback",
            description="中期动量为正且价格回踩 MA20 附近时买入，跌破 MA20 退出",
            indicators=["return_pct", "price_vs_ma_pct"],
            entry_conditions=[
                StrategyCondition(indicator="return_pct", operator="gt", value=0, window=60),
                StrategyCondition(indicator="price_vs_ma_pct", operator="between", value=[-2, 1], window=20),
            ],
            exit_conditions=[
                StrategyCondition(indicator="price_vs_ma_pct", operator="lt", value=-3, window=20)
            ],
            stop_loss_pct=0.06,
            **common,
        ),
    )


def _ma_strategy(
    name: str,
    label: str,
    fast: int,
    slow: int,
    common: dict[str, Any],
) -> StrategySpec:
    alias = f"spread_{fast}_{slow}"
    return StrategySpec(
        name=name,
        description=f"{label} 金叉持有、死叉退出",
        indicators=[alias],
        indicator_specs=[
            IndicatorSpec(
                name="ma_spread_pct",
                alias=alias,
                role="entry",
                params={"fast_window": fast, "slow_window": slow},
            )
        ],
        entry_conditions=[StrategyCondition(indicator=alias, operator="gt", value=0)],
        exit_conditions=[StrategyCondition(indicator=alias, operator="lte", value=0)],
        **common,
    )


def default_cost_scenarios(asset_type: AssetType | str) -> tuple[CostScenario, ...]:
    kind = AssetType(asset_type)
    tax = 0.001 if kind == AssetType.STOCK else 0.0
    transfer = 0.00002 if kind == AssetType.STOCK else 0.0
    return (
        CostScenario(
            name="low",
            slippage_bps=2,
            buy_commission_rate=0.00015,
            sell_commission_rate=0.00015,
            minimum_commission=0,
            stamp_tax_rate=tax,
            transfer_fee_rate=transfer,
        ),
        CostScenario(
            name="base",
            slippage_bps=5,
            buy_commission_rate=0.0003,
            sell_commission_rate=0.0003,
            minimum_commission=5,
            stamp_tax_rate=tax,
            transfer_fee_rate=transfer,
        ),
        CostScenario(
            name="stress",
            slippage_bps=15,
            buy_commission_rate=0.0006,
            sell_commission_rate=0.0006,
            minimum_commission=5,
            stamp_tax_rate=tax,
            transfer_fee_rate=transfer,
        ),
    )


def build_comparison_spec(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    asset_type: AssetType | str,
    initial_capital: float = 1_000_000,
    strategies: Iterable[StrategySpec] | None = None,
    objective: str = "",
) -> StrategyComparisonSpec:
    kind = AssetType(asset_type)
    return StrategyComparisonSpec(
        ticker=ticker,
        asset_type=kind,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        ranking_metric=_ranking_metric_for_objective(objective),
        strategies=tuple(strategies or standard_strategy_suite(kind)),
        task_contract=strategy_comparison_contract(),
        cost_scenarios=default_cost_scenarios(kind),
    )


def _ranking_metric_for_objective(objective: str) -> str:
    """Translate the user's comparison goal into an explicit ranking metric."""
    normalized = objective.lower()
    if any(term in normalized for term in ("样本外", "稳健性", "稳健表现")):
        return "out_of_sample_return"
    if any(term in normalized for term in ("calmar", "卡玛", "回撤收益")):
        return "calmar_ratio"
    if any(term in normalized for term in ("sharpe", "夏普", "风险收益", "风险调整")):
        return "sharpe_ratio"
    return "total_return"


def _account_config(spec: StrategyComparisonSpec, scenario: CostScenario) -> SimulationAccountConfig:
    return SimulationAccountConfig(
        initial_cash=spec.initial_capital,
        asset_type=spec.asset_type,
        fill_time=spec.fill_time,
        slippage_bps=scenario.slippage_bps,
        buy_commission_rate=scenario.buy_commission_rate,
        sell_commission_rate=scenario.sell_commission_rate,
        minimum_commission=scenario.minimum_commission,
        stamp_tax_rate=scenario.stamp_tax_rate,
        transfer_fee_rate=scenario.transfer_fee_rate,
        max_single_position_pct=0.95,
        max_total_position_pct=0.95,
    )


async def compare_strategies(spec: StrategyComparisonSpec) -> dict[str, Any]:
    """Run the formal comparison on one immutable dataset and verify completion."""
    prepared = await prepare_single_backtest_data(
        ticker=spec.ticker,
        start_date=spec.start_date,
        end_date=spec.end_date,
        asset_type=spec.asset_type,
    )
    frame, snapshot = prepared
    base = next(item for item in spec.cost_scenarios if item.name == "base")

    results = []
    base_results: dict[str, dict[str, Any]] = {}
    for strategy in spec.strategies:
        result = await run_backtest(
            ticker=spec.ticker,
            start_date=spec.start_date,
            end_date=spec.end_date,
            asset_type=spec.asset_type,
            initial_capital=spec.initial_capital,
            fill_time=spec.fill_time,
            strategy_name=strategy.name,
            strategy_spec=strategy.model_dump(mode="json"),
            prepared_data=prepared,
            account_config=_account_config(spec, base),
        )
        base_results[strategy.name] = result
        results.append(_comparison_row(strategy, result))

    benchmark = next(item for item in results if item["strategy_name"] == spec.benchmark)
    for item in results:
        item["excess_return"] = round(item["total_return"] - benchmark["total_return"], 6)
        item["metrics"]["excess_return"] = item["excess_return"]
        item["diagnostics"] = _curve_diagnostics(item["equity_curve"], frame, spec.out_of_sample_ratio)

    cost_analysis = await _run_cost_scenarios(spec, prepared, base_results)
    cost_consistency = _base_cost_consistency(results, cost_analysis.get("base", []))
    sensitivity = await _run_parameter_sensitivity(spec, prepared, base)
    ranking = [
        item["strategy_name"]
        for item in sorted(
            results,
            key=lambda row: (_ranking_value(row, spec.ranking_metric), row["total_return"]),
            reverse=True,
        )
    ]
    actual_years = _history_years(snapshot)
    payload: dict[str, Any] = {
        "data_type": "strategy_backtest_comparison",
        "ticker": spec.ticker,
        "asset_type": spec.asset_type.value,
        "start_date": spec.start_date,
        "end_date": spec.end_date,
        "actual_start_date": snapshot["actual_start_date"],
        "actual_end_date": snapshot["actual_end_date"],
        "history_years": round(actual_years, 2),
        "initial_capital": spec.initial_capital,
        "strategy_count": len(results),
        "benchmark": spec.benchmark,
        "ranking_metric": spec.ranking_metric,
        "ranking_label": {
            "total_return": "总收益率",
            "sharpe_ratio": "夏普比率",
            "calmar_ratio": "卡玛比率",
            "out_of_sample_return": "样本外收益率",
        }[spec.ranking_metric],
        "task_contract": spec.task_contract.model_dump(mode="json"),
        "data_snapshot": snapshot,
        "execution": _account_config(spec, base).effective_trading_rules(spec.asset_type).model_dump(mode="json")
        | {"fill_time": spec.fill_time},
        "comparisons": results,
        "ranking": ranking,
        "cost_scenarios": cost_analysis,
        "cost_consistency": cost_consistency,
        "parameter_sensitivity": sensitivity,
    }
    payload["acceptance"] = _acceptance(spec, payload).model_dump(mode="json")
    return payload


def _comparison_row(strategy: StrategySpec, result: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        key: result.get(key)
        for key in (
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
            "exposure",
            "turnover",
            "total_fees",
        )
    }
    return {
        "strategy_name": strategy.name,
        "display_name": strategy.description.split("，", 1)[0] or strategy.name,
        "description": strategy.description,
        "strategy_spec": strategy.model_dump(mode="json"),
        "entry_rules": [item.model_dump(mode="json") for item in strategy.entry_conditions],
        "exit_rules": [item.model_dump(mode="json") for item in strategy.exit_conditions],
        **metrics,
        "metrics": metrics,
        "final_value": result.get("final_value", 0),
        "total_trades": result.get("total_trades", 0),
        "equity_curve": result.get("equity_curve", []),
        "drawdown_curve": _drawdown_curve(result.get("equity_curve", [])),
        "trades": result.get("trades", []),
        "error": result.get("error"),
    }


def _ranking_value(row: dict[str, Any], metric: str) -> float:
    if metric == "out_of_sample_return":
        value = row.get("diagnostics", {}).get("out_of_sample", {}).get("out_of_sample_return")
    else:
        value = row.get(metric)
    return float(value) if value is not None else float("-inf")


async def _run_cost_scenarios(
    spec: StrategyComparisonSpec,
    prepared: tuple[pd.DataFrame, dict[str, Any]],
    base_results: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for scenario in spec.cost_scenarios:
        rows = []
        for strategy in spec.strategies:
            if scenario.name == "base":
                # The formal result already used the base account rules. Reusing it
                # guarantees that the comparison table and base-cost row are identical.
                result = base_results[strategy.name]
            else:
                # Re-run the strategy engine under each cost model. Replaying only
                # binary positions loses price-dependent stop-loss/take-profit rules.
                result = await run_backtest(
                    ticker=spec.ticker,
                    start_date=spec.start_date,
                    end_date=spec.end_date,
                    asset_type=spec.asset_type,
                    initial_capital=spec.initial_capital,
                    fill_time=spec.fill_time,
                    strategy_name=strategy.name,
                    strategy_spec=strategy.model_dump(mode="json"),
                    prepared_data=prepared,
                    account_config=_account_config(spec, scenario),
                )
            rows.append(
                {
                    "strategy_name": strategy.name,
                    "total_return": result.get("total_return"),
                    "max_drawdown": result.get("max_drawdown"),
                    "sharpe_ratio": result.get("sharpe_ratio"),
                    "total_fees": result.get("total_fees"),
                    "final_value": result.get("final_value"),
                    "total_trades": result.get("total_trades"),
                }
            )
        output[scenario.name] = rows
    return output


def _base_cost_consistency(
    comparisons: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify the base-cost table is the same execution result shown to users."""
    by_name = {row.get("strategy_name"): row for row in base_rows}
    mismatches = []
    fields = ("total_return", "max_drawdown", "total_fees", "final_value", "total_trades")
    for comparison in comparisons:
        name = comparison.get("strategy_name")
        base = by_name.get(name)
        if base is None:
            mismatches.append({"strategy_name": name, "field": "row", "reason": "missing"})
            continue
        for field in fields:
            left = comparison.get(field)
            right = base.get(field)
            if left is None or right is None or not np.isclose(float(left), float(right), rtol=0, atol=1e-12):
                mismatches.append(
                    {"strategy_name": name, "field": field, "comparison": left, "base_scenario": right}
                )
    return {"passed": not mismatches, "mismatches": mismatches}


def _parameter_variants(strategy: StrategySpec) -> list[StrategySpec]:
    variants: list[StrategySpec] = []
    if strategy.name.startswith("ma_"):
        pairs = [(3, 15), (5, 20), (10, 30)] if strategy.name == "ma_5_20" else [(15, 50), (20, 60), (30, 90)]
        for fast, slow in pairs:
            variants.append(_ma_strategy(f"{strategy.name}_{fast}_{slow}", f"MA{fast}/{slow}", fast, slow, {
                "asset_types": strategy.asset_types,
                "position_size_pct": strategy.position_size_pct,
                "source": "yaml",
            }))
    else:
        windows = {
            "momentum_20": (10, 20, 60),
            "bollinger_reversal": (15, 20, 30),
            "breakout_20": (10, 20, 55),
        }.get(strategy.name)
        if windows:
            for window in windows:
                clone = strategy.model_copy(deep=True)
                clone.name = f"{strategy.name}_{window}"
                for condition in [*clone.entry_conditions, *clone.exit_conditions]:
                    if condition.window is not None:
                        condition.window = window
                variants.append(clone)
        elif strategy.name == "rsi_reversal":
            for entry, exit_value in ((25, 50), (30, 55), (35, 60)):
                clone = strategy.model_copy(deep=True)
                clone.name = f"{strategy.name}_{entry}_{exit_value}"
                clone.entry_conditions[0].value = float(entry)
                clone.exit_conditions[0].value = float(exit_value)
                variants.append(clone)
        elif strategy.name == "trend_pullback":
            for window in (10, 20, 30):
                clone = strategy.model_copy(deep=True)
                clone.name = f"{strategy.name}_{window}"
                for condition in [*clone.entry_conditions, *clone.exit_conditions]:
                    if condition.indicator == "price_vs_ma_pct":
                        condition.window = window
                variants.append(clone)
    return variants


async def _run_parameter_sensitivity(
    spec: StrategyComparisonSpec,
    prepared: tuple[pd.DataFrame, dict[str, Any]],
    base: CostScenario,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for strategy in spec.strategies:
        variants = _parameter_variants(strategy)
        if not variants:
            output[strategy.name] = {"status": "not_applicable", "variants": []}
            continue
        rows = []
        for variant in variants:
            result = await run_backtest(
                ticker=spec.ticker,
                start_date=spec.start_date,
                end_date=spec.end_date,
                asset_type=spec.asset_type,
                initial_capital=spec.initial_capital,
                fill_time=spec.fill_time,
                strategy_name=variant.name,
                strategy_spec=variant.model_dump(mode="json"),
                prepared_data=prepared,
                account_config=_account_config(spec, base),
            )
            rows.append(
                {
                    "variant": variant.name,
                    "total_return": result["total_return"],
                    "sharpe_ratio": result["sharpe_ratio"],
                }
            )
        returns = [float(row["total_return"]) for row in rows]
        output[strategy.name] = {
            "status": "stable" if len(returns) > 1 and np.std(returns) <= 0.15 else "sensitive",
            "return_std": round(float(np.std(returns)), 6),
            "variants": rows,
        }
    return output


def _curve_diagnostics(curve: list[dict[str, Any]], frame: pd.DataFrame, oos_ratio: float) -> dict[str, Any]:
    values = np.array([float(item["value"]) for item in curve], dtype=float)
    if values.size < 3:
        return {"out_of_sample": {}, "rolling": [], "regimes": {}, "confidence": {}}
    split = max(2, min(len(values) - 1, int(len(values) * (1 - oos_ratio))))
    returns = np.diff(values) / values[:-1]
    benchmark_returns = pd.to_numeric(frame["close"], errors="coerce").pct_change().fillna(0).to_numpy()
    rolling = []
    for end in range(252, len(values) + 1, 63):
        sample = values[end - 252 : end]
        rolling.append(
            {
                "end_date": curve[end - 1]["date"],
                "total_return": round(float(sample[-1] / sample[0] - 1), 6),
                "max_drawdown": round(_max_drawdown(sample), 6),
            }
        )
    regimes: dict[str, dict[str, float | int]] = {}
    if len(benchmark_returns) == len(values):
        trend = pd.Series(benchmark_returns).rolling(60).sum().fillna(0).to_numpy()[1:]
        volatility = pd.Series(benchmark_returns).rolling(20).std().fillna(0).to_numpy()[1:]
        vol_median = float(np.median(volatility))
        labels = np.where(volatility > vol_median, "high_volatility", np.where(trend >= 0, "uptrend", "downtrend"))
        for label in ("uptrend", "downtrend", "high_volatility"):
            sample = returns[labels == label]
            regimes[label] = {
                "days": int(sample.size),
                "annualized_return": round(float(sample.mean() * 252), 6) if sample.size else 0.0,
                "win_day_rate": round(float((sample > 0).mean()), 6) if sample.size else 0.0,
            }
    rng = np.random.default_rng(20260822)
    bootstrap = []
    if returns.size:
        for _ in range(500):
            sample = rng.choice(returns, size=returns.size, replace=True)
            bootstrap.append(float(sample.mean() * 252))
    return {
        "out_of_sample": {
            "split_date": curve[split]["date"],
            "in_sample_return": round(float(values[split - 1] / values[0] - 1), 6),
            "out_of_sample_return": round(float(values[-1] / values[split - 1] - 1), 6),
            "out_of_sample_max_drawdown": round(_max_drawdown(values[split - 1 :]), 6),
        },
        "rolling": rolling,
        "regimes": regimes,
        "confidence": {
            "annualized_return_ci_95": [
                round(float(np.percentile(bootstrap, 2.5)), 6),
                round(float(np.percentile(bootstrap, 97.5)), 6),
            ]
            if bootstrap
            else [],
            "method": "fixed-seed daily-return bootstrap",
        },
    }


def _drawdown_curve(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    peak = 0.0
    output = []
    for item in curve:
        value = float(item["value"])
        peak = max(peak, value)
        output.append({"date": item["date"], "value": round(value / peak - 1, 6) if peak else 0.0})
    return output


def _max_drawdown(values: np.ndarray) -> float:
    peaks = np.maximum.accumulate(values)
    return float(np.max((peaks - values) / peaks)) if values.size and np.all(peaks > 0) else 0.0


def _history_years(snapshot: dict[str, Any]) -> float:
    start = date.fromisoformat(str(snapshot["actual_start_date"]))
    end = date.fromisoformat(str(snapshot["actual_end_date"]))
    return max((end - start).days / 365.25, 0)


def _acceptance(spec: StrategyComparisonSpec, payload: dict[str, Any]) -> TaskAcceptance:
    contract = spec.task_contract
    rows = payload.get("comparisons", [])
    metrics_present = all(
        all(metric in row.get("metrics", {}) for metric in contract.required_metrics if metric != "excess_return")
        and "excess_return" in row
        for row in rows
    )
    checks = {
        "minimum_strategy_count": len(rows) >= contract.minimum_strategy_count,
        "required_benchmark": any(row.get("strategy_name") == contract.required_benchmark for row in rows),
        "minimum_history_years": float(payload.get("history_years", 0)) >= contract.minimum_history_years,
        "required_metrics": metrics_present,
        "comparison_table": bool(rows),
        "equity_curves": all(bool(row.get("equity_curve")) for row in rows),
        "drawdown_curves": all(bool(row.get("drawdown_curve")) for row in rows),
        "cost_scenarios": set(payload.get("cost_scenarios", {})) >= {"low", "base", "stress"},
        "base_cost_consistency": payload.get("cost_consistency", {}).get("passed") is True,
        "out_of_sample": all(bool(row.get("diagnostics", {}).get("out_of_sample")) for row in rows),
        "stability": bool(payload.get("parameter_sensitivity")),
        "shared_data_snapshot": bool(payload.get("data_snapshot", {}).get("sha256")),
    }
    return TaskAcceptance(
        satisfied=all(checks.values()),
        checks=checks,
        missing=[name for name, passed in checks.items() if not passed],
    )
