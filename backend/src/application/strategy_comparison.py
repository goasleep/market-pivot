"""Reproducible multi-strategy comparison and robustness diagnostics."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from application.comparison_artifacts import create_comparison_artifacts
from data.history_validation import prepare_cross_validated_backtest_data as prepare_single_backtest_data
from data.market_index import async_get_market_index_history
from engine.backtester import run_backtest
from llm.service import get_llm_service
from models.schemas import (
    AssetType,
    IndicatorSpec,
    SimulationAccountConfig,
    StrategySpec,
)
from models.strategy_research import (
    ComparisonConclusion,
    CostScenario,
    StrategyComparisonSpec,
    TaskAcceptance,
    strategy_comparison_contract,
)


def standard_strategy_suite(asset_type: AssetType | str) -> tuple[StrategySpec, ...]:
    """Return transparent benchmark strategies with materially different signals."""
    kind = AssetType(asset_type)
    return (
        _single_expression_strategy(
            name="buy_hold",
            description="首个可交易日买入并持有，作为含成本基准",
            asset_type=kind,
            expression=_compare("close", "gt", 0),
        ),
        _ma_strategy("ma_5_20", "MA5/20 趋势", 5, 20, kind),
        _ma_strategy("ma_20_60", "MA20/60 趋势", 20, 60, kind),
        _entry_exit_strategy(
            name="momentum_20",
            description="20 日动量为正时持有，动量转负时退出",
            asset_type=kind,
            entry=_compare("return_pct", "gt", 0, 20),
            exit=_compare("return_pct", "lte", 0, 20),
        ),
        _entry_exit_strategy(
            name="momentum_252",
            description="252 日动量为正时持有，作为长周期趋势对照",
            asset_type=kind,
            entry=_compare("return_pct", "gt", 0, 252),
            exit=_compare("return_pct", "lte", 0, 252),
        ),
        _entry_exit_strategy(
            name="rsi_reversal",
            description="RSI 低于 30 时买入，恢复至 55 时退出",
            asset_type=kind,
            entry=_compare("rsi", "lt", 30, 14),
            exit=_compare("rsi", "gte", 55, 14),
            stop_loss_pct=0.08,
        ),
        _entry_exit_strategy(
            name="bollinger_reversal",
            description="价格低于布林中轨两个标准差时买入，回归中轨时退出",
            asset_type=kind,
            entry=_compare("bollinger_zscore", "lte", -2, 20),
            exit=_compare("bollinger_zscore", "gte", 0, 20),
            stop_loss_pct=0.08,
        ),
        _entry_exit_strategy(
            name="breakout_20",
            description="收盘价突破此前 20 日高点时买入，跌破 MA20 时退出",
            asset_type=kind,
            entry=_compare("rolling_breakout_pct", "gt", 0, 20),
            exit=_compare("price_vs_ma_pct", "lt", 0, 20),
        ),
        _entry_exit_strategy(
            name="trend_pullback",
            description="中期动量为正且价格回踩 MA20 附近时买入，跌破 MA20 退出",
            asset_type=kind,
            entry={
                "type": "all",
                "children": [
                    _compare("return_pct", "gt", 0, 60),
                    _compare("price_vs_ma_pct", "between", [-2, 1], 20),
                ],
            },
            exit=_compare("price_vs_ma_pct", "lt", -3, 20),
            stop_loss_pct=0.06,
        ),
        _volatility_strategy(
            name="volatility_target_15",
            description="20 日波动率目标 15%，每周调整 0% 至 95% 的目标仓位",
            asset_type=kind,
        ),
        _volatility_strategy(
            name="trend_volatility_target",
            description="位于 MA60 上方时采用 15% 波动率目标，否则空仓",
            asset_type=kind,
            trend_window=60,
        ),
    )


def _compare(indicator: str, operator: str, value: float | list[float], window: int | None = None) -> dict:
    left = {"type": "indicator", "indicator": indicator}
    if window is not None:
        left["window"] = window
    return {
        "type": "compare",
        "left": left,
        "operator": operator,
        "right": {"type": "constant", "value": value},
    }


def _single_expression_strategy(
    *, name: str, description: str, asset_type: AssetType, expression: dict
) -> StrategySpec:
    return StrategySpec(
        name=name,
        description=description,
        asset_types=[asset_type],
        components=[
            {
                "id": "signal",
                "expression": expression,
                "score_when_true": 1,
                "score_when_false": -1,
            }
        ],
        source="yaml",
    )


def _entry_exit_strategy(
    *,
    name: str,
    description: str,
    asset_type: AssetType,
    entry: dict,
    exit: dict,
    stop_loss_pct: float | None = None,
    indicator_specs: list[IndicatorSpec] | None = None,
) -> StrategySpec:
    return StrategySpec(
        name=name,
        description=description,
        asset_types=[asset_type],
        indicator_specs=indicator_specs or [],
        stop_loss_pct=stop_loss_pct,
        components=[
            {"id": "entry", "expression": entry, "score_when_true": 1, "score_when_false": 0},
            {"id": "exit", "expression": exit, "score_when_true": -1, "score_when_false": 0},
        ],
        fusion={"type": "priority", "entry_threshold": 0.25, "exit_threshold": -0.25, "conflict_policy": "exit"},
        source="yaml",
    )


def _volatility_strategy(
    *, name: str, description: str, asset_type: AssetType, trend_window: int = 0, target: float = 0.15
) -> StrategySpec:
    params = {"volatility_window": 20, "target_volatility": target, "max_exposure": 0.95}
    if trend_window:
        params["trend_window"] = trend_window
    return StrategySpec(
        name=name,
        description=description,
        asset_types=[asset_type],
        components=[
            {
                "id": "volatility_target",
                "type": "python",
                "plugin": "core.volatility_target",
                "plugin_version": "1.0.0",
                "params": params,
            }
        ],
        position_policy={"mode": "continuous", "max_exposure": 0.95, "rebalance_frequency": "weekly"},
        source="yaml",
    )


def _ma_strategy(
    name: str,
    label: str,
    fast: int,
    slow: int,
    asset_type: AssetType,
) -> StrategySpec:
    alias = f"spread_{fast}_{slow}"
    return _entry_exit_strategy(
        name=name,
        description=f"{label} 金叉持有、死叉退出",
        asset_type=asset_type,
        indicator_specs=[
            IndicatorSpec(
                name="ma_spread_pct",
                alias=alias,
                role="entry",
                params={"fast_window": fast, "slow_window": slow},
            )
        ],
        entry=_compare(alias, "gt", 0),
        exit=_compare(alias, "lte", 0),
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
    market_benchmark_ticker: str = "000300",
    market_benchmark_name: str = "沪深300",
) -> StrategyComparisonSpec:
    kind = AssetType(asset_type)
    warmup_start = (date.fromisoformat(start_date) - timedelta(days=450)).isoformat()
    return StrategyComparisonSpec(
        ticker=ticker,
        asset_type=kind,
        start_date=start_date,
        end_date=end_date,
        requested_start_date=start_date,
        warmup_start_date=warmup_start,
        evaluation_start_date=start_date,
        evaluation_end_date=end_date,
        warmup_bars=252,
        initial_capital=initial_capital,
        market_benchmark_ticker=market_benchmark_ticker,
        market_benchmark_name=market_benchmark_name,
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


async def compare_strategies(
    spec: StrategyComparisonSpec,
    *,
    publish_artifacts: bool = False,
    generate_explanation: bool = False,
) -> dict[str, Any]:
    """Run the formal comparison on one immutable dataset and verify completion."""
    prepared_bundle = await prepare_single_backtest_data(
        ticker=spec.ticker,
        start_date=spec.warmup_start_date or spec.start_date,
        end_date=spec.evaluation_end_date or spec.end_date,
        asset_type=spec.asset_type,
    )
    cross_validated_bundle = len(prepared_bundle) == 3
    if cross_validated_bundle:
        prepared_frame, snapshot, cross_validation = prepared_bundle
    else:  # Compatibility for deterministic tests and injected legacy providers.
        prepared_frame, snapshot = prepared_bundle
        cross_validation = {
            "status": "unverified",
            "selected_source": snapshot.get("source", "injected"),
            "selection_reason": "仅提供一个注入的数据快照",
            "rule_version": "history-cross-validation-v1",
            "candidates": [],
            "comparison": {"status": "unverified"},
            "differences": [],
        }
    requested_start = spec.evaluation_start_date or spec.requested_start_date or spec.start_date
    before_requested = prepared_frame[prepared_frame["date"] < requested_start]
    if not cross_validated_bundle:
        evaluation_start_candidate = requested_start
    elif len(before_requested) >= spec.warmup_bars:
        evaluation_start_candidate = requested_start
    elif len(prepared_frame) > spec.warmup_bars:
        evaluation_start_candidate = str(prepared_frame.iloc[spec.warmup_bars]["date"])
    else:
        evaluation_start_candidate = str(prepared_frame.iloc[-1]["date"])
    frame = prepared_frame[prepared_frame["date"] <= (spec.evaluation_end_date or spec.end_date)].reset_index(drop=True)
    eligible_evaluation_rows = frame[frame["date"] >= evaluation_start_candidate]
    evaluation_start = (
        str(eligible_evaluation_rows.iloc[0]["date"])
        if not eligible_evaluation_rows.empty
        else str(frame.iloc[-1]["date"])
    )
    prepared = (
        prepared_bundle
        if not cross_validated_bundle and len(frame) == len(prepared_frame)
        else (frame, snapshot)
    )
    evaluation_frame = frame[frame["date"] >= evaluation_start].reset_index(drop=True)
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
            evaluation_start_date=evaluation_start,
        )
        base_results[strategy.name] = result
        results.append(_comparison_row(strategy, result))

    benchmark = next(item for item in results if item["strategy_name"] == spec.benchmark)
    for item in results:
        item["excess_return"] = round(item["total_return"] - benchmark["total_return"], 6)
        item["metrics"]["excess_return"] = item["excess_return"]
        item["diagnostics"] = _curve_diagnostics(item["equity_curve"], evaluation_frame, spec.out_of_sample_ratio)

    cost_analysis = await _run_cost_scenarios(spec, prepared, base_results, evaluation_start)
    cost_consistency = _base_cost_consistency(results, cost_analysis.get("base", []))
    sensitivity = await _run_parameter_sensitivity(spec, prepared, base, evaluation_start)
    market_benchmark = await _run_market_benchmark(spec, results, base, evaluation_start)
    ranking = [
        item["strategy_name"]
        for item in sorted(
            results,
            key=lambda row: (_ranking_value(row, spec.ranking_metric), row["total_return"]),
            reverse=True,
        )
    ]
    actual_years = max(
        (
            date.fromisoformat(str(snapshot["actual_end_date"]))
            - date.fromisoformat(str(evaluation_start))
        ).days
        / 365.25,
        0,
    )
    payload: dict[str, Any] = {
        "comparison_id": f"strategy-comparison-{uuid4().hex[:16]}",
        "data_type": "strategy_backtest_comparison",
        "ticker": spec.ticker,
        "asset_type": spec.asset_type.value,
        "start_date": evaluation_start,
        "end_date": spec.end_date,
        "requested_start_date": spec.requested_start_date or spec.start_date,
        "warmup_start_date": snapshot["actual_start_date"],
        "evaluation_start_date": evaluation_start,
        "evaluation_end_date": spec.evaluation_end_date or spec.end_date,
        "warmup_bars": len(frame[frame["date"] < evaluation_start]),
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
        "data_validation": cross_validation,
        "execution": _account_config(spec, base).effective_trading_rules(spec.asset_type).model_dump(mode="json")
        | {"fill_time": spec.fill_time},
        "price_curve": [
            {"date": str(row["date"]), "value": round(float(row["close"]), 6)}
            for _, row in evaluation_frame.iterrows()
        ],
        "comparisons": results,
        "ranking": ranking,
        "cost_scenarios": cost_analysis,
        "cost_consistency": cost_consistency,
        "parameter_sensitivity": sensitivity,
        "market_benchmark": market_benchmark,
    }
    payload["market_regime_attribution"] = build_market_regime_attribution(payload)
    payload["trade_attribution"] = build_trade_attribution(payload)
    payload["robustness_assessments"] = build_robustness_assessments(payload)
    payload["strategy_assessments"] = build_strategy_assessments(payload)
    payload["acceptance"] = _acceptance(spec, payload).model_dump(mode="json")
    conclusion = build_comparison_conclusion(payload, minimum_history_years=spec.task_contract.minimum_history_years)
    payload["conclusion"] = conclusion.model_dump(mode="json")
    payload["research_decision"] = build_research_decision(payload)
    if generate_explanation:
        conclusion = await enrich_comparison_conclusion(conclusion, payload)
    payload["conclusion"] = conclusion.model_dump(mode="json")
    payload["artifacts"] = []
    if publish_artifacts:
        try:
            payload["artifacts"] = await create_comparison_artifacts(
                payload,
                selected_rows=frame.to_dict(orient="records"),
            )
        except Exception as exc:
            payload["artifact_error"] = str(exc)[:500]
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
        "entry_rules": [
            item.expression.model_dump(mode="json")
            for item in strategy.components
            if item.id == "entry" and item.expression is not None
        ],
        "exit_rules": [
            item.expression.model_dump(mode="json")
            for item in strategy.components
            if item.id == "exit" and item.expression is not None
        ],
        **metrics,
        "metrics": metrics,
        "final_value": result.get("final_value", 0),
        "total_trades": result.get("total_trades", 0),
        "equity_curve": result.get("equity_curve", []),
        "signal_curve": result.get("signal_curve", []),
        "drawdown_curve": _drawdown_curve(result.get("equity_curve", [])),
        "trades": result.get("trades", []),
        "error": result.get("error"),
    }


def _compounded_return(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float((1 + clean).prod() - 1) if not clean.empty else 0.0


def build_market_regime_attribution(payload: dict[str, Any]) -> dict[str, Any]:
    """Attribute strategy returns to deterministic trend and volatility regimes."""
    price_rows = [row for row in payload.get("price_curve", []) if isinstance(row, dict)]
    if len(price_rows) < 3:
        return {"rule_version": "market-regime-v2", "available": False, "reason": "价格序列不足"}
    frame = pd.DataFrame(
        {
            "date": [str(row.get("date")) for row in price_rows],
            "close": [float(row.get("value") or 0) for row in price_rows],
        }
    )
    returns = frame["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    trend_window = min(60, max(5, len(frame) // 5))
    volatility_window = min(20, max(5, len(frame) // 10))
    trend = frame["close"].pct_change(trend_window)
    expanding_trend = frame["close"] / float(frame["close"].iloc[0]) - 1
    trend = trend.fillna(expanding_trend)
    volatility = returns.rolling(volatility_window, min_periods=max(3, volatility_window // 2)).std() * np.sqrt(252)
    valid_volatility = volatility.dropna()
    volatility_threshold = float(valid_volatility.quantile(0.67)) if not valid_volatility.empty else 0.0
    trend_threshold = max(0.02, 0.05 * trend_window / 60)
    direction = np.where(
        trend >= trend_threshold,
        "uptrend",
        np.where(trend <= -trend_threshold, "downtrend", "range"),
    )
    high_volatility = (volatility >= volatility_threshold) & volatility.notna() if volatility_threshold else False
    frame["benchmark_return"] = returns
    frame["direction"] = direction
    frame["high_volatility"] = high_volatility
    direction_labels = {"uptrend": "上涨趋势", "downtrend": "下跌趋势", "range": "横盘震荡"}
    distribution = [
        {
            "regime": label,
            "label": display,
            "days": int((frame["direction"] == label).sum()),
            "share": round(float((frame["direction"] == label).mean()), 6),
        }
        for label, display in direction_labels.items()
    ]
    distribution.append(
        {
            "regime": "high_volatility",
            "label": "高波动",
            "days": int(frame["high_volatility"].sum()),
            "share": round(float(frame["high_volatility"].mean()), 6),
        }
    )
    strategy_rows = []
    for strategy in payload.get("comparisons", []):
        if not isinstance(strategy, dict):
            continue
        equity = pd.Series(
            {
                str(point.get("date")): float(point.get("value"))
                for point in strategy.get("equity_curve", [])
                if isinstance(point, dict) and point.get("date") is not None and point.get("value") is not None
            }
        )
        aligned_equity = frame["date"].map(equity).ffill().bfill()
        strategy_returns = aligned_equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        exposure_by_date = {
            str(point.get("date")): point.get("actual_exposure", point.get("target_exposure"))
            for point in strategy.get("signal_curve", [])
            if isinstance(point, dict) and point.get("date") is not None
        }
        if not any(value is not None for value in exposure_by_date.values()):
            exposure_by_date = {
                str(point.get("date")): point.get("exposure")
                for point in strategy.get("equity_curve", [])
                if isinstance(point, dict) and point.get("date") is not None
            }
        exposure = pd.to_numeric(frame["date"].map(exposure_by_date), errors="coerce")
        regimes = []
        masks = [(label, display, frame["direction"] == label) for label, display in direction_labels.items()]
        masks.append(("high_volatility", "高波动", frame["high_volatility"]))
        for label, display, mask in masks:
            strategy_sample = strategy_returns[mask]
            benchmark_sample = frame.loc[mask, "benchmark_return"]
            strategy_return = _compounded_return(strategy_sample)
            benchmark_return = _compounded_return(benchmark_sample)
            regimes.append(
                {
                    "regime": label,
                    "label": display,
                    "days": int(mask.sum()),
                    "strategy_return": round(strategy_return, 6),
                    "benchmark_return": round(benchmark_return, 6),
                    "excess_return": round(strategy_return - benchmark_return, 6),
                    "win_day_rate": round(float((strategy_sample > 0).mean()), 6)
                    if not strategy_sample.empty
                    else 0.0,
                    "average_exposure": round(float(exposure[mask].dropna().mean()), 6)
                    if not exposure[mask].dropna().empty
                    else None,
                    "worst_day": round(float(strategy_sample.min()), 6) if not strategy_sample.empty else 0.0,
                }
            )
        strategy_rows.append(
            {
                "strategy_name": strategy.get("strategy_name"),
                "display_name": strategy.get("display_name"),
                "regimes": regimes,
            }
        )
    return {
        "rule_version": "market-regime-v2",
        "available": True,
        "trend_window": trend_window,
        "trend_threshold": round(trend_threshold, 6),
        "volatility_window": volatility_window,
        "high_volatility_threshold": round(volatility_threshold, 6),
        "distribution": distribution,
        "strategies": strategy_rows,
    }


def _holding_days(buy_date: str, sell_date: str) -> int:
    try:
        return max((date.fromisoformat(sell_date) - date.fromisoformat(buy_date)).days, 0)
    except ValueError:
        return 0


def _one_strategy_trade_attribution(strategy: dict[str, Any]) -> dict[str, Any]:
    lots: deque[dict[str, Any]] = deque()
    matched = []
    for trade in strategy.get("trades", []):
        if not isinstance(trade, dict):
            continue
        action = str(trade.get("action") or "").lower()
        shares = int(trade.get("shares") or 0)
        price = float(trade.get("price") or 0)
        if shares <= 0 or price <= 0:
            continue
        costs = sum(float(trade.get(key) or 0) for key in ("commission", "tax", "transfer_fee"))
        if action == "buy":
            lots.append(
                {
                    "date": str(trade.get("date") or ""),
                    "shares": shares,
                    "cost_per_share": (price * shares + costs) / shares,
                }
            )
            continue
        if action != "sell":
            continue
        remaining = shares
        net_sell_per_share = (price * shares - costs) / shares
        while remaining > 0 and lots:
            lot = lots[0]
            matched_shares = min(remaining, int(lot["shares"]))
            cost_basis = float(lot["cost_per_share"]) * matched_shares
            proceeds = net_sell_per_share * matched_shares
            pnl = proceeds - cost_basis
            matched.append(
                {
                    "buy_date": lot["date"],
                    "sell_date": str(trade.get("date") or ""),
                    "shares": matched_shares,
                    "holding_days": _holding_days(str(lot["date"]), str(trade.get("date") or "")),
                    "pnl": round(pnl, 6),
                    "return_pct": round(pnl / cost_basis, 6) if cost_basis else 0.0,
                }
            )
            lot["shares"] = int(lot["shares"]) - matched_shares
            remaining -= matched_shares
            if int(lot["shares"]) <= 0:
                lots.popleft()
    profits = [float(item["pnl"]) for item in matched if float(item["pnl"]) > 0]
    losses = [float(item["pnl"]) for item in matched if float(item["pnl"]) < 0]
    consecutive_losses = 0
    max_consecutive_losses = 0
    for item in matched:
        consecutive_losses = consecutive_losses + 1 if float(item["pnl"]) < 0 else 0
        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    total_positive = sum(profits)
    top_profit = sum(sorted(profits, reverse=True)[:3])
    return {
        "strategy_name": strategy.get("strategy_name"),
        "display_name": strategy.get("display_name"),
        "method": "FIFO matched fills including recorded fees",
        "closed_trade_segments": len(matched),
        "open_shares": sum(int(lot["shares"]) for lot in lots),
        "realized_pnl": round(sum(float(item["pnl"]) for item in matched), 6),
        "win_rate": round(len(profits) / len(matched), 6) if matched else None,
        "average_win": round(float(np.mean(profits)), 6) if profits else None,
        "average_loss": round(float(np.mean(losses)), 6) if losses else None,
        "payoff_ratio": round(float(np.mean(profits)) / abs(float(np.mean(losses))), 6)
        if profits and losses
        else None,
        "average_holding_days": round(float(np.mean([item["holding_days"] for item in matched])), 2)
        if matched
        else None,
        "max_consecutive_losses": max_consecutive_losses,
        "top3_profit_concentration": round(top_profit / total_positive, 6) if total_positive > 0 else None,
        "matched_trades": matched,
        "best_trades": sorted(matched, key=lambda item: float(item["pnl"]), reverse=True)[:3],
        "worst_trades": sorted(matched, key=lambda item: float(item["pnl"]))[:3],
    }


def build_trade_attribution(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [
        _one_strategy_trade_attribution(strategy)
        for strategy in payload.get("comparisons", [])
        if isinstance(strategy, dict)
    ]
    return {"rule_version": "fifo-trade-attribution-v1", "strategies": rows}


def build_robustness_assessments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sensitivity = payload.get("parameter_sensitivity") or {}
    stress_by_name = {
        str(row.get("strategy_name")): row
        for row in (payload.get("cost_scenarios", {}).get("stress") or [])
        if isinstance(row, dict)
    }
    ranking = {str(name): index for index, name in enumerate(payload.get("ranking") or [], 1)}
    output = []
    for row in payload.get("comparisons", []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("strategy_name"))
        diagnostics = row.get("diagnostics") or {}
        rolling = [item for item in diagnostics.get("rolling", []) if isinstance(item, dict)]
        rolling_returns = [float(item["total_return"]) for item in rolling if item.get("total_return") is not None]
        parameter = sensitivity.get(name) or {}
        variants = [item for item in parameter.get("variants", []) if isinstance(item, dict)]
        variant_returns = [float(item["total_return"]) for item in variants if item.get("total_return") is not None]
        oos = diagnostics.get("out_of_sample", {}).get("out_of_sample_return")
        stress_return = stress_by_name.get(name, {}).get("total_return")
        conditions = {
            "out_of_sample_positive": oos is not None and float(oos) > 0,
            "rolling_positive_majority": bool(rolling_returns)
            and sum(value > 0 for value in rolling_returns) / len(rolling_returns) >= 0.6,
            "parameters_stable": parameter.get("status") in {"stable", "not_applicable"},
            "stress_cost_positive": stress_return is not None and float(stress_return) > 0,
        }
        passed = sum(conditions.values())
        grade = "strong" if passed == 4 else "moderate" if passed >= 2 else "weak"
        output.append(
            {
                "strategy_name": name,
                "display_name": row.get("display_name"),
                "rank": ranking.get(name),
                "grade": grade,
                "checks": conditions,
                "rolling_window_count": len(rolling_returns),
                "rolling_positive_ratio": round(
                    sum(value > 0 for value in rolling_returns) / len(rolling_returns), 6
                )
                if rolling_returns
                else None,
                "rolling_median_return": round(float(np.median(rolling_returns)), 6)
                if rolling_returns
                else None,
                "rolling_worst_return": round(min(rolling_returns), 6) if rolling_returns else None,
                "parameter_status": parameter.get("status", "unknown"),
                "parameter_variant_count": len(variant_returns),
                "parameter_return_range": [round(min(variant_returns), 6), round(max(variant_returns), 6)]
                if variant_returns
                else [],
                "out_of_sample_return": oos,
                "stress_total_return": stress_return,
                "cost_degradation": round(float(row.get("total_return") or 0) - float(stress_return), 6)
                if stress_return is not None
                else None,
                "annualized_return_ci_95": diagnostics.get("confidence", {}).get("annualized_return_ci_95") or [],
            }
        )
    return sorted(output, key=lambda item: (item.get("rank") or 10_000, str(item["strategy_name"])))


def _strategy_context(strategy_name: str) -> tuple[str, str]:
    contexts = {
        "buy_hold": (
            "持续上行或大级别趋势行情，用于观察标的本身的收益与完整回撤",
            "下跌和长时间震荡阶段会持续暴露，无法主动控制回撤",
        ),
        "ma_5_20": (
            "中短期方向明确、趋势延续性较强的行情",
            "横盘震荡时均线频繁交叉，容易反复买卖并累积成本",
        ),
        "ma_20_60": (
            "持续时间较长的中期趋势行情",
            "信号确认较慢，快速反转或 V 形修复时可能晚进晚出",
        ),
        "momentum_20": (
            "短中期动量持续、涨跌方向较清晰的行情",
            "动量在零轴附近反复切换时容易产生来回交易",
        ),
        "momentum_252": (
            "长周期趋势稳定、愿意降低交易频率的行情",
            "对新趋势和快速反转反应较慢，可能错过行情早段",
        ),
        "rsi_reversal": (
            "震荡或超跌后容易均值修复的行情",
            "单边下跌中超卖可以持续，抄底信号可能过早",
        ),
        "bollinger_reversal": (
            "价格围绕中枢震荡、极端偏离后容易回归的行情",
            "趋势突破阶段价格可能持续偏离中轨，反转假设会失效",
        ),
        "breakout_20": (
            "趋势启动或加速、突破后有延续性的行情",
            "假突破和震荡区间会导致追高后快速止损",
        ),
        "trend_pullback": (
            "中期上升趋势中的有序回踩行情",
            "趋势已反转却被误判为回踩时，入场后可能继续下跌",
        ),
        "volatility_target_15": (
            "波动水平变化明显、希望主动约束仓位风险的行情",
            "急跌后快速反弹时仓位恢复偏慢，可能损失上涨弹性",
        ),
        "trend_volatility_target": (
            "趋势向上且需要随波动动态控制仓位的行情",
            "V 形反转或均线附近震荡时，趋势过滤会造成空仓或反复切换",
        ),
    }
    return contexts.get(
        strategy_name,
        ("满足该策略入场条件且信号具有延续性的行情", "信号快速反转或市场结构变化时可能失效"),
    )


def _metric_ranks(
    rows: list[dict[str, Any]],
    value_getter,
    *,
    reverse: bool,
) -> dict[str, int]:
    available = [(row, value_getter(row)) for row in rows]
    available = [(row, float(value)) for row, value in available if value is not None]
    ordered = sorted(available, key=lambda item: item[1], reverse=reverse)
    return {str(row.get("strategy_name")): index for index, (row, _) in enumerate(ordered, 1)}


def build_strategy_assessments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Explain each strategy from frozen definitions and relative backtest evidence."""
    rows = [row for row in payload.get("comparisons", []) if isinstance(row, dict) and not row.get("error")]
    if not rows:
        return []
    stress_by_name = {
        str(row.get("strategy_name")): row
        for row in (payload.get("cost_scenarios", {}).get("stress") or [])
        if isinstance(row, dict)
    }
    sensitivity = payload.get("parameter_sensitivity") or {}
    regime_by_name = {
        str(item.get("strategy_name")): item
        for item in (payload.get("market_regime_attribution", {}).get("strategies") or [])
        if isinstance(item, dict)
    }
    trade_by_name = {
        str(item.get("strategy_name")): item
        for item in (payload.get("trade_attribution", {}).get("strategies") or [])
        if isinstance(item, dict)
    }
    robustness_by_name = {
        str(item.get("strategy_name")): item
        for item in (payload.get("robustness_assessments") or [])
        if isinstance(item, dict)
    }
    ranking = {str(name): index for index, name in enumerate(payload.get("ranking") or [], 1)}
    count = len(rows)
    top_count = min(3, count)
    return_ranks = _metric_ranks(rows, lambda row: row.get("total_return"), reverse=True)
    sharpe_ranks = _metric_ranks(rows, lambda row: row.get("sharpe_ratio"), reverse=True)
    drawdown_ranks = _metric_ranks(rows, lambda row: row.get("max_drawdown"), reverse=False)
    oos_ranks = _metric_ranks(
        rows,
        lambda row: row.get("diagnostics", {}).get("out_of_sample", {}).get("out_of_sample_return"),
        reverse=True,
    )
    stress_ranks = _metric_ranks(
        rows,
        lambda row: stress_by_name.get(str(row.get("strategy_name")), {}).get("total_return"),
        reverse=True,
    )
    turnovers = [float(row["turnover"]) for row in rows if row.get("turnover") is not None]
    median_turnover = float(np.median(turnovers)) if turnovers else 0.0
    assessments = []
    for row in rows:
        name = str(row.get("strategy_name"))
        display_name = str(row.get("display_name") or name)
        total_return = float(row.get("total_return") or 0)
        max_drawdown = float(row.get("max_drawdown") or 0)
        sharpe = row.get("sharpe_ratio")
        sharpe_value = float(sharpe) if sharpe is not None else None
        oos = row.get("diagnostics", {}).get("out_of_sample", {}).get("out_of_sample_return")
        oos_value = float(oos) if oos is not None else None
        stress_return = stress_by_name.get(name, {}).get("total_return")
        stress_value = float(stress_return) if stress_return is not None else None
        strengths = []
        if return_ranks.get(name, count + 1) <= top_count:
            strengths.append(f"总收益 {total_return:+.2%}，排名 {return_ranks[name]}/{count}")
        if sharpe_value is not None and sharpe_ranks.get(name, count + 1) <= top_count:
            strengths.append(f"夏普 {sharpe_value:.3f}，排名 {sharpe_ranks[name]}/{count}")
        if drawdown_ranks.get(name, count + 1) <= top_count:
            strengths.append(f"最大回撤 {max_drawdown:.2%}，控制排名 {drawdown_ranks[name]}/{count}")
        if oos_value is not None and oos_ranks.get(name, count + 1) <= top_count:
            strengths.append(f"样本外收益 {oos_value:+.2%}，排名 {oos_ranks[name]}/{count}")
        if stress_value is not None and stress_ranks.get(name, count + 1) <= top_count:
            strengths.append(f"压力成本收益 {stress_value:+.2%}，排名 {stress_ranks[name]}/{count}")
        if not strengths and total_return > 0:
            strengths.append(f"评价期仍取得 {total_return:+.2%} 正收益")

        regime_rows = [
            item
            for item in (regime_by_name.get(name, {}).get("regimes") or [])
            if isinstance(item, dict) and item.get("regime") in {"uptrend", "downtrend", "range"}
        ]
        strongest_regime = max(regime_rows, key=lambda item: float(item.get("strategy_return") or 0), default=None)
        weakest_regime = min(regime_rows, key=lambda item: float(item.get("strategy_return") or 0), default=None)
        if strongest_regime and float(strongest_regime.get("strategy_return") or 0) > 0:
            strengths.append(
                f"{strongest_regime.get('label')}阶段收益 {float(strongest_regime['strategy_return']):+.2%}"
            )

        weaknesses = []
        if total_return < 0:
            weaknesses.append(f"评价期亏损 {total_return:.2%}")
        elif return_ranks.get(name, 0) > max(top_count, int(np.ceil(count * 0.67))):
            weaknesses.append(f"总收益仅排名 {return_ranks[name]}/{count}")
        if sharpe_value is not None and sharpe_value <= 0:
            weaknesses.append(f"夏普为 {sharpe_value:.3f}，风险补偿不足")
        if oos_value is not None and oos_value < 0:
            weaknesses.append(f"样本外收益为 {oos_value:.2%}")
        if stress_value is not None and stress_value < 0:
            weaknesses.append(f"压力成本下收益降至 {stress_value:.2%}")
        turnover = float(row.get("turnover") or 0)
        if median_turnover > 0 and turnover > median_turnover * 1.5:
            weaknesses.append(f"换手率 {turnover:.2f}，高于策略中位数，成本敏感")
        sensitivity_status = str((sensitivity.get(name) or {}).get("status") or "unknown")
        if sensitivity_status not in {"stable", "unknown"}:
            weaknesses.append(f"参数敏感性状态为 {sensitivity_status}")
        if weakest_regime and float(weakest_regime.get("strategy_return") or 0) < 0:
            weaknesses.append(
                f"{weakest_regime.get('label')}阶段收益 {float(weakest_regime['strategy_return']):.2%}"
            )
        trade_attribution = trade_by_name.get(name) or {}
        concentration = trade_attribution.get("top3_profit_concentration")
        if concentration is not None and float(concentration) > 0.6:
            weaknesses.append(f"前三笔盈利贡献 {float(concentration):.1%}，收益集中度偏高")
        payoff_ratio = trade_attribution.get("payoff_ratio")
        if payoff_ratio is not None and float(payoff_ratio) >= 1.5:
            strengths.append(f"平均盈亏比 {float(payoff_ratio):.2f}")
        max_consecutive_losses = int(trade_attribution.get("max_consecutive_losses") or 0)
        if max_consecutive_losses >= 3:
            weaknesses.append(f"最长连续亏损 {max_consecutive_losses} 笔")
        if not weaknesses:
            weaknesses.append("当前样本未暴露突出短板，仍需关注策略固有失效场景")

        leading_dimensions = sum(
            rank_map.get(name, count + 1) <= top_count
            for rank_map in (sharpe_ranks, drawdown_ranks, oos_ranks, stress_ranks)
        )
        robustness = robustness_by_name.get(name) or {}
        robustness_grade = str(robustness.get("grade") or "unknown")
        if total_return < 0 and (sharpe_value is None or sharpe_value <= 0):
            verdict = "当前不建议作为主策略"
        elif leading_dimensions >= 2 and total_return >= 0 and robustness_grade != "weak":
            verdict = "优先候选"
        elif return_ranks.get(name, count + 1) <= top_count or leading_dimensions >= 1:
            verdict = "有条件候选"
        else:
            verdict = "对照或备选"
        suitable_market, failure_mode = _strategy_context(name)
        assessments.append(
            {
                "strategy_name": name,
                "display_name": display_name,
                "rank": ranking.get(name, return_ranks.get(name, count)),
                "mechanism": row.get("description") or display_name,
                "suitable_market": suitable_market,
                "failure_mode": failure_mode,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "verdict": verdict,
                "why_good": "；".join(strengths),
                "why_bad": "；".join(weaknesses),
                "strongest_regime": strongest_regime,
                "weakest_regime": weakest_regime,
                "trade_attribution": trade_attribution,
                "robustness_grade": robustness_grade,
            }
        )
    return sorted(assessments, key=lambda item: (int(item.get("rank") or count), str(item["strategy_name"])))


def build_research_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn frozen diagnostics into falsification tests and a deterministic review gate."""
    conclusion = payload.get("conclusion") or {}
    preferred = conclusion.get("risk_adjusted_winner") or conclusion.get("robustness_winner") or {}
    preferred_name = str(preferred.get("strategy_name") or (payload.get("ranking") or [""])[0])
    assessments = {
        str(item.get("strategy_name")): item
        for item in payload.get("strategy_assessments", [])
        if isinstance(item, dict)
    }
    robustness = {
        str(item.get("strategy_name")): item
        for item in payload.get("robustness_assessments", [])
        if isinstance(item, dict)
    }
    trades = {
        str(item.get("strategy_name")): item
        for item in payload.get("trade_attribution", {}).get("strategies", [])
        if isinstance(item, dict)
    }
    assessment = assessments.get(preferred_name) or {}
    robust = robustness.get(preferred_name) or {}
    trade = trades.get(preferred_name) or {}
    weakest_regime = assessment.get("weakest_regime") or {}
    falsification_risks = [
        f"如果后续处于{assessment.get('failure_mode')}，当前推荐依据可能失效。"
        if assessment.get("failure_mode")
        else "如果市场结构发生变化，当前推荐依据可能失效。"
    ]
    if weakest_regime and float(weakest_regime.get("strategy_return") or 0) < 0:
        falsification_risks.append(
            f"该策略在{weakest_regime.get('label')}阶段收益为 "
            f"{float(weakest_regime.get('strategy_return') or 0):.2%}；若未来该阶段占比上升，整体优势可能消失。"
        )
    concentration = trade.get("top3_profit_concentration")
    if concentration is not None and float(concentration) > 0.5:
        falsification_risks.append(
            f"前三笔盈利贡献 {float(concentration):.1%}；移除少数大盈利后，结论可能反转。"
        )
    if robust.get("grade") == "weak":
        falsification_risks.append("滚动、样本外、参数或压力成本检查多数未通过，当前优势可能依赖特定样本。")
    validation = payload.get("data_validation") or {}
    if validation.get("status") != "verified":
        falsification_risks.append("行情交叉核验未达到 verified，换用独立数据源可能改变排名。")

    next_experiments = []
    if weakest_regime:
        next_experiments.append(
            {
                "id": "weak-regime-walk-forward",
                "question": f"策略在{weakest_regime.get('label')}阶段是否仍有可接受表现？",
                "method": "按时间顺序滚动切分，并单独报告该市场阶段的收益、回撤、仓位和相对标的超额。",
                "success_criteria": "多个滚动窗口中结论方向一致，且最弱阶段不再持续侵蚀全部趋势期收益。",
            }
        )
    if int(trade.get("closed_trade_segments") or 0) > 0:
        next_experiments.append(
            {
                "id": "leave-top-trades-out",
                "question": "策略收益是否依赖少数大盈利交易？",
                "method": "依次移除盈利最大的 1 笔和 3 笔闭合交易，重新计算收益、夏普和最大回撤。",
                "success_criteria": "移除前三笔盈利后仍保持正收益，且策略相对排名不发生根本反转。",
            }
        )
    if robust.get("parameter_status") not in {"stable", "not_applicable"}:
        next_experiments.append(
            {
                "id": "walk-forward-parameters",
                "question": "当前参数是否只是样本内偶然最优？",
                "method": "只在训练窗口选择参数，在后续窗口冻结执行，并与相邻参数同步比较。",
                "success_criteria": "相邻参数多数保持正收益，且样本外排名与当前结论一致。",
            }
        )
    if validation.get("status") != "verified":
        next_experiments.append(
            {
                "id": "independent-source-replay",
                "question": "更换独立行情源后结论是否可复现？",
                "method": "使用相同策略版本、区间和成交假设，在独立行情源上完整重放。",
                "success_criteria": "核心指标差异可解释，首选策略与主要风险结论不变。",
            }
        )
    next_experiments.append(
        {
            "id": "forward-paper-observation",
            "question": "冻结策略在未来未见数据上是否保持行为一致？",
            "method": "冻结策略版本进入模拟盘，记录每次信号、成交、滑点偏差和失效原因，不在观察期调参。",
            "success_criteria": "完成至少一个完整入场—退出周期，实际执行偏差未突破既定成本压力假设。",
        }
    )

    robust_checks = robust.get("checks") or {}
    closed_segments = int(trade.get("closed_trade_segments") or 0)
    gate_checks = {
        "task_acceptance": payload.get("acceptance", {}).get("satisfied") is True,
        "official_sample": conclusion.get("official") is True,
        "data_cross_validation": validation.get("status") == "verified",
        "out_of_sample_positive": robust_checks.get("out_of_sample_positive") is True,
        "rolling_positive_majority": robust_checks.get("rolling_positive_majority") is True,
        "parameters_stable": robust_checks.get("parameters_stable") is True,
        "stress_cost_positive": robust_checks.get("stress_cost_positive") is True,
        "trade_sample_sufficient": preferred_name == "buy_hold" or closed_segments >= 20,
        "profit_not_over_concentrated": concentration is None or float(concentration) <= 0.5,
    }
    missing = [name for name, passed in gate_checks.items() if not passed]
    gate_status = "eligible_for_manual_review" if not missing else "research_only"
    return {
        "rule_version": "strategy-research-decision-v1",
        "preferred_strategy": preferred_name,
        "preferred_display_name": preferred.get("display_name") or assessment.get("display_name") or preferred_name,
        "current_verdict": assessment.get("verdict") or "待判断",
        "robustness_grade": robust.get("grade") or "unknown",
        "falsification_risks": falsification_risks,
        "next_experiments": next_experiments,
        "deployment_gate": {
            "status": gate_status,
            "checks": gate_checks,
            "missing": missing,
            "message": (
                "可提交人工审核；仍不得自动部署。"
                if gate_status == "eligible_for_manual_review"
                else "仅限研究与模拟验证，补齐缺失检查后再考虑人工审核。"
            ),
        },
    }


def _metric_difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        return round(float(left) - float(right), 6)
    except (TypeError, ValueError):
        return None


async def _run_market_benchmark(
    spec: StrategyComparisonSpec,
    target_rows: list[dict[str, Any]],
    base_cost: CostScenario,
    evaluation_start: str,
) -> dict[str, Any]:
    """Apply every strategy to a broad-market index without blocking target conclusions on failure."""
    try:
        market_frame, snapshot = await async_get_market_index_history(
            spec.market_benchmark_ticker,
            start_date=spec.warmup_start_date or spec.start_date,
            end_date=spec.evaluation_end_date or spec.end_date,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "ticker": spec.market_benchmark_ticker,
            "name": spec.market_benchmark_name,
            "error": str(exc)[:500],
            "comparisons": [],
        }

    market_frame = market_frame[
        market_frame["date"] <= (spec.evaluation_end_date or spec.end_date)
    ].reset_index(drop=True)
    evaluation_rows = market_frame[market_frame["date"] >= evaluation_start]
    if len(evaluation_rows) < 2:
        return {
            "status": "unavailable",
            "ticker": spec.market_benchmark_ticker,
            "name": spec.market_benchmark_name,
            "error": "同期大盘指数不足 2 个可用交易日",
            "snapshot": snapshot,
            "comparisons": [],
        }
    market_evaluation_start = str(evaluation_rows.iloc[0]["date"])
    prepared = (market_frame, snapshot)
    target_by_name = {str(row.get("strategy_name")): row for row in target_rows}
    comparisons = []
    for strategy in spec.strategies:
        target = target_by_name.get(strategy.name, {})
        try:
            market_result = await run_backtest(
                ticker=spec.market_benchmark_ticker,
                start_date=spec.start_date,
                end_date=spec.end_date,
                asset_type=spec.asset_type,
                initial_capital=spec.initial_capital,
                fill_time=spec.fill_time,
                strategy_name=strategy.name,
                strategy_spec=strategy.model_dump(mode="json"),
                prepared_data=prepared,
                account_config=_account_config(spec, base_cost),
                evaluation_start_date=market_evaluation_start,
            )
            market_error = market_result.get("error")
        except Exception as exc:
            market_result = {}
            market_error = str(exc)[:500]
        market_total_return = None if market_error else market_result.get("total_return")
        market_max_drawdown = None if market_error else market_result.get("max_drawdown")
        market_sharpe_ratio = None if market_error else market_result.get("sharpe_ratio")
        comparisons.append(
            {
                "strategy_name": strategy.name,
                "display_name": strategy.description.split("，", 1)[0] or strategy.name,
                "asset_total_return": target.get("total_return"),
                "market_total_return": market_total_return,
                "excess_return": _metric_difference(target.get("total_return"), market_total_return),
                "asset_max_drawdown": target.get("max_drawdown"),
                "market_max_drawdown": market_max_drawdown,
                "drawdown_improvement": _metric_difference(market_max_drawdown, target.get("max_drawdown")),
                "asset_sharpe_ratio": target.get("sharpe_ratio"),
                "market_sharpe_ratio": market_sharpe_ratio,
                "asset_equity_curve": target.get("equity_curve") or [],
                "market_equity_curve": [] if market_error else market_result.get("equity_curve") or [],
                "market_error": market_error,
            }
        )
    errors = [row for row in comparisons if row.get("market_error")]
    target_dates = {str(point.get("date")) for row in target_rows for point in (row.get("equity_curve") or [])}
    market_dates = set(evaluation_rows["date"].astype(str))
    coverage_ratio = len(target_dates & market_dates) / max(len(target_dates), 1)
    return {
        "status": "partial" if errors else "available",
        "ticker": spec.market_benchmark_ticker,
        "name": spec.market_benchmark_name,
        "evaluation_start_date": market_evaluation_start,
        "evaluation_end_date": str(evaluation_rows.iloc[-1]["date"]),
        "coverage_ratio": round(coverage_ratio, 6),
        "snapshot": snapshot,
        "comparisons": comparisons,
        "simulation_note": "大盘指数不可直接交易；此处使用相同策略、评价期和成交成本进行指数代理模拟。",
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
    evaluation_start_date: str,
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
                    evaluation_start_date=evaluation_start_date,
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
    volatility_component = next(
        (item for item in strategy.components if item.plugin == "core.volatility_target"),
        None,
    )
    if volatility_component is not None:
        for target in (0.10, 0.15, 0.20):
            clone = strategy.model_copy(deep=True)
            clone.name = f"{strategy.name}_{int(target * 100)}"
            next(item for item in clone.components if item.plugin == "core.volatility_target").params[
                "target_volatility"
            ] = target
            variants.append(clone)
        return variants
    if strategy.name.startswith("ma_"):
        pairs = [(3, 15), (5, 20), (10, 30)] if strategy.name == "ma_5_20" else [(15, 50), (20, 60), (30, 90)]
        for fast, slow in pairs:
            variants.append(
                _ma_strategy(
                    f"{strategy.name}_{fast}_{slow}",
                    f"MA{fast}/{slow}",
                    fast,
                    slow,
                    strategy.asset_types[0],
                )
            )
    else:
        windows = {
            "momentum_20": (10, 20, 60),
            "momentum_252": (126, 189, 252),
            "bollinger_reversal": (15, 20, 30),
            "breakout_20": (10, 20, 55),
        }.get(strategy.name)
        if windows:
            for window in windows:
                clone = strategy.model_copy(deep=True)
                clone.name = f"{strategy.name}_{window}"
                for component in clone.components:
                    _update_expression_window(component.expression, window)
                variants.append(clone)
        elif strategy.name == "rsi_reversal":
            for entry, exit_value in ((25, 50), (30, 55), (35, 60)):
                clone = strategy.model_copy(deep=True)
                clone.name = f"{strategy.name}_{entry}_{exit_value}"
                clone.components[0].expression.right.value = float(entry)
                clone.components[1].expression.right.value = float(exit_value)
                variants.append(clone)
        elif strategy.name == "trend_pullback":
            for window in (10, 20, 30):
                clone = strategy.model_copy(deep=True)
                clone.name = f"{strategy.name}_{window}"
                for component in clone.components:
                    _update_expression_window(component.expression, window, indicator="price_vs_ma_pct")
                variants.append(clone)
    return variants


def _update_expression_window(expression, window: int, *, indicator: str | None = None) -> None:
    if expression is None:
        return
    for operand in (expression.left, expression.right):
        if operand is not None and operand.type == "indicator" and (
            indicator is None or operand.indicator == indicator
        ):
            operand.window = window
    for child in expression.children:
        _update_expression_window(child, window, indicator=indicator)
    _update_expression_window(expression.expression, window, indicator=indicator)


async def _run_parameter_sensitivity(
    spec: StrategyComparisonSpec,
    prepared: tuple[pd.DataFrame, dict[str, Any]],
    base: CostScenario,
    evaluation_start_date: str,
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
                evaluation_start_date=evaluation_start_date,
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
        "cross_validation_attempted": bool(payload.get("data_validation", {}).get("rule_version")),
        "fair_evaluation_period": all(
            row.get("equity_curve")
            and row["equity_curve"][0].get("date") == payload.get("evaluation_start_date")
            for row in rows
        ),
    }
    return TaskAcceptance(
        satisfied=all(checks.values()),
        checks=checks,
        missing=[name for name, passed in checks.items() if not passed],
    )


def _winner(row: dict[str, Any] | None, metric: str, value: Any) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "strategy_name": row.get("strategy_name"),
        "display_name": row.get("display_name"),
        "metric": metric,
        "value": value,
    }


def _display_name(row: dict[str, Any] | None) -> str:
    if not row:
        return "该策略"
    return str(row.get("display_name") or row.get("strategy_name") or "该策略")


def _comparison_recommendations(
    *,
    total: dict[str, Any],
    sharpe: dict[str, Any] | None,
    drawdown: dict[str, Any],
    out_of_sample: dict[str, Any] | None,
    robustness: dict[str, Any],
    official: bool,
) -> list[str]:
    """Create a deterministic fallback recommendation from frozen winners."""
    primary = sharpe or drawdown or robustness or out_of_sample or total
    primary_key = primary.get("strategy_name")
    dimensions = [
        label
        for label, row in (
            ("风险调整表现", sharpe),
            ("回撤控制", drawdown),
            ("压力成本稳健性", robustness),
            ("样本外表现", out_of_sample),
        )
        if row and row.get("strategy_name") == primary_key
    ]
    basis = "、".join(dimensions) or "当前样本的综合风险收益表现"
    recommendations = [
        f"我的首选是{_display_name(primary)}：它在{basis}上领先，建议将其作为下一轮模拟验证的主策略。"
    ]
    if total.get("strategy_name") != primary_key:
        total_return = float(total.get("total_return") or 0)
        total_drawdown = float(total.get("max_drawdown") or 0)
        recommendations.append(
            f"进攻型备选是{_display_name(total)}：本期总收益为 {total_return:+.2%}、最大回撤为 "
            f"{total_drawdown:.2%}；是否接受这组收益回撤交换由你决定。"
        )
    if not official:
        recommendations.append("当前证据未达到正式验证门槛；我的建议是先保留为模拟候选，暂不进入部署准入。")
    return recommendations


def build_comparison_conclusion(
    payload: dict[str, Any],
    *,
    minimum_history_years: float = 5.0,
) -> ComparisonConclusion:
    rows = [row for row in payload.get("comparisons", []) if not row.get("error")]
    validation = payload.get("data_validation") or {}
    official = (
        float(payload.get("history_years", 0)) >= minimum_history_years
        and validation.get("status") != "conflict"
        and bool(rows)
    )
    warnings = []
    if validation.get("status") != "verified":
        warnings.append(f"历史行情交叉核验状态为 {validation.get('status', 'unknown')}。")
    history_years = float(payload.get("history_years", 0))
    if history_years < minimum_history_years:
        warnings.append(
            f"当前评价期约 {history_years:g} 年，低于 {minimum_history_years:g} 年正式验证标准；"
            "仍按现有数据给出分维度结果，但应降低置信度并避免外推。"
        )
    market_benchmark = payload.get("market_benchmark") or {}
    if market_benchmark.get("status") == "unavailable":
        warnings.append(
            f"同期大盘基准 {market_benchmark.get('name') or market_benchmark.get('ticker') or ''} 不可用："
            f"{market_benchmark.get('error') or '未取得指数行情'}。当前标的结论仍保留。"
        )
    elif market_benchmark.get("status") == "partial":
        warnings.append("同期大盘的部分策略模拟失败；可用的同策略比较仍保留，其余维度不据此下结论。")
    limitations = [
        "固定参数历史模拟不代表未来表现。",
        "日线回测不模拟盘口排队、部分成交和真实流动性冲击。",
        "样本外结果是固定策略留出段诊断，不等同于独立训练后的实盘验证。",
    ]
    if not rows:
        return ComparisonConclusion(
            official=False,
            tradeoffs=["没有可比较的策略结果，无法基于当前数据计算分维度表现。"],
            data_warnings=warnings,
            limitations=limitations,
        )

    total = max(
        rows,
        key=lambda row: float(row["total_return"]) if row.get("total_return") is not None else float("-inf"),
    )
    sharpe_rows = [row for row in rows if row.get("sharpe_ratio") is not None]
    sharpe = max(sharpe_rows, key=lambda row: float(row["sharpe_ratio"])) if sharpe_rows else None
    drawdown = min(
        rows,
        key=lambda row: float(row["max_drawdown"]) if row.get("max_drawdown") is not None else float("inf"),
    )
    oos_rows = [
        row
        for row in rows
        if row.get("diagnostics", {}).get("out_of_sample", {}).get("out_of_sample_return") is not None
    ]
    oos = (
        max(
            oos_rows,
            key=lambda row: float(row["diagnostics"]["out_of_sample"]["out_of_sample_return"]),
        )
        if oos_rows
        else None
    )
    stable = [
        row
        for row in rows
        if payload.get("parameter_sensitivity", {}).get(row.get("strategy_name"), {}).get("status") == "stable"
    ]
    robustness_pool = stable or rows
    stress_by_name = {
        row.get("strategy_name"): row for row in payload.get("cost_scenarios", {}).get("stress", [])
    }
    robustness = max(
        robustness_pool,
        key=lambda row: (
            float(stress_by_name[row.get("strategy_name")]["total_return"])
            if stress_by_name.get(row.get("strategy_name"), {}).get("total_return") is not None
            else float("-inf")
        ),
    )
    current_findings = "；".join(
        f"{label}为{row.get('display_name') or row.get('strategy_name')}"
        for label, row in (
            ("绝对收益领先", total),
            ("风险调整表现领先", sharpe),
            ("回撤控制领先", drawdown),
            ("样本外表现领先", oos),
            ("压力成本下稳健性领先", robustness),
        )
        if row
    )
    return ComparisonConclusion(
        official=official,
        absolute_return_winner=_winner(total, "total_return", total.get("total_return")),
        risk_adjusted_winner=_winner(sharpe, "sharpe_ratio", sharpe.get("sharpe_ratio") if sharpe else None),
        drawdown_winner=_winner(drawdown, "max_drawdown", drawdown.get("max_drawdown")),
        out_of_sample_winner=_winner(
            oos,
            "out_of_sample_return",
            oos.get("diagnostics", {}).get("out_of_sample", {}).get("out_of_sample_return") if oos else None,
        ),
        robustness_winner=_winner(
            robustness,
            "stress_total_return",
            stress_by_name.get(robustness.get("strategy_name"), {}).get("total_return"),
        ),
        tradeoffs=[
            f"基于当前可用样本的分维度结果：{current_findings}。",
            *(
                []
                if official
                else [
                    "已使用当前可用数据完成分维度比较；数据风险会降低结论置信度，但不会替代或抹去当前样本给出的结果。"
                ]
            ),
        ],
        data_warnings=warnings,
        limitations=limitations,
        recommendations=_comparison_recommendations(
            total=total,
            sharpe=sharpe,
            drawdown=drawdown,
            out_of_sample=oos,
            robustness=robustness,
            official=official,
        ),
    )


async def enrich_comparison_conclusion(
    conclusion: ComparisonConclusion,
    payload: dict[str, Any],
) -> ComparisonConclusion:
    """Let the Agent recommend from frozen facts without changing winners or metrics."""
    stress_by_name = {
        row.get("strategy_name"): row for row in payload.get("cost_scenarios", {}).get("stress", [])
    }
    prompt = {
        "ticker": payload.get("ticker"),
        "evaluation_period": [payload.get("evaluation_start_date"), payload.get("evaluation_end_date")],
        "frozen_conclusion": conclusion.model_dump(mode="json", exclude={"recommendations", "interpretations"}),
        "strategy_metrics": [
            {
                "strategy_name": row.get("strategy_name"),
                "display_name": row.get("display_name"),
                "total_return": row.get("total_return"),
                "max_drawdown": row.get("max_drawdown"),
                "sharpe_ratio": row.get("sharpe_ratio"),
                "calmar_ratio": row.get("calmar_ratio"),
                "out_of_sample_return": row.get("diagnostics", {})
                .get("out_of_sample", {})
                .get("out_of_sample_return"),
                "stress_total_return": stress_by_name.get(row.get("strategy_name"), {}).get("total_return"),
            }
            for row in payload.get("comparisons", [])
            if not row.get("error")
        ],
        "strategy_assessments": payload.get("strategy_assessments") or [],
        "market_regime_attribution": payload.get("market_regime_attribution") or {},
        "trade_attribution": payload.get("trade_attribution") or {},
        "robustness_assessments": payload.get("robustness_assessments") or [],
        "research_decision": payload.get("research_decision") or {},
        "instruction": (
            "给出 1 至 3 条直接、可执行的策略建议。明确首选策略、依据和适用条件；可以给一个备选。"
            "不要向用户解释‘没有唯一最好策略’、‘不同指标代表不同取舍’等常识，也不要重复通用风险免责声明。"
            "建议只供用户判断，不得修改冻结的排名、指标或数值。"
        ),
    }
    try:
        raw = await get_llm_service().chat_json(
            json.dumps(prompt, ensure_ascii=False, default=str),
            system=(
                '只返回 JSON：{"recommendations":["..."]}。直接提出建议，不要教育用户或解释显而易见的道理；'
                "不得创造或修改策略排名、指标和数值。"
            ),
        )
        items = raw.get("recommendations") if isinstance(raw, dict) else None
        forbidden = ("不存在唯一", "不同评价维度", "不同指标代表", "历史表现不代表未来")
        if isinstance(items, list):
            recommendations = [
                item[:300]
                for item in items[:3]
                if isinstance(item, str) and item.strip() and not any(phrase in item for phrase in forbidden)
            ]
            if recommendations:
                return conclusion.model_copy(update={"recommendations": recommendations})
    except Exception:
        pass
    return conclusion
