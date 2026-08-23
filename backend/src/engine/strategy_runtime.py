"""Shared deterministic strategy and portfolio execution planning."""

from __future__ import annotations

from collections.abc import Mapping
from math import sqrt
from typing import Any

import pandas as pd

from models.schemas import (
    AssetType,
    Decision,
    PortfolioSpec,
    PortfolioState,
    Position,
    PriceEvidence,
    SimulationAccountConfig,
    StrategySpec,
    TradeDecision,
    TradePlan,
)
from strategies.compiler import evaluate_strategy


def decision_from_strategy(
    spec: StrategySpec,
    history: pd.DataFrame,
    *,
    asset_type: AssetType,
    ticker: str,
    current_price: float,
    position: Position | None = None,
    has_position: bool | None = None,
) -> tuple[TradeDecision, dict[str, Any]]:
    """Evaluate the exact deployed DSL and return its auditable decision."""
    holding = bool(position) if has_position is None else has_position
    evaluation = evaluate_strategy(spec, history, asset_type=asset_type)
    exit_reason = None
    if position:
        if position.stop_loss is not None and current_price <= position.stop_loss:
            exit_reason = "stop_loss_triggered"
        elif position.take_profit is not None and current_price >= position.take_profit:
            exit_reason = "take_profit_triggered"
    if holding and (exit_reason or evaluation.get("exit_matched")):
        evaluation = {**evaluation, "exit_reason": exit_reason or "strategy_exit_conditions"}
        exit_logic_label = "全部" if spec.exit_condition_logic == "all" else "任一"
        return (
            TradeDecision(
                ticker=ticker,
                asset_type=asset_type,
                decision=Decision.SELL,
                reasoning=f"策略 {spec.name} 的退出条件{exit_logic_label}满足。",
            ),
            evaluation,
        )
    if holding or not evaluation.get("matched"):
        return TradeDecision(ticker=ticker, asset_type=asset_type, decision=Decision.HOLD), evaluation
    stop = current_price * (1 - spec.stop_loss_pct) if spec.stop_loss_pct is not None else None
    target = current_price * (1 + spec.take_profit_pct) if spec.take_profit_pct is not None else None
    as_of = str(history.iloc[-1].get("date", "")) if not history.empty else ""
    entry_logic_label = "全部" if spec.entry_condition_logic == "all" else "任一"
    return (
        TradeDecision(
            ticker=ticker,
            asset_type=asset_type,
            decision=Decision.BUY,
            reasoning=f"策略 {spec.name} 的入场条件{entry_logic_label}满足。",
            plan=TradePlan(
                entry_price=current_price,
                stop_loss=stop,
                take_profit=target,
                position_size=spec.position_size_pct,
                entry_explanation="使用当日收盘价作为结构化策略入场基准。",
                stop_loss_explanation=(
                    f"按入场价下方 {spec.stop_loss_pct:.1%} 设置止损。"
                    if spec.stop_loss_pct is not None
                    else "该策略未设置固定比例止损。"
                ),
                take_profit_explanation=(
                    f"按入场价上方 {spec.take_profit_pct:.1%} 设置止盈。"
                    if spec.take_profit_pct is not None
                    else "该策略未设置固定比例止盈。"
                ),
                price_evidence=[
                    PriceEvidence(
                        metric="close",
                        value=current_price,
                        source="strategy/normalized_history",
                        as_of=as_of,
                        calculation="当前收盘价作为结构化策略入场基准",
                    )
                ],
            ),
        ),
        evaluation,
    )


def target_exposure_from_strategy(spec: StrategySpec, history: pd.DataFrame) -> float | None:
    """Return a bounded exposure for dynamic position models.

    ``None`` keeps legacy decision-based strategies on their existing path.
    The calculation only consumes the supplied historical prefix.
    """
    model = spec.position_model
    if model is None or model.type == "fixed":
        return None
    close = pd.to_numeric(history.get("close"), errors="coerce").dropna()
    required = max(model.volatility_window + 1, model.trend_window if model.type == "trend_volatility_target" else 0)
    if len(close) < required:
        return 0.0
    returns = close.pct_change(fill_method=None).dropna().tail(model.volatility_window)
    realized = float(returns.std(ddof=0) * sqrt(252)) if len(returns) >= model.volatility_window else 0.0
    if not pd.notna(realized) or realized <= 0:
        exposure = model.max_exposure
    else:
        exposure = model.target_volatility / realized
    if model.type == "trend_volatility_target":
        trend = float(close.tail(model.trend_window).mean())
        if float(close.iloc[-1]) <= trend:
            exposure = 0.0
    return round(min(max(float(exposure), model.min_exposure), model.max_exposure), 10)


def plan_rebalance(
    portfolio: PortfolioState,
    rules: SimulationAccountConfig,
    target: Mapping[str, float],
    prices: Mapping[str, float],
    decisions: Mapping[str, TradeDecision] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic sell-first paper-order proposals without mutation."""
    if portfolio.total_value <= 0:
        return []
    min_lot = rules.effective_trading_rules(rules.asset_type).min_lot
    target_shares: dict[str, int] = {}
    for ticker, weight in target.items():
        price = float(prices.get(ticker, 0))
        if price > 0 and weight > 0:
            target_shares[ticker] = int(portfolio.total_value * float(weight) / price // min_lot * min_lot)
    proposals: list[dict[str, Any]] = []
    positions = {position.ticker: position for position in portfolio.positions}
    for ticker, position in positions.items():
        excess = position.shares - target_shares.get(ticker, 0)
        if excess > 0 and ticker in prices:
            proposals.append({"ticker": ticker, "side": "sell", "shares": excess, "price": float(prices[ticker])})
    for ticker, desired in target_shares.items():
        additional = desired - (positions[ticker].shares if ticker in positions else 0)
        if additional <= 0:
            continue
        decision = (decisions or {}).get(ticker)
        proposals.append(
            {
                "ticker": ticker,
                "side": "buy",
                "shares": additional,
                "price": float(prices[ticker]),
                "stop_loss": decision.stop_loss if decision else None,
                "take_profit": decision.take_profit if decision else None,
            }
        )
    return proposals


def target_weights_for_decisions(
    decisions: Mapping[str, TradeDecision],
    portfolio: PortfolioState,
    spec: PortfolioSpec,
) -> dict[str, float]:
    positions = {position.ticker for position in portfolio.positions}
    candidates: list[tuple[int, float, str]] = []
    for ticker, decision in decisions.items():
        if decision.decision == Decision.SELL:
            continue
        if decision.decision == Decision.BUY:
            candidates.append((1, float(decision.confidence), ticker))
        elif ticker in positions:
            candidates.append((0, float(decision.confidence), ticker))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = [ticker for _, _, ticker in candidates[: spec.max_positions]]
    if not selected:
        return {}
    investable = 1 - spec.cash_reserve
    weight = min(spec.max_position_weight, investable / len(selected))
    return {ticker: round(weight, 10) for ticker in selected}
