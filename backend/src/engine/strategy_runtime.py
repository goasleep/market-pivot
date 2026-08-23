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
    StrategyIntent,
    StrategyRuntimeState,
    StrategySignal,
    StrategySpec,
    TradeDecision,
    TradePlan,
)
from strategies.compiler import evaluate_expression, evaluate_strategy, validate_strategy_spec
from strategies.plugin_registry import get_strategy_plugin


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


def evaluate_strategy_intent(
    spec: StrategySpec,
    history: pd.DataFrame,
    *,
    asset_type: AssetType | str,
    current_exposure: float = 0.0,
    state: StrategyRuntimeState | None = None,
) -> tuple[StrategyIntent, StrategyRuntimeState, dict[str, Any]]:
    """Evaluate StrategySpec v2 into a continuous, pre-trade target exposure."""

    if spec.schema_version != 2:
        raise ValueError("evaluate_strategy_intent 只接受 StrategySpec v2")
    kind = AssetType(asset_type)
    if kind not in spec.asset_types:
        raise ValueError(f"策略不支持资产类型 {kind.value}")
    errors = validate_strategy_spec(spec, available_columns=set(history.columns))
    if errors:
        raise ValueError("策略未通过可执行性校验: " + "; ".join(errors))
    state = state.model_copy(deep=True) if state is not None else StrategyRuntimeState()
    current_date = str(history.iloc[-1].get("date", "")) if not history.empty else ""
    if state.last_evaluated_date == current_date and state.last_output:
        intent = StrategyIntent.model_validate(state.last_output)
        return intent, state, intent.trace

    before = state.lifecycle
    signals: list[StrategySignal] = []
    expression_traces: dict[str, Any] = {}
    variable_updates: dict[str, Any] = {}
    for component in spec.components:
        if component.type == "dsl":
            trace = evaluate_expression(
                component.expression,
                history,
                indicator_specs=spec.indicator_specs,
            )
            expression_traces[component.id] = trace
            signal = StrategySignal(
                component_id=component.id,
                score=component.score_when_true if trace["matched"] else component.score_when_false,
                confidence=1.0,
                reasons=[component.expression.description or f"DSL 表达式 {component.id}"],
            )
        else:
            plugin = get_strategy_plugin(component.plugin or "", component.plugin_version)
            raw = plugin(history.copy(deep=False), dict(component.params), dict(state.variables))
            payload = raw.model_dump(mode="python") if isinstance(raw, StrategySignal) else dict(raw)
            payload["component_id"] = component.id
            signal = StrategySignal.model_validate(payload)
        signals.append(signal)
        variable_updates.update(signal.state_updates)

    signal_components = [
        (component, signal)
        for component, signal in zip(spec.components, signals)
        if component.role == "signal"
    ]
    router_signals = [signal for component, signal in zip(spec.components, signals) if component.role == "router"]
    position_signals = [
        (component, signal)
        for component, signal in zip(spec.components, signals)
        if component.role == "position" and signal.target_exposure is not None
    ]
    fusion = spec.fusion
    position = spec.position_policy
    if fusion is None or position is None:
        raise ValueError("StrategySpec v2 缺少 fusion 或 position_policy")
    fused_score = _fuse_scores(signal_components, fusion.type)
    if router_signals:
        router_multiplier = min(max((sum(item.score for item in router_signals) / len(router_signals) + 1) / 2, 0), 1)
        fused_score *= router_multiplier
    target_hints = [
        (component.weight, float(signal.target_exposure))
        for component, signal in [*signal_components, *position_signals]
        if signal.target_exposure is not None and component.weight > 0
    ]
    if target_hints:
        raw_target = sum(weight * target for weight, target in target_hints) / sum(
            weight for weight, _ in target_hints
        )
    else:
        raw_target = max(0.0, fused_score) * position.max_exposure

    previous_target = min(max(float(current_exposure), 0.0), 1.0)
    component_scores = [signal.score for _, signal in signal_components]
    has_conflict = any(score > 0 for score in component_scores) and any(score < 0 for score in component_scores)
    suppress_thresholds = False
    if has_conflict and fusion.conflict_policy == "hold":
        raw_target = previous_target
        suppress_thresholds = True
    elif has_conflict and fusion.conflict_policy == "reduce":
        raw_target = min(raw_target, previous_target * 0.5)
    elif has_conflict and fusion.conflict_policy == "exit":
        raw_target = 0.0

    current_price = float(history.iloc[-1]["close"]) if not history.empty else 0.0
    exit_reason = None
    if previous_target > 0 and state.entry_price:
        if spec.stop_loss_pct is not None and current_price <= state.entry_price * (1 - spec.stop_loss_pct):
            exit_reason = "stop_loss_triggered"
        elif spec.take_profit_pct is not None and current_price >= state.entry_price * (1 + spec.take_profit_pct):
            exit_reason = "take_profit_triggered"
    exiting = previous_target > 0 and (
        bool(exit_reason)
        or (has_conflict and fusion.conflict_policy == "exit")
        or (not suppress_thresholds and fused_score <= fusion.exit_threshold)
    )
    entering = previous_target <= 0 and not has_conflict and fused_score >= fusion.entry_threshold
    if state.cooldown_remaining > 0:
        target = 0.0
        state.cooldown_remaining -= 1
    elif previous_target <= 0 and not entering:
        target = 0.0
    elif exiting:
        target = 0.0
        state.cooldown_remaining = (spec.state_policy.cooldown_bars_after_exit if spec.state_policy else 0)
    else:
        target = min(max(raw_target, position.min_exposure), position.max_exposure)
        target = min(target, previous_target + position.max_increase_per_rebalance)
        target = max(target, previous_target - position.max_decrease_per_rebalance)
        if abs(target - previous_target) < position.minimum_change:
            target = previous_target

    target = round(min(max(float(target), 0.0), position.max_exposure), 10)
    tolerance = max(position.minimum_change, 1e-8)
    if target > current_exposure + tolerance:
        decision = Decision.BUY
    elif target < current_exposure - tolerance:
        decision = Decision.SELL
    else:
        decision = Decision.HOLD
    after = "cooldown" if state.cooldown_remaining > 0 and target == 0 else "active" if target > 0 else "flat"
    state.lifecycle = after
    state.target_exposure = target
    if previous_target <= 0 and target > 0:
        state.entry_price = current_price
    elif target == 0:
        state.entry_price = None
    state.peak_price = _next_peak_price(state.peak_price, history, active=target > 0)
    state.bars_in_state = state.bars_in_state + 1 if before == after else 1
    state.variables.update(variable_updates)
    state.last_evaluated_date = current_date
    trace = {
        "schema_version": 2,
        "fusion": fusion.model_dump(mode="json"),
        "fused_score": round(float(fused_score), 10),
        "raw_target_exposure": round(float(raw_target), 10),
        "current_exposure": round(float(current_exposure), 10),
        "expression_traces": expression_traces,
        "router_count": len(router_signals),
        "component_conflict": has_conflict,
        "exit_reason": exit_reason,
    }
    intent = StrategyIntent(
        decision=decision,
        target_exposure=target,
        score=round(min(max(float(fused_score), -1.0), 1.0), 10),
        confidence=round(sum(item.confidence for item in signals) / len(signals), 10) if signals else 0.0,
        state_before=before,
        state_after=after,
        component_signals=signals,
        trace=trace,
    )
    state.last_output = intent.model_dump(mode="json")
    return intent, state, trace


def _fuse_scores(components, method: str) -> float:
    weighted = [(float(component.weight), float(signal.score)) for component, signal in components]
    if not weighted:
        return 0.0
    if method == "majority_vote":
        votes = [1.0 if score > 0 else -1.0 if score < 0 else 0.0 for _, score in weighted]
        return sum(votes) / len(votes)
    if method == "priority":
        return next((score for _, score in weighted if score != 0), 0.0)
    denominator = sum(weight for weight, _ in weighted)
    return sum(weight * score for weight, score in weighted) / denominator if denominator else 0.0


def _next_peak_price(previous: float | None, history: pd.DataFrame, *, active: bool) -> float | None:
    if not active or history.empty or "close" not in history:
        return None
    current = float(history.iloc[-1]["close"])
    return max(previous or current, current)


def decision_from_intent(
    spec: StrategySpec,
    intent: StrategyIntent,
    *,
    ticker: str,
    asset_type: AssetType,
    current_price: float,
) -> TradeDecision:
    """Expose a v2 target change through the existing audit and Agent-gate contract."""

    stop = current_price * (1 - spec.stop_loss_pct) if intent.decision == Decision.BUY and spec.stop_loss_pct else None
    target = (
        current_price * (1 + spec.take_profit_pct)
        if intent.decision == Decision.BUY and spec.take_profit_pct
        else None
    )
    return TradeDecision(
        ticker=ticker,
        asset_type=asset_type,
        decision=intent.decision,
        confidence=intent.confidence,
        reasoning=(
            f"混合策略 {spec.name} 目标仓位调整为 {intent.target_exposure:.1%}，融合得分 {intent.score:.3f}。"
        ),
        plan=TradePlan(
            entry_price=current_price if intent.decision == Decision.BUY else None,
            stop_loss=stop,
            take_profit=target,
            position_size=intent.target_exposure,
            position_strategy="连续目标仓位",
        ),
    )


def normalize_target_exposures(
    targets: Mapping[str, float],
    *,
    max_position_weight: float = 0.95,
    max_positions: int = 100,
    cash_reserve: float = 0.05,
) -> dict[str, float]:
    """Bound long-only component targets before they reach the order planner."""

    ranked = sorted(
        ((ticker, min(max(float(value), 0.0), max_position_weight)) for ticker, value in targets.items()),
        key=lambda item: (-item[1], item[0]),
    )
    selected = [(ticker, value) for ticker, value in ranked[:max_positions] if value > 0]
    total = sum(value for _, value in selected)
    investable = 1 - cash_reserve
    scale = min(1.0, investable / total) if total > 0 else 1.0
    return {ticker: round(value * scale, 10) for ticker, value in selected}


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
