"""Portfolio construction and rebalancing helpers for multi-asset backtests."""

from __future__ import annotations

from typing import Mapping

from engine.trading_engine import TimeAwareTradingEngine
from models.schemas import PortfolioSpec, TradeDecision


def target_weights(
    decisions: Mapping[str, TradeDecision],
    engine: TimeAwareTradingEngine,
    spec: PortfolioSpec,
) -> dict[str, float]:
    """Build equal-weight targets from symbol decisions and current holdings.

    HOLD keeps an existing position eligible, BUY adds a candidate, and SELL
    removes a symbol from the target portfolio.  Ranking is deterministic so
    the same experiment can be replayed byte-for-byte.
    """
    positions = {position.ticker: position for position in engine.portfolio.positions}
    candidates: list[tuple[int, float, str]] = []
    for ticker, decision in decisions.items():
        if decision.decision.value == "sell":
            continue
        if decision.decision.value == "buy":
            candidates.append((1, float(decision.confidence), ticker))
        elif ticker in positions:
            candidates.append((0, float(decision.confidence), ticker))

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = [ticker for _, _, ticker in candidates[: spec.max_positions]]
    if not selected:
        return {}

    investable_weight = 1.0 - spec.cash_reserve
    equal_weight = min(spec.max_position_weight, investable_weight / len(selected))
    return {ticker: round(equal_weight, 10) for ticker in selected}


def rebalance_portfolio(
    engine: TimeAwareTradingEngine,
    target: Mapping[str, float],
    prices: Mapping[str, float],
    decisions: Mapping[str, TradeDecision] | None = None,
    trade_date: str = "",
) -> None:
    """Move the simulated account toward target market-value weights.

    Sells are submitted before buys to release cash.  The trading engine still
    enforces lot size, T+1 availability, fees, and available cash, so this
    helper only proposes target quantities and never bypasses execution rules.
    """
    total_value = engine.portfolio.total_value
    if total_value <= 0:
        return

    rules = engine.rules.effective_trading_rules(engine.rules.asset_type)
    min_lot = rules.min_lot
    target_shares: dict[str, int] = {}
    for ticker, weight in target.items():
        price = float(prices.get(ticker, 0.0))
        if price <= 0 or weight <= 0:
            continue
        target_shares[ticker] = int(total_value * float(weight) / price // min_lot * min_lot)

    # Reduce positions that are above target, or no longer belong to target.
    for position in list(engine.portfolio.positions):
        desired = target_shares.get(position.ticker, 0)
        excess = position.shares - desired
        if excess <= 0 or position.ticker not in prices:
            continue
        engine.sell(position.ticker, excess, float(prices[position.ticker]), trade_date)

    # Add positions after sells have released cash.
    for ticker, desired in target_shares.items():
        price = float(prices[ticker])
        current = engine._find_position(ticker)
        current_shares = current.shares if current else 0
        additional = desired - current_shares
        if additional <= 0:
            continue
        decision = decisions.get(ticker) if decisions else None
        plan = decision.plan if decision else None
        engine.buy(
            ticker,
            additional,
            price,
            trade_date,
            stop_loss=plan.stop_loss if plan else None,
            take_profit=plan.take_profit if plan else None,
        )

    # Newly created positions have no mark price yet; use the execution price
    # for same-close snapshots and for the next allocation calculation.
    engine.update_prices(dict(prices), trigger_exits=False)


def enforce_max_position_weight(
    engine: TimeAwareTradingEngine,
    max_weight: float,
    prices: Mapping[str, float],
    trade_date: str = "",
) -> None:
    """Trim end-of-day drift above the configured hard position limit."""
    total_value = engine.portfolio.total_value
    if total_value <= 0:
        return
    min_lot = engine.rules.effective_trading_rules(engine.rules.asset_type).min_lot
    for position in list(engine.portfolio.positions):
        price = float(prices.get(position.ticker, 0.0))
        if price <= 0:
            continue
        allowed = int(total_value * max_weight / price // min_lot * min_lot)
        if position.shares > allowed:
            engine.sell(position.ticker, position.shares - allowed, price, trade_date)
    engine.update_prices(dict(prices), trigger_exits=False)


def portfolio_snapshot(engine: TimeAwareTradingEngine, date: str) -> dict:
    """Return an auditable end-of-day portfolio snapshot."""
    total_value = engine.portfolio.total_value
    positions = []
    for position in engine.portfolio.positions:
        market_value = position.market_value
        positions.append(
            {
                "date": date,
                "ticker": position.ticker,
                "asset_type": position.asset_type.value,
                "shares": position.shares,
                "price": position.current_price,
                "market_value": round(market_value, 2),
                "weight": round(market_value / total_value, 8) if total_value else 0.0,
            }
        )
    return {
        "date": date,
        "cash": round(engine.portfolio.cash, 2),
        "total_value": round(total_value, 2),
        "cash_weight": round(engine.portfolio.cash / total_value, 8) if total_value else 0.0,
        "positions": positions,
    }
