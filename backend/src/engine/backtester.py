"""Backtest engine - runs Agent workflow over historical data.

Flow:
  1. Fetch historical data for the stock
  2. For each trading day (at decision_interval):
     a. Set engine date (prevent data leakage)
     b. Run agent workflow
     c. Execute trade decision
     d. Record portfolio snapshot
  3. Calculate performance metrics
  4. Return results
"""

import asyncio
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from application.research import research_service
from data.backtest_data import BacktestDataError, prepare_backtest_data
from data.fund_provider import async_get_fund_history
from data.market_context import build_market_context
from data.stock_provider import async_get_stock_history
from engine.portfolio_allocator import (
    enforce_max_position_weight,
    portfolio_snapshot,
    rebalance_portfolio,
    target_weights,
)
from engine.trading_engine import TimeAwareTradingEngine, decision_shares
from models.schemas import (
    AssetType,
    Decision,
    PortfolioSpec,
    PriceEvidence,
    SimulationAccountConfig,
    StrategySpec,
    TradeDecision,
    TradePlan,
)
from strategies.compiler import evaluate_strategy, strategy_from_mapping

# Kept as a module alias for existing test doubles and callers that patch the
# workflow. Production invocation is routed through ResearchService below.
workflow = research_service.workflow


async def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 1_000_000,
    initial_position: int = 0,
    decision_interval: int = 1,
    fill_time: str = "next_open",
    strategy_name: str | None = None,
    num_news: int = 5,
    progress_callback=None,
    asset_type: AssetType | str = AssetType.STOCK,
    strategy_spec: dict | None = None,
    capture_data: bool = False,
) -> dict:
    """Run backtest over historical data.

    Args:
        ticker: Stock code
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting cash
        initial_position: Starting shares
        decision_interval: Run agent every N trading days
        num_news: Number of news articles to fetch per run
        progress_callback: async callback(stage, message) for progress updates

    Returns:
        Dict with backtest results
    """
    asset_type = AssetType(asset_type)
    executable_spec = (
        strategy_from_mapping(strategy_spec, source="llm")
        if strategy_spec
        else None
    )
    logger.info(f"Backtest {asset_type.value}:{ticker} {start_date} -> {end_date}")

    if decision_interval < 1:
        raise ValueError("decision_interval must be at least 1")
    if fill_time not in {"next_open", "same_close"}:
        raise ValueError("fill_time must be next_open or same_close")

    # 1. Fetch historical data
    if progress_callback:
        await progress_callback("data_fetch", f"Fetching history for {ticker}...")

    if asset_type == AssetType.STOCK:
        df = await async_get_stock_history(ticker, start_date=start_date, end_date=end_date)
    else:
        df = await async_get_fund_history(
            ticker,
            asset_type=asset_type.value,
            start_date=start_date,
            end_date=end_date,
        )
    if df.empty or len(df) < 5:
        return _empty_result(ticker, start_date, end_date, initial_capital)

    try:
        df, data_snapshot = prepare_backtest_data(
            df,
            ticker=ticker,
            asset_type=asset_type.value,
            start_date=start_date,
            end_date=end_date,
            adjustment="qfq" if asset_type == AssetType.STOCK else "provider_default",
        )
    except BacktestDataError as exc:
        logger.warning("Backtest data rejected for {}: {}", ticker, exc)
        return _empty_result(ticker, start_date, end_date, initial_capital, error=str(exc))

    # Filter to date range
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)

    trading_dates = df["date"].tolist()
    logger.info(f"Backtest period: {len(trading_dates)} trading days")

    # 2. Initialize engine
    engine = TimeAwareTradingEngine(
        initial_capital=initial_capital,
        rules=SimulationAccountConfig(initial_cash=initial_capital, asset_type=asset_type),
    )
    engine.set_available_dates(trading_dates)

    # Set initial position if provided
    if initial_position > 0:
        first_price = float(df.iloc[0]["close"])
        engine.seed_position(ticker, initial_position, first_price)

    # 3. Iterate over trading days
    equity_curve = []
    day_count = 0
    pending_decision = None

    for i, row in df.iterrows():
        current_date = row["date"]
        current_price = float(row["close"])

        engine.advance_to_date(current_date)
        if pending_decision is not None:
            _execute_decision(engine, pending_decision, ticker, float(row["open"]), current_date)
            pending_decision = None
        engine.update_prices({ticker: current_price})

        # Run agent at interval
        if day_count % decision_interval == 0:
            if progress_callback:
                pct = engine.progress_pct
                await progress_callback(
                    "agent_run",
                    f"Day {day_count + 1}/{len(trading_dates)} ({pct:.0f}%): Agent analyzing...",
                )

            try:
                context = await build_market_context(
                    ticker,
                    asset_type=asset_type,
                    as_of_date=current_date,
                    current_price=current_price,
                    history_df=df.iloc[: i + 1],
                    include_live_enrichment=False,
                )
                state = {
                    "ticker": ticker,
                    "asset_type": asset_type.value,
                    "current_price": current_price,
                    "as_of_date": current_date,
                    "is_backtest": True,
                    "strategy_name": strategy_name,
                    "market_context": context,
                    "progress": [],
                }
                if executable_spec:
                    decision = _decision_from_strategy(
                        executable_spec,
                        df.iloc[: i + 1],
                        asset_type=asset_type,
                        ticker=ticker,
                        current_price=current_price,
                        has_position=bool(engine._find_position(ticker)),
                    )
                else:
                    result = await research_service.run_state(
                        state,
                        trace_config=research_service.build_trace_config(
                            ticker,
                            asset_type,
                            run_name="backtest-agent-analysis",
                            tags=["backtest", "agent"],
                            metadata={
                                "ticker": ticker,
                                "asset_type": asset_type.value,
                                "as_of_date": current_date,
                                "strategy": strategy_name or "auto",
                            },
                        ),
                        workflow_override=workflow,
                    )
                    decision = result.get("final_decision")

                if decision and decision.decision != Decision.HOLD:
                    if fill_time == "same_close":
                        _execute_decision(engine, decision, ticker, current_price, current_date)
                    else:
                        pending_decision = decision
            except Exception as e:
                logger.error(f"Agent error on {current_date}: {e}")

        equity_curve.append(
            {
                "date": current_date,
                "value": round(engine.portfolio.total_value, 2),
            }
        )

        day_count += 1

    # 4. Calculate metrics
    if progress_callback:
        await progress_callback("calculating", "Calculating performance metrics...")

    metrics = _calc_metrics(equity_curve, engine.portfolio.trades, initial_capital)

    payload = {
        "ticker": ticker,
        "asset_type": asset_type.value,
        "strategy_spec": executable_spec.model_dump(mode="json") if executable_spec else None,
        "data_snapshot": data_snapshot,
        "buy_hold_return": round(float(df.iloc[-1]["close"] / df.iloc[0]["close"] - 1), 6),
        "execution": _execution_manifest(engine, fill_time),
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "final_value": round(engine.portfolio.total_value, 2),
        "total_return": metrics["total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "win_rate": metrics["win_rate"],
        "realized_pnl": metrics["realized_pnl"],
        "total_fees": metrics["total_fees"],
        "total_trades": len(engine.portfolio.trades),
        "equity_curve": equity_curve,
        "trades": [t.model_dump() for t in engine.portfolio.trades],
    }
    if capture_data:
        payload["_data_snapshot_rows"] = df.to_dict(orient="records")
    return payload


async def run_pool_backtest(
    tickers: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 1_000_000,
    decision_interval: int = 1,
    fill_time: str = "next_open",
    strategy_name: str | None = None,
    progress_callback=None,
    asset_type: AssetType | str = AssetType.STOCK,
    strategy_spec: dict | None = None,
    portfolio_spec: dict | PortfolioSpec | None = None,
    capture_data: bool = False,
) -> dict:
    """Run the same Agent and execution semantics across a stock pool.

    Each symbol receives an as-of historical context.  Decisions are merged
    into one portfolio, next-open orders remain pending until that symbol has
    a subsequent bar, and the shared ``decision_shares`` helper keeps sizing
    consistent with the persistent simulation account.
    """

    asset_type = AssetType(asset_type)
    portfolio = (
        portfolio_spec
        if isinstance(portfolio_spec, PortfolioSpec)
        else PortfolioSpec.model_validate(portfolio_spec)
        if portfolio_spec is not None
        else None
    )
    executable_spec = strategy_from_mapping(strategy_spec, source="llm") if strategy_spec else None
    if not tickers:
        raise ValueError("tickers 不能为空")
    if decision_interval < 1:
        raise ValueError("decision_interval must be at least 1")
    if fill_time not in {"next_open", "same_close"}:
        raise ValueError("fill_time must be next_open or same_close")
    symbols = list(
        dict.fromkeys(
            ticker.strip().lower().removeprefix("sh").removeprefix("sz").zfill(6)
            for ticker in tickers
        )
    )
    if progress_callback:
        await progress_callback("data_fetch", f"Fetching history for {len(symbols)} symbols...")
    frames = await asyncio.gather(
        *(
            async_get_stock_history(symbol, start_date=start_date, end_date=end_date)
            if asset_type == AssetType.STOCK
            else async_get_fund_history(
                symbol,
                asset_type=asset_type.value,
                start_date=start_date,
                end_date=end_date,
            )
            for symbol in symbols
        )
    )
    valid: dict[str, pd.DataFrame] = {}
    data_snapshots = []
    data_rejections: list[dict[str, str]] = []
    for symbol, frame in zip(symbols, frames):
        if frame.empty or len(frame) < 5:
            data_rejections.append({"ticker": symbol, "reason": "历史数据为空或不足 5 行"})
            continue
        try:
            item, snapshot = prepare_backtest_data(
                frame,
                ticker=symbol,
                asset_type=asset_type.value,
                start_date=start_date,
                end_date=end_date,
                adjustment="qfq" if asset_type == AssetType.STOCK else "provider_default",
            )
        except BacktestDataError as exc:
            logger.warning("Skipping invalid backtest data for {}: {}", symbol, exc)
            data_rejections.append({"ticker": symbol, "reason": str(exc)})
            continue
        valid[symbol] = item
        data_snapshots.append(snapshot)
    if not valid:
        return _empty_pool_result(symbols, start_date, end_date, initial_capital)

    trading_dates = sorted({day for frame in valid.values() for day in frame["date"].tolist()})
    engine = TimeAwareTradingEngine(
        initial_capital=initial_capital,
        rules=SimulationAccountConfig(initial_cash=initial_capital, asset_type=asset_type),
    )
    engine.set_available_dates(trading_dates)
    pending: dict[str, object] = {}
    pending_target: tuple[dict[str, float], dict[str, TradeDecision]] | None = None
    equity_curve: list[dict] = []
    portfolio_history: list[dict] = []
    target_weights_history: list[dict] = []

    for day_count, current_date in enumerate(trading_dates):
        engine.advance_to_date(current_date)
        rows_today: dict[str, pd.Series] = {}
        for symbol, frame in valid.items():
            matches = frame[frame["date"] == current_date]
            if not matches.empty:
                rows_today[symbol] = matches.iloc[-1]

        if portfolio is not None and pending_target is not None:
            pending_weights, pending_decisions = pending_target
            rebalance_portfolio(
                engine,
                pending_weights,
                {symbol: float(row["open"]) for symbol, row in rows_today.items()},
                pending_decisions,
                current_date,
            )
            pending_target = None
        else:
            for symbol, decision in list(pending.items()):
                row = rows_today.get(symbol)
                if row is not None:
                    _execute_decision(engine, decision, symbol, float(row["open"]), current_date)
                    pending.pop(symbol, None)
        engine.update_prices({symbol: float(row["close"]) for symbol, row in rows_today.items()})

        if day_count % decision_interval == 0:
            day_decisions: dict[str, TradeDecision] = {
                position.ticker: TradeDecision(
                    ticker=position.ticker,
                    asset_type=position.asset_type,
                    decision=Decision.HOLD,
                )
                for position in engine.portfolio.positions
            }
            for symbol, row in rows_today.items():
                if progress_callback:
                    await progress_callback(
                        "agent_run",
                        f"{current_date}: Agent analyzing {symbol} ({day_count + 1}/{len(trading_dates)})...",
                    )
                try:
                    frame = valid[symbol]
                    history_until_day = frame[frame["date"] <= current_date]
                    current_price = float(row["close"])
                    context = await build_market_context(
                        symbol,
                        asset_type=asset_type,
                        as_of_date=current_date,
                        current_price=current_price,
                        history_df=history_until_day,
                        include_live_enrichment=False,
                    )
                    state = {
                        "ticker": symbol,
                        "asset_type": asset_type.value,
                        "current_price": current_price,
                        "as_of_date": current_date,
                        "is_backtest": True,
                        "strategy_name": strategy_name,
                        "market_context": context,
                        "progress": [],
                    }
                    if executable_spec:
                        decision = _decision_from_strategy(
                            executable_spec,
                            history_until_day,
                            asset_type=asset_type,
                            ticker=symbol,
                            current_price=current_price,
                            has_position=bool(engine._find_position(symbol)),
                        )
                    else:
                        result = await research_service.run_state(
                            state,
                            trace_config=research_service.build_trace_config(
                                symbol,
                                asset_type,
                                run_name="backtest-agent-analysis",
                                tags=["backtest", "agent", "pool"],
                                metadata={
                                    "ticker": symbol,
                                    "asset_type": asset_type.value,
                                    "as_of_date": current_date,
                                    "strategy": strategy_name or "auto",
                                },
                            ),
                            workflow_override=workflow,
                        )
                        decision = result.get("final_decision")
                    if decision is None:
                        decision = TradeDecision(ticker=symbol, asset_type=asset_type, decision=Decision.HOLD)
                    day_decisions[symbol] = decision
                    if portfolio is None and decision.decision != Decision.HOLD:
                        if fill_time == "same_close":
                            _execute_decision(engine, decision, symbol, current_price, current_date)
                        else:
                            pending[symbol] = decision
                except Exception as exc:
                    logger.error(f"Agent error on {symbol} {current_date}: {exc}")

            if portfolio is not None and _should_rebalance(day_count, portfolio.rebalance_frequency):
                weights = target_weights(day_decisions, engine, portfolio)
                target_weights_history.append(
                    {"date": current_date, "weights": weights, "cash_reserve": portfolio.cash_reserve}
                )
                close_prices = {symbol: float(row["close"]) for symbol, row in rows_today.items()}
                if fill_time == "same_close":
                    rebalance_portfolio(engine, weights, close_prices, day_decisions, current_date)
                else:
                    pending_target = (weights, day_decisions)

        if portfolio is not None:
            enforce_max_position_weight(
                engine,
                portfolio.max_position_weight,
                {symbol: float(row["close"]) for symbol, row in rows_today.items()},
                current_date,
            )

        snapshot = portfolio_snapshot(engine, current_date)
        portfolio_history.append(snapshot)
        equity_curve.append({"date": current_date, "value": snapshot["total_value"]})

    if progress_callback:
        await progress_callback("calculating", "Calculating pool performance metrics...")
    metrics = _calc_metrics(equity_curve, engine.portfolio.trades, initial_capital)
    payload = {
        "ticker": "pool",
        "mode": "portfolio" if portfolio is not None else "pool",
        "asset_type": asset_type.value,
        "strategy_spec": executable_spec.model_dump(mode="json") if executable_spec else None,
        "portfolio_spec": portfolio.model_dump(mode="json") if portfolio else None,
        "data_snapshots": data_snapshots,
        "data_rejections": data_rejections,
        "execution": _execution_manifest(engine, fill_time),
        "tickers": list(valid),
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "final_value": round(engine.portfolio.total_value, 2),
        "total_return": metrics["total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "win_rate": metrics["win_rate"],
        "realized_pnl": metrics["realized_pnl"],
        "total_fees": metrics["total_fees"],
        "total_trades": len(engine.portfolio.trades),
        "equity_curve": equity_curve,
        "trades": [trade.model_dump() for trade in engine.portfolio.trades],
    }
    if portfolio is not None:
        payload["portfolio_history"] = portfolio_history
        payload["target_weights_history"] = target_weights_history
        payload["symbol_metrics"] = _symbol_metrics(valid, engine)
    if capture_data:
        payload["_data_snapshot_rows"] = {
            symbol: frame.to_dict(orient="records") for symbol, frame in valid.items()
        }
    return payload


def _execute_decision(engine, decision, ticker: str, price: float, date: str):
    """Execute a trade decision."""
    shares = decision_shares(engine.portfolio, engine.rules, decision, price)
    if shares <= 0:
        return
    if decision.decision == Decision.BUY:
        engine.buy(
            ticker,
            shares,
            price,
            date,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
        )
    elif decision.decision == Decision.SELL:
        engine.sell(ticker, shares, price, date)


def _execution_manifest(engine: TimeAwareTradingEngine, fill_time: str) -> dict[str, Any]:
    rules = engine.rules.effective_trading_rules(engine.rules.asset_type)
    return {
        "fill_time": fill_time,
        "slippage_bps": rules.slippage_bps,
        "buy_commission_rate": rules.buy_commission_rate,
        "sell_commission_rate": rules.sell_commission_rate,
        "minimum_commission": rules.minimum_commission,
        "stamp_tax_rate": rules.stamp_tax_rate,
        "transfer_fee_rate": rules.transfer_fee_rate,
        "min_lot": rules.min_lot,
        "t_plus_one": rules.t_plus_one,
    }


def _decision_from_strategy(
    spec: StrategySpec,
    history: pd.DataFrame,
    *,
    asset_type: AssetType,
    ticker: str,
    current_price: float,
    has_position: bool,
) -> TradeDecision:
    """Turn a validated LLM/user strategy spec into a deterministic decision."""
    evaluation = evaluate_strategy(spec, history, asset_type=asset_type)
    if has_position and evaluation.get("exit_matched"):
        return TradeDecision(
            ticker=ticker,
            asset_type=asset_type,
            decision=Decision.SELL,
            reasoning=f"策略 {spec.name} 的退出条件全部满足。",
        )
    if has_position or not evaluation.get("matched"):
        return TradeDecision(ticker=ticker, asset_type=asset_type, decision=Decision.HOLD)
    stop = current_price * (1 - (spec.stop_loss_pct or 0.08))
    target = current_price * (1 + (spec.take_profit_pct or 0.16))
    as_of = str(history.iloc[-1].get("date", "")) if not history.empty else ""
    evidence = [
        PriceEvidence(
            metric="close",
            value=current_price,
            source="strategy/backtest_history",
            as_of=as_of,
            calculation="当前收盘价作为结构化策略入场基准",
        )
    ]
    return TradeDecision(
        ticker=ticker,
        asset_type=asset_type,
        decision=Decision.BUY,
        reasoning=f"策略 {spec.name} 的入场条件全部满足。",
        plan=TradePlan(
            entry_price=current_price,
            stop_loss=stop,
            take_profit=target,
            position_size=spec.position_size_pct,
            entry_explanation="使用回测当日收盘价作为入场基准。",
            stop_loss_explanation=f"按入场价下方 {spec.stop_loss_pct or 0.08:.1%} 设置止损。",
            take_profit_explanation=f"按入场价上方 {spec.take_profit_pct or 0.16:.1%} 设置止盈。",
            price_evidence=evidence,
        ),
    )


def _calc_metrics(equity_curve: list[dict], trades: list, initial_capital: float) -> dict:
    """Calculate backtest performance metrics."""
    if not equity_curve:
        return {
            "total_return": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "realized_pnl": 0,
            "total_fees": 0,
        }

    values = [e["value"] for e in equity_curve]
    final_value = values[-1]
    total_return = (final_value - initial_capital) / initial_capital

    # Max drawdown
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (daily, annualized)
    if len(values) > 2:
        returns = np.diff(values) / np.array(values[:-1])
        if returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # FIFO realized P&L, including transaction costs and partial sells.
    buy_lots: dict[str, deque[tuple[int, float, float]]] = {}
    realized_pnl = 0.0
    winning_sells = 0
    sell_count = 0
    total_fees = 0.0
    for t in trades:
        d = t if isinstance(t, dict) else t.model_dump()
        total_fees += float(d.get("commission", 0)) + float(d.get("tax", 0))
        if d["action"] == "buy":
            buy_lots.setdefault(d["ticker"], deque()).append(
                (int(d["shares"]), float(d["price"]), float(d.get("commission", 0)))
            )
        elif d["action"] == "sell":
            remaining = int(d["shares"])
            sell_price = float(d["price"])
            sell_commission = float(d.get("commission", 0))
            sell_tax = float(d.get("tax", 0))
            cost_basis = 0.0
            matched = 0
            lots = buy_lots.get(d["ticker"], deque())
            while remaining > 0 and lots:
                lot_shares, lot_price, lot_commission = lots[0]
                used = min(remaining, lot_shares)
                cost_basis += used * lot_price + lot_commission * used / max(lot_shares, 1)
                matched += used
                remaining -= used
                if used == lot_shares:
                    lots.popleft()
                else:
                    lots[0] = (
                        lot_shares - used,
                        lot_price,
                        lot_commission * (lot_shares - used) / lot_shares,
                    )
            if matched:
                pnl = matched * sell_price - cost_basis - sell_commission - sell_tax
                realized_pnl += pnl
                sell_count += 1
                winning_sells += int(pnl > 0)

    win_rate = winning_sells / sell_count if sell_count else 0.0

    return {
        "total_return": round(total_return, 6),
        "max_drawdown": round(max_dd, 6),
        "sharpe_ratio": round(float(sharpe), 4) if sharpe else None,
        "win_rate": round(win_rate, 4),
        "realized_pnl": round(realized_pnl, 2),
        "total_fees": round(total_fees, 2),
    }


def _should_rebalance(day_count: int, frequency: str) -> bool:
    """Use trading-day periods so missing market dates do not shift schedules."""
    periods = {"daily": 1, "weekly": 5, "monthly": 21, "manual": 0}
    period = periods.get(frequency, 5)
    return period > 0 and day_count % period == 0


def _symbol_metrics(frames: dict[str, pd.DataFrame], engine: TimeAwareTradingEngine) -> list[dict]:
    """Summarise per-symbol attribution inputs for the portfolio report."""
    trades = [trade.model_dump() for trade in engine.portfolio.trades]
    positions = {position.ticker: position for position in engine.portfolio.positions}
    metrics = []
    for ticker, frame in frames.items():
        first_close = float(frame.iloc[0]["close"])
        last_close = float(frame.iloc[-1]["close"])
        ticker_trades = [trade for trade in trades if trade["ticker"] == ticker]
        position = positions.get(ticker)
        metrics.append(
            {
                "ticker": ticker,
                "buy_hold_return": round(last_close / first_close - 1, 6) if first_close else 0.0,
                "trade_count": len(ticker_trades),
                "buy_count": sum(1 for trade in ticker_trades if str(trade["action"]) in {"buy", "Decision.BUY"}),
                "sell_count": sum(1 for trade in ticker_trades if str(trade["action"]) in {"sell", "Decision.SELL"}),
                "final_shares": position.shares if position else 0,
                "final_market_value": round(position.market_value, 2) if position else 0.0,
                "final_weight": 0.0,
            }
        )
    total_value = engine.portfolio.total_value
    for item in metrics:
        if total_value:
            item["final_weight"] = round(item["final_market_value"] / total_value, 8)
    return metrics


def _empty_result(ticker, start_date, end_date, initial_capital, error: str | None = None) -> dict:
    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "final_value": initial_capital,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": None,
        "win_rate": 0.0,
        "realized_pnl": 0.0,
        "total_fees": 0.0,
        "total_trades": 0,
        "equity_curve": [],
        "trades": [],
        "error": error,
    }


def _empty_pool_result(tickers, start_date, end_date, initial_capital) -> dict:
    result = _empty_result("pool", start_date, end_date, initial_capital)
    result["tickers"] = tickers
    return result
