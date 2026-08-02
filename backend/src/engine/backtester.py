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
import json
from datetime import datetime
from loguru import logger
import pandas as pd
import numpy as np

from models.schemas import Decision
from data.akshare_provider import get_stock_history
from engine.trading_engine import TimeAwareTradingEngine
from graph.workflow import workflow


async def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 1_000_000,
    initial_position: int = 0,
    decision_interval: int = 1,
    num_news: int = 5,
    progress_callback=None,
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
    logger.info(f"Backtest {ticker} {start_date} -> {end_date}")

    # 1. Fetch historical data
    if progress_callback:
        await progress_callback("data_fetch", f"Fetching history for {ticker}...")

    df = get_stock_history(ticker, start_date=start_date, end_date=end_date)
    if df.empty or len(df) < 5:
        return _empty_result(ticker, start_date, end_date, initial_capital)

    # Filter to date range
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)

    trading_dates = df["date"].tolist()
    logger.info(f"Backtest period: {len(trading_dates)} trading days")

    # 2. Initialize engine
    engine = TimeAwareTradingEngine(initial_capital=initial_capital)
    engine.set_available_dates(trading_dates)

    # Set initial position if provided
    if initial_position > 0:
        first_price = float(df.iloc[0]["close"])
        from models.schemas import Position

        engine.portfolio.positions.append(Position(ticker=ticker, shares=initial_position, avg_cost=first_price))

    # 3. Iterate over trading days
    equity_curve = []
    day_count = 0

    for i, row in df.iterrows():
        current_date = row["date"]
        current_price = float(row["close"])

        engine.advance_to_date(current_date)
        engine.update_prices({ticker: current_price})

        # Record equity snapshot
        equity_curve.append(
            {
                "date": current_date,
                "value": round(engine.portfolio.total_value, 2),
            }
        )

        # Run agent at interval
        if day_count % decision_interval == 0 and not engine.is_finished:
            if progress_callback:
                pct = engine.progress_pct
                await progress_callback(
                    "agent_run",
                    f"Day {day_count + 1}/{len(trading_dates)} ({pct:.0f}%): Agent analyzing...",
                )

            try:
                # Run agent workflow
                state = {
                    "ticker": ticker,
                    "current_price": current_price,
                    "progress": [],
                }
                result = await workflow.ainvoke(state)
                decision = result.get("final_decision")

                if decision and decision.decision != Decision.HOLD:
                    _execute_decision(engine, decision, ticker, current_price, current_date)
            except Exception as e:
                logger.error(f"Agent error on {current_date}: {e}")

        day_count += 1

    # 4. Calculate metrics
    if progress_callback:
        await progress_callback("calculating", "Calculating performance metrics...")

    metrics = _calc_metrics(equity_curve, engine.portfolio.trades, initial_capital)

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "final_value": round(engine.portfolio.total_value, 2),
        "total_return": metrics["total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "win_rate": metrics["win_rate"],
        "total_trades": len(engine.portfolio.trades),
        "equity_curve": equity_curve,
        "trades": [t.model_dump() for t in engine.portfolio.trades],
    }


def _execute_decision(engine, decision, ticker: str, price: float, date: str):
    """Execute a trade decision."""
    if decision.decision == Decision.BUY:
        # Calculate shares based on position_size
        position_pct = decision.position_size or 0.2
        max_invest = engine.portfolio.cash * position_pct
        shares = int(max_invest / price)
        if shares >= 100:
            engine.buy(ticker, shares, price, date)

    elif decision.decision == Decision.SELL:
        pos = engine._find_position(ticker)
        if pos:
            # Sell a portion (or all if stop loss hit)
            if decision.stop_loss and price <= decision.stop_loss:
                engine.sell(ticker, pos.shares, price, date)  # full sell
            else:
                sell_shares = pos.shares // 2  # sell half
                if sell_shares >= 100:
                    engine.sell(ticker, sell_shares, price, date)


def _calc_metrics(equity_curve: list[dict], trades: list, initial_capital: float) -> dict:
    """Calculate backtest performance metrics."""
    if not equity_curve:
        return {"total_return": 0, "max_drawdown": 0, "win_rate": 0}

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

    # Win rate
    buy_trades = {}
    sell_results = []
    for t in trades:
        d = t if isinstance(t, dict) else t.model_dump()
        if d["action"] == "buy":
            buy_trades[d["ticker"]] = d["price"]
        elif d["action"] == "sell" and d["ticker"] in buy_trades:
            sell_results.append(d["price"] > buy_trades[d["ticker"]])

    win_rate = sum(sell_results) / len(sell_results) if sell_results else 0.0

    return {
        "total_return": round(total_return, 6),
        "max_drawdown": round(max_dd, 6),
        "sharpe_ratio": round(float(sharpe), 4) if sharpe else None,
        "win_rate": round(win_rate, 4),
    }


def _empty_result(ticker, start_date, end_date, initial_capital) -> dict:
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
        "total_trades": 0,
        "equity_curve": [],
        "trades": [],
    }
