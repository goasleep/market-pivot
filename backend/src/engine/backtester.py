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

import numpy as np
import pandas as pd
from loguru import logger

from data.akshare_provider import async_get_fund_history, async_get_stock_history
from data.market_context import build_market_context
from engine.trading_engine import TimeAwareTradingEngine, decision_shares
from graph.workflow import workflow
from models.schemas import AssetType, Decision, SimulationAccountConfig


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

    # Filter to date range
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)

    trading_dates = df["date"].tolist()
    logger.info(f"Backtest period: {len(trading_dates)} trading days")

    # 2. Initialize engine
    engine = TimeAwareTradingEngine(
        initial_capital=initial_capital,
        rules=SimulationAccountConfig(initial_cash=initial_capital),
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
                result = await workflow.ainvoke(state)
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

    return {
        "ticker": ticker,
        "asset_type": asset_type.value,
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
) -> dict:
    """Run the same Agent and execution semantics across a stock pool.

    Each symbol receives an as-of historical context.  Decisions are merged
    into one portfolio, next-open orders remain pending until that symbol has
    a subsequent bar, and the shared ``decision_shares`` helper keeps sizing
    consistent with the persistent simulation account.
    """

    asset_type = AssetType(asset_type)
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
    for symbol, frame in zip(symbols, frames):
        if frame.empty or len(frame) < 5:
            continue
        item = frame.copy()
        item["date"] = pd.to_datetime(item["date"]).dt.strftime("%Y-%m-%d")
        valid[symbol] = item.sort_values("date").reset_index(drop=True)
    if not valid:
        return _empty_pool_result(symbols, start_date, end_date, initial_capital)

    trading_dates = sorted({day for frame in valid.values() for day in frame["date"].tolist()})
    engine = TimeAwareTradingEngine(
        initial_capital=initial_capital,
        rules=SimulationAccountConfig(initial_cash=initial_capital),
    )
    engine.set_available_dates(trading_dates)
    pending: dict[str, object] = {}
    equity_curve: list[dict] = []

    for day_count, current_date in enumerate(trading_dates):
        engine.advance_to_date(current_date)
        rows_today: dict[str, pd.Series] = {}
        for symbol, frame in valid.items():
            matches = frame[frame["date"] == current_date]
            if not matches.empty:
                rows_today[symbol] = matches.iloc[-1]

        for symbol, decision in list(pending.items()):
            row = rows_today.get(symbol)
            if row is not None:
                _execute_decision(engine, decision, symbol, float(row["open"]), current_date)
                pending.pop(symbol, None)
        engine.update_prices({symbol: float(row["close"]) for symbol, row in rows_today.items()})

        if day_count % decision_interval == 0:
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
                    result = await workflow.ainvoke(
                        {
                            "ticker": symbol,
                            "asset_type": asset_type.value,
                            "current_price": current_price,
                            "as_of_date": current_date,
                            "is_backtest": True,
                            "strategy_name": strategy_name,
                            "market_context": context,
                            "progress": [],
                        }
                    )
                    decision = result.get("final_decision")
                    if decision and decision.decision != Decision.HOLD:
                        if fill_time == "same_close":
                            _execute_decision(engine, decision, symbol, current_price, current_date)
                        else:
                            pending[symbol] = decision
                except Exception as exc:
                    logger.error(f"Agent error on {symbol} {current_date}: {exc}")

        equity_curve.append({"date": current_date, "value": round(engine.portfolio.total_value, 2)})

    if progress_callback:
        await progress_callback("calculating", "Calculating pool performance metrics...")
    metrics = _calc_metrics(equity_curve, engine.portfolio.trades, initial_capital)
    return {
        "ticker": "pool",
        "asset_type": asset_type.value,
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


def _execute_decision(engine, decision, ticker: str, price: float, date: str):
    """Execute a trade decision."""
    shares = decision_shares(engine.portfolio, engine.rules, decision, price)
    if shares <= 0:
        return
    if decision.decision == Decision.BUY:
        engine.buy(ticker, shares, price, date)
    elif decision.decision == Decision.SELL:
        engine.sell(ticker, shares, price, date)


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
        "realized_pnl": 0.0,
        "total_fees": 0.0,
        "total_trades": 0,
        "equity_curve": [],
        "trades": [],
    }


def _empty_pool_result(tickers, start_date, end_date, initial_capital) -> dict:
    result = _empty_result("pool", start_date, end_date, initial_capital)
    result["tickers"] = tickers
    return result
