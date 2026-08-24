"""Deterministic PNG charts used as visual evidence for financial agents."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd


class ChartDataUnavailableError(ValueError):
    """Raised when source data cannot support a truthful chart."""


@dataclass(frozen=True)
class RenderedFinancialChart:
    name: str
    content: bytes
    metadata: dict[str, Any]


def _history_frame(records: list[dict[str, Any]], *, limit: int, minimum: int) -> pd.DataFrame:
    if not records:
        raise ChartDataUnavailableError("历史行情为空")
    frame = pd.DataFrame(records)
    required = ("date", "open", "high", "low", "close", "volume")
    if any(column not in frame.columns for column in required):
        raise ChartDataUnavailableError("历史行情缺少 OHLCV 字段")
    frame = frame.loc[:, required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_values("date").drop_duplicates("date", keep="last")
    frame = frame[
        (frame["open"] > 0)
        & (frame["high"] > 0)
        & (frame["low"] > 0)
        & (frame["close"] > 0)
        & (frame["volume"] >= 0)
        & (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
    ].tail(limit)
    if len(frame) < minimum:
        raise ChartDataUnavailableError(f"有效行情不足 {minimum} 个交易日")
    return frame.set_index("date")


def _save_figure(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def _metadata(frame: pd.DataFrame, *, chart_type: str, adjustment: str) -> dict[str, Any]:
    return {
        "chart_type": chart_type,
        "chart_version": "1.0",
        "start_date": frame.index[0].strftime("%Y-%m-%d"),
        "end_date": frame.index[-1].strftime("%Y-%m-%d"),
        "row_count": len(frame),
        "adjustment": adjustment,
        "width": 1400,
        "height": 900,
    }


def render_technical_chart(
    ticker: str,
    asset_type: str,
    records: list[dict[str, Any]],
) -> RenderedFinancialChart:
    """Render candles, moving averages, volume and MACD for the latest 120 sessions."""
    frame = _history_frame(records, limit=120, minimum=20)
    chart = frame.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    close = chart["Close"]
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    histogram = (dif - dea) * 2
    histogram_colors = np.where(histogram >= 0, "#dc2626", "#16a34a")
    volume_ma20 = chart["Volume"].rolling(20).mean()
    add_plots = [
        mpf.make_addplot(volume_ma20, panel=1, color="#f59e0b", width=1.1, ylabel="Volume"),
        mpf.make_addplot(dif, panel=2, color="#2563eb", width=1.1, ylabel="MACD"),
        mpf.make_addplot(dea, panel=2, color="#f59e0b", width=1.1),
        mpf.make_addplot(histogram, panel=2, type="bar", color=histogram_colors, alpha=0.65),
    ]
    market_colors = mpf.make_marketcolors(
        up="#dc2626",
        down="#16a34a",
        edge="inherit",
        wick="inherit",
        volume="inherit",
    )
    style = mpf.make_mpf_style(
        marketcolors=market_colors,
        gridstyle="--",
        gridcolor="#dbe3ef",
        facecolor="white",
        figcolor="white",
        y_on_right=False,
    )
    moving_averages = tuple(window for window in (5, 20, 60) if len(chart) >= window)
    title = (
        f"{ticker} {asset_type.upper()} | Daily Candles | MA{moving_averages} | "
        f"Red=Up Green=Down | Close={close.iloc[-1]:.4f}"
    )
    fig, axes = mpf.plot(
        chart,
        type="candle",
        style=style,
        mav=moving_averages,
        volume=True,
        addplot=add_plots,
        panel_ratios=(6, 2, 2),
        figsize=(14, 9),
        returnfig=True,
        title=title,
        datetime_format="%Y-%m-%d",
        xrotation=15,
        tight_layout=True,
        warn_too_much_data=200,
    )
    axes[0].axhline(close.iloc[-1], color="#64748b", linestyle=":", linewidth=1)
    adjustment = "qfq" if asset_type == "stock" else "none"
    metadata = _metadata(frame, chart_type="technical", adjustment=adjustment)
    metadata["indicators"] = [*(f"MA{item}" for item in moving_averages), "volume_ma20", "MACD"]
    return RenderedFinancialChart(
        name=f"{ticker}-technical-{metadata['end_date']}.png",
        content=_save_figure(fig),
        metadata=metadata,
    )


def calculate_market_risk_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate deterministic short/medium-term risk metrics from closing prices."""
    frame = _history_frame(records, limit=120, minimum=20)
    close = frame["close"]
    returns = close.pct_change().dropna()
    running_max = close.cummax()
    drawdown = close / running_max - 1
    volatility_20 = returns.rolling(20).std() * math.sqrt(252)
    return {
        "return_5d_pct": round(float((close.iloc[-1] / close.iloc[-6] - 1) * 100), 2) if len(close) >= 6 else None,
        "return_20d_pct": round(float((close.iloc[-1] / close.iloc[-21] - 1) * 100), 2)
        if len(close) >= 21
        else None,
        "max_drawdown_120d_pct": round(float(drawdown.min() * 100), 2),
        "current_drawdown_pct": round(float(drawdown.iloc[-1] * 100), 2),
        "volatility_20d_annualized_pct": round(float(volatility_20.iloc[-1] * 100), 2)
        if not pd.isna(volatility_20.iloc[-1])
        else None,
        "down_days_20": int((returns.tail(20) < 0).sum()),
    }


def render_risk_chart(
    ticker: str,
    asset_type: str,
    records: list[dict[str, Any]],
) -> RenderedFinancialChart:
    """Render normalized price, drawdown and rolling volatility."""
    frame = _history_frame(records, limit=120, minimum=20)
    close = frame["close"]
    normalized = close / close.iloc[0] * 100
    drawdown = (close / close.cummax() - 1) * 100
    volatility = close.pct_change().rolling(20).std() * math.sqrt(252) * 100

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [4, 2, 2]})
    fig.suptitle(f"{ticker} {asset_type.upper()} | 120-Session Risk Evidence", fontsize=15, fontweight="bold")
    axes[0].plot(normalized.index, normalized, color="#2563eb", linewidth=1.8, label="Normalized Price (100)")
    axes[0].axhline(100, color="#94a3b8", linestyle=":", linewidth=1)
    axes[0].legend(loc="upper left")
    axes[0].set_ylabel("Index")
    axes[1].fill_between(drawdown.index, drawdown, 0, color="#dc2626", alpha=0.28)
    axes[1].plot(drawdown.index, drawdown, color="#dc2626", linewidth=1)
    axes[1].set_ylabel("Drawdown %")
    axes[2].plot(volatility.index, volatility, color="#7c3aed", linewidth=1.5, label="20D Annualized Volatility")
    axes[2].legend(loc="upper left")
    axes[2].set_ylabel("Volatility %")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
    fig.autofmt_xdate(rotation=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    adjustment = "qfq" if asset_type == "stock" else "none"
    metadata = _metadata(frame, chart_type="risk", adjustment=adjustment)
    metadata["metrics"] = calculate_market_risk_metrics(records)
    return RenderedFinancialChart(
        name=f"{ticker}-risk-{metadata['end_date']}.png",
        content=_save_figure(fig),
        metadata=metadata,
    )


def render_fund_structure_chart(
    ticker: str,
    asset_type: str,
    price_records: list[dict[str, Any]],
    nav_records: list[dict[str, Any]],
) -> RenderedFinancialChart:
    """Render exchange price versus unit NAV and the resulting premium/discount."""
    if asset_type not in {"etf", "lof"}:
        raise ChartDataUnavailableError("基金结构图仅适用于 ETF 或 LOF")
    prices = pd.DataFrame(price_records)
    nav = pd.DataFrame(nav_records)
    if not {"date", "close"}.issubset(prices.columns) or not {"date", "unit_nav"}.issubset(nav.columns):
        raise ChartDataUnavailableError("缺少场内价格或单位净值字段")
    prices = prices[["date", "close"]].copy()
    nav = nav[["date", "unit_nav"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    nav["unit_nav"] = pd.to_numeric(nav["unit_nav"], errors="coerce")
    merged = (
        prices.merge(nav, on="date", how="inner")
        .dropna()
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )
    merged = merged[(merged["close"] > 0) & (merged["unit_nav"] > 0)].tail(60).set_index("date")
    if len(merged) < 5:
        raise ChartDataUnavailableError("场内价格与单位净值的有效重叠日期不足 5 日")
    price_index = merged["close"] / merged["close"].iloc[0] * 100
    nav_index = merged["unit_nav"] / merged["unit_nav"].iloc[0] * 100
    premium = (merged["close"] / merged["unit_nav"] - 1) * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [5, 3]})
    fig.suptitle(f"{ticker} {asset_type.upper()} | Exchange Price vs Unit NAV", fontsize=15, fontweight="bold")
    axes[0].plot(price_index.index, price_index, color="#2563eb", linewidth=1.8, label="Exchange Price (100)")
    axes[0].plot(nav_index.index, nav_index, color="#f59e0b", linewidth=1.8, label="Unit NAV (100)")
    axes[0].legend(loc="upper left")
    axes[0].set_ylabel("Normalized Index")
    colors = np.where(premium >= 0, "#dc2626", "#16a34a")
    axes[1].bar(premium.index, premium, color=colors, alpha=0.72, label="Premium / Discount")
    axes[1].axhline(0, color="#64748b", linewidth=1)
    axes[1].set_ylabel("Premium %")
    axes[1].legend(loc="upper left")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
    fig.autofmt_xdate(rotation=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    metadata = {
        "chart_type": "fund_structure",
        "chart_version": "1.0",
        "start_date": merged.index[0].strftime("%Y-%m-%d"),
        "end_date": merged.index[-1].strftime("%Y-%m-%d"),
        "row_count": len(merged),
        "adjustment": "none",
        "width": 1400,
        "height": 900,
        "latest_premium_pct": round(float(premium.iloc[-1]), 4),
    }
    return RenderedFinancialChart(
        name=f"{ticker}-fund-structure-{metadata['end_date']}.png",
        content=_save_figure(fig),
        metadata=metadata,
    )
