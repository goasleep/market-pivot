"""Broad-market index history used as an optional strategy comparison benchmark."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import pandas as pd

DEFAULT_MARKET_BENCHMARK_TICKER = "000300"
DEFAULT_MARKET_BENCHMARK_NAME = "沪深300"


class MarketIndexDataError(RuntimeError):
    """Raised when no usable broad-market index history can be obtained."""


def _normalize_index_history(frame: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    normalized = frame.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
    ).copy()
    if normalized.empty or not {"date", "close"}.issubset(normalized.columns):
        return pd.DataFrame()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["date", "close"])
    normalized["date"] = normalized["date"].dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low"):
        if column not in normalized.columns:
            normalized[column] = normalized["close"]
        else:
            normalized[column] = normalized[column].fillna(normalized["close"])
    if "volume" not in normalized.columns:
        normalized["volume"] = 0.0
    normalized["ticker"] = ticker
    normalized = normalized.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    normalized.attrs["source_metadata"] = {
        "source_id": source,
        "source_name": "东方财富（AkShare）" if source == "eastmoney" else "新浪财经（AkShare）",
        "endpoint": "index_zh_a_hist" if source == "eastmoney" else "stock_zh_index_daily",
    }
    return normalized


def get_market_index_history(
    ticker: str = DEFAULT_MARKET_BENCHMARK_TICKER,
    *,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one broad-market index with a second independent endpoint fallback."""
    import akshare as ak

    code = ticker.strip().lower().removeprefix("sh").removeprefix("sz").zfill(6)
    compact_start = start_date.replace("-", "")
    compact_end = end_date.replace("-", "")
    errors = []
    frame = pd.DataFrame()
    source = "eastmoney"
    try:
        frame = _normalize_index_history(
            ak.index_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=compact_start,
                end_date=compact_end,
            ),
            code,
            source,
        )
    except Exception as exc:
        errors.append(f"东方财富指数接口: {exc}")
    if frame.empty:
        source = "sina"
        try:
            market = "sh" if code.startswith(("0", "5", "6", "9")) else "sz"
            frame = _normalize_index_history(ak.stock_zh_index_daily(symbol=f"{market}{code}"), code, source)
        except Exception as exc:
            errors.append(f"新浪指数接口: {exc}")
    if frame.empty:
        raise MarketIndexDataError("；".join(errors) or f"指数 {code} 没有可用历史数据")
    frame = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)].reset_index(drop=True)
    if len(frame) < 2:
        raise MarketIndexDataError(f"指数 {code} 在请求区间内不足 2 个交易日")
    records = frame[["date", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
    snapshot = {
        "ticker": code,
        "source": source,
        "actual_start_date": str(frame.iloc[0]["date"]),
        "actual_end_date": str(frame.iloc[-1]["date"]),
        "row_count": len(frame),
        "sha256": hashlib.sha256(
            json.dumps(records, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "source_metadata": dict(frame.attrs.get("source_metadata") or {}),
    }
    return frame, snapshot


async def async_get_market_index_history(*args, **kwargs) -> tuple[pd.DataFrame, dict[str, Any]]:
    return await asyncio.to_thread(get_market_index_history, *args, **kwargs)
