"""Historical data normalization, quality checks, and reproducible snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd


class BacktestDataError(ValueError):
    """Raised when historical data is not safe to use for a backtest."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_backtest_data(
    frame: pd.DataFrame,
    *,
    ticker: str,
    asset_type: str,
    start_date: str,
    end_date: str,
    source: str = "akshare",
    adjustment: str = "provider_default",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize a provider frame and return a content-addressed data manifest.

    The manifest is deliberately derived from the exact rows consumed by the
    engine.  It makes a result auditable even when the upstream provider later
    revises historical data.
    """
    if frame is None or frame.empty:
        raise BacktestDataError(f"{ticker} 没有可用历史数据")

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise BacktestDataError(f"{ticker} 历史数据缺少字段: {', '.join(missing)}")

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    numeric_columns = ["open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized["date"].isna().any():
        raise BacktestDataError(f"{ticker} 历史数据包含无效日期")
    normalized = normalized.sort_values("date").reset_index(drop=True)
    if start_date:
        normalized = normalized[normalized["date"] >= start_date]
    if end_date:
        normalized = normalized[normalized["date"] <= end_date]
    normalized = normalized.reset_index(drop=True)
    if normalized.empty:
        raise BacktestDataError(f"{ticker} 在请求区间内没有可用历史数据")
    duplicate_dates = normalized.loc[normalized["date"].duplicated(), "date"].tolist()
    if duplicate_dates:
        raise BacktestDataError(f"{ticker} 历史数据存在重复交易日: {duplicate_dates[:3]}")

    positive = normalized[["open", "high", "low", "close"]].gt(0).all(axis=1)
    if not positive.all():
        raise BacktestDataError(f"{ticker} 历史数据包含非正价格")
    valid_ohlc = (
        (normalized["high"] >= normalized[["open", "close"]].max(axis=1))
        & (normalized["low"] <= normalized[["open", "close"]].min(axis=1))
        & (normalized["high"] >= normalized["low"])
    )
    if not valid_ohlc.all():
        raise BacktestDataError(f"{ticker} 历史数据存在不一致的 OHLC 关系")
    if normalized["volume"].isna().any() or normalized["volume"].lt(0).any():
        raise BacktestDataError(f"{ticker} 历史数据包含无效成交量")

    selected_columns = [column for column in numeric_columns if column in normalized.columns]
    selected_columns = ["date", *selected_columns]
    records = normalized[selected_columns].where(pd.notna(normalized[selected_columns]), None).to_dict(orient="records")
    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    metadata = {
        "source": source,
        "adjustment": adjustment,
        "fetched_at": _utc_now(),
        "ticker": ticker,
        "asset_type": asset_type,
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "actual_start_date": records[0]["date"],
        "actual_end_date": records[-1]["date"],
        "row_count": len(records),
        "columns": selected_columns,
        "sha256": digest,
        "quality": {
            "status": "valid",
            "duplicate_dates": 0,
            "invalid_prices": 0,
            "invalid_ohlc": 0,
            "invalid_volume": 0,
        },
    }
    return normalized, metadata
