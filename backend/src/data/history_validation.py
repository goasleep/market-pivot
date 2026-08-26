"""Independent historical-source reconciliation for formal backtests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import date
from typing import Any

import akshare as ak
import pandas as pd

from data.backtest_data import BacktestDataError, prepare_backtest_data
from data.exchange_fund_provider import async_get_exchange_fund_history
from data.stock_provider import async_get_stock_history
from models.schemas import AssetType
from models.strategy_research import CrossValidationReport

RULE_VERSION = "history-cross-validation-v1"
SOURCE_PRIORITY = {"eastmoney": 0, "tencent": 1, "sina": 2}


def _normalize_tencent(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    columns = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    normalized = frame.rename(columns=columns).copy()
    normalized["ticker"] = ticker
    normalized.attrs["source_metadata"] = {
        "source_id": "tencent",
        "source_name": "腾讯证券",
        "endpoint": "stock_zh_a_hist_tx",
        "fallback": False,
        "source_chain": ["tencent"],
        "cache": "miss",
    }
    return normalized


def _fetch_tencent_history(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    code = ticker.strip().lower().removeprefix("sh").removeprefix("sz").zfill(6)
    market = "sh" if code.startswith(("5", "6", "9")) else "sz"
    frame = ak.stock_zh_a_hist_tx(
        symbol=f"{market}{code}",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
        timeout=15,
    )
    return _normalize_tencent(frame, code)


async def _fetch_candidates(
    ticker: str,
    start_date: str,
    end_date: str,
    asset_type: AssetType,
) -> list[tuple[str, pd.DataFrame | Exception]]:
    primary = (
        async_get_stock_history(ticker, start_date=start_date, end_date=end_date, adjust="qfq")
        if asset_type == AssetType.STOCK
        else async_get_exchange_fund_history(
            ticker,
            asset_type=asset_type.value,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
    )
    results = await asyncio.gather(
        primary,
        asyncio.to_thread(_fetch_tencent_history, ticker, start_date, end_date),
        return_exceptions=True,
    )
    return [("primary", results[0]), ("tencent", results[1])]


def _source_id(label: str, frame: pd.DataFrame) -> str:
    metadata = dict(frame.attrs.get("source_metadata") or {})
    return str(metadata.get("source_id") or ("eastmoney" if label == "primary" else label))


def _return_comparison(left: pd.DataFrame, right: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lhs = left[["date", "close"]].rename(columns={"close": "left_close"})
    rhs = right[["date", "close"]].rename(columns={"close": "right_close"})
    merged = lhs.merge(rhs, on="date", how="inner").sort_values("date").reset_index(drop=True)
    maximum_rows = max(len(left), len(right), 1)
    coverage = len(merged) / maximum_rows
    if len(merged) < 3:
        return {
            "status": "conflict",
            "overlap_ratio": round(coverage, 6),
            "common_dates": len(merged),
            "end_date_lag_days": 999,
            "median_abs_return_diff": None,
            "within_one_pct_ratio": 0.0,
        }, []

    merged["left_return"] = pd.to_numeric(merged["left_close"], errors="coerce").pct_change(fill_method=None)
    merged["right_return"] = pd.to_numeric(merged["right_close"], errors="coerce").pct_change(fill_method=None)
    merged["abs_return_diff"] = (merged["left_return"] - merged["right_return"]).abs()
    valid = merged.dropna(subset=["abs_return_diff"])
    median_diff = float(valid["abs_return_diff"].median()) if not valid.empty else 1.0
    within_one = float((valid["abs_return_diff"] <= 0.01).mean()) if not valid.empty else 0.0
    left_end = date.fromisoformat(str(left.iloc[-1]["date"]))
    right_end = date.fromisoformat(str(right.iloc[-1]["date"]))
    earlier_end = min(left_end, right_end).isoformat()
    later_end = max(left_end, right_end).isoformat()
    union_dates = set(left["date"].astype(str)) | set(right["date"].astype(str))
    lag = sum(earlier_end < day <= later_end for day in union_dates)
    if coverage >= 0.95 and lag <= 1 and median_diff <= 0.001 and within_one >= 0.99:
        status = "verified"
    elif coverage < 0.80 or lag > 3 or median_diff > 0.0025 or within_one < 0.95:
        status = "conflict"
    else:
        status = "degraded"
    differences = [
        {
            "date": str(row.date),
            "left_return": None if pd.isna(row.left_return) else round(float(row.left_return), 8),
            "right_return": None if pd.isna(row.right_return) else round(float(row.right_return), 8),
            "abs_return_diff": None if pd.isna(row.abs_return_diff) else round(float(row.abs_return_diff), 8),
        }
        for row in merged.itertuples(index=False)
    ]
    return {
        "status": status,
        "overlap_ratio": round(coverage, 6),
        "common_dates": len(merged),
        "end_date_lag_days": lag,
        "median_abs_return_diff": round(median_diff, 8),
        "within_one_pct_ratio": round(within_one, 6),
    }, differences


def _candidate_score(
    snapshot: dict[str, Any],
    *,
    requested_end: str,
    maximum_rows: int,
    consistency_status: str,
) -> float:
    end_lag = abs((date.fromisoformat(requested_end) - date.fromisoformat(snapshot["actual_end_date"])).days)
    recency = 30.0 if end_lag <= 1 else 20.0 if end_lag <= 3 else max(0.0, 20.0 - end_lag)
    coverage = 30.0 * min(float(snapshot["row_count"]) / max(maximum_rows, 1), 1.0)
    completeness = 20.0 if set(("date", "open", "high", "low", "close", "volume")) <= set(snapshot["columns"]) else 0.0
    consistency = {"verified": 20.0, "degraded": 10.0, "conflict": 0.0, "unverified": 5.0}[consistency_status]
    return round(recency + coverage + completeness + consistency, 4)


async def prepare_cross_validated_backtest_data(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    asset_type: AssetType | str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Fetch independent candidates, select deterministically, and freeze the winner."""
    kind = AssetType(asset_type)
    raw_candidates = await _fetch_candidates(ticker, start_date, end_date, kind)
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for label, raw in raw_candidates:
        if isinstance(raw, Exception) or raw is None or raw.empty:
            failures.append({"source_id": label, "eligible": False, "issues": [str(raw)[:500]]})
            continue
        source_id = _source_id(label, raw)
        if source_id in seen_sources:
            failures.append(
                {
                    "source_id": source_id,
                    "eligible": False,
                    "issues": ["与另一候选实际来自同一上游，未作为独立交叉核验来源"],
                }
            )
            continue
        metadata = dict(raw.attrs.get("source_metadata") or {})
        try:
            frame, snapshot = prepare_backtest_data(
                raw,
                ticker=ticker,
                asset_type=kind.value,
                start_date=start_date,
                end_date=end_date,
                source=source_id,
                source_metadata=metadata,
                adjustment="none" if source_id == "sina" else "qfq",
            )
        except BacktestDataError as exc:
            failures.append({"source_id": source_id, "eligible": False, "issues": [str(exc)]})
            continue
        seen_sources.add(source_id)
        candidates.append({"source_id": source_id, "frame": frame, "snapshot": snapshot, "metadata": metadata})
    if not candidates:
        raise BacktestDataError(f"{ticker} 所有历史行情源均不可用")

    comparison: dict[str, Any] = {"status": "unverified"}
    differences: list[dict[str, Any]] = []
    if len(candidates) >= 2:
        comparison, differences = _return_comparison(candidates[0]["frame"], candidates[1]["frame"])
        comparison.update(
            {
                "left_source": candidates[0]["source_id"],
                "right_source": candidates[1]["source_id"],
            }
        )
        differences = [
            {
                "left_source": candidates[0]["source_id"],
                "right_source": candidates[1]["source_id"],
                **item,
            }
            for item in differences
        ]
    maximum_rows = max(int(item["snapshot"]["row_count"]) for item in candidates)
    consistency = str(comparison["status"])
    for item in candidates:
        item["score"] = _candidate_score(
            item["snapshot"],
            requested_end=end_date,
            maximum_rows=maximum_rows,
            consistency_status=consistency,
        )
    candidates.sort(
        key=lambda item: (
            -item["score"],
            SOURCE_PRIORITY.get(str(item["source_id"]), 99),
            item["source_id"],
        )
    )
    selected = candidates[0]
    if consistency == "conflict" and len(candidates) > 1 and selected["score"] - candidates[1]["score"] < 10:
        raise BacktestDataError(f"{ticker} 独立行情源存在严重冲突，无法形成可信正式排名")
    status = "unverified" if len(candidates) == 1 else "verified" if consistency == "verified" else "degraded"
    candidate_reports = []
    for item in candidates:
        snapshot = item["snapshot"]
        candidate_reports.append(
            {
                "source_id": item["source_id"],
                "source_name": item["metadata"].get("source_name", item["source_id"]),
                "eligible": True,
                "selected": item is selected,
                "quality_score": item["score"],
                "actual_start_date": snapshot["actual_start_date"],
                "actual_end_date": snapshot["actual_end_date"],
                "row_count": snapshot["row_count"],
                "adjustment": snapshot["adjustment"],
                "sha256": snapshot["sha256"],
                "issues": [],
            }
        )
    candidate_reports.extend(
        {
            "source_id": item["source_id"],
            "source_name": item["source_id"],
            "eligible": False,
            "selected": False,
            "quality_score": 0.0,
            "row_count": 0,
            "issues": item["issues"],
        }
        for item in failures
    )
    report = CrossValidationReport.model_validate(
        {
            "status": status,
            "selected_source": selected["source_id"],
            "selection_reason": (
                "独立来源核验通过，按质量分选择"
                if status == "verified"
                else "按确定性质量分选择，结果需结合数据警告解读"
            ),
            "rule_version": RULE_VERSION,
            "candidates": candidate_reports,
            "comparison": comparison,
            "differences": differences,
        }
    ).model_dump(mode="json")
    selected_snapshot = dict(selected["snapshot"])
    selected_snapshot["cross_validation_sha256"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()
    return selected["frame"], selected_snapshot, report
