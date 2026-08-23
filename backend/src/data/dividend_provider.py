"""AkShare adapter for normalized A-share cash-dividend records."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

_COLUMN_ALIASES = {
    "ticker": ("代码", "股票代码", "证券代码"),
    "name": ("名称", "股票简称", "证券简称"),
    "cash": ("现金分红-现金分红比例", "现金分红比例", "派息比例"),
    "status": ("方案进度", "实施进度", "状态"),
    "record_date": ("股权登记日",),
    "ex_dividend_date": ("除权除息日",),
}


def _pick(row: dict[str, Any], key: str) -> Any:
    for alias in _COLUMN_ALIASES[key]:
        if alias in row:
            return row.get(alias)
    return None


def normalize_dividend_records(raw_records: list[dict[str, Any]], fiscal_year: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in raw_records:
        ticker = str(_pick(raw, "ticker") or "").strip().removesuffix(".0").zfill(6)
        if len(ticker) != 6 or not ticker.isdigit():
            continue
        try:
            cash_per_ten = float(_pick(raw, "cash") or 0)
        except (TypeError, ValueError):
            cash_per_ten = 0.0
        result.append(
            {
                "ticker": ticker,
                "name": str(_pick(raw, "name") or "").strip(),
                "fiscal_year": fiscal_year,
                "cash_dividend_per_10_shares": cash_per_ten,
                "cash_dividend_per_share": cash_per_ten / 10.0,
                "status": str(_pick(raw, "status") or "").strip(),
                "record_date": str(_pick(raw, "record_date") or "").strip() or None,
                "ex_dividend_date": str(_pick(raw, "ex_dividend_date") or "").strip() or None,
            }
        )
    return result


def _default_fetcher(year: int) -> list[dict[str, Any]]:
    import akshare as ak

    frame = ak.stock_fhps_em(date=f"{year}1231")
    return frame.to_dict(orient="records")


async def fetch_cash_dividends(
    periods: list[int],
    *,
    fetcher: Callable[[int], list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Fetch years sequentially to avoid hammering the upstream endpoint."""
    provider = fetcher or _default_fetcher
    rows: list[dict[str, Any]] = []
    counts: dict[int, int] = {}
    for year in periods:
        raw = await asyncio.to_thread(provider, year)
        normalized = normalize_dividend_records(raw, year)
        counts[year] = len(normalized)
        rows.extend(normalized)
    return rows, counts
