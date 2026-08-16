"""Polars-powered ETF/LOF candidate screening.

The screener deliberately stops at candidate selection.  It does not turn a
high score into a buy order; the research workflow must still inspect the
candidate and produce an evidence-backed decision.
"""

from __future__ import annotations

from typing import Any

import polars as pl

_NUMERIC_COLUMNS = (
    "price",
    "pct_chg",
    "amount",
    "volume",
    "turnover",
    "total_mv",
    "discount_rate",
    "iopv",
)
_SORT_COLUMNS = {
    "screen_score",
    "amount",
    "pct_chg",
    "turnover",
    "total_mv",
    "discount_rate",
}


def _numeric_expr(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Float64, strict=False).fill_null(0.0)


class FundScreener:
    """Screen an ETF/LOF market snapshot without making a trade decision."""

    def screen_snapshot(
        self,
        records: list[dict[str, Any]],
        *,
        asset_type: str,
        min_pct_chg: float | None = None,
        max_pct_chg: float | None = None,
        min_amount: float | None = None,
        min_turnover: float | None = None,
        keyword: str | None = None,
        sort_by: str = "screen_score",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return a ranked candidate list from one market-wide snapshot.

        The score is intentionally transparent: liquidity contributes 45%,
        turnover 20%, and one-day momentum 35%.  The score is a screening
        aid only; it is not a buy signal.
        """
        if asset_type not in {"etf", "lof"}:
            raise ValueError("FundScreener only supports etf and lof")
        if not records:
            return []

        frame = pl.DataFrame(records)
        for name in _NUMERIC_COLUMNS:
            if name not in frame.columns:
                frame = frame.with_columns(pl.lit(0.0).alias(name))
        frame = frame.with_columns([_numeric_expr(name).alias(name) for name in _NUMERIC_COLUMNS])

        if "name" not in frame.columns:
            frame = frame.with_columns(pl.lit("").alias("name"))
        if keyword:
            frame = frame.filter(
                pl.col("name")
                .cast(pl.String)
                .str.to_lowercase()
                .str.contains(keyword.strip().lower(), literal=True)
            )
        if min_pct_chg is not None:
            frame = frame.filter(pl.col("pct_chg") >= min_pct_chg)
        if max_pct_chg is not None:
            frame = frame.filter(pl.col("pct_chg") <= max_pct_chg)
        if min_amount is not None:
            frame = frame.filter(pl.col("amount") >= min_amount)
        if min_turnover is not None:
            frame = frame.filter(pl.col("turnover") >= min_turnover)
        if frame.is_empty():
            return []

        max_amount = float(frame.select(pl.col("amount").max()).item() or 0.0)
        max_turnover = float(frame.select(pl.col("turnover").max()).item() or 0.0)
        amount_scale = max(max_amount, 1.0)
        turnover_scale = max(max_turnover, 1.0)

        frame = frame.with_columns(
            (
                (pl.col("amount") / amount_scale * 45.0)
                + (pl.col("turnover") / turnover_scale * 20.0)
                + (((pl.col("pct_chg").clip(-5.0, 5.0) + 5.0) / 10.0) * 35.0)
            )
            .round(2)
            .alias("screen_score")
        )

        if sort_by not in _SORT_COLUMNS:
            sort_by = "screen_score"
        frame = frame.sort(sort_by, descending=True).head(max(1, min(limit, 50)))

        output: list[dict[str, Any]] = []
        for row in frame.to_dicts():
            reasons = [
                f"成交额 {row['amount']:.0f} 元",
                f"换手率 {row['turnover']:.2f}%",
                f"当日涨跌幅 {row['pct_chg']:.2f}%",
            ]
            if row.get("iopv", 0.0):
                reasons.append(f"IOPV {row['iopv']:.4f}")
            if row.get("discount_rate", 0.0):
                reasons.append(f"折溢价率 {row['discount_rate']:.2f}%")
            row["screen_score"] = float(row.get("screen_score", 0.0))
            row["screen_basis"] = reasons
            output.append(row)
        return output
