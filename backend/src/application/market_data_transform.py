"""Deterministic Polars execution for the bounded financial transform DSL."""

from __future__ import annotations

from typing import Any

import polars as pl

from models.financial_task import FinancialTaskSpec
from models.market_data import TaskAcceptanceResult

IMPLEMENTED_STATUS_TOKENS = ("实施", "已实施", "实施完成")


def execute_transform(rows: list[dict[str, Any]], spec: FinancialTaskSpec) -> list[dict[str, Any]]:
    """Execute declared operators without generated Python or arbitrary expressions."""
    if not rows:
        return []
    frame = pl.DataFrame(rows)
    for instruction in spec.transforms:
        params = instruction.params
        if instruction.operator == "filter":
            field = str(params["field"])
            op = str(params["op"])
            value = params.get("value")
            if op == "gt":
                frame = frame.filter(pl.col(field) > value)
            elif op == "eq":
                frame = frame.filter(pl.col(field) == value)
            elif op == "in":
                frame = frame.filter(pl.col(field).is_in(value or []))
            else:
                raise ValueError(f"不支持的过滤操作: {op}")
        elif instruction.operator == "group_by":
            by = [str(item) for item in params.get("by", [])]
            expressions = []
            for field, aggregation in (params.get("aggregations") or {}).items():
                if aggregation != "sum":
                    raise ValueError(f"不支持的聚合操作: {aggregation}")
                expressions.append(pl.col(str(field)).sum().alias(str(field)))
            frame = frame.group_by(by).agg(expressions)
        elif instruction.operator == "require_period_coverage":
            entity = str(params["entity"])
            period = str(params["period"])
            required = [int(item) for item in params.get("periods", [])]
            eligible = (
                frame.filter(pl.col("cash_dividend_per_share") > 0)
                .group_by(entity)
                .agg(pl.col(period).n_unique().alias("_period_count"))
                .filter(pl.col("_period_count") == len(required))
                .select(entity)
            )
            frame = frame.join(eligible, on=entity, how="inner")
        elif instruction.operator == "pivot":
            frame = frame.pivot(
                on=str(params["columns"]),
                index=[str(item) for item in params.get("index", [])],
                values=str(params["values"]),
                aggregate_function="sum",
            )
        elif instruction.operator == "derive":
            sum_fields = [str(item) for item in params.get("sum_fields", [])]
            frame = frame.with_columns(
                pl.sum_horizontal([pl.col(field).fill_null(0) for field in sum_fields]).alias(str(params["field"]))
            )
        elif instruction.operator == "sort":
            frame = frame.sort(
                [str(item) for item in params.get("by", [])],
                descending=[bool(item) for item in params.get("descending", [])],
            )
        elif instruction.operator == "select":
            frame = frame.select([str(item) for item in params.get("columns", [])])
        elif instruction.operator == "head":
            frame = frame.head(int(params.get("count") or 30))
        else:
            raise ValueError(f"不支持的变换操作: {instruction.operator}")
    return frame.to_dicts()


def prepare_dividend_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply provider-semantic filters before the general transform pipeline."""
    return [
        row
        for row in rows
        if any(token in str(row.get("status") or "") for token in IMPLEMENTED_STATUS_TOKENS)
        and float(row.get("cash_dividend_per_10_shares") or 0) > 0
    ]


def evaluate_acceptance(
    spec: FinancialTaskSpec,
    rows: list[dict[str, Any]],
    *,
    returned_periods: list[int],
) -> TaskAcceptanceResult:
    checks: list[dict[str, Any]] = []
    issues: list[str] = []
    for criterion in spec.acceptance:
        params = criterion.params
        passed = True
        if criterion.criterion == "dataset_resolved":
            passed = bool(spec.primary_dataset_id) and spec.primary_dataset_id == params.get("dataset_id")
        elif criterion.criterion == "period_coverage":
            required = {int(item) for item in params.get("periods", [])}
            passed = required <= set(returned_periods)
        elif criterion.criterion == "non_empty":
            passed = bool(rows)
        elif criterion.criterion == "fields_present":
            required_fields = {str(item) for item in params.get("fields", [])}
            passed = bool(rows) and required_fields <= set(rows[0])
        elif criterion.criterion == "sorted":
            field = str(params["field"])
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            passed = values == sorted(values, reverse=bool(params.get("descending")))
        checks.append({"criterion": criterion.criterion, "passed": passed, "params": params})
        if not passed:
            issues.append(f"验收未通过: {criterion.criterion}")
    if not returned_periods:
        status = "data_unavailable"
    elif not rows:
        status = "invalid_result" if set(spec.periods) <= set(returned_periods) else "partial"
    elif all(item["passed"] for item in checks):
        status = "satisfied"
    else:
        status = "partial"
    return TaskAcceptanceResult(status=status, satisfied=status == "satisfied", checks=checks, issues=issues)
