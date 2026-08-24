"""Generic dataset query service with deterministic transformation and acceptance."""

from __future__ import annotations

import csv
import io
from datetime import date
from typing import Any

from application.data_analysis_sandbox import run_data_analysis
from application.market_data_transform import (
    evaluate_acceptance,
    execute_transform,
    prepare_dividend_source_rows,
)
from data.dividend_provider import fetch_cash_dividends
from data.market_data_catalog import market_data_catalog
from data.source_registry import provenance
from models.financial_task import FinancialTaskSpec
from models.market_data import DatasetCoverage, MarketDataResult


class MarketDataQueryError(ValueError):
    """A stable, user-safe market-data capability or contract error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def rows_to_csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


class MarketDataQueryService:
    async def execute(self, spec: FinancialTaskSpec) -> tuple[MarketDataResult, str | None]:
        dataset_id = spec.primary_dataset_id
        if not dataset_id:
            raise MarketDataQueryError("invalid_task_spec", "市场数据查询缺少已解析的 dataset_id")
        try:
            dataset = market_data_catalog.get(dataset_id)
        except ValueError as exc:
            raise MarketDataQueryError(
                "dataset_unavailable",
                f"当前市场数据目录未登记数据集 {dataset_id}",
            ) from exc
        if not market_data_catalog.supports_dataset_asset_type(dataset, spec.asset_type.value):
            supported = "、".join(dataset.asset_types)
            raise MarketDataQueryError(
                "dataset_asset_mismatch",
                f"数据集 {dataset_id} 仅支持 {supported}，不支持 {spec.asset_type.value}",
            )
        if dataset_id != "cn_a_share_cash_dividends":
            raise MarketDataQueryError(
                "dataset_adapter_unavailable",
                f"数据集 {dataset_id} 尚未配置查询适配器",
            )

        source_rows, counts = await fetch_cash_dividends(spec.periods)
        returned_periods = sorted(year for year, count in counts.items() if count > 0)
        prepared = prepare_dividend_source_rows(source_rows)
        transformed = (
            await run_data_analysis(spec.analysis_fallback_source, prepared)
            if spec.analysis_fallback_source
            else execute_transform(prepared, spec)
        )
        acceptance = evaluate_acceptance(spec, transformed, returned_periods=returned_periods)
        missing = sorted(set(spec.periods) - set(returned_periods))
        coverage_status = "complete" if not missing else "partial" if returned_periods else "unavailable"
        columns = spec.output.columns or (list(transformed[0]) if transformed else [])
        result = MarketDataResult(
            dataset_id=dataset_id,
            available=bool(transformed),
            rows=[],
            preview=transformed[: spec.output.preview_limit],
            schema_fields=columns,
            coverage=DatasetCoverage(
                requested_periods=spec.periods,
                returned_periods=returned_periods,
                missing_periods=missing,
                source_rows=len(source_rows),
                result_rows=len(transformed),
                status=coverage_status,
            ),
            acceptance=acceptance,
            provenance=provenance(
                "akshare",
                as_of=date.today().isoformat(),
                freshness="provider_snapshot",
                status="available" if returned_periods else "unavailable",
            ),
            semantics=spec.dataset_requirements[0].semantics,
        )
        csv_content = rows_to_csv(transformed, columns) if transformed and spec.output.include_full_artifact else None
        return result, csv_content


market_data_query_service = MarketDataQueryService()
