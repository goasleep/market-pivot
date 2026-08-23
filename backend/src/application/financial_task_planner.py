"""Compile a user objective into a data-oriented financial task contract."""

from __future__ import annotations

from datetime import date

from data.market_data_catalog import market_data_catalog
from models.financial_task import (
    AcceptanceCriterion,
    DatasetRequirement,
    FinancialOperation,
    FinancialTaskSpec,
    OutputContract,
    ResearchAssetType,
    TransformInstruction,
)


def latest_complete_fiscal_years(as_of: date, count: int = 6) -> list[int]:
    latest = as_of.year - 1
    return list(range(latest - count + 1, latest + 1))


def compile_financial_task_spec(request: dict, *, as_of: date | None = None) -> FinancialTaskSpec | None:
    """Resolve task semantics from the dataset catalog.

    Returning ``None`` means the request should continue through the established
    quote/history/news/backtest research planner. It never means ``help``.
    """
    message = str(request.get("message") or "").strip()
    if not message:
        return None
    asset_type = str(request.get("asset_type") or "stock")
    matches = market_data_catalog.search(message, asset_type=asset_type, limit=1)
    if not matches:
        return None
    dataset = matches[0]
    if dataset.dataset_id != "cn_a_share_cash_dividends":
        return None
    years = latest_complete_fiscal_years(as_of or date.today(), 6)
    return FinancialTaskSpec(
        objective=message,
        operation=FinancialOperation.RANK,
        asset_type=ResearchAssetType.STOCK,
        dataset_requirements=[
            DatasetRequirement(
                concept="A股已实施现金分红",
                dataset_id=dataset.dataset_id,
                required_fields=[
                    "ticker",
                    "name",
                    "fiscal_year",
                    "cash_dividend_per_10_shares",
                    "status",
                ],
                semantics={
                    "universe": "SSE/SZSE/BSE listed A-shares returned by the provider",
                    "dividend_status": "implemented cash dividends only",
                    "cash_basis": "pre-tax cash dividend",
                    "unit": "CNY per share",
                    "multiple_plans": "sum implemented plans within the same fiscal year",
                },
            )
        ],
        periods=years,
        filters=[
            {"field": "status", "op": "implemented"},
            {"field": "cash_dividend_per_10_shares", "op": "gt", "value": 0},
        ],
        transforms=[
            TransformInstruction(
                operator="group_by",
                params={
                    "by": ["ticker", "name", "fiscal_year"],
                    "aggregations": {"cash_dividend_per_share": "sum"},
                },
            ),
            TransformInstruction(
                operator="require_period_coverage",
                params={"entity": "ticker", "period": "fiscal_year", "periods": years},
            ),
            TransformInstruction(
                operator="pivot",
                params={
                    "index": ["ticker", "name"],
                    "columns": "fiscal_year",
                    "values": "cash_dividend_per_share",
                },
            ),
            TransformInstruction(
                operator="derive",
                params={"field": "six_year_total", "sum_fields": [str(year) for year in years]},
            ),
            TransformInstruction(
                operator="sort",
                params={"by": ["six_year_total", "ticker"], "descending": [True, False]},
            ),
        ],
        acceptance=[
            AcceptanceCriterion(criterion="dataset_resolved", params={"dataset_id": dataset.dataset_id}),
            AcceptanceCriterion(criterion="period_coverage", params={"periods": years}),
            AcceptanceCriterion(criterion="non_empty"),
            AcceptanceCriterion(
                criterion="fields_present",
                params={"fields": ["ticker", "name", *[str(year) for year in years], "six_year_total"]},
            ),
            AcceptanceCriterion(
                criterion="sorted", params={"field": "six_year_total", "descending": True}
            ),
        ],
        output=OutputContract(
            format="table",
            preview_limit=30,
            include_full_artifact=True,
            columns=["ticker", "name", *[str(year) for year in years], "six_year_total"],
        ),
        assumptions=[
            f"近6个完整财年按 {years[0]}—{years[-1]} 解释",
            "仅统计已实施的税前现金分红；送股、转增及仅预案未实施方案不计入",
            "同一公司同一报告年度多次实施现金分红按每股金额求和",
        ],
    )
