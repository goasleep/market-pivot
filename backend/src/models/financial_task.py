"""Structured contract for provider-neutral financial research tasks."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class FinancialOperation(str, Enum):
    SCREEN = "screen"
    RANK = "rank"
    AGGREGATE = "aggregate"
    TIME_SERIES = "time_series"
    COMPARE = "compare"
    ANALYZE = "analyze"
    BACKTEST = "backtest"


class ResearchAssetType(str, Enum):
    """Research scope is broader than the orderable exchange asset model."""

    STOCK = "stock"
    ETF = "etf"
    LOF = "lof"
    OPEN_FUND = "open_fund"


class DatasetRequirement(BaseModel):
    concept: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    required_fields: list[str] = Field(default_factory=list)
    semantics: dict[str, str] = Field(default_factory=dict)


class TransformInstruction(BaseModel):
    operator: Literal[
        "filter",
        "group_by",
        "require_period_coverage",
        "pivot",
        "derive",
        "sort",
        "select",
        "head",
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class AcceptanceCriterion(BaseModel):
    criterion: Literal[
        "dataset_resolved",
        "period_coverage",
        "non_empty",
        "fields_present",
        "sorted",
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class OutputContract(BaseModel):
    format: Literal["table", "series", "narrative", "artifact"] = "table"
    preview_limit: int = Field(default=30, ge=1, le=200)
    include_full_artifact: bool = True
    columns: list[str] = Field(default_factory=list)


class FinancialTaskSpec(BaseModel):
    """Machine-checkable task description produced before data access."""

    objective: str = Field(min_length=1)
    operation: FinancialOperation
    asset_type: ResearchAssetType
    dataset_requirements: list[DatasetRequirement] = Field(min_length=1)
    periods: list[int] = Field(default_factory=list)
    filters: list[dict[str, Any]] = Field(default_factory=list)
    transforms: list[TransformInstruction] = Field(default_factory=list)
    acceptance: list[AcceptanceCriterion] = Field(default_factory=list)
    output: OutputContract = Field(default_factory=OutputContract)
    assumptions: list[str] = Field(default_factory=list)
    analysis_fallback_source: str | None = Field(default=None, max_length=20_000)

    @property
    def primary_dataset_id(self) -> str | None:
        return self.dataset_requirements[0].dataset_id if self.dataset_requirements else None
