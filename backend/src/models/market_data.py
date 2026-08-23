"""Dataset catalog and generic market-data query contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetField(BaseModel):
    name: str
    dtype: str
    description: str
    unit: str | None = None
    aliases: list[str] = Field(default_factory=list)


class DatasetDefinition(BaseModel):
    dataset_id: str
    title: str
    description: str
    asset_types: list[str]
    aliases: list[str] = Field(default_factory=list)
    fields: list[DatasetField]
    provider_ids: list[str]
    capabilities: list[str] = Field(default_factory=list)
    temporal_field: str | None = None


class MarketDataQuery(BaseModel):
    dataset_id: str
    periods: list[int] = Field(default_factory=list)
    filters: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1, le=100_000)


class DatasetCoverage(BaseModel):
    requested_periods: list[int] = Field(default_factory=list)
    returned_periods: list[int] = Field(default_factory=list)
    missing_periods: list[int] = Field(default_factory=list)
    source_rows: int = 0
    result_rows: int = 0
    status: Literal["complete", "partial", "unavailable"] = "unavailable"


class TaskAcceptanceResult(BaseModel):
    status: Literal["satisfied", "partial", "data_unavailable", "invalid_result"]
    satisfied: bool
    checks: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class MarketDataResult(BaseModel):
    data_type: str = "market_dataset"
    dataset_id: str
    available: bool
    rows: list[dict[str, Any]] = Field(default_factory=list)
    preview: list[dict[str, Any]] = Field(default_factory=list)
    schema_fields: list[str] = Field(default_factory=list)
    coverage: DatasetCoverage
    acceptance: TaskAcceptanceResult
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    semantics: dict[str, str] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
