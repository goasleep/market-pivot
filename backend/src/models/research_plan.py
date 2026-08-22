"""Serializable execution contracts for market Deep Research plans."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.schemas import AssetType
from models.strategy_research import TaskContract

ResearchDepth = Literal["quick", "standard", "deep"]
ResearchStepKind = Literal[
    "instrument_profile",
    "market_snapshot",
    "price_history",
    "fund_nav",
    "liquidity",
    "technical",
    "fundamentals",
    "news",
    "methodology",
    "backtest",
    "comparison",
    "comprehensive_analysis",
    "risk",
    "synthesis",
    "report",
]
ResearchStepStatus = Literal["pending", "running", "completed", "failed", "skipped"]
RecoveryClassification = Literal["transient", "correctable", "terminal", "unknown"]
RecoveryAction = Literal["retry", "adjust", "abort"]


class EvidenceRef(BaseModel):
    source: str
    source_type: Literal["market_data", "web", "methodology", "backtest", "derived", "artifact"]
    as_of: str | None = None
    retrieved_at: str
    data_status: str = "available"
    url: str | None = None
    artifact_id: str | None = None
    content_hash: str | None = None


class ResearchStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=80)
    kind: ResearchStepKind
    title: str = Field(min_length=1, max_length=160)
    depends_on: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
    max_attempts: int = Field(default=2, ge=1, le=2)


class ResearchPlan(BaseModel):
    plan_id: str
    objective: str
    asset_type: AssetType
    tickers: list[str]
    as_of_date: str
    depth: ResearchDepth
    task_contract: TaskContract | None = None
    revision: int = Field(default=1, ge=1)
    steps: tuple[ResearchStep, ...]

    @model_validator(mode="after")
    def validate_dag(self) -> "ResearchPlan":
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("研究计划包含重复 step id")
        known = set(ids)
        for step in self.steps:
            if not step.success_criteria:
                raise ValueError(f"步骤 {step.id} 必须声明成功标准")
            missing = set(step.depends_on) - known
            if missing:
                raise ValueError(f"步骤 {step.id} 依赖不存在的步骤: {sorted(missing)}")
            if step.id in step.depends_on:
                raise ValueError(f"步骤 {step.id} 不能依赖自身")

        visiting: set[str] = set()
        visited: set[str] = set()
        by_id = {step.id: step for step in self.steps}

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("研究计划依赖图存在循环")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in by_id[step_id].depends_on:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in ids:
            visit(step_id)
        return self


class StepRecovery(BaseModel):
    attempt: int = Field(ge=1)
    classification: RecoveryClassification
    action: RecoveryAction
    summary: str = Field(min_length=1, max_length=500)
    error: str = Field(default="", max_length=500)
    input_patch: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    step_id: str
    status: ResearchStepStatus
    attempt: int = 0
    summary: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    failure_context: dict[str, Any] = Field(default_factory=dict)
    recovery_history: list[StepRecovery] = Field(default_factory=list)


class ResearchBudget(BaseModel):
    max_steps: int
    max_tool_calls: int
    max_replans: int
    deadline_seconds: int
    max_parallel: int = 4
