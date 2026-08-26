"""Typed contracts shared by the Financial Harness components."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

AssetScope = Literal["stock", "etf", "lof", "open_fund", "any"]
SkillDomain = Literal["stock", "exchange_fund", "open_fund", "shared"]
BudgetProfile = Literal["quick", "standard", "deep"]
DEEP_EXECUTION_MINUTES = 30
ExecutionMode = Literal[
    "direct_response",
    "artifact_generation",
    "evidence_research",
    "backtest_execution",
    "mixed_workflow",
    "simulation_read",
    "simulation_write",
]
EvidenceStatus = Literal["available", "limited", "unavailable", "conflicting"]


class HarnessBudget(BaseModel):
    max_steps: int = Field(ge=1, le=100)
    max_tool_calls: int = Field(ge=0, le=200)
    deadline_seconds: int = Field(ge=1, le=3600)
    max_parallel: int = Field(default=4, ge=1, le=16)
    max_replans: int = Field(default=1, ge=0, le=5)


BUDGETS: dict[BudgetProfile, HarnessBudget] = {
    "quick": HarnessBudget(max_steps=4, max_tool_calls=6, deadline_seconds=60, max_parallel=2, max_replans=0),
    "standard": HarnessBudget(
        max_steps=12,
        max_tool_calls=100,
        deadline_seconds=600,
        max_parallel=4,
        max_replans=1,
    ),
    "deep": HarnessBudget(
        max_steps=20,
        max_tool_calls=48,
        deadline_seconds=DEEP_EXECUTION_MINUTES * 60,
        max_parallel=4,
        max_replans=2,
    ),
}


class ToolDescriptor(BaseModel):
    """Public metadata for one trusted Python tool."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    capability_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    asset_types: tuple[AssetScope, ...] = ("any",)
    data_types: tuple[str, ...] = ()
    read_only: bool = True
    cost: Literal["low", "medium", "high"] = "low"
    description: str = ""


class SkillManifest(BaseModel):
    """Validated declarative skill package; it never contains executable code."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    version: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1000)
    domain: SkillDomain = "shared"
    asset_types: tuple[AssetScope, ...] = ("any",)
    product_categories: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = Field(min_length=1)
    requires: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    evidence_types: tuple[str, ...] = ()
    output_fields: tuple[str, ...] = ()
    validators: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    cost: Literal["low", "medium", "high"] = "low"
    enabled: bool = True
    allow_side_effects: bool = False
    composite: bool = False
    composes: tuple[str, ...] = ()
    instructions_file: str = "instructions.md"
    instructions: str = ""


class HarnessTaskContract(BaseModel):
    """Machine-checkable definition of what one user task may and must do."""

    contract_version: str = "2.0"
    contract_id: str = Field(default_factory=lambda: f"contract-{uuid4().hex}")
    objective: str = Field(min_length=1)
    asset_type: AssetScope = "stock"
    fund_domain: Literal["exchange_fund", "open_fund"] | None = None
    product_category: str = "unknown"
    pricing_basis: Literal["market_price", "nav", "money_yield"] = "market_price"
    tickers: tuple[str, ...] = ()
    intent: str = "analyze"
    execution_mode: ExecutionMode = "direct_response"
    deliverables: tuple[str, ...] = ()
    required_outputs: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    allowed_capabilities: tuple[str, ...] = ()
    forbidden_capabilities: tuple[str, ...] = ()
    evidence_requirements: tuple[str, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    resolve_representative_product: bool = False
    allow_mutations: bool = False
    budget_profile: BudgetProfile = "standard"
    acceptance_profile: str = "default"
    source_task_spec: dict[str, Any] | None = None
    routing: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_capability_bounds(self):
        required = set(self.required_capabilities)
        allowed = set(self.allowed_capabilities)
        forbidden = set(self.forbidden_capabilities)
        if required & forbidden:
            raise ValueError("required_capabilities 与 forbidden_capabilities 冲突")
        if allowed and not required <= allowed:
            raise ValueError("required_capabilities 必须包含在 allowed_capabilities 中")
        return self

    @property
    def budget(self) -> HarnessBudget:
        return BUDGETS[self.budget_profile]


class HarnessStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    capability_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    skill_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    title: str = Field(min_length=1, max_length=160)
    depends_on: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    inputs: dict[str, Any] = Field(default_factory=dict)
    success_criteria: tuple[str, ...] = ()
    max_attempts: int = Field(default=1, ge=1, le=2)


class HarnessPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(min_length=1, max_length=120)
    objective: str = Field(min_length=1)
    contract_id: str = ""
    budget_profile: BudgetProfile = "standard"
    selected_skills: tuple[str, ...] = ()
    revision: int = Field(default=1, ge=1)
    steps: tuple[HarnessStep, ...]

    @model_validator(mode="after")
    def validate_dag(self):
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("HarnessPlan 包含重复 step id")
        known = set(ids)
        by_id = {step.id: step for step in self.steps}
        for step in self.steps:
            missing = set(step.depends_on) - known
            if missing:
                raise ValueError(f"步骤 {step.id} 依赖不存在: {sorted(missing)}")
            if step.id in step.depends_on:
                raise ValueError(f"步骤 {step.id} 不能依赖自身")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("HarnessPlan 存在循环依赖")
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


class EvidenceRecord(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"evidence-{uuid4().hex}")
    capability_id: str
    tool_name: str
    source_type: str
    status: EvidenceStatus = "available"
    as_of: str | None = None
    fetched_at: str | None = None
    freshness: str | None = None
    sources: tuple[dict[str, Any], ...] = ()
    summary: str = ""
    artifact_ids: tuple[str, ...] = ()
    content_hash: str | None = None
    raw_result: str | None = Field(default=None, exclude=True)


class ValidatorResult(BaseModel):
    validator_id: str
    satisfied: bool
    missing: tuple[str, ...] = ()
    reason: str = ""


class AcceptanceResult(BaseModel):
    outcome: Literal["satisfied", "partial", "needs_input", "data_unavailable", "failed"]
    satisfied: bool
    terminal: bool
    validator_results: tuple[ValidatorResult, ...] = ()
    evidence_coverage: dict[str, Any] = Field(default_factory=dict)
    missing: tuple[str, ...] = ()
    next_action: str = ""
    reason: str = ""
