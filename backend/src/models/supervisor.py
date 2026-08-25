"""Contracts shared by the single Supervisor Agent and its completion judge."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SupervisorOutcome(str, Enum):
    SATISFIED = "satisfied"
    PARTIAL = "partial"
    NEEDS_INPUT = "needs_input"
    DATA_UNAVAILABLE = "data_unavailable"
    FAILED = "failed"


class ExecutionMode(str, Enum):
    """Model-selected execution shape for one user request."""

    DIRECT_RESPONSE = "direct_response"
    ARTIFACT_GENERATION = "artifact_generation"
    EVIDENCE_RESEARCH = "evidence_research"
    BACKTEST_EXECUTION = "backtest_execution"
    MIXED_WORKFLOW = "mixed_workflow"
    SUPERVISOR_DECIDES = "supervisor_decides"


class TaskRoutingDecision(BaseModel):
    """Structured routing decision produced by the configured model."""

    mode: ExecutionMode
    requires_tools: bool = False
    allow_research_plan: bool = False
    deliverables: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @classmethod
    def supervisor_fallback(cls) -> "TaskRoutingDecision":
        return cls(
            mode=ExecutionMode.SUPERVISOR_DECIDES,
            requires_tools=False,
            allow_research_plan=True,
            deliverables=[],
            reason="任务分类暂不可用，由Supervisor根据原始请求决定执行方式",
            confidence=0.0,
        )


class TaskContract(BaseModel):
    objective: str
    deliverables: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    requires_tools: bool = False
    resolve_representative_product: bool = False
    missing_inputs: list[str] = Field(default_factory=list)
    source_task_spec: dict[str, Any] | None = None
    routing: TaskRoutingDecision | None = None


class CompletionResult(BaseModel):
    outcome: SupervisorOutcome
    satisfied: bool
    terminal: bool
    missing: list[str] = Field(default_factory=list)
    next_action: str = ""
    reason: str = ""
