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


class TaskContract(BaseModel):
    objective: str
    deliverables: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    requires_tools: bool = False
    resolve_representative_product: bool = False
    missing_inputs: list[str] = Field(default_factory=list)
    source_task_spec: dict[str, Any] | None = None


class CompletionResult(BaseModel):
    outcome: SupervisorOutcome
    satisfied: bool
    terminal: bool
    missing: list[str] = Field(default_factory=list)
    next_action: str = ""
    reason: str = ""
