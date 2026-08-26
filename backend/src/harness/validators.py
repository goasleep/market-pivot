"""Trusted deterministic validator registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from harness.models import EvidenceRecord, SkillManifest

Validator = Callable[..., Any]


def covered_capabilities(
    skills: Iterable[SkillManifest],
    evidence: Iterable[EvidenceRecord],
) -> set[str]:
    """Resolve atomic evidence into completed declarative composite capabilities."""
    covered = {record.capability_id for record in evidence if record.status == "available"}
    composites = [skill for skill in skills if skill.composite]
    changed = True
    while changed:
        changed = False
        for skill in composites:
            if set(skill.composes) <= covered and not set(skill.capabilities) <= covered:
                covered.update(skill.capabilities)
                changed = True
    return covered


class ValidatorRegistry:
    def __init__(self) -> None:
        self._validators: dict[str, Validator] = {}

    def register(self, validator_id: str, validator: Validator) -> None:
        if not validator_id:
            raise ValueError("validator_id 不能为空")
        if validator_id in self._validators:
            raise ValueError(f"重复验证器: {validator_id}")
        self._validators[validator_id] = validator

    def has(self, validator_id: str) -> bool:
        return validator_id in self._validators

    def get(self, validator_id: str) -> Validator:
        try:
            return self._validators[validator_id]
        except KeyError as exc:
            raise ValueError(f"未知验证器: {validator_id}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._validators))


validator_registry = ValidatorRegistry()
