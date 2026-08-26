"""Assemble policy and selected Skill context under a deterministic budget."""

from __future__ import annotations

import json
from pathlib import Path

from harness.models import HarnessPlan, HarnessTaskContract, SkillManifest

_ROOT = Path(__file__).resolve().parents[2]
POLICY_ROOT = _ROOT / "resources" / "harness_policies"


class ContextAssembler:
    def assemble(
        self,
        contract: HarnessTaskContract,
        plan: HarnessPlan,
        skills: tuple[SkillManifest, ...],
        *,
        approval_status: str = "not_required",
        max_chars: int = 36_000,
    ) -> str:
        protected = (
            "任务合同："
            + json.dumps(contract.model_dump(mode="json"), ensure_ascii=False)
            + "\n执行计划："
            + json.dumps(plan.model_dump(mode="json"), ensure_ascii=False)
            + f"\n审批状态：{approval_status}\n"
        )
        policies = []
        for path in sorted(POLICY_ROOT.glob("*.md")):
            policies.append(path.read_text(encoding="utf-8").strip())
        instructions = [f"## Skill {skill.id}@{skill.version}\n{skill.instructions}" for skill in skills]
        optional = "\n\n".join([*policies, *instructions])
        remaining = max(0, max_chars - len(protected))
        return protected + optional[:remaining]


context_assembler = ContextAssembler()
