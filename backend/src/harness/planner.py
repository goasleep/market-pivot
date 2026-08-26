"""Constrained Skill selection and deterministic DAG planning."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable

from harness.models import HarnessPlan, HarnessStep, HarnessTaskContract, SkillManifest
from harness.registry import SkillRegistry
from llm.service import get_llm_service


class SkillSelector:
    def select(self, contract: HarnessTaskContract, registry: SkillRegistry) -> tuple[SkillManifest, ...]:
        forbidden = set(contract.forbidden_capabilities)
        requested = [item for item in contract.required_capabilities if item not in forbidden]
        if not requested:
            return ()
        skills = registry.resolve_capabilities(
            requested,
            asset_type=contract.asset_type,
            product_category=contract.product_category,
        )
        if any(not set(skill.capabilities).isdisjoint(forbidden) for skill in skills):
            raise ValueError("Skill 依赖闭包包含合同禁止能力")
        if any(skill.allow_side_effects for skill in skills) and not contract.allow_mutations:
            raise ValueError("只读合同不能选择副作用 Skill")
        return skills


class HarnessPlanner:
    """Generate an auditable plan from the selected dependency closure."""

    def deterministic_plan(
        self,
        contract: HarnessTaskContract,
        skills: Iterable[SkillManifest],
    ) -> HarnessPlan:
        ordered = tuple(skills)
        if len(ordered) > contract.budget.max_steps:
            raise ValueError("Skill 依赖计划超过任务步骤预算")
        selected_ids = {skill.id for skill in ordered}
        steps = tuple(
            HarnessStep(
                id=f"step-{index + 1}",
                capability_id=next(
                    (item for item in skill.capabilities if item in contract.required_capabilities),
                    skill.capabilities[0],
                ),
                skill_id=skill.id,
                title=skill.title,
                depends_on=tuple(
                    f"step-{next(i for i, candidate in enumerate(ordered, 1) if candidate.id == dependency)}"
                    for dependency in skill.requires
                    if dependency in selected_ids
                ),
                tool_names=skill.tools,
                success_criteria=skill.output_fields or skill.evidence_types,
                max_attempts=1 if skill.cost == "high" or skill.allow_side_effects else 2,
            )
            for index, skill in enumerate(ordered)
        )
        total_calls = sum(min(len(step.tool_names), step.max_attempts) for step in steps)
        if total_calls > contract.budget.max_tool_calls:
            raise ValueError("确定性计划超过工具调用预算")
        return HarnessPlan(
            plan_id=f"plan-{contract.contract_id}",
            objective=contract.objective,
            contract_id=contract.contract_id,
            budget_profile=contract.budget_profile,
            selected_skills=tuple(skill.id for skill in ordered),
            steps=steps,
        )

    def validate_candidate(
        self,
        contract: HarnessTaskContract,
        skills: tuple[SkillManifest, ...],
        payload: object,
    ) -> HarnessPlan:
        plan = HarnessPlan.model_validate(payload)
        by_id = {skill.id: skill for skill in skills}
        if set(plan.selected_skills) != set(by_id):
            raise ValueError("Planner selected_skills 必须等于已过滤的 Skill 闭包")
        if len(plan.steps) > contract.budget.max_steps:
            raise ValueError("Planner 计划超过步骤预算")
        allowed = set(contract.allowed_capabilities)
        covered: set[str] = set()
        calls = 0
        for step in plan.steps:
            skill = by_id.get(step.skill_id)
            if skill is None:
                raise ValueError(f"Planner 使用未知 Skill: {step.skill_id}")
            if allowed and step.capability_id not in allowed and step.capability_id not in skill.capabilities:
                raise ValueError(f"Planner 使用合同外能力: {step.capability_id}")
            if not set(step.tool_names) <= set(skill.tools):
                raise ValueError(f"Planner 为 {step.skill_id} 注入未声明工具")
            covered.update(skill.capabilities)
            calls += min(len(step.tool_names), step.max_attempts)
        if not set(contract.required_capabilities) <= covered:
            raise ValueError("Planner 未覆盖 required_capabilities")
        if calls > contract.budget.max_tool_calls:
            raise ValueError("Planner 计划超过工具调用预算")
        return plan

    async def constrained_plan(
        self,
        contract: HarnessTaskContract,
        skills: tuple[SkillManifest, ...],
    ) -> HarnessPlan:
        fallback = self.deterministic_plan(contract, skills)
        if len(skills) < 6:
            return fallback
        prompt = json.dumps(
            {
                "contract": contract.model_dump(mode="json"),
                "candidate_skills": [
                    {
                        "id": skill.id,
                        "version": skill.version,
                        "title": skill.title,
                        "capabilities": skill.capabilities,
                        "requires": skill.requires,
                        "tools": skill.tools,
                        "cost": skill.cost,
                    }
                    for skill in skills
                ],
                "fallback_plan": fallback.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
        repair = ""
        for _attempt in range(2):
            try:
                payload = await asyncio.wait_for(
                    get_llm_service().chat_json(
                        prompt + repair,
                        system=(
                            "你是受约束的 Financial Harness Planner。只返回 HarnessPlan JSON；"
                            "不能新增 Skill、capability 或 tool。保持依赖无环、覆盖全部 required_capabilities，"
                            "只读独立步骤可并行，写操作必须依赖所有前置读取。不要输出思维链。"
                        ),
                    ),
                    timeout=min(10, max(5, contract.budget.deadline_seconds // 20)),
                )
                return self.validate_candidate(contract, skills, payload)
            except ValueError as exc:
                repair = f"\n上次计划无效：{str(exc)[:500]}。只能修复一次；严格沿用 fallback_plan 的字段结构。"
            except Exception:
                return fallback
        return fallback


skill_selector = SkillSelector()
harness_planner = HarnessPlanner()
