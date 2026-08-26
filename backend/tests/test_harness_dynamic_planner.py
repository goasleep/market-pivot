import pytest

from harness import planner as planner_module
from harness.models import HarnessStep, HarnessTaskContract, SkillManifest
from harness.planner import HarnessPlanner


def _skills():
    return tuple(
        SkillManifest(
            id=f"skill.{index}",
            version="1",
            title=f"Skill {index}",
            description="fixture",
            capabilities=(f"cap.{index}",),
            tools=(f"tool_{index}",),
        )
        for index in range(2)
    )


def test_planner_rejects_unknown_tool_injection():
    contract = HarnessTaskContract(
        objective="fixture",
        required_capabilities=("cap.0", "cap.1"),
        allowed_capabilities=("cap.0", "cap.1"),
    )
    skills = _skills()
    fallback = HarnessPlanner().deterministic_plan(contract, skills)
    bad = fallback.model_copy(
        update={
            "steps": (
                HarnessStep(
                    id="bad",
                    capability_id="cap.0",
                    skill_id="skill.0",
                    title="bad",
                    tool_names=("unknown_tool",),
                ),
                fallback.steps[1],
            )
        }
    )
    with pytest.raises(ValueError, match="未声明工具"):
        HarnessPlanner().validate_candidate(contract, skills, bad.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_planner_unavailability_falls_back_without_repair_retry(monkeypatch):
    calls = 0

    class BrokenPlannerService:
        async def chat_json(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise TimeoutError("planner timeout")

    skills = tuple(
        SkillManifest(
            id=f"skill.{index}",
            version="1",
            title=f"Skill {index}",
            description="fixture",
            capabilities=(f"cap.{index}",),
        )
        for index in range(6)
    )
    contract = HarnessTaskContract(
        objective="fixture",
        required_capabilities=tuple(f"cap.{index}" for index in range(6)),
        allowed_capabilities=tuple(f"cap.{index}" for index in range(6)),
    )
    monkeypatch.setattr(planner_module, "get_llm_service", lambda: BrokenPlannerService())
    plan = await HarnessPlanner().constrained_plan(contract, skills)
    assert calls == 1
    assert plan.selected_skills == tuple(skill.id for skill in skills)
