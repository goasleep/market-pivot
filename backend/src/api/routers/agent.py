"""Read-only public Financial Harness capability catalog."""

from fastapi import APIRouter

from harness.runtime import get_harness_registry

router = APIRouter()


@router.get("/capabilities")
async def list_agent_capabilities():
    skills = get_harness_registry().list()
    return {
        "capabilities": [
            {
                "skill_id": skill.id,
                "title": skill.title,
                "version": skill.version,
                "domain": skill.domain,
                "asset_types": list(skill.asset_types),
                "status": "enabled" if skill.enabled else "disabled",
                "data_requirements": list(skill.evidence_types),
                "capability_ids": list(skill.capabilities),
            }
            for skill in skills
        ]
    }
