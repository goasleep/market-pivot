from pathlib import Path

import pytest

from agents.asset_requests import AssetAgentRequest, AssetIntent
from harness.bootstrap import build_default_catalog, build_default_validators, load_default_skills
from harness.compiler import harness_task_compiler
from harness.models import EvidenceRecord, SkillManifest
from harness.planner import harness_planner, skill_selector
from harness.registry import SkillRegistry
from harness.validators import covered_capabilities
from models.schemas import AssetType
from models.supervisor import ExecutionMode, TaskRoutingDecision


def _routing(*, requires_tools: bool = True) -> TaskRoutingDecision:
    return TaskRoutingDecision(
        mode=ExecutionMode.EVIDENCE_RESEARCH if requires_tools else ExecutionMode.DIRECT_RESPONSE,
        requires_tools=requires_tools,
        allow_research_plan=requires_tools,
    )


def _request(message: str, asset_type: AssetType = AssetType.ETF) -> AssetAgentRequest:
    return AssetAgentRequest(
        message=message,
        history=[],
        intent=AssetIntent.ANALYZE,
        tickers=("510300",),
        asset_type=asset_type,
        intent_confirmed=True,
    )


def test_packaged_fund_comprehensive_skill_is_tool_free_composite():
    registry = load_default_skills(catalog=build_default_catalog())
    skill = registry.get("exchange_fund.comprehensive_analysis")

    assert skill.composite is True
    assert skill.tools == ()
    assert "exchange_fund.profile" in skill.composes
    assert "risk.metrics" in skill.composes
    assert "exchange_fund.comprehensive" in skill.validators


def test_registry_rejects_composite_with_direct_tools(tmp_path: Path):
    skill_dir = tmp_path / "invalid"
    skill_dir.mkdir()
    (skill_dir / "instructions.md").write_text("invalid", encoding="utf-8")
    (skill_dir / "skill.yaml").write_text(
        """
id: invalid.composite
version: 1
title: Invalid
description: Invalid composite
capabilities: [invalid.composite]
requires: [market.quote]
composite: true
composes: [market.quote]
tools: [get_realtime_quote]
""".strip(),
        encoding="utf-8",
    )
    catalog = build_default_catalog()
    base = load_default_skills(catalog=catalog)
    manifests = [*base.list(public_only=False)]
    invalid = SkillManifest.model_validate(
        {
            "id": "invalid.composite",
            "version": "1",
            "title": "Invalid",
            "description": "Invalid composite",
            "capabilities": ["invalid.composite"],
            "requires": ["market.quote"],
            "composite": True,
            "composes": ["market.quote"],
            "tools": ["get_realtime_quote"],
        }
    )
    with pytest.raises(ValueError, match="组合 Skill.*直接工具"):
        SkillRegistry((*manifests, invalid), catalog=catalog, validators=base.validators)


@pytest.mark.parametrize("asset_type", [AssetType.ETF, AssetType.LOF])
def test_fund_analysis_routes_to_composite_and_never_stock_graph(asset_type: AssetType):
    contract = harness_task_compiler.compile(_request("综合分析 510300 的趋势、流动性和风险", asset_type), _routing())
    assert contract.required_capabilities == ("exchange_fund.comprehensive_analysis",)
    assert "stock.comprehensive_analysis" in contract.forbidden_capabilities

    registry = load_default_skills(catalog=build_default_catalog())
    skills = skill_selector.select(contract, registry)
    plan = harness_planner.deterministic_plan(contract, skills)
    assert "exchange_fund.comprehensive_analysis" in plan.selected_skills
    assert "stock.comprehensive_analysis" not in plan.selected_skills
    composite_step = next(step for step in plan.steps if step.skill_id == "exchange_fund.comprehensive_analysis")
    assert composite_step.tool_names == ()
    assert len(composite_step.depends_on) == len(registry.get("exchange_fund.comprehensive_analysis").requires)


def test_fund_comprehensive_adds_only_requested_optional_branches():
    contract = harness_task_compiler.compile(
        _request("综合分析 513100 QDII ETF 的折溢价、跟踪误差和公告风险"),
        _routing(),
    )
    assert set(contract.required_capabilities) == {
        "exchange_fund.comprehensive_analysis",
        "exchange_fund.tracking_quality",
        "exchange_fund.event_risk",
        "exchange_fund.premium_discount",
    }
    assert "exchange_fund.portfolio_fit" not in contract.required_capabilities
    assert "exchange_fund.exposure" not in contract.required_capabilities


def test_multi_branch_fund_analysis_uses_deep_budget():
    contract = harness_task_compiler.compile(
        _request("分析ETF调仓公告、跟踪误差和折溢价风险"),
        _routing(),
    )

    assert contract.budget_profile == "deep"


def test_direct_fund_explanation_does_not_select_comprehensive_skill():
    contract = harness_task_compiler.compile(_request("解释 ETF 综合分析通常看什么"), _routing(requires_tools=False))
    assert contract.required_capabilities == ()


def test_composite_coverage_requires_all_declared_child_evidence():
    skill = SkillManifest(
        id="exchange_fund.comprehensive_analysis",
        version="1",
        title="Fund",
        description="Fund composite",
        capabilities=("exchange_fund.comprehensive_analysis",),
        requires=("profile", "risk"),
        composite=True,
        composes=("exchange_fund.profile", "risk.metrics"),
    )
    evidence = (
        EvidenceRecord(
            capability_id="exchange_fund.profile",
            tool_name="profile",
            source_type="profile",
            status="available",
        ),
    )
    assert covered_capabilities((skill,), evidence) == {"exchange_fund.profile"}

    complete = (
        *evidence,
        EvidenceRecord(
            capability_id="risk.metrics",
            tool_name="risk",
            source_type="risk",
            status="available",
        ),
    )
    assert covered_capabilities((skill,), complete) == {
        "exchange_fund.profile",
        "risk.metrics",
        "exchange_fund.comprehensive_analysis",
    }


def test_fund_comprehensive_validator_requires_verified_identity_and_core_evidence():
    contract = harness_task_compiler.compile(_request("综合分析 510300"), _routing())
    registry = load_default_skills(catalog=build_default_catalog())
    composite = registry.get("exchange_fund.comprehensive_analysis")
    evidence = tuple(
        EvidenceRecord(
            capability_id=capability,
            tool_name=capability,
            source_type=capability,
            status="available",
            summary='{"verified": true}' if capability == "exchange_fund.profile" else "{}",
        )
        for capability in composite.composes
    )
    result = build_default_validators().get("exchange_fund.comprehensive")(contract, "answer", evidence)
    assert result.satisfied is True
    assert result.missing == ()

    missing_profile = tuple(record for record in evidence if record.capability_id != "exchange_fund.profile")
    result = build_default_validators().get("exchange_fund.comprehensive")(contract, "answer", missing_profile)
    assert result.satisfied is False
    assert "exchange_fund.profile" in result.missing
    assert "verified_fund_instrument" in result.missing
