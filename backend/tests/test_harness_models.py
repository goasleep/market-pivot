from __future__ import annotations

import pytest
from pydantic import ValidationError

from harness.models import BUDGETS, HarnessPlan, HarnessStep, SkillManifest


def test_standard_harness_budget_allows_one_hundred_tool_calls():
    assert BUDGETS["standard"].max_tool_calls == 100


def test_harness_plan_rejects_cycles_and_unknown_dependencies():
    with pytest.raises(ValidationError, match="依赖不存在"):
        HarnessPlan(
            plan_id="plan-missing",
            objective="分析ETF",
            steps=(
                HarnessStep(
                    id="profile",
                    capability_id="exchange_fund.profile",
                    skill_id="exchange_fund.profile",
                    title="核对产品资料",
                    depends_on=("missing",),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="存在循环"):
        HarnessPlan(
            plan_id="plan-cycle",
            objective="分析ETF",
            steps=(
                HarnessStep(
                    id="a",
                    capability_id="exchange_fund.profile",
                    skill_id="exchange_fund.profile",
                    title="A",
                    depends_on=("b",),
                ),
                HarnessStep(
                    id="b",
                    capability_id="exchange_fund.liquidity_cost",
                    skill_id="exchange_fund.liquidity_cost",
                    title="B",
                    depends_on=("a",),
                ),
            ),
        )


def test_skill_manifest_has_safe_defaults():
    manifest = SkillManifest(
        id="market.quote",
        version="1.0.0",
        title="行情快照",
        description="获取可验证行情",
        capabilities=("market.quote",),
        tools=("get_realtime_quote",),
    )

    assert manifest.enabled is True
    assert manifest.allow_side_effects is False
    assert manifest.instructions_file == "instructions.md"
