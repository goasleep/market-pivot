from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.tools import tool

from harness.bootstrap import build_default_catalog, load_default_skills
from harness.models import SkillManifest, ToolDescriptor
from harness.registry import SkillRegistry, ToolCatalog
from harness.validators import ValidatorRegistry


@tool
async def read_value(query: str) -> str:
    """Read a deterministic value."""
    return query


@tool
async def submit_simulation_order(ticker: str) -> str:
    """Submit a paper-trading order."""
    return ticker


def _write_skill(root: Path, body: str, instructions: str = "使用结构化数据。") -> None:
    skill = root / "example"
    skill.mkdir(parents=True)
    (skill / "skill.yaml").write_text(body, encoding="utf-8")
    (skill / "instructions.md").write_text(instructions, encoding="utf-8")


def test_skill_registry_loads_instructions_and_dependency_closure(tmp_path):
    catalog = ToolCatalog()
    catalog.register(
        read_value,
        ToolDescriptor(
            name="read_value",
            capability_id="data.read",
            asset_types=("etf",),
            data_types=("market_data",),
        ),
    )
    _write_skill(
        tmp_path,
        """
id: data.read
version: 1.0.0
title: 数据读取
description: 读取结构化数据
asset_types: [etf]
capabilities: [data.read]
tools: [read_value]
validators: [evidence.required]
""".strip(),
    )
    validators = ValidatorRegistry()
    validators.register("evidence.required", lambda *_args, **_kwargs: None)

    registry = SkillRegistry.load([tmp_path], catalog=catalog, validators=validators)

    skill = registry.get("data.read")
    assert skill.instructions == "使用结构化数据。"
    assert registry.resolve_capabilities(("data.read",), asset_type="etf") == (skill,)
    assert catalog.tools_for_skills((skill,)) == [read_value]


def test_skill_registry_rejects_unknown_tool_and_validator(tmp_path):
    _write_skill(
        tmp_path,
        """
id: broken.skill
version: 1.0.0
title: 错误能力
description: 引用不存在的运行组件
capabilities: [broken.skill]
tools: [missing_tool]
validators: [missing.validator]
""".strip(),
    )

    with pytest.raises(ValueError, match="未知工具"):
        SkillRegistry.load([tmp_path], catalog=ToolCatalog(), validators=ValidatorRegistry())


def test_skill_registry_rejects_unacknowledged_side_effect_tool(tmp_path):
    catalog = ToolCatalog()
    catalog.register(
        submit_simulation_order,
        ToolDescriptor(
            name="submit_simulation_order",
            capability_id="simulation.write",
            asset_types=("stock", "etf", "lof"),
            data_types=("simulation",),
            read_only=False,
        ),
    )
    _write_skill(
        tmp_path,
        """
id: simulation.write
version: 1.0.0
title: 模拟盘写操作
description: 提交纸面交易订单
capabilities: [simulation.write]
tools: [submit_simulation_order]
""".strip(),
    )

    with pytest.raises(ValueError, match="allow_side_effects"):
        SkillRegistry.load([tmp_path], catalog=catalog, validators=ValidatorRegistry())


def test_registry_rejects_dependency_cycles():
    left = SkillManifest(
        id="left",
        version="1.0.0",
        title="Left",
        description="Left",
        capabilities=("left",),
        requires=("right",),
    )
    right = SkillManifest(
        id="right",
        version="1.0.0",
        title="Right",
        description="Right",
        capabilities=("right",),
        requires=("left",),
    )

    with pytest.raises(ValueError, match="循环依赖"):
        SkillRegistry((left, right), catalog=ToolCatalog(), validators=ValidatorRegistry())


def test_packaged_skills_resolve_to_a_minimal_bound_tool_surface():
    catalog = build_default_catalog()
    registry = load_default_skills(catalog=catalog)

    quote = registry.resolve_capabilities(("market.quote",), asset_type="etf")

    assert [skill.id for skill in quote] == ["market.quote"]
    assert quote[0].tools == ("get_realtime_quote",)
