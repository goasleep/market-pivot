"""Safe tool catalog and declarative financial skill registry."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from langchain_core.tools import StructuredTool

from harness.models import SkillManifest, ToolDescriptor
from harness.validators import ValidatorRegistry
from tools.policies import tool_policy


class ToolCatalog:
    """Metadata is static; request-scoped callables can be rebound safely."""

    def __init__(self) -> None:
        self._descriptors: dict[str, ToolDescriptor] = {}
        self._tools: dict[str, StructuredTool] = {}

    def register_descriptor(self, descriptor: ToolDescriptor) -> None:
        existing = self._descriptors.get(descriptor.name)
        if existing is not None and existing != descriptor:
            raise ValueError(f"重复工具描述且内容冲突: {descriptor.name}")
        policy = tool_policy(descriptor.name)
        if policy.side_effect and descriptor.read_only:
            raise ValueError(f"工具 {descriptor.name} 有副作用，不能声明为 read_only")
        self._descriptors[descriptor.name] = descriptor

    def register(self, tool: StructuredTool, descriptor: ToolDescriptor) -> None:
        if tool.name != descriptor.name:
            raise ValueError(f"工具名称不一致: {tool.name} != {descriptor.name}")
        self.register_descriptor(descriptor)
        self._tools[tool.name] = tool

    def bind(self, tools: Iterable[StructuredTool]) -> "ToolCatalog":
        for tool in tools:
            if tool.name not in self._descriptors:
                raise ValueError(f"请求绑定了未登记工具: {tool.name}")
            self._tools[tool.name] = tool
        return self

    def has_descriptor(self, name: str) -> bool:
        return name in self._descriptors

    def descriptor(self, name: str) -> ToolDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as exc:
            raise ValueError(f"未知工具: {name}") from exc

    def tool(self, name: str) -> StructuredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ValueError(f"工具尚未绑定到当前请求: {name}") from exc

    def tools_for_skills(self, skills: Iterable[SkillManifest]) -> list[StructuredTool]:
        names: list[str] = []
        for skill in skills:
            for name in skill.tools:
                if name not in names:
                    names.append(name)
        return [self.tool(name) for name in names]

    def public_status(self) -> dict[str, int]:
        return {"descriptors": len(self._descriptors), "bound_tools": len(self._tools)}


class SkillRegistry:
    def __init__(
        self,
        skills: Iterable[SkillManifest],
        *,
        catalog: ToolCatalog,
        validators: ValidatorRegistry,
    ) -> None:
        self.catalog = catalog
        self.validators = validators
        self._skills: dict[str, SkillManifest] = {}
        for skill in skills:
            if skill.id in self._skills:
                raise ValueError(f"重复 Skill ID: {skill.id}")
            self._skills[skill.id] = skill
        self._validate_components()
        self._validate_dependencies()

    @classmethod
    def load(
        cls,
        roots: Iterable[Path],
        *,
        catalog: ToolCatalog,
        validators: ValidatorRegistry,
    ) -> "SkillRegistry":
        skills: list[SkillManifest] = []
        for root in roots:
            root = Path(root).expanduser().resolve()
            if not root.exists():
                continue
            for manifest_path in sorted(root.rglob("skill.yaml")):
                payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
                manifest = SkillManifest.model_validate(payload)
                instruction_path = (manifest_path.parent / manifest.instructions_file).resolve()
                try:
                    instruction_path.relative_to(manifest_path.parent.resolve())
                except ValueError as exc:
                    raise ValueError(f"Skill {manifest.id} instructions_file 越界") from exc
                if not instruction_path.is_file():
                    raise ValueError(f"Skill {manifest.id} 缺少指令文件: {instruction_path}")
                skills.append(
                    manifest.model_copy(update={"instructions": instruction_path.read_text(encoding="utf-8").strip()})
                )
        return cls(skills, catalog=catalog, validators=validators)

    def _validate_components(self) -> None:
        for skill in self._skills.values():
            if skill.id.startswith(("fund.", "etf.")):
                raise ValueError(f"Skill {skill.id} 使用了已废弃的基金领域前缀")
            if any(capability.startswith(("fund.", "etf.")) for capability in skill.capabilities):
                raise ValueError(f"Skill {skill.id} 暴露了已废弃的基金 Capability ID")
            if "fund" in skill.asset_types:
                raise ValueError(f"Skill {skill.id} 不能使用模糊 asset_type=fund")
            expected_domain = (
                "open_fund"
                if skill.id.startswith("open_fund.")
                else "exchange_fund"
                if skill.id.startswith("exchange_fund.")
                else "stock"
                if skill.id.startswith("stock.")
                else "shared"
            )
            if skill.domain != expected_domain:
                raise ValueError(f"Skill {skill.id} 的 domain 必须是 {expected_domain}")
            if skill.composite and skill.tools:
                raise ValueError(f"组合 Skill {skill.id} 不能声明直接工具")
            if skill.composite and (not skill.requires or not skill.composes):
                raise ValueError(f"组合 Skill {skill.id} 必须声明 requires 和 composes")
            if not skill.composite and skill.composes:
                raise ValueError(f"普通 Skill {skill.id} 不能声明 composes")
            for name in skill.tools:
                if not self.catalog.has_descriptor(name):
                    raise ValueError(f"Skill {skill.id} 引用了未知工具: {name}")
                descriptor = self.catalog.descriptor(name)
                if skill.domain == "open_fund" and (
                    descriptor.capability_id.startswith(("market.", "exchange_fund."))
                    or name.startswith(("get_realtime_", "get_exchange_fund_", "calculate_exchange_fund_"))
                ):
                    raise ValueError(f"Open Fund Skill {skill.id} 不能依赖场内价格、盘口或折溢价工具 {name}")
                if skill.domain == "exchange_fund" and (
                    descriptor.capability_id.startswith("open_fund.") or name.startswith("get_open_fund_")
                ):
                    raise ValueError(f"Exchange Fund Skill {skill.id} 不能依赖场外申赎或货币收益工具 {name}")
                if not descriptor.read_only and not skill.allow_side_effects:
                    raise ValueError(f"Skill {skill.id} 使用副作用工具 {name}，必须声明 allow_side_effects")
            for validator_id in skill.validators:
                if not self.validators.has(validator_id):
                    raise ValueError(f"Skill {skill.id} 引用了未知验证器: {validator_id}")

    def _validate_dependencies(self) -> None:
        for skill in self._skills.values():
            missing = set(skill.requires) - set(self._skills)
            if missing:
                raise ValueError(f"Skill {skill.id} 依赖不存在: {sorted(missing)}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(skill_id: str) -> None:
            if skill_id in visiting:
                raise ValueError("Skill Registry 存在循环依赖")
            if skill_id in visited:
                return
            visiting.add(skill_id)
            for dependency in self._skills[skill_id].requires:
                visit(dependency)
            visiting.remove(skill_id)
            visited.add(skill_id)

        for skill_id in self._skills:
            visit(skill_id)

        def dependency_closure(skill: SkillManifest) -> set[str]:
            resolved: set[str] = set()

            def include(dependency_id: str) -> None:
                if dependency_id in resolved:
                    return
                resolved.add(dependency_id)
                for nested in self._skills[dependency_id].requires:
                    include(nested)

            for dependency_id in skill.requires:
                include(dependency_id)
            return resolved

        for skill in self._skills.values():
            if not skill.composite:
                continue
            dependency_capabilities = {
                capability
                for dependency_id in dependency_closure(skill)
                for capability in self._skills[dependency_id].capabilities
            }
            missing_capabilities = set(skill.composes) - dependency_capabilities
            if missing_capabilities:
                raise ValueError(
                    f"组合 Skill {skill.id} 的 composes 不在依赖闭包内: {sorted(missing_capabilities)}"
                )

    def get(self, skill_id: str) -> SkillManifest:
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise ValueError(f"未知 Skill: {skill_id}") from exc

    def list(self, *, public_only: bool = True) -> tuple[SkillManifest, ...]:
        values = (skill for skill in self._skills.values() if skill.enabled or not public_only)
        return tuple(sorted(values, key=lambda item: item.id))

    def resolve_capabilities(
        self,
        capability_ids: Iterable[str],
        *,
        asset_type: str,
        product_category: str = "unknown",
    ) -> tuple[SkillManifest, ...]:
        requested = tuple(dict.fromkeys(capability_ids))
        selected: dict[str, SkillManifest] = {}

        def include(skill: SkillManifest) -> None:
            if skill.id in selected:
                return
            if not skill.enabled:
                raise ValueError(f"Skill 已禁用: {skill.id}")
            if "any" not in skill.asset_types and asset_type not in skill.asset_types:
                raise ValueError(f"Skill {skill.id} 不支持资产类型 {asset_type}")
            if (
                skill.product_categories
                and product_category != "unknown"
                and product_category not in skill.product_categories
            ):
                raise ValueError(f"Skill {skill.id} 不支持产品类别 {product_category}")
            for dependency_id in skill.requires:
                include(self.get(dependency_id))
            selected[skill.id] = skill

        for capability_id in requested:
            candidates = [skill for skill in self._skills.values() if capability_id in skill.capabilities]
            if not candidates:
                raise ValueError(f"没有 Skill 提供能力: {capability_id}")
            compatible = [
                skill
                for skill in candidates
                if ("any" in skill.asset_types or asset_type in skill.asset_types)
                and (
                    not skill.product_categories
                    or product_category == "unknown"
                    or product_category in skill.product_categories
                )
            ]
            if not compatible:
                raise ValueError(f"能力 {capability_id} 不支持资产类型 {asset_type}")
            include(sorted(compatible, key=lambda item: (item.cost, item.id))[0])
        return tuple(selected.values())

    def public_status(self) -> dict[str, object]:
        enabled = [skill for skill in self._skills.values() if skill.enabled]
        return {
            "healthy": True,
            "skill_count": len(enabled),
            "versions": {skill.id: skill.version for skill in enabled},
        }
