"""Application-level Harness registry lifecycle and public health metadata."""

from __future__ import annotations

from harness.bootstrap import build_default_catalog, load_default_skills
from harness.registry import SkillRegistry

HARNESS_VERSION = "2.0.0"
_registry: SkillRegistry | None = None


def initialize_harness_registry() -> SkillRegistry:
    global _registry
    _registry = load_default_skills(catalog=build_default_catalog())
    return _registry


def get_harness_registry() -> SkillRegistry:
    return _registry or initialize_harness_registry()


def harness_status() -> dict[str, object]:
    registry = get_harness_registry()
    return {"version": HARNESS_VERSION, "registry": registry.public_status()}
