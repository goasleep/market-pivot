"""Small persistent registry for executable strategy definitions."""

from __future__ import annotations

import json
from pathlib import Path

from models.schemas import StrategySpec


class StrategyRegistry:
    """Store versioned YAML/user/LLM strategy specs outside the Agent prompt."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._items: dict[str, StrategySpec] = {}
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._items = {name: StrategySpec.model_validate(spec) for name, spec in payload.items()}
        except (OSError, ValueError, TypeError):
            self._items = {}

    def register(self, spec: StrategySpec) -> StrategySpec:
        self._items[spec.name] = spec
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {name: item.model_dump(mode="json") for name, item in self._items.items()},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return spec

    def get(self, name: str) -> StrategySpec | None:
        return self._items.get(name)

    def list(self) -> list[StrategySpec]:
        return list(self._items.values())
