"""Strategy / Skill manager.

Loads YAML strategy files from strategies/ directory, supports:
- Active strategy selection (by name, alias, or market regime)
- Dynamic system prompt injection
- Strategy listing for API
"""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from models.schemas import StrategySpec
from strategies.compiler import strategy_from_mapping
from strategies.registry import StrategyRegistry

# ---------------------------------------------------------------------------
# Data models (plain dataclasses, no Pydantic needed here)
# ---------------------------------------------------------------------------

_STRATEGIES_DIR = Path(__file__).parent

# Cache: name -> dict (raw YAML)
_loaded: dict[str, dict[str, Any]] = {}
_loaded_once = False
_runtime_registry = StrategyRegistry(Path(__file__).resolve().parents[2] / "data" / "strategies.json")


def _load_all() -> None:
    """Load all .yaml strategy files from the strategies/ directory."""
    global _loaded_once
    if _loaded_once:
        return
    for yml_file in sorted(_STRATEGIES_DIR.glob("*.yaml")):
        try:
            with open(yml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            name = data.get("name", yml_file.stem)
            _loaded[name] = data
            logger.debug(f"Loaded strategy: {name} from {yml_file.name}")
        except Exception as e:
            logger.error(f"Failed to load strategy {yml_file}: {e}")
    _loaded_once = True


def list_strategies(active_only: bool = False) -> list[dict[str, Any]]:
    """Return list of strategy metadata."""
    _load_all()
    result = []
    for name, data in _loaded.items():
        if active_only and not data.get("default_active", False):
            continue
        result.append(
            {
                "name": name,
                "display_name": data.get("display_name", name),
                "description": data.get("description", ""),
                "category": data.get("category", ""),
                "default_active": data.get("default_active", False),
                "default_router": data.get("default_router", False),
                "priority": data.get("default_priority", 99),
                "market_regimes": data.get("market_regimes", []),
                "aliases": data.get("aliases", []),
            }
        )
    return result


def get_strategy(name: str) -> dict[str, Any] | None:
    """Get a single strategy by name or alias."""
    _load_all()
    if name in _loaded:
        return _loaded[name]
    # Search aliases
    for data in _loaded.values():
        if name in data.get("aliases", []):
            return data
    return None


def get_strategy_spec(name: str, *, source: str | None = None) -> StrategySpec | None:
    """Return the validated executable definition for a named strategy."""
    runtime = _runtime_registry.get(name)
    if runtime is not None:
        return runtime
    data = get_strategy(name)
    if not data:
        return None
    try:
        return strategy_from_mapping(data, source=source or "yaml")
    except Exception as exc:
        logger.warning("Strategy {} has no executable DSL: {}", name, exc)
        return None


def register_strategy_spec(spec: StrategySpec) -> StrategySpec:
    """Persist an LLM/user-generated strategy for later backtests."""
    return _runtime_registry.register(spec)


def get_active_strategies(market_regime: str | None = None) -> list[dict[str, Any]]:
    """Get strategies that should be active.

    If market_regime is given, also filter by market_regimes field.
    Falls back to default_active strategies if no regime match.
    """
    _load_all()
    # First try regime-matched strategies
    if market_regime:
        matched = [
            s
            for s in _loaded.values()
            if market_regime in s.get("market_regimes", []) and s.get("default_router", False)
        ]
        if matched:
            return sorted(matched, key=lambda s: s.get("default_priority", 99))

    # Fallback: default_active strategies
    active = [s for s in _loaded.values() if s.get("default_active", False)]
    return sorted(active, key=lambda s: s.get("default_priority", 99))


def build_strategy_prompt(strategies: list[dict[str, Any]]) -> str:
    """Build the strategy injection text for system prompt.

    Args:
        strategies: List of strategy dicts (from get_active_strategies or manual selection)

    Returns:
        Formatted text to append to agent system prompts
    """
    if not strategies:
        return ""

    sections = ["\n\n## 激活的交易策略\n"]
    for s in strategies:
        display = s.get("display_name", s.get("name", ""))
        instructions = s.get("instructions", "")
        sections.append(f"### {display}\n{instructions}\n")

    return "\n".join(sections)


def get_strategy_instructions(
    strategy_name: str | None = None,
    market_regime: str | None = None,
) -> str:
    """Convenience: get formatted strategy instructions for prompt injection.

    If strategy_name is given, use that specific strategy.
    Otherwise auto-select based on market_regime or defaults.
    """
    _load_all()

    if strategy_name:
        s = get_strategy(strategy_name)
        if s:
            return build_strategy_prompt([s])
        logger.warning(f"Strategy not found: {strategy_name}")
        return ""

    strategies = get_active_strategies(market_regime)
    return build_strategy_prompt(strategies)
