"""Execution policy metadata for tools that can change application state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ConfirmationMode = Literal["never", "always"]


@dataclass(frozen=True)
class ToolPolicy:
    """Declare authorization and confirmation requirements for one tool."""

    requires_explicit_request: bool = False
    confirmation: ConfirmationMode = "never"
    side_effect: bool = False


TOOL_POLICIES: dict[str, ToolPolicy] = {
    "create_simulation_account": ToolPolicy(
        requires_explicit_request=True,
        confirmation="always",
        side_effect=True,
    ),
    "deploy_backtest_experiment": ToolPolicy(
        requires_explicit_request=True,
        confirmation="always",
        side_effect=True,
    ),
    "set_strategy_deployment_status": ToolPolicy(
        requires_explicit_request=True,
        confirmation="always",
        side_effect=True,
    ),
    "submit_simulation_order": ToolPolicy(
        requires_explicit_request=True,
        confirmation="always",
        side_effect=True,
    ),
    "cancel_simulation_order": ToolPolicy(
        requires_explicit_request=True,
        confirmation="always",
        side_effect=True,
    ),
}


def tool_policy(name: str) -> ToolPolicy:
    return TOOL_POLICIES.get(name, ToolPolicy())


def tool_requires_confirmation(name: str) -> bool:
    return tool_policy(name).confirmation == "always"
