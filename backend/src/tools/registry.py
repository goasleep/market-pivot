"""Central tool registry assembled from business-oriented groups."""

from langchain_core.tools import StructuredTool

from tools import artifacts, assets, data, methodology, research, simulation
from tools.market_data import build_market_data_tools
from tools.policies import tool_policy


def build_artifact_tool(
    *,
    conversation_id: str | None = None,
    task_id: str | None = None,
) -> StructuredTool:
    """Return the scope-bound artifact saving tool."""
    return artifacts.build_artifact_tool(conversation_id=conversation_id, task_id=task_id)


def build_artifact_tools(
    *,
    conversation_id: str | None = None,
    task_id: str | None = None,
) -> list[StructuredTool]:
    """Return scope-bound artifact tools for one chat task."""
    return artifacts.build_artifact_tools(conversation_id=conversation_id, task_id=task_id)


def build_chat_tools(
    analysis_tool: StructuredTool,
    artifact_tool: StructuredTool | None = None,
    artifact_tools: list[StructuredTool] | None = None,
    *,
    allow_mutating_tools: bool = True,
    conversation_id: str | None = None,
    task_id: str | None = None,
) -> list[StructuredTool]:
    """Build the provider-agnostic tool surface from business groups."""
    tools: list[StructuredTool] = []
    seen: set[str] = set()
    for candidate in [
        *assets.TOOLS,
        *data.TOOLS,
        *build_market_data_tools(conversation_id=conversation_id, task_id=task_id),
        *methodology.TOOLS,
        *research.TOOLS,
        *simulation.TOOLS,
        analysis_tool,
    ]:
        if tool_policy(candidate.name).side_effect and not allow_mutating_tools:
            continue
        if candidate.name not in seen:
            tools.append(candidate)
            seen.add(candidate.name)
    for candidate in artifact_tools or [artifact_tool or artifacts.TOOLS[0]]:
        if candidate.name not in seen:
            tools.append(candidate)
            seen.add(candidate.name)
    return tools
