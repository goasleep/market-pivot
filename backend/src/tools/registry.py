"""Central tool registry assembled from business-oriented groups."""

from langchain_core.tools import StructuredTool

from tools import artifacts, assets, data, exchange_fund, methodology, open_fund, research, simulation
from tools.market_data import build_market_data_tools
from tools.policies import tool_policy


def build_artifact_tools(
    *,
    conversation_id: str | None = None,
    task_id: str | None = None,
) -> list[StructuredTool]:
    """Return scope-bound artifact tools for one chat task."""
    return artifacts.build_artifact_tools(conversation_id=conversation_id, task_id=task_id)


def build_named_tools(
    names: set[str],
    analysis_tool: StructuredTool,
    *,
    artifact_tools: list[StructuredTool] | None = None,
    allow_mutating_tools: bool = False,
    conversation_id: str | None = None,
    task_id: str | None = None,
) -> list[StructuredTool]:
    """Build only the request-scoped tools selected by the Financial Harness."""
    candidates: list[StructuredTool] = [
        *assets.TOOLS,
        *exchange_fund.TOOLS,
        *open_fund.TOOLS,
        *data.TOOLS,
        *methodology.TOOLS,
        *research.TOOLS,
        *simulation.TOOLS,
        analysis_tool,
    ]
    market_data_names = {"search_market_data_catalog", "query_market_data"}
    if names & market_data_names:
        candidates.extend(build_market_data_tools(conversation_id=conversation_id, task_id=task_id))
    if names & {"save_artifacts", "list_artifacts", "read_artifact", "create_chart_artifact"}:
        candidates.extend(
            artifact_tools or artifacts.build_artifact_tools(conversation_id=conversation_id, task_id=task_id)
        )
    selected: list[StructuredTool] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.name not in names or candidate.name in seen:
            continue
        if tool_policy(candidate.name).side_effect and not allow_mutating_tools:
            continue
        selected.append(candidate)
        seen.add(candidate.name)
    missing = names - seen
    if missing:
        raise ValueError(f"Harness 选择了无法构建的工具: {sorted(missing)}")
    return selected
