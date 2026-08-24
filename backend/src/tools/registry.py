"""Central tool registry assembled from business-oriented groups."""

from langchain_core.tools import StructuredTool

from data.market_data_catalog import market_data_catalog
from models.fund_task import FundTaskKind, FundTaskSpec
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


_FUND_TASK_TOOL_ALLOWLISTS: dict[FundTaskKind, set[str]] = {
    FundTaskKind.INSTRUMENT_RESEARCH: {
        "get_realtime_quote",
        "get_historical_prices",
        "get_fund_nav_history",
        "get_fundamentals",
        "compare_quotes",
        "compute_technical_indicators",
        "calculate_risk_metrics",
        "build_trade_plan",
        "run_backtest",
        "design_and_run_backtest",
        "compare_strategy_backtests",
        "design_and_run_sandbox_strategy",
        "search_web",
        "fetch_web_content",
        "search_methodology",
        "run_fund_or_stock_analysis",
        "save_artifacts",
        "list_artifacts",
        "read_artifact",
        "create_chart_artifact",
    },
    FundTaskKind.UNIVERSE_RESEARCH: {
        "screen_assets",
        "search_market_data_catalog",
        "query_market_data",
        "search_web",
        "fetch_web_content",
        "save_artifacts",
        "list_artifacts",
        "read_artifact",
    },
    FundTaskKind.EVENT_RESEARCH: {
        "get_fundamentals",
        "get_fund_nav_history",
        "search_web",
        "fetch_web_content",
        "save_artifacts",
        "list_artifacts",
        "read_artifact",
    },
    FundTaskKind.SIMULATION_QUERY: {
        "get_simulation_portfolio",
        "get_simulation_orders",
        "list_simulation_accounts",
        "list_strategy_deployments",
    },
    FundTaskKind.SIMULATION_MUTATION: {
        "get_simulation_portfolio",
        "get_simulation_orders",
        "list_simulation_accounts",
        "list_strategy_deployments",
        "create_simulation_account",
        "deploy_backtest_experiment",
        "set_strategy_deployment_status",
        "submit_simulation_order",
        "cancel_simulation_order",
    },
}


def build_task_tools(
    task_spec: FundTaskSpec,
    analysis_tool: StructuredTool,
    artifact_tools: list[StructuredTool] | None = None,
    *,
    allow_mutating_tools: bool = False,
    conversation_id: str | None = None,
    task_id: str | None = None,
) -> list[StructuredTool]:
    """Expose only tools allowed by the compiled fund task contract."""
    candidates = build_chat_tools(
        analysis_tool,
        artifact_tools=artifact_tools,
        allow_mutating_tools=allow_mutating_tools,
        conversation_id=conversation_id,
        task_id=task_id,
    )
    allowed = _FUND_TASK_TOOL_ALLOWLISTS.get(task_spec.task_kind, set())
    market_data_asset_type = (
        task_spec.subject.product_type if task_spec.subject.product_type in {"etf", "lof"} else "fund"
    )
    return [
        tool
        for tool in candidates
        if tool.name in allowed
        and not (
            tool.name == "query_market_data"
            and task_spec.task_kind == FundTaskKind.UNIVERSE_RESEARCH
            and not market_data_catalog.supports_asset_type(market_data_asset_type)
        )
    ]
