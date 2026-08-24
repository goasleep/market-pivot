"""LangGraph workflow: multi-agent collaboration pipeline.

Flow:
  market_data → [technical, fundamentals, sentiment] (parallel)
               → debate (bull vs bear)
               → risk_manager
               → portfolio_manager (final decision)
"""

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from agents.debate_room import debate
from agents.fundamentals_analyst import analyze_stage as fund_analyze
from agents.portfolio_manager import decide as pm_decide
from agents.risk_manager import assess_stage as risk_assess
from agents.sentiment_analyst import analyze as sent_analyze
from agents.technical_analyst import analyze_stage as tech_analyze
from data.market_context import build_market_context
from models.schemas import AgentReport, AgentStageResult, AssetType, MarketContext, TradeDecision


class WorkflowState(TypedDict, total=False):
    ticker: str
    asset_type: str
    conversation_history: list[dict[str, str]]
    investor_context: dict[str, Any]
    current_price: float
    as_of_date: str | None
    market_context: MarketContext
    is_backtest: bool
    conversation_id: str
    task_id: str
    asset_data: dict[str, Any]
    strategy_name: str  # optional strategy override
    technical_report: AgentReport
    fundamentals_report: AgentReport
    sentiment_report: AgentReport
    analyst_reports: dict[str, AgentReport]
    debate_report: AgentReport
    risk_report: AgentReport
    final_decision: TradeDecision
    progress: Annotated[list[dict[str, str]], operator.add]
    visual_artifacts: Annotated[list[dict[str, Any]], operator.add]


# --- Node functions ---


async def fetch_market_data(state: WorkflowState) -> dict:
    """Node 1: Fetch market data."""
    ticker = state["ticker"]
    logger.info(f"[Graph] Fetching market data for {ticker}")

    context = state.get("market_context")
    if context is None:
        context = await build_market_context(
            ticker,
            asset_type=state.get("asset_type", AssetType.STOCK.value),
            as_of_date=state.get("as_of_date"),
            current_price=state.get("current_price"),
        )
    price = context.current_price

    return {
        "market_context": context,
        "asset_data": context.realtime,
        "current_price": price,
        "progress": [{"stage": "market_data", "message": f"Current price: {price}"}],
    }


async def run_technical(state: WorkflowState) -> dict:
    """Node 2a: Technical analysis."""
    result = await tech_analyze(
        state["ticker"],
        strategy_name=state.get("strategy_name"),
        context=state.get("market_context"),
        conversation_id=state.get("conversation_id"),
        task_id=state.get("task_id"),
    )
    stage = result if isinstance(result, AgentStageResult) else AgentStageResult(report=result)
    return {
        "technical_report": stage.report,
        "visual_artifacts": stage.artifacts,
        "progress": [{"stage": "technical", "message": stage.report.reasoning[:80]}],
    }


async def run_fundamentals(state: WorkflowState) -> dict:
    """Node 2b: Fundamentals analysis."""
    result = await fund_analyze(
        state["ticker"],
        context=state.get("market_context"),
        conversation_id=state.get("conversation_id"),
        task_id=state.get("task_id"),
    )
    stage = result if isinstance(result, AgentStageResult) else AgentStageResult(report=result)
    return {
        "fundamentals_report": stage.report,
        "visual_artifacts": stage.artifacts,
        "progress": [{"stage": "fundamentals", "message": stage.report.reasoning[:80]}],
    }


async def run_sentiment(state: WorkflowState) -> dict:
    """Node 2c: Sentiment analysis."""
    report = await sent_analyze(state["ticker"], context=state.get("market_context"))
    return {
        "sentiment_report": report,
        "progress": [{"stage": "sentiment", "message": report.reasoning[:80]}],
    }


async def merge_analysts(state: WorkflowState) -> dict:
    """Node 3: Merge analyst reports and run debate."""
    reports = {
        "technical": state["technical_report"],
        "fundamentals": state["fundamentals_report"],
        "sentiment": state["sentiment_report"],
    }

    debate_report = await debate(
        state["ticker"],
        reports,
        asset_type=state.get("asset_type", AssetType.STOCK.value),
    )

    return {
        "analyst_reports": reports,
        "debate_report": debate_report,
        "progress": [{"stage": "debate", "message": debate_report.reasoning[:80]}],
    }


async def run_risk(state: WorkflowState) -> dict:
    """Node 4: Risk assessment."""
    result = await risk_assess(
        state["ticker"],
        state.get("analyst_reports", {}),
        state.get("debate_report"),
        asset_type=state.get("asset_type", AssetType.STOCK.value),
        context=state.get("market_context"),
        conversation_id=state.get("conversation_id"),
        task_id=state.get("task_id"),
    )
    stage = result if isinstance(result, AgentStageResult) else AgentStageResult(report=result)
    return {
        "risk_report": stage.report,
        "visual_artifacts": stage.artifacts,
        "progress": [{"stage": "risk", "message": stage.report.reasoning[:80]}],
    }


async def run_portfolio_manager(state: WorkflowState) -> dict:
    """Node 5: Final decision."""
    decision = await pm_decide(
        state["ticker"],
        state.get("analyst_reports", {}),
        state.get("debate_report"),
        state.get("risk_report"),
        state.get("current_price", 0.0),
        asset_type=state.get("asset_type", AssetType.STOCK.value),
        conversation_history=state.get("conversation_history", []),
        investor_context=state.get("investor_context", {}),
        strategy_name=state.get("strategy_name"),
        market_regime=(state.get("market_context").market_regime if state.get("market_context") else None),
        market_context=state.get("market_context"),
    )
    return {
        "final_decision": decision,
        "progress": [{"stage": "portfolio", "message": decision.reasoning[:80]}],
    }


# --- Build graph ---


def build_workflow(checkpointer: Any | None = None):
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(WorkflowState)

    # Add nodes
    graph.add_node("market_data", fetch_market_data)
    graph.add_node("technical", run_technical)
    graph.add_node("fundamentals", run_fundamentals)
    graph.add_node("sentiment", run_sentiment)
    graph.add_node("merge_debate", merge_analysts)
    graph.add_node("risk", run_risk)
    graph.add_node("portfolio", run_portfolio_manager)

    # Edges
    graph.set_entry_point("market_data")

    # market_data -> parallel analysts
    graph.add_edge("market_data", "technical")
    graph.add_edge("market_data", "fundamentals")
    graph.add_edge("market_data", "sentiment")

    # All analysts -> merge_debate
    graph.add_edge("technical", "merge_debate")
    graph.add_edge("fundamentals", "merge_debate")
    graph.add_edge("sentiment", "merge_debate")

    # debate -> risk -> portfolio -> end
    graph.add_edge("merge_debate", "risk")
    graph.add_edge("risk", "portfolio")
    graph.add_edge("portfolio", END)

    return graph.compile(checkpointer=checkpointer)


# Background callers (backtests and scheduled simulation runs) deliberately use
# an uncheckpointed graph.  Their durable audit state lives in the application
# database; writing one LangGraph thread per symbol/date would create a large,
# non-resumable checkpoint leak.
workflow = build_workflow()
interactive_workflow = build_workflow()


def configure_workflow(checkpointer: Any | None) -> None:
    """Recompile only the interactive graph with the application saver."""
    global interactive_workflow
    interactive_workflow = build_workflow(checkpointer)


def get_workflow(*, checkpointed: bool = False):
    return interactive_workflow if checkpointed else workflow
