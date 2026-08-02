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
from agents.fundamentals_analyst import analyze as fund_analyze
from agents.portfolio_manager import decide as pm_decide
from agents.risk_manager import assess as risk_assess
from agents.sentiment_analyst import analyze as sent_analyze
from agents.technical_analyst import analyze as tech_analyze
from data.market_context import build_market_context
from models.schemas import AgentReport, MarketContext, TradeDecision


class WorkflowState(TypedDict, total=False):
    ticker: str
    current_price: float
    as_of_date: str | None
    market_context: MarketContext
    is_backtest: bool
    stock_data: dict[str, Any]
    strategy_name: str  # optional strategy override
    technical_report: AgentReport
    fundamentals_report: AgentReport
    sentiment_report: AgentReport
    analyst_reports: dict[str, AgentReport]
    debate_report: AgentReport
    risk_report: AgentReport
    final_decision: TradeDecision
    progress: Annotated[list[dict[str, str]], operator.add]


# --- Node functions ---


async def fetch_market_data(state: WorkflowState) -> dict:
    """Node 1: Fetch market data."""
    ticker = state["ticker"]
    logger.info(f"[Graph] Fetching market data for {ticker}")

    context = state.get("market_context")
    if context is None:
        context = await build_market_context(
            ticker,
            as_of_date=state.get("as_of_date"),
            current_price=state.get("current_price"),
        )
    price = context.current_price

    return {
        "market_context": context,
        "stock_data": context.realtime,
        "current_price": price,
        "progress": [{"stage": "market_data", "message": f"Current price: {price}"}],
    }


async def run_technical(state: WorkflowState) -> dict:
    """Node 2a: Technical analysis."""
    report = await tech_analyze(
        state["ticker"],
        strategy_name=state.get("strategy_name"),
        context=state.get("market_context"),
    )
    return {
        "technical_report": report,
        "progress": [{"stage": "technical", "message": report.reasoning[:80]}],
    }


async def run_fundamentals(state: WorkflowState) -> dict:
    """Node 2b: Fundamentals analysis."""
    report = await fund_analyze(state["ticker"], context=state.get("market_context"))
    return {
        "fundamentals_report": report,
        "progress": [{"stage": "fundamentals", "message": report.reasoning[:80]}],
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

    # Run debate
    debate_report = await debate(state["ticker"], reports)

    return {
        "analyst_reports": reports,
        "debate_report": debate_report,
        "progress": [{"stage": "debate", "message": debate_report.reasoning[:80]}],
    }


async def run_risk(state: WorkflowState) -> dict:
    """Node 4: Risk assessment."""
    report = await risk_assess(
        state["ticker"],
        state.get("analyst_reports", {}),
        state.get("debate_report"),
    )
    return {
        "risk_report": report,
        "progress": [{"stage": "risk", "message": report.reasoning[:80]}],
    }


async def run_portfolio_manager(state: WorkflowState) -> dict:
    """Node 5: Final decision."""
    decision = await pm_decide(
        state["ticker"],
        state.get("analyst_reports", {}),
        state.get("debate_report"),
        state.get("risk_report"),
        state.get("current_price", 0.0),
        strategy_name=state.get("strategy_name"),
        market_regime=(state.get("market_context").market_regime if state.get("market_context") else None),
    )
    return {
        "final_decision": decision,
        "progress": [{"stage": "portfolio", "message": decision.reasoning[:80]}],
    }


# --- Build graph ---


def build_workflow():
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

    return graph.compile()


# Compiled workflow singleton
workflow = build_workflow()
