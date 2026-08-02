"""Analysis router - run multi-agent analysis for a stock."""

import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from loguru import logger

from graph.workflow import workflow

router = APIRouter()


class AnalysisRequest(BaseModel):
    ticker: str = Field(..., description="A-share stock code, e.g. 000737")
    show_reasoning: bool = Field(default=True, description="Include agent reasoning in output")
    strategy: str | None = Field(default=None, description="Strategy name override, e.g. 'bull_trend'")


class AnalysisResult(BaseModel):
    ticker: str
    decision: str  # buy / sell / hold
    confidence: float
    target_price: float | None = None
    stop_loss: float | None = None
    position_size: float | None = None
    reasoning: str
    agent_reports: dict[str, str] = {}
    dashboard: dict | None = None  # full structured dashboard


@router.post("/run")
async def run_analysis(req: AnalysisRequest):
    """Run multi-agent analysis and return result."""
    logger.info(f"Analysis request: {req.ticker}")

    state = {
        "ticker": req.ticker,
        "current_price": 0.0,
        "progress": [],
    }
    if req.strategy:
        state["strategy_name"] = req.strategy

    result = await workflow.ainvoke(state)
    decision = result.get("final_decision")

    if not decision:
        return AnalysisResult(
            ticker=req.ticker,
            decision="hold",
            confidence=0.0,
            reasoning="Analysis pipeline returned no decision.",
        )

    return AnalysisResult(
        ticker=decision.ticker,
        decision=decision.decision.value,
        confidence=decision.confidence,
        target_price=decision.target_price,
        stop_loss=decision.stop_loss,
        position_size=decision.position_size,
        reasoning=decision.reasoning,
        agent_reports=decision.agent_reports if req.show_reasoning else {},
        dashboard=decision.dashboard.model_dump() if decision.dashboard else None,
    )


@router.get("/stream")
async def stream_analysis(ticker: str):
    """Run analysis with SSE streaming progress via GET (EventSource compatible)."""
    import asyncio

    async def event_generator():
        stages = [
            ("market_data", "Fetching market data..."),
            ("technical", "Technical analyst working..."),
            ("fundamentals", "Fundamentals analyst working..."),
            ("sentiment", "Sentiment analyst working..."),
            ("debate", "Bull vs Bear debate..."),
            ("risk", "Risk manager assessing..."),
            ("portfolio", "Portfolio manager deciding..."),
        ]

        # Send staged progress events
        for stage, msg in stages:
            yield {
                "event": "progress",
                "data": json.dumps({"stage": stage, "message": msg}, ensure_ascii=False),
            }
            await asyncio.sleep(0.3)

        # Run actual workflow
        try:
            state = {"ticker": ticker, "current_price": 0.0, "progress": []}
            result = await workflow.ainvoke(state)
            decision = result.get("final_decision")

            if decision:
                yield {
                    "event": "complete",
                    "data": json.dumps(
                        {
                            "ticker": decision.ticker,
                            "decision": decision.decision.value,
                            "confidence": decision.confidence,
                            "target_price": decision.target_price,
                            "stop_loss": decision.stop_loss,
                            "position_size": decision.position_size,
                            "reasoning": decision.reasoning,
                            "agent_reports": decision.agent_reports,
                            "dashboard": decision.dashboard.model_dump() if decision.dashboard else None,
                        },
                        ensure_ascii=False,
                    ),
                }
            else:
                yield {
                    "event": "complete",
                    "data": json.dumps(
                        {
                            "ticker": ticker,
                            "decision": "hold",
                            "confidence": 0.0,
                            "reasoning": "No decision returned.",
                        },
                        ensure_ascii=False,
                    ),
                }
        except Exception as e:
            logger.error(f"Stream analysis error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
