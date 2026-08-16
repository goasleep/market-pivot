"""Analysis router - run multi-agent analysis for a stock."""

import json

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from application.research import research_service
from models.schemas import AssetType

router = APIRouter()


class AnalysisRequest(BaseModel):
    ticker: str = Field(..., description="Stock or exchange-traded fund code")
    asset_type: AssetType = Field(default=AssetType.STOCK)
    show_reasoning: bool = Field(default=True, description="Include agent reasoning in output")
    strategy: str | None = Field(default=None, description="Strategy name override, e.g. 'bull_trend'")
    holding_period_days: int | None = Field(default=None, ge=1, le=365)
    available_capital: float | None = Field(default=None, gt=0)
    max_loss_pct: float | None = Field(default=None, ge=0, le=1)
    current_position_pct: float | None = Field(default=None, ge=0, le=1)
    entry_price: float | None = Field(default=None, gt=0)


class AnalysisResult(BaseModel):
    ticker: str
    asset_type: AssetType = AssetType.STOCK
    decision: str  # buy / sell / hold
    confidence: float
    target_price: float | None = None
    stop_loss: float | None = None
    position_size: float | None = None
    reasoning: str
    agent_reports: dict[str, str] = Field(default_factory=dict)
    dashboard: dict | None = None  # full structured dashboard
    data_status: dict = Field(default_factory=dict)
    artifacts: list[dict] = Field(default_factory=list)


@router.post("/run")
async def run_analysis(req: AnalysisRequest):
    """Run multi-agent analysis and return result."""
    logger.info(f"Analysis request: {req.ticker}")

    investor_context = req.model_dump(
        include={
            "holding_period_days",
            "available_capital",
            "max_loss_pct",
            "current_position_pct",
            "entry_price",
        },
        exclude_none=True,
    )
    result = await research_service.run(req.ticker, req.strategy, req.asset_type, investor_context)
    decision = result.get("final_decision")

    if not decision:
        return AnalysisResult(
            ticker=req.ticker,
            decision="hold",
            confidence=0.0,
            reasoning="Analysis pipeline returned no decision.",
        )

    market_context = result.get("market_context")
    artifacts = await research_service.create_artifacts(
        decision,
        market_context,
        source="analysis-api",
    )

    return AnalysisResult(
        **research_service.decision_payload(
            decision,
            market_context,
            show_reasoning=req.show_reasoning,
            artifacts=artifacts,
        )
    )


@router.get("/stream")
async def stream_analysis(
    ticker: str,
    asset_type: AssetType = AssetType.STOCK,
    holding_period_days: int | None = None,
    available_capital: float | None = None,
    max_loss_pct: float | None = None,
    current_position_pct: float | None = None,
    entry_price: float | None = None,
):
    """Run analysis with SSE streaming progress via GET (EventSource compatible)."""
    async def event_generator():
        stage_names = {"merge_debate": "debate"}
        final_state: dict = {}
        try:
            investor_context = {
                key: value
                for key, value in {
                    "holding_period_days": holding_period_days,
                    "available_capital": available_capital,
                    "max_loss_pct": max_loss_pct,
                    "current_position_pct": current_position_pct,
                    "entry_price": entry_price,
                }.items()
                if value is not None
            }
            async for update in research_service.stream(
                ticker,
                asset_type=asset_type,
                investor_context=investor_context,
            ):
                node = update["node"]
                node_update = update["update"]
                final_state = update["state"]
                progress_items = node_update.get("progress", []) if isinstance(node_update, dict) else []
                if progress_items:
                    progress = progress_items[-1]
                    yield {
                        "event": "progress",
                        "data": json.dumps(
                            {
                                "stage": stage_names.get(node, node),
                                "message": progress.get("message", ""),
                            },
                            ensure_ascii=False,
                        ),
                    }

            decision = final_state.get("final_decision")

            if decision:
                market_context = final_state.get("market_context")
                artifacts = await research_service.create_artifacts(
                    decision,
                    market_context,
                    source="analysis-stream",
                )
                payload = research_service.decision_payload(
                    decision,
                    market_context,
                    artifacts=artifacts,
                )
                yield {
                    "event": "complete",
                    "data": json.dumps(payload, ensure_ascii=False),
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
