"""Risk Manager Agent.

Assesses overall risk and suggests position sizing and stop-loss.
"""

import json

from loguru import logger

from agents.prompt_context import INVESTOR_CONTEXT
from application.visual_evidence import visual_evidence_service
from charts.financial import ChartDataUnavailableError, calculate_market_risk_metrics
from llm import LLMService, get_llm_service
from models.schemas import AgentReport, AgentStageResult, AssetType, Decision, MarketContext

SYSTEM_PROMPT = """You are a professional A-share risk manager.
You assess risk levels and recommend position sizing, stop-loss, and risk controls.
You must respond in Chinese.
You must respond with valid JSON in this format:
{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.0-1.0,
  "reasoning": "risk assessment in Chinese",
  "risk_level": "low" | "medium" | "high",
  "max_position_pct": 0.0-1.0,
  "stop_loss_pct": 0.0-1.0,
  "risk_factors": ["factor1", "factor2"]
}""" + INVESTOR_CONTEXT


async def assess(
    ticker: str,
    agent_reports: dict[str, AgentReport],
    debate_report: AgentReport | None = None,
    llm: LLMService | None = None,
    asset_type: AssetType | str = AssetType.STOCK,
    context: MarketContext | None = None,
) -> AgentReport:
    """Run risk assessment and return only the report for legacy callers."""
    stage = await assess_stage(
        ticker,
        agent_reports,
        debate_report,
        llm=llm,
        asset_type=asset_type,
        context=context,
    )
    return stage.report


async def assess_stage(
    ticker: str,
    agent_reports: dict[str, AgentReport],
    debate_report: AgentReport | None = None,
    llm: LLMService | None = None,
    asset_type: AssetType | str = AssetType.STOCK,
    context: MarketContext | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
) -> AgentStageResult:
    """Run risk assessment based on all agent reports."""
    logger.info(f"[RiskManager] Assessing {ticker}")

    # Compile all reports
    all_reports = []
    for name, r in agent_reports.items():
        all_reports.append(f"## {name}\nSignal: {r.signal.value}\nConfidence: {r.confidence:.2f}\n{r.reasoning}")
    if debate_report:
        all_reports.append(f"## Debate Outcome\nSignal: {debate_report.signal.value}\n{debate_report.reasoning}")

    analysis = "\n\n".join(all_reports)
    asset_label = (
        "股票"
        if AssetType(asset_type) == AssetType.STOCK
        else f"{AssetType(asset_type).value.upper()} 场内基金"
    )
    risk_metrics: dict = {}
    if context and context.history:
        try:
            risk_metrics = calculate_market_risk_metrics(context.history)
        except ChartDataUnavailableError:
            risk_metrics = {}

    prompt = f"""Assess the risk for {asset_label} {ticker} based on all analyst reports:

{analysis}

Deterministic market risk metrics:
{json.dumps(risk_metrics, ensure_ascii=False)}

Provide risk assessment as JSON. Focus on:
1. Overall risk level (low/medium/high)
2. Maximum recommended position (% of total portfolio)
3. Stop-loss percentage suggestion
4. Key risk factors to monitor
5. Whether to proceed with the trade

Conservative when signals conflict, more confident when signals align.
"""

    artifacts: list[dict] = []
    try:
        llm_service = llm or get_llm_service()
        supports_images = bool(getattr(llm_service, "supports_vision", lambda: False)())
        evidence = None
        if context and context.history:
            evidence = await visual_evidence_service.prepare_risk(
                ticker=ticker,
                asset_type=AssetType(asset_type).value,
                history=context.history,
                conversation_id=conversation_id,
                task_id=task_id,
            )
            if evidence:
                artifacts.append(evidence.artifact)
        if supports_images and evidence and evidence.model_url:
            result = await llm_service.chat_json_with_images(
                prompt + "\nThe attached chart visualizes normalized price, drawdown and rolling volatility.",
                [evidence.model_url],
                system=SYSTEM_PROMPT,
            )
        else:
            result = await llm_service.chat_json(prompt, system=SYSTEM_PROMPT)
        report = AgentReport(
            agent_name="risk_manager",
            signal=Decision(result.get("signal", "hold")),
            confidence=float(result.get("confidence", 0.5)),
            reasoning=result.get("reasoning", ""),
            key_data={
                "risk_level": result.get("risk_level", "medium"),
                "max_position_pct": float(result.get("max_position_pct", 0.3)),
                "stop_loss_pct": float(result.get("stop_loss_pct", 0.08)),
                "risk_factors": result.get("risk_factors", []),
                "market_risk_metrics": risk_metrics,
            },
        )
        return AgentStageResult(report=report, artifacts=artifacts)
    except Exception as e:
        logger.error(f"[RiskManager] LLM error: {e}")
        return AgentStageResult(
            report=AgentReport(
                agent_name="risk_manager",
                reasoning="风险评估模型暂时不可用，已采用保守风险上限。",
                key_data={
                    "degraded": True,
                    "reason": "llm_unavailable",
                    "risk_level": "high",
                    "max_position_pct": 0.2,
                    "stop_loss_pct": 0.05,
                    "risk_factors": ["风险评估模型不可用"],
                    "market_risk_metrics": risk_metrics,
                },
            ),
            artifacts=artifacts,
        )
