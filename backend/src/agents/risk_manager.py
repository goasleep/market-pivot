"""Risk Manager Agent.

Assesses overall risk and suggests position sizing and stop-loss.
"""

from loguru import logger

from llm.deepseek import chat_json
from models.schemas import AgentReport, Decision

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
}"""


async def assess(
    ticker: str,
    agent_reports: dict[str, AgentReport],
    debate_report: AgentReport | None = None,
) -> AgentReport:
    """Run risk assessment based on all agent reports."""
    logger.info(f"[RiskManager] Assessing {ticker}")

    # Compile all reports
    all_reports = []
    for name, r in agent_reports.items():
        all_reports.append(f"## {name}\nSignal: {r.signal.value}\nConfidence: {r.confidence:.2f}\n{r.reasoning}")
    if debate_report:
        all_reports.append(f"## Debate Outcome\nSignal: {debate_report.signal.value}\n{debate_report.reasoning}")

    analysis = "\n\n".join(all_reports)

    prompt = f"""Assess the risk for A-share stock {ticker} based on all analyst reports:

{analysis}

Provide risk assessment as JSON. Focus on:
1. Overall risk level (low/medium/high)
2. Maximum recommended position (% of total portfolio)
3. Stop-loss percentage suggestion
4. Key risk factors to monitor
5. Whether to proceed with the trade

Conservative when signals conflict, more confident when signals align.
"""

    try:
        result = await chat_json(prompt, system=SYSTEM_PROMPT)
        return AgentReport(
            agent_name="risk_manager",
            signal=Decision(result.get("signal", "hold")),
            confidence=float(result.get("confidence", 0.5)),
            reasoning=result.get("reasoning", ""),
            key_data={
                "risk_level": result.get("risk_level", "medium"),
                "max_position_pct": float(result.get("max_position_pct", 0.3)),
                "stop_loss_pct": float(result.get("stop_loss_pct", 0.08)),
                "risk_factors": result.get("risk_factors", []),
            },
        )
    except Exception as e:
        logger.error(f"[RiskManager] LLM error: {e}")
        return AgentReport(agent_name="risk_manager", reasoning=f"Risk assessment error: {e}")
