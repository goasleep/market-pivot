"""Portfolio Manager Agent.

Final decision maker: synthesizes all agent reports into a concrete trade decision.
Outputs a structured DecisionDashboard with 6 blocks.
"""

from loguru import logger
from models.schemas import (
    AgentReport,
    Decision,
    TradeDecision,
    DecisionDashboard,
    CoreConclusion,
    DataPerspective,
    Intelligence,
    BattlePlan,
    PhaseDecision,
    SignalAttribution,
    SignalType,
)
from llm.deepseek import chat_json
from strategies.skill_manager import get_strategy_instructions


SYSTEM_PROMPT = """You are a professional A-share portfolio manager.
You make the final trading decision based on all analyst reports.
You synthesize technical, fundamental, sentiment analysis, debate outcome, and risk assessment.
You must respond in Chinese.
You must respond with valid JSON only (no markdown, no explanation outside JSON).

The JSON must contain both a simplified decision and a detailed dashboard:

{
  "decision": "buy" | "sell" | "hold",
  "confidence": 0.0-1.0,
  "target_price": number | null,
  "stop_loss": number | null,
  "position_size": 0.0-1.0,
  "reasoning": "决策综述，200-400字",

  "dashboard": {
    "core_conclusion": {
      "signal": "strong_buy" | "buy" | "watch" | "reduce" | "sell" | "strong_sell",
      "confidence": 0.0-1.0,
      "one_line_summary": "一句话总结结论",
      "position_advice": "仓位建议描述"
    },
    "data_perspective": {
      "trend_status": "趋势状态，如：多头排列、空头排列、震荡",
      "price_position": "价格位置，如：接近MA20支撑、远离均线超买",
      "volume_analysis": "量能分析，如：放量上涨、缩量回调",
      "chip_structure": "筹码结构，如：筹码集中、上方套牢盘多"
    },
    "intelligence": {
      "latest_news": ["最新新闻摘要1", "摘要2"],
      "risk_alerts": ["风险警报1", "警报2"],
      "positive_catalysts": ["利好催化1"],
      "earnings_outlook": "盈利展望描述"
    },
    "battle_plan": {
      "entry_price": number | null,
      "stop_loss": number | null,
      "take_profit": number | null,
      "position_strategy": "仓位策略，如：分批建仓、一次到位、空仓观望",
      "action_items": ["行动项1", "行动项2"]
    },
    "phase_decision": {
      "pre_market": "盘前观察条件",
      "intraday": "盘中执行计划",
      "post_market": "盘后复盘要点"
    },
    "signal_attribution": {
      "technical_score": -100 to 100,
      "sentiment_score": -100 to 100,
      "fundamental_score": -100 to 100,
      "market_regime_score": -100 to 100
    }
  }
}
"""


def _parse_dashboard(raw: dict) -> DecisionDashboard:
    """Parse LLM JSON output into DecisionDashboard model."""
    db = raw.get("dashboard", {})

    cc = db.get("core_conclusion", {})
    dp = db.get("data_perspective", {})
    intel = db.get("intelligence", {})
    bp = db.get("battle_plan", {})
    ph = db.get("phase_decision", {})
    sa = db.get("signal_attribution", {})

    # Parse signal type with fallback
    try:
        signal = SignalType(cc.get("signal", "watch"))
    except ValueError:
        signal = SignalType.WATCH

    return DecisionDashboard(
        core_conclusion=CoreConclusion(
            signal=signal,
            confidence=float(cc.get("confidence", 0.5)),
            one_line_summary=cc.get("one_line_summary", ""),
            position_advice=cc.get("position_advice", ""),
        ),
        data_perspective=DataPerspective(
            trend_status=dp.get("trend_status", ""),
            price_position=dp.get("price_position", ""),
            volume_analysis=dp.get("volume_analysis", ""),
            chip_structure=dp.get("chip_structure", ""),
        ),
        intelligence=Intelligence(
            latest_news=intel.get("latest_news", []),
            risk_alerts=intel.get("risk_alerts", []),
            positive_catalysts=intel.get("positive_catalysts", []),
            earnings_outlook=intel.get("earnings_outlook", ""),
        ),
        battle_plan=BattlePlan(
            entry_price=bp.get("entry_price"),
            stop_loss=bp.get("stop_loss"),
            take_profit=bp.get("take_profit"),
            position_strategy=bp.get("position_strategy", ""),
            action_items=bp.get("action_items", []),
        ),
        phase_decision=PhaseDecision(
            pre_market=ph.get("pre_market", ""),
            intraday=ph.get("intraday", ""),
            post_market=ph.get("post_market", ""),
        ),
        signal_attribution=SignalAttribution(
            technical_score=float(sa.get("technical_score", 0)),
            sentiment_score=float(sa.get("sentiment_score", 0)),
            fundamental_score=float(sa.get("fundamental_score", 0)),
            market_regime_score=float(sa.get("market_regime_score", 0)),
        ),
    )


async def decide(
    ticker: str,
    agent_reports: dict[str, AgentReport],
    debate_report: AgentReport | None = None,
    risk_report: AgentReport | None = None,
    current_price: float = 0.0,
    strategy_name: str | None = None,
) -> TradeDecision:
    """Make final trading decision with structured dashboard.

    Args:
        ticker: Stock code
        agent_reports: Dict of agent_name -> AgentReport
        debate_report: Debate outcome
        risk_report: Risk assessment
        current_price: Current stock price
        strategy_name: Optional strategy to inject into prompt
    """
    logger.info(f"[PortfolioManager] Deciding on {ticker}, strategy={strategy_name or 'auto'}")

    # Compile all reports
    sections = []
    for name, r in agent_reports.items():
        sections.append(f"## {name}\nSignal: {r.signal.value}\nConfidence: {r.confidence:.2f}\n{r.reasoning}")
    if debate_report:
        sections.append(f"## Debate Outcome\nSignal: {debate_report.signal.value}\n{debate_report.reasoning}")
    if risk_report:
        sections.append(
            f"## Risk Manager\nSignal: {risk_report.signal.value}\n"
            f"Risk Level: {risk_report.key_data.get('risk_level', 'medium')}\n"
            f"Max Position: {risk_report.key_data.get('max_position_pct', 0.3) * 100:.0f}%\n"
            f"Stop Loss: {risk_report.key_data.get('stop_loss_pct', 0.08) * 100:.1f}%\n"
            f"{risk_report.reasoning}"
        )

    analysis = "\n\n".join(sections)

    prompt = f"""Make the final trading decision for A-share stock {ticker}.

Current price: {current_price}

All analyst reports:
{analysis}

Your decision must consider:
1. Weight of evidence across all agents
2. Risk-adjusted return potential
3. Position sizing (respect risk manager's limits)
4. Entry/exit levels (target price, stop loss)
5. Overall portfolio impact

Make your final decision as JSON.
"""

    try:
        # Inject strategy instructions into system prompt
        strategy_text = get_strategy_instructions(strategy_name=strategy_name)
        full_system = SYSTEM_PROMPT + strategy_text
        result = await chat_json(prompt, system=full_system)
        dashboard = _parse_dashboard(result)
        return TradeDecision(
            ticker=ticker,
            decision=Decision(result.get("decision", "hold")),
            confidence=float(result.get("confidence", 0.5)),
            target_price=result.get("target_price"),
            stop_loss=result.get("stop_loss"),
            position_size=result.get("position_size"),
            reasoning=result.get("reasoning", ""),
            agent_reports={name: r.reasoning for name, r in agent_reports.items()},
            dashboard=dashboard,
        )
    except Exception as e:
        logger.error(f"[PortfolioManager] LLM error: {e}")
        return TradeDecision(
            ticker=ticker,
            reasoning=f"Decision error: {e}",
            agent_reports={name: r.reasoning for name, r in agent_reports.items()},
        )
