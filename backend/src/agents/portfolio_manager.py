"""Portfolio Manager Agent.

Final decision maker: synthesizes all agent reports into a concrete trade decision.
Outputs a structured DecisionDashboard with 6 blocks.
"""

import json

from loguru import logger

from agents.prompt_context import INVESTOR_CONTEXT
from domain.decision_policy import DecisionValidator
from llm import LLMService, get_llm_service
from models.schemas import (
    AgentReport,
    AssetType,
    BattlePlan,
    CoreConclusion,
    DataPerspective,
    Decision,
    DecisionDashboard,
    Intelligence,
    MarketContext,
    PhaseDecision,
    SignalAttribution,
    SignalType,
    StrategyPlan,
    StrategySpec,
    TradeDecision,
    TradePlan,
)
from strategies.skill_manager import get_strategy_instructions, get_strategy_spec, register_strategy_spec

SYSTEM_PROMPT = """You are a professional A-share portfolio manager.
You make the final trading decision based on all analyst reports.
You synthesize technical, fundamental, sentiment analysis, debate outcome, and risk assessment.
You must respond in Chinese.
You must respond with valid JSON only (no markdown, no explanation outside JSON).

The JSON must contain both a simplified decision and a detailed dashboard:

{
  "decision": "buy" | "sell" | "hold",
  "confidence": 0.0-1.0,
  "entry_price": number | null,
  "target_price": number | null,
  "stop_loss": number | null,
  "take_profit": number | null,
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
      "action_items": ["行动项1", "行动项2"],
      "entry_explanation": "入场价为什么是这个数字；必须引用输入数据和计算过程",
      "stop_loss_explanation": "止损价为什么是这个数字；必须引用输入数据和计算过程",
      "take_profit_explanation": "止盈价为什么是这个数字；必须引用输入数据和计算过程",
      "price_evidence": [
        {"metric": "MA20", "value": 0, "source": "market_context/history", "as_of": "YYYY-MM-DD", "calculation": ""}
      ]
    },
    "strategy_plan": {
      "name": "本次采用或临时构建的策略名称",
      "thesis": "策略假设",
      "entry_conditions": ["入场条件"],
      "exit_conditions": ["退出条件"],
      "indicators_used": ["MA20", "ATR14"],
      "data_basis": [
        {"metric": "close", "value": 0, "source": "market_context/realtime", "as_of": "YYYY-MM-DD", "calculation": ""}
      ],
      "spec": {
        "name": "strategy_name",
        "version": "1.0.0",
        "asset_types": ["etf", "lof"],
        "indicators": ["return_pct", "price_vs_ma_pct"],
        "entry_conditions": [{"indicator": "return_pct", "operator": "gt", "value": 0, "window": 20}],
        "exit_conditions": [],
        "stop_loss_pct": 0.08,
        "take_profit_pct": 0.16,
        "position_size_pct": 0.2,
        "rebalance_frequency": "daily",
        "source": "llm"
      }
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
""" + INVESTOR_CONTEXT


def _parse_dashboard(raw: dict) -> DecisionDashboard:
    """Parse LLM JSON output into DecisionDashboard model."""
    db = raw.get("dashboard", {})

    cc = db.get("core_conclusion", {})
    dp = db.get("data_perspective", {})
    intel = db.get("intelligence", {})
    bp = db.get("battle_plan", {})
    ph = db.get("phase_decision", {})
    sa = db.get("signal_attribution", {})
    strategy = db.get("strategy_plan", {})

    # Parse signal type with fallback
    try:
        signal = SignalType(cc.get("signal", "watch"))
    except ValueError:
        signal = SignalType.WATCH

    spec = None
    if isinstance(strategy.get("spec"), dict):
        try:
            spec = StrategySpec.model_validate(strategy["spec"])
        except Exception as exc:
            logger.warning("Invalid LLM strategy spec: {}", exc)

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
            entry_explanation=bp.get("entry_explanation", ""),
            stop_loss_explanation=bp.get("stop_loss_explanation", ""),
            take_profit_explanation=bp.get("take_profit_explanation", ""),
            price_evidence=bp.get("price_evidence", []),
        ),
        strategy_plan=StrategyPlan(
            name=strategy.get("name", ""),
            thesis=strategy.get("thesis", ""),
            entry_conditions=strategy.get("entry_conditions", []),
            exit_conditions=strategy.get("exit_conditions", []),
            indicators_used=strategy.get("indicators_used", []),
            data_basis=strategy.get("data_basis", []),
            spec=spec,
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


def _market_facts(
    market_context: MarketContext | None,
    agent_reports: dict[str, AgentReport],
) -> str:
    """Expose compact, traceable facts to the final LLM decision."""
    if market_context is None:
        return "market_context unavailable"
    technical = agent_reports.get("technical")
    payload = {
        "ticker": market_context.ticker,
        "asset_type": market_context.asset_type.value,
        "as_of_date": market_context.as_of_date,
        "current_price": market_context.current_price,
        "market_regime": market_context.market_regime,
        "data_status": market_context.data_status,
        "realtime": market_context.realtime,
        "fund_data": market_context.fund_data.model_dump(mode="json") if market_context.fund_data else None,
        "financial": market_context.financial,
        "recent_history": market_context.history[-30:],
        "technical_indicators": technical.key_data if technical else {},
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _buy_plan_has_evidence(dashboard: DecisionDashboard) -> bool:
    """Require a complete, ordered and traceable price plan for a buy."""
    battle = dashboard.battle_plan
    levels = (battle.entry_price, battle.stop_loss, battle.take_profit)
    explanations = (
        battle.entry_explanation,
        battle.stop_loss_explanation,
        battle.take_profit_explanation,
    )
    evidence = battle.price_evidence
    if not all(level is not None and level > 0 for level in levels):
        return False
    if not all(explanation.strip() for explanation in explanations):
        return False
    if not battle.stop_loss < battle.entry_price < battle.take_profit:
        return False
    return bool(
        evidence
        and all(
            item.metric
            and item.source
            and item.as_of
            for item in evidence
        )
    )


async def decide(
    ticker: str,
    agent_reports: dict[str, AgentReport],
    debate_report: AgentReport | None = None,
    risk_report: AgentReport | None = None,
    current_price: float = 0.0,
    strategy_name: str | None = None,
    market_regime: str | None = None,
    asset_type: AssetType | str = AssetType.STOCK,
    conversation_history: list[dict[str, str]] | None = None,
    investor_context: dict | None = None,
    llm: LLMService | None = None,
    market_context: MarketContext | None = None,
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

    asset_type = AssetType(asset_type)
    asset_label = "A-share stock" if asset_type == AssetType.STOCK else f"{asset_type.value.upper()} fund"
    context_text = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in (conversation_history or [])[-8:]
        if item.get("content")
    )
    investor_text = "\n".join(f"{key}: {value}" for key, value in (investor_context or {}).items())
    market_facts = _market_facts(market_context, agent_reports)
    prompt = f"""Make the final trading decision for the {asset_label} {ticker}.

Current price: {current_price}

Investor constraints:
{investor_text or "not provided; use conservative defaults"}

Recent conversation context:
{context_text or "none"}

All analyst reports:
{analysis}

Traceable market facts (do not invent values outside this block):
{market_facts}

Your decision must consider:
1. Weight of evidence across all agents
2. Risk-adjusted return potential
3. Position sizing (respect risk manager's limits)
4. Entry/exit levels (target price, stop loss)
5. Overall portfolio impact

The strategy and price levels may be your judgment, but every non-null price
must be justified by explicit facts above.  Explain the source date, the
indicator or support/resistance used, and the calculation or comparison that
led to the number.  If the data is insufficient, return null and explain why.

Make your final decision as JSON.
"""

    try:
        # Inject strategy instructions into system prompt
        strategy_text = get_strategy_instructions(
            strategy_name=strategy_name,
            market_regime=market_regime,
        )
        selected_spec = get_strategy_spec(strategy_name) if strategy_name else None
        spec_text = (
            json.dumps(selected_spec.model_dump(mode="json"), ensure_ascii=False)
            if selected_spec
            else "none"
        )
        prompt += f"\n\nExecutable strategy specification:\n{spec_text}\n"
        full_system = SYSTEM_PROMPT + strategy_text
        llm_service = llm or get_llm_service()
        result = await llm_service.chat_json(prompt, system=full_system)
        dashboard = _parse_dashboard(result)
        if dashboard.strategy_plan.spec is not None:
            register_strategy_spec(dashboard.strategy_plan.spec)
        requested_decision = Decision(result.get("decision", "hold"))
        decision = requested_decision
        reasoning = result.get("reasoning", "")
        if requested_decision == Decision.BUY and not _buy_plan_has_evidence(dashboard):
            logger.warning(
                "[PortfolioManager] Buy plan for {} lacked complete price evidence; downgraded to hold",
                ticker,
            )
            decision = Decision.HOLD
            reasoning = (
                f"{reasoning}\n\n系统校验：原始买入建议未提供完整且有日期依据的入场、止损、止盈方案，"
                "因此本次降级为观望。"
            ).strip()
        position_size = result.get("position_size")
        if position_size is not None and risk_report:
            risk_limit = float(risk_report.key_data.get("max_position_pct", 1.0))
            position_size = min(max(float(position_size), 0.0), max(risk_limit, 0.0))
        plan = TradePlan(
            entry_price=result.get("entry_price")
            if result.get("entry_price") is not None
            else dashboard.battle_plan.entry_price,
            stop_loss=result.get("stop_loss")
            if result.get("stop_loss") is not None
            else dashboard.battle_plan.stop_loss,
            take_profit=result.get("take_profit")
            if result.get("take_profit") is not None
            else dashboard.battle_plan.take_profit,
            position_size=position_size,
            position_strategy=dashboard.battle_plan.position_strategy,
            action_items=dashboard.battle_plan.action_items,
            entry_explanation=dashboard.battle_plan.entry_explanation,
            stop_loss_explanation=dashboard.battle_plan.stop_loss_explanation,
            take_profit_explanation=dashboard.battle_plan.take_profit_explanation,
            price_evidence=dashboard.battle_plan.price_evidence,
        )
        decision_payload = TradeDecision(
            ticker=ticker,
            asset_type=asset_type,
            decision=decision,
            confidence=float(result.get("confidence", 0.5)),
            plan=plan,
            reasoning=reasoning,
            agent_reports={name: r.reasoning for name, r in agent_reports.items()},
            dashboard=dashboard,
        )
        issues = DecisionValidator.validate(decision_payload, current_price=current_price)
        if decision_payload.decision == Decision.BUY and issues:
            messages = "；".join(issue.message for issue in issues)
            logger.warning("[PortfolioManager] Decision validation failed for {}: {}", ticker, messages)
            decision_payload = decision_payload.model_copy(
                update={
                    "decision": Decision.HOLD,
                    "reasoning": f"{decision_payload.reasoning}\n\n系统校验：{messages}".strip(),
                }
            )
        return decision_payload
    except Exception as e:
        logger.error(f"[PortfolioManager] LLM error: {e}")
        return TradeDecision(
            ticker=ticker,
            asset_type=asset_type,
            decision=Decision.HOLD,
            confidence=0.0,
            reasoning="组合决策模型暂时不可用，本次已降级为观望；结构化行情仍可单独参考。",
            agent_reports={name: r.reasoning for name, r in agent_reports.items()},
        )
