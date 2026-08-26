"""Fundamentals Analyst Agent.

Analyzes financial indicators: PE, PB, ROE, revenue growth, profit margin, etc.
"""

import asyncio
import json

from loguru import logger

from agents.prompt_context import INVESTOR_CONTEXT
from application.visual_evidence import visual_evidence_service
from data.stock_provider import async_get_financial_data, async_get_stock_realtime
from llm import LLMService, get_llm_service
from models.schemas import AgentReport, AgentStageResult, AssetType, Decision, MarketContext

SYSTEM_PROMPT = (
    """You are a professional A-share fundamentals analyst.
You analyze company financials including PE/PB ratios, ROE, revenue growth, profit margins, debt ratios.
You provide buy/sell/hold signals based on fundamental analysis.
You must respond in Chinese.
You must respond with valid JSON in this format:
{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.0-1.0,
  "reasoning": "detailed analysis in Chinese",
  "key_metrics": {"pe": ..., "pb": ..., "roe": ...}
}"""
    + INVESTOR_CONTEXT
)

FUND_SYSTEM_PROMPT = (
    """You are a professional exchange-traded fund structure analyst.
You analyze short-to-medium-term ETF/LOF price behavior, unit NAV, premium/discount and liquidity.
Do not invent holdings or benchmark facts that are absent from the input. You must respond in Chinese.
You must respond with valid JSON in this format:
{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.0-1.0,
  "reasoning": "detailed analysis in Chinese",
  "key_metrics": {"premium_discount": "...", "liquidity": "...", "nav_tracking": "..."}
}"""
    + INVESTOR_CONTEXT
)


async def analyze(
    ticker: str,
    context: MarketContext | None = None,
    llm: LLMService | None = None,
) -> AgentReport:
    """Run fundamental/fund-structure analysis for legacy callers."""
    stage = await analyze_stage(ticker, context=context, llm=llm)
    return stage.report


async def analyze_stage(
    ticker: str,
    context: MarketContext | None = None,
    llm: LLMService | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
) -> AgentStageResult:
    """Run fundamentals analysis on a stock."""
    logger.info(f"[FundamentalsAgent] Analyzing {ticker}")

    if context is None:
        fin, rt = await asyncio.gather(
            async_get_financial_data(ticker),
            async_get_stock_realtime(ticker),
        )
    else:
        fin, rt = context.financial, context.realtime

    llm_service = llm or get_llm_service()
    artifacts: list[dict] = []
    if context and context.asset_type != AssetType.STOCK:
        fund_payload = context.fund_data.model_dump(mode="json") if context.fund_data else {}
        fund_facts = {key: value for key, value in fund_payload.items() if key != "nav_history"}
        text_prompt = f"""Analyze the {context.asset_type.value.upper()} fund {ticker} for short-to-medium-term trading.

Fund facts supplied by the backend:
{json.dumps(fund_payload, ensure_ascii=False, default=str)}

Realtime exchange data:
{json.dumps(context.realtime, ensure_ascii=False, default=str)}

Return the required JSON. Focus on price versus NAV, premium/discount persistence, liquidity, tracking behavior,
holding-period risks and conditions for entry or exit. Do not infer missing holdings or index composition.
"""
        try:
            result: dict
            supports_images = bool(getattr(llm_service, "supports_vision", lambda: False)())
            evidence = None
            nav_history = fund_payload.get("nav_history") or []
            if nav_history:
                evidence = await visual_evidence_service.prepare_fund_structure(
                    ticker=ticker,
                    asset_type=context.asset_type.value,
                    history=context.history,
                    nav_history=nav_history,
                    conversation_id=conversation_id,
                    task_id=task_id,
                )
                if evidence:
                    artifacts.append(evidence.artifact)
            if supports_images and evidence and evidence.model_url:
                result = await llm_service.chat_json_with_images(
                    f"""Analyze the {context.asset_type.value.upper()} fund {ticker} for short-to-medium-term trading.

Exact fund facts supplied by the backend (the historical NAV series is represented by the attached chart):
{json.dumps(fund_facts, ensure_ascii=False, default=str)}

Realtime exchange data:
{json.dumps(context.realtime, ensure_ascii=False, default=str)}

The attached chart is deterministic price/NAV visual evidence. Return the required JSON and focus on price versus
NAV, premium/discount persistence, liquidity, tracking behavior, holding-period risks and entry/exit conditions.
Do not infer missing holdings or index composition.
""",
                    [evidence.model_url],
                    system=FUND_SYSTEM_PROMPT,
                )
            else:
                result = await llm_service.chat_json(text_prompt, system=FUND_SYSTEM_PROMPT)
            return AgentStageResult(
                report=AgentReport(
                    agent_name="fund_structure",
                    signal=Decision(result.get("signal", "hold")),
                    confidence=float(result.get("confidence", 0.5)),
                    reasoning=result.get("reasoning", ""),
                    key_data=result.get("key_metrics", {}),
                ),
                artifacts=artifacts,
            )
        except Exception as e:
            logger.error(f"[FundamentalsAgent] fund structure LLM error: {e}")
            return AgentStageResult(
                report=AgentReport(
                    agent_name="fund_structure",
                    reasoning="基金结构分析模型暂时不可用，已按中性信号降级。",
                    key_data={"degraded": True, "reason": "llm_unavailable"},
                ),
                artifacts=artifacts,
            )

    prompt = f"""Analyze the A-share stock {ticker} using fundamental analysis.

Realtime market data:
- Current price: {rt.get("price", "N/A")}
- PE (dynamic): {rt.get("pe", "N/A")}
- PB: {rt.get("pb", "N/A")}
- Total market value: {rt.get("total_mv", "N/A")}
- Circulating market value: {rt.get("circ_mv", "N/A")}

Financial indicators:
- ROE: {fin.get("roe", "N/A")}%
- Gross profit margin: {fin.get("gross_profit_margin", "N/A")}%
- Net profit margin: {fin.get("net_profit_margin", "N/A")}%
- Debt ratio: {fin.get("debt_ratio", "N/A")}%

Provide your analysis as JSON. Focus on:
1. Valuation (PE/PB vs industry average)
2. Profitability (ROE, margins)
3. Growth potential
4. Financial health (debt ratio)
5. Overall fundamental quality

Signal: buy if undervalued with strong fundamentals, sell if overvalued or deteriorating, hold if fair.
"""

    try:
        result = await llm_service.chat_json(prompt, system=SYSTEM_PROMPT)
        return AgentStageResult(
            report=AgentReport(
                agent_name="fundamentals",
                signal=Decision(result.get("signal", "hold")),
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", ""),
                key_data=result.get("key_metrics", {}),
            )
        )
    except Exception as e:
        logger.error(f"[FundamentalsAgent] LLM error: {e}")
        return AgentStageResult(
            report=AgentReport(
                agent_name="fundamentals",
                reasoning="基本面分析模型暂时不可用，已按中性信号降级。",
                key_data={"degraded": True, "reason": "llm_unavailable"},
            )
        )
