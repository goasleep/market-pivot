"""Fundamentals Analyst Agent.

Analyzes financial indicators: PE, PB, ROE, revenue growth, profit margin, etc.
"""

import asyncio

from loguru import logger

from llm import LLMService, get_llm_service
from models.schemas import AgentReport, Decision, MarketContext

SYSTEM_PROMPT = """You are a professional A-share fundamentals analyst.
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


async def analyze(
    ticker: str,
    context: MarketContext | None = None,
    llm: LLMService | None = None,
) -> AgentReport:
    """Run fundamentals analysis on a stock."""
    logger.info(f"[FundamentalsAgent] Analyzing {ticker}")

    if context is None:
        from data.akshare_provider import async_get_financial_data, async_get_stock_realtime

        fin, rt = await asyncio.gather(
            async_get_financial_data(ticker),
            async_get_stock_realtime(ticker),
        )
    else:
        fin, rt = context.financial, context.realtime

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
        llm_service = llm or get_llm_service()
        result = await llm_service.chat_json(prompt, system=SYSTEM_PROMPT)
        return AgentReport(
            agent_name="fundamentals",
            signal=Decision(result.get("signal", "hold")),
            confidence=float(result.get("confidence", 0.5)),
            reasoning=result.get("reasoning", ""),
            key_data=result.get("key_metrics", {}),
        )
    except Exception as e:
        logger.error(f"[FundamentalsAgent] LLM error: {e}")
        return AgentReport(agent_name="fundamentals", reasoning=f"Analysis error: {e}")
