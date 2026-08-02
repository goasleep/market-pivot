"""Debate Room - Bull vs Bear.

Two agents debate from opposing perspectives, then an LLM judge evaluates.
"""

from loguru import logger
from models.schemas import AgentReport, Decision
from llm.deepseek import chat, chat_json


BULL_SYSTEM = """You are a bullish (bull) A-share investment researcher.
Your job is to find reasons to BUY the stock based on the provided analysis data.
Be persuasive but factual. Focus on opportunities, growth potential, positive catalysts.
Respond in Chinese, 300-500 words.
"""

BEAR_SYSTEM = """You are a bearish (bear) A-share investment researcher.
Your job is to find reasons to SELL the stock based on the provided analysis data.
Be persuasive but factual. Focus on risks, overvaluation, negative signals, downside potential.
Respond in Chinese, 300-500 words.
"""

JUDGE_SYSTEM = """You are a neutral investment debate judge.
You evaluate bull and bear arguments and make a balanced assessment.
You must respond with valid JSON only:
{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.0-1.0,
  "bull_score": 0-10,
  "bear_score": 0-10,
  "reasoning": "summary of debate outcome in Chinese"
}
"""


async def debate(
    ticker: str,
    agent_reports: dict[str, AgentReport],
    rounds: int = 2,
) -> AgentReport:
    """Run bull vs bear debate.

    Args:
        ticker: Stock code
        agent_reports: Dict of agent_name -> AgentReport from analysts
        rounds: Number of debate rounds

    Returns:
        AgentReport with debate outcome
    """
    logger.info(f"[DebateRoom] Debating {ticker}, rounds={rounds}")

    # Compile analysis summary
    analysis_summary = "\n\n".join(
        f"## {name}\nSignal: {r.signal.value} (confidence: {r.confidence:.2f})\n{r.reasoning}"
        for name, r in agent_reports.items()
    )

    # Round 1: Bull and Bear present initial arguments
    bull_prompt = f"""Based on the following analysis for stock {ticker}, present your bullish case:

{analysis_summary}

Argue why this stock should be bought. Focus on the strongest evidence.
"""
    bull_arg = await chat(bull_prompt, system=BULL_SYSTEM)

    bear_prompt = f"""Based on the following analysis for stock {ticker}, present your bearish case:

{analysis_summary}

Argue why this stock should be sold. Focus on the strongest evidence.
"""
    bear_arg = await chat(bear_prompt, system=BEAR_SYSTEM)

    # Round 2: Rebuttal
    if rounds >= 2:
        bull_rebuttal_prompt = f"""The bear argues:
{bear_arg}

Rebut their arguments and strengthen your bull case for {ticker}.
"""
        bull_arg = await chat(bull_rebuttal_prompt, system=BULL_SYSTEM)

        bear_rebuttal_prompt = f"""The bull argues:
{bull_arg}

Rebut their arguments and strengthen your bear case for {ticker}.
"""
        bear_arg = await chat(bear_rebuttal_prompt, system=BEAR_SYSTEM)

    # Judge evaluates
    judge_prompt = f"""Evaluate the bull vs bear debate for stock {ticker}.

Bull argument:
{bull_arg}

Bear argument:
{bear_arg}

Original analysis:
{analysis_summary}

Make your judgment as JSON. Score both sides 0-10 and give final signal.
"""
    try:
        result = await chat_json(judge_prompt, system=JUDGE_SYSTEM)
        return AgentReport(
            agent_name="debate",
            signal=Decision(result.get("signal", "hold")),
            confidence=float(result.get("confidence", 0.5)),
            reasoning=result.get("reasoning", ""),
            key_data={
                "bull_score": result.get("bull_score", 5),
                "bear_score": result.get("bear_score", 5),
                "bull_argument": bull_arg,
                "bear_argument": bear_arg,
            },
        )
    except Exception as e:
        logger.error(f"[DebateRoom] Judge error: {e}")
        return AgentReport(
            agent_name="debate",
            reasoning=f"Debate judge error: {e}",
        )
