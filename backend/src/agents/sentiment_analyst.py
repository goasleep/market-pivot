"""Sentiment Analyst Agent.

Analyzes news sentiment and market sentiment for a stock.
"""

from loguru import logger
from models.schemas import AgentReport, Decision
from data.akshare_provider import get_stock_news
from llm.deepseek import chat_json


SYSTEM_PROMPT = """You are a professional A-share market sentiment analyst.
You analyze news articles, market sentiment, and public opinion to gauge market mood.
You provide buy/sell/hold signals based on sentiment analysis.
You must respond in Chinese.
You must respond with valid JSON in this format:
{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.0-1.0,
  "reasoning": "detailed analysis in Chinese",
  "sentiment_score": -1.0 to 1.0,
  "key_themes": ["theme1", "theme2"]
}"""


async def analyze(ticker: str, num_news: int = 10) -> AgentReport:
    """Run sentiment analysis on a stock based on recent news."""
    logger.info(f"[SentimentAgent] Analyzing {ticker}")

    # Fetch news
    news = get_stock_news(ticker, limit=num_news)
    if not news:
        return AgentReport(
            agent_name="sentiment",
            reasoning="No recent news found, sentiment neutral.",
            confidence=0.3,
        )

    # Format news for prompt
    news_text = "\n".join(f"[{n['date']}] {n['title']}\n{n['content'][:200]}" for n in news)

    prompt = f"""Analyze the market sentiment for A-share stock {ticker} based on recent news.

Recent news articles:
{news_text}

Provide your analysis as JSON. Focus on:
1. Overall sentiment (positive/negative/neutral)
2. Key themes in the news
3. Potential market impact
4. Any risk events or catalysts

Signal: buy if sentiment is strongly positive, sell if strongly negative, hold if mixed.
"""

    try:
        result = await chat_json(prompt, system=SYSTEM_PROMPT)
        return AgentReport(
            agent_name="sentiment",
            signal=Decision(result.get("signal", "hold")),
            confidence=float(result.get("confidence", 0.5)),
            reasoning=result.get("reasoning", ""),
            key_data={
                "sentiment_score": result.get("sentiment_score", 0),
                "key_themes": result.get("key_themes", []),
            },
        )
    except Exception as e:
        logger.error(f"[SentimentAgent] LLM error: {e}")
        return AgentReport(agent_name="sentiment", reasoning=f"Analysis error: {e}")
