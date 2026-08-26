"""Technical Analyst Agent.

Analyzes technical indicators: MACD, RSI, KDJ, Bollinger Bands, MA, volume.
"""

import pandas as pd
from loguru import logger

from agents.prompt_context import INVESTOR_CONTEXT
from application.visual_evidence import visual_evidence_service
from data.stock_provider import async_get_stock_history
from llm import LLMService, get_llm_service
from models.schemas import AgentReport, AgentStageResult, AssetType, Decision, MarketContext
from strategies.strategy_registry import get_strategy_instructions

_BASE_PROMPT = (
    """You are a professional A-share technical analyst.
You analyze technical indicators including MACD, RSI, KDJ, Bollinger Bands, moving averages, and volume patterns.
You provide clear buy/sell/hold signals based on technical analysis.
You must respond in Chinese.
You must respond with valid JSON in this format:
{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.0-1.0,
  "reasoning": "detailed analysis in Chinese",
  "key_indicators": {"macd": "...", "rsi": ..., "kdj": "..."}
}"""
    + INVESTOR_CONTEXT
)


def _build_system_prompt(strategy_name: str | None = None, market_regime: str | None = None) -> str:
    """Build system prompt with strategy instructions injected."""
    strategy_text = get_strategy_instructions(strategy_name=strategy_name, market_regime=market_regime)
    return _BASE_PROMPT + strategy_text


async def analyze(
    ticker: str,
    days: int = 120,
    strategy_name: str | None = None,
    context: MarketContext | None = None,
    llm: LLMService | None = None,
) -> AgentReport:
    """Run technical analysis and return only the report for legacy callers."""
    stage = await analyze_stage(
        ticker,
        days=days,
        strategy_name=strategy_name,
        context=context,
        llm=llm,
    )
    return stage.report


async def analyze_stage(
    ticker: str,
    days: int = 120,
    strategy_name: str | None = None,
    context: MarketContext | None = None,
    llm: LLMService | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
) -> AgentStageResult:
    """Run technical analysis on a stock.

    Args:
        ticker: Stock code
        days: Number of days of history to analyze
        strategy_name: Optional strategy name to use (e.g. "bull_trend")
    """
    logger.info(f"[TechnicalAgent] Analyzing {ticker}, strategy={strategy_name or 'auto'}")

    if context is None:
        df = await async_get_stock_history(ticker, start_date="", end_date="")
    else:
        df = pd.DataFrame(context.history)
    if df.empty:
        return AgentStageResult(report=AgentReport(agent_name="technical", reasoning="No data available"))

    # Calculate technical indicators
    indicators = calculate_technical_indicators(df.tail(days))

    # Build prompt
    if "pct_chg" not in df.columns:
        df = df.copy()
        close = pd.to_numeric(df["close"], errors="coerce")
        df["pct_chg"] = close.pct_change(fill_method=None).mul(100).fillna(0.0)
    recent_data = df.tail(20)[["date", "close", "volume", "pct_chg"]].to_dict(orient="records")
    asset_label = (
        "A-share stock"
        if not context or context.asset_type == AssetType.STOCK
        else f"{context.asset_type.value.upper()} fund"
    )
    text_prompt = f"""Analyze the {asset_label} {ticker} using technical analysis.

Recent price data (last 20 days):
{recent_data}

Technical indicators:
{indicators}

Provide your analysis as JSON. Focus on:
1. Trend direction (based on MA5, MA10, MA20, MA60)
2. Momentum (MACD, RSI, KDJ)
3. Volatility (Bollinger Bands width)
4. Volume patterns
5. Support/resistance levels

Signal: buy if technicals suggest upward momentum, sell if downward, hold if mixed.
"""

    artifacts: list[dict] = []
    try:
        system_prompt = _build_system_prompt(strategy_name, context.market_regime if context else None)
        llm_service = llm or get_llm_service()
        result: dict
        supports_images = bool(getattr(llm_service, "supports_vision", lambda: False)())
        evidence = await visual_evidence_service.prepare_technical(
            ticker=ticker,
            asset_type=(context.asset_type.value if context else AssetType.STOCK.value),
            history=df.tail(days).to_dict(orient="records"),
            conversation_id=conversation_id,
            task_id=task_id,
        )
        if evidence:
            artifacts.append(evidence.artifact)
        if supports_images and evidence and evidence.model_url:
            visual_prompt = f"""Analyze the {asset_label} {ticker} using the attached deterministic chart.

The chart contains daily candles, MA lines, volume and MACD. Red candles are up and green candles are down.
Exact latest indicators (use these values for numeric claims):
{indicators}

Return the required JSON. Evaluate trend, momentum, volatility, volume, support/resistance and signal.
"""
            result = await llm_service.chat_json_with_images(
                visual_prompt,
                [evidence.model_url],
                system=system_prompt,
            )
        else:
            result = await llm_service.chat_json(text_prompt, system=system_prompt)
        report = AgentReport(
            agent_name="technical",
            signal=Decision(result.get("signal", "hold")),
            confidence=float(result.get("confidence", 0.5)),
            reasoning=result.get("reasoning", ""),
            key_data=result.get("key_indicators", {}),
        )
        return AgentStageResult(report=report, artifacts=artifacts)
    except Exception as e:
        logger.error(f"[TechnicalAgent] LLM error: {e}")
        return AgentStageResult(
            report=AgentReport(
                agent_name="technical",
                reasoning="技术分析模型暂时不可用，已按中性信号降级；请以结构化行情和确定性指标为准。",
                key_data={"degraded": True, "reason": "llm_unavailable"},
            ),
            artifacts=artifacts,
        )


def calculate_technical_indicators(df: pd.DataFrame) -> dict:
    """Calculate technical indicators from OHLCV data."""
    if df.empty:
        return {}

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # Moving averages
    ma5 = close.rolling(5).mean().iloc[-1] if len(close) >= 5 else 0
    ma10 = close.rolling(10).mean().iloc[-1] if len(close) >= 10 else 0
    ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else 0
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else 0

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2

    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # KDJ
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9).replace(0, 1e-10) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d

    # Bollinger Bands (20, 2)
    ma_boll = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = ma_boll + 2 * std
    lower = ma_boll - 2 * std

    return {
        "current_price": float(close.iloc[-1]),
        "MA5": round(float(ma5), 2),
        "MA10": round(float(ma10), 2),
        "MA20": round(float(ma20), 2),
        "MA60": round(float(ma60), 2) if ma60 else None,
        "MACD_DIF": round(float(dif.iloc[-1]), 4),
        "MACD_DEA": round(float(dea.iloc[-1]), 4),
        "MACD_hist": round(float(macd.iloc[-1]), 4),
        "RSI_14": round(float(rsi.iloc[-1]), 2) if not rsi.empty else None,
        "KDJ_K": round(float(k.iloc[-1]), 2),
        "KDJ_D": round(float(d.iloc[-1]), 2),
        "KDJ_J": round(float(j.iloc[-1]), 2),
        "BOLL_upper": round(float(upper.iloc[-1]), 2) if not upper.empty else None,
        "BOLL_lower": round(float(lower.iloc[-1]), 2) if not lower.empty else None,
        "BOLL_mid": round(float(ma_boll.iloc[-1]), 2) if not ma_boll.empty else None,
        "avg_volume_20": float(volume.tail(20).mean()),
    }


# Backward-compatible alias for existing callers and tests.
_calc_indicators = calculate_technical_indicators
