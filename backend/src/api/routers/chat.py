"""Stock Agent chat API with SSE text and inline widgets."""

import asyncio
import json
from typing import Literal

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agents.stock_agent import (
    StockAgentRequest,
    StockIntent,
    stock_agent,
)
from data.akshare_provider import (
    async_get_stock_history,
    get_breaker_status,
)
from models.schemas import AssetType
from strategies.skill_manager import list_strategies
from widgets.renderer import (
    render_agent_pipeline,
    render_breaker_status,
    render_decision_dashboard,
    render_mini_chart,
    render_signal_gauge,
    render_stock_card,
    render_strategy_selector,
)

router = APIRouter()


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="A stock or exchange-traded fund request")
    strategy: str | None = Field(default=None, description="Strategy override")
    history: list[ChatHistoryItem] = Field(default_factory=list, description="Recent conversation context")
    conversation_id: str | None = Field(default=None, description="Client-side conversation identifier")
    asset_type: AssetType | None = Field(default=None, description="Optional stock, ETF, or LOF override")


PIPELINE_STAGES = [
    {"name": "market_data", "label": "Market Data", "status": "pending"},
    {"name": "technical", "label": "Technical", "status": "pending"},
    {"name": "fundamentals", "label": "Fundamentals", "status": "pending"},
    {"name": "sentiment", "label": "Sentiment", "status": "pending"},
    {"name": "debate", "label": "Debate", "status": "pending"},
    {"name": "risk", "label": "Risk", "status": "pending"},
    {"name": "portfolio", "label": "Portfolio", "status": "pending"},
]

_INTENT_LABELS = {
    StockIntent.ANALYZE: "股票综合分析",
    StockIntent.QUOTE: "实时行情查询",
    StockIntent.HISTORY: "历史走势查询",
    StockIntent.NEWS: "个股新闻查询",
    StockIntent.STRATEGIES: "交易策略查询",
    StockIntent.PORTFOLIO: "组合信息查询",
    StockIntent.BACKTEST: "策略回测",
    StockIntent.COMPARE: "股票对比",
    StockIntent.HELP: "能力说明",
}


def _event(event: str, payload: dict) -> dict:
    return {"event": event, "data": json.dumps(payload, ensure_ascii=False)}


def _done() -> dict:
    return _event("done", {})


def _format_price(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _quote_text(ticker: str, quote: dict) -> str:
    name = quote.get("name") or ""
    change = quote.get("pct_chg")
    change_text = f"{float(change):+.2f}%" if change is not None else "-"
    return (
        f"**{name or ticker}（{ticker}）** 当前价 ¥{_format_price(quote.get('price'))}，"
        f"涨跌幅 {change_text}。开盘 ¥{_format_price(quote.get('open'))}，"
        f"最高 ¥{_format_price(quote.get('high'))}，最低 ¥{_format_price(quote.get('low'))}。"
    )


def _news_text(ticker: str, news: list[dict]) -> str:
    if not news:
        return f"暂时没有获取到 {ticker} 的最新新闻。"
    lines = [f"**{ticker} 最新新闻**"]
    for item in news[:8]:
        title = item.get("title") or "未命名新闻"
        date = item.get("date") or ""
        source = item.get("source") or ""
        suffix = " · ".join(part for part in (date, source) if part)
        lines.append(f"- **{title}**{f'（{suffix}）' if suffix else ''}")
    return "\n".join(lines)


def _history_text(ticker: str, history) -> str:
    if history.empty:
        return f"暂时没有获取到 {ticker} 的历史行情数据。"
    latest = history.iloc[-1]
    date = latest.get("date", "")
    close = _format_price(latest.get("close"))
    pct = latest.get("pct_chg")
    pct_text = f"{float(pct):+.2f}%" if pct is not None else "-"
    return f"**{ticker} 历史走势** 最近交易日 {date}，收盘价 ¥{close}，涨跌幅 {pct_text}。"


async def _analysis_events(request: StockAgentRequest):
    """Yield the stock analysis pipeline and final decision widgets."""
    ticker = request.ticker
    assert ticker is not None

    yield _event("text", {"text": f"股票 Agent：开始分析 {ticker}。"})
    stages = [stage.copy() for stage in PIPELINE_STAGES]
    stage_names = {"merge_debate": "debate"}
    accumulated: dict = {}
    realtime_sent = False
    status_sent = False
    async for update in stock_agent.analyze_stream(request):
        if not status_sent and update.get("data_status"):
            status = update["data_status"]
            yield _event(
                "text",
                {
                    "text": (
                        "数据状态："
                        f"历史={'正常' if status.get('history') else '缺失'}，"
                        f"实时={'正常' if status.get('realtime') else '缺失'}，"
                        f"财务={'正常' if status.get('financial') else '不适用/缺失'}，"
                        f"新闻={'正常' if status.get('news') else '缺失'}。"
                    )
                },
            )
            status_sent = True
        if not realtime_sent and update.get("realtime"):
            yield _event(
                "widget",
                {"type": "stock_card", "html": render_stock_card(update["realtime"])},
            )
            realtime_sent = True

        node = stage_names.get(update["node"], update["node"])
        for stage in stages:
            if stage["name"] == node:
                stage["status"] = "done"
                break
        current_stage = next(
            (stage["name"] for stage in stages if stage["status"] == "pending"),
            "",
        )
        yield _event(
            "widget",
            {"type": "agent_pipeline", "html": render_agent_pipeline(stages, current_stage)},
        )
        accumulated = update.get("state", accumulated)

    decision = accumulated.get("final_decision")

    if not decision:
        yield _event("text", {"text": "分析流程完成，但没有返回最终决策。"})
        yield _done()
        return

    dashboard = decision.dashboard.model_dump() if decision.dashboard else None
    if dashboard:
        yield _event(
            "widget",
            {"type": "decision_dashboard", "html": render_decision_dashboard(dashboard)},
        )
        attribution = dashboard.get("signal_attribution", {})
        if attribution:
            yield _event(
                "widget",
                {"type": "signal_gauge", "html": render_signal_gauge(attribution)},
            )

    decision_label = {"buy": "买入", "sell": "卖出", "hold": "观望"}.get(
        decision.decision.value, decision.decision.value
    )
    summary = f"**{decision_label}** {ticker} | 置信度：{decision.confidence:.0%}"
    if decision.target_price:
        summary += f" | 目标价：¥{decision.target_price}"
    if decision.stop_loss:
        summary += f" | 止损价：¥{decision.stop_loss}"
    if decision.position_size:
        summary += f" | 建议仓位：{decision.position_size:.0%}"
    yield _event("text", {"text": summary})
    if decision.reasoning:
        yield _event("text", {"text": f"\n\n**决策依据：**\n{decision.reasoning}"})

    yield _event(
        "complete",
        {
            "task": "analysis",
            "ticker": decision.ticker,
            "decision": decision.decision.value,
            "confidence": decision.confidence,
            "target_price": decision.target_price,
            "stop_loss": decision.stop_loss,
            "position_size": decision.position_size,
            "reasoning": decision.reasoning,
            "agent_reports": decision.agent_reports,
            "dashboard": dashboard,
        },
    )
    yield _done()


@router.post("/send")
async def chat_send(req: ChatRequest):
    """Route a stock-related message and return an SSE response."""
    logger.info(f"[StockAgent] Message: {req.message}")

    async def event_generator():
        request = stock_agent.prepare(
            message=req.message,
            history=[item.model_dump() for item in req.history],
            strategy=req.strategy,
            conversation_id=req.conversation_id,
            asset_type=req.asset_type,
        )
        try:
            yield _event("text", {"text": "A-Share Agent：正在让模型判断任务并选择数据工具。"})
            async for event in stock_agent.chat(request):
                yield _event(event["type"], {"text": event["text"]})
        except Exception as exc:
            logger.exception(f"[StockAgent] Task failed: {exc}")
            yield _event("text", {"text": f"Agent 执行失败：{exc}"})
        yield _done()

    return EventSourceResponse(event_generator())


@router.get("/widgets/strategies")
async def get_strategy_widget():
    """Get the strategy selector widget."""
    strategies = await asyncio.to_thread(list_strategies)
    return {"html": render_strategy_selector(strategies)}


@router.get("/widgets/breakers")
async def get_breaker_widget():
    """Get the circuit breaker status widget."""
    return {"html": render_breaker_status(get_breaker_status())}


@router.get("/widgets/mini_chart/{ticker}")
async def get_mini_chart_widget(ticker: str):
    """Get a mini sparkline chart for a stock."""
    history = await async_get_stock_history(ticker)
    if history.empty:
        return {"html": '<div style="color:#94a3b8;font-size:12px">No data</div>'}
    return {"html": render_mini_chart(history.tail(30)["close"].tolist())}
