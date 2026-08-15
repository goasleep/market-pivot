"""Stock Agent chat API with SSE text and inline widgets."""

import asyncio
import json
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agents.stock_agent import (
    StockAgentRequest,
    StockIntent,
    stock_agent,
)
from application.chat_service import ChatTaskInput, chat_store, chat_task_manager
from data.akshare_provider import (
    async_get_stock_history,
    get_breaker_status,
)
from models.schemas import AssetType
from strategies.skill_manager import list_strategies
from widgets.a2ui import (
    CATALOG,
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
    parts: list[dict] | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="A stock or exchange-traded fund request")
    strategy: str | None = Field(default=None, description="Strategy override")
    history: list[ChatHistoryItem] = Field(default_factory=list, description="Recent conversation context")
    conversation_id: str | None = Field(default=None, description="Client-side conversation identifier")
    asset_type: AssetType | None = Field(default=None, description="Optional stock, ETF, or LOF override")
    task_id: str | None = Field(default=None, description="Client-generated task identifier")


class ConversationUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)


class A2UIActionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    surface_id: str = Field(alias="surfaceId", min_length=1, max_length=200)
    context: dict = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


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
                {"type": "a2ui", "messages": render_stock_card(update["realtime"])},
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
            {"type": "a2ui", "messages": render_agent_pipeline(stages, current_stage)},
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
            {"type": "a2ui", "messages": render_decision_dashboard(dashboard)},
        )
        attribution = dashboard.get("signal_attribution", {})
        if attribution:
            yield _event(
                "widget",
                {"type": "a2ui", "messages": render_signal_gauge(attribution)},
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
    """Create a durable Agent task and stream its events to this subscriber."""
    logger.info(f"[StockAgent] Message: {req.message}")
    conversation_id = req.conversation_id or f"conversation-{uuid4().hex}"
    task_id = req.task_id or f"task-{uuid4().hex}"
    history = [item.model_dump(exclude_none=True) for item in req.history]
    try:
        _, assistant_message_id = chat_store.prepare_task(
            conversation_id=conversation_id,
            task_id=task_id,
            message=req.message,
            history=history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    task_input = ChatTaskInput(
        task_id=task_id,
        conversation_id=conversation_id,
        message=req.message,
        history=history,
        strategy=req.strategy,
        asset_type=req.asset_type,
        assistant_message_id=assistant_message_id,
    )

    async def event_generator():
        await chat_task_manager.start(task_input)
        async for event in chat_task_manager.subscribe(task_id):
            yield event

    return EventSourceResponse(event_generator())


@router.post("/tasks/{task_id}/cancel")
async def cancel_chat_task(task_id: str):
    """Cancel a server-owned chat task without relying on the SSE client connection."""
    result = await chat_task_manager.cancel(task_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail="聊天任务不存在")
    return result


@router.get("/tasks/{task_id}")
async def get_chat_task(task_id: str):
    """Return durable status for a chat task after a browser refresh."""
    result = chat_store.get_task(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="聊天任务不存在")
    return result


@router.get("/tasks/{task_id}/stream")
async def reconnect_chat_task(request: Request, task_id: str, last_event_id: int = 0):
    """Reconnect a browser to an existing server-owned chat SSE task."""
    record = chat_store.get_task(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="聊天任务不存在")

    header_cursor = request.headers.get("last-event-id")
    if header_cursor and header_cursor.isdigit():
        last_event_id = max(last_event_id, int(header_cursor))

    async def event_generator():
        async for event in chat_task_manager.subscribe(task_id, after_sequence=last_event_id):
            yield event

    return EventSourceResponse(event_generator())


@router.get("/conversations")
async def list_chat_conversations():
    """List durable chat history for the current local installation."""
    return {"conversations": chat_store.list_conversations()}


@router.get("/conversations/{conversation_id}")
async def get_chat_conversation(conversation_id: str):
    """Load one durable conversation, including partial or cancelled messages."""
    conversation = chat_store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conversation


@router.patch("/conversations/{conversation_id}")
async def rename_chat_conversation(conversation_id: str, req: ConversationUpdate):
    if not chat_store.rename_conversation(conversation_id, req.title):
        raise HTTPException(status_code=404, detail="会话不存在")
    return chat_store.get_conversation(conversation_id)


@router.delete("/conversations/{conversation_id}")
async def delete_chat_conversation(conversation_id: str):
    if chat_store.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if not chat_store.delete_conversation(conversation_id):
        raise HTTPException(status_code=409, detail="会话仍有任务运行，无法删除")
    return {"conversation_id": conversation_id, "deleted": True}


@router.get("/widgets/strategies")
async def get_strategy_widget():
    """Get a complete A2UI strategy surface."""
    strategies = await asyncio.to_thread(list_strategies)
    return {"protocol": "a2ui", "catalog": CATALOG, "messages": render_strategy_selector(strategies)}


@router.get("/widgets/breakers")
async def get_breaker_widget():
    """Get a complete A2UI breaker-status surface."""
    return {"protocol": "a2ui", "catalog": CATALOG, "messages": render_breaker_status(get_breaker_status())}


@router.get("/widgets/mini_chart/{ticker}")
async def get_mini_chart_widget(ticker: str):
    """Get a native A2UI sparkline surface for a stock."""
    history = await async_get_stock_history(ticker)
    if history.empty:
        return {"protocol": "a2ui", "catalog": CATALOG, "messages": render_mini_chart([])}
    return {
        "protocol": "a2ui",
        "catalog": CATALOG,
        "messages": render_mini_chart(history.tail(30)["close"].tolist()),
    }


@router.get("/a2ui/catalog")
async def get_a2ui_catalog():
    """Publish the catalog understood by the native frontend renderer."""
    return CATALOG


@router.post("/a2ui/actions")
async def handle_a2ui_action(req: A2UIActionRequest):
    """Receive native A2UI actions without executing agent-provided code."""
    logger.info("[A2UI] action={} surface={} context_keys={}", req.name, req.surface_id, list(req.context))
    return {"accepted": True, "name": req.name, "surfaceId": req.surface_id}
