"""Stock Agent chat API with SSE text and inline widgets."""

from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from application.chat_service import ChatTaskInput, chat_store, chat_task_manager
from application.chat_widgets import chat_widget_service
from models.schemas import AssetType

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


class ChatInteractionResponse(BaseModel):
    interaction_id: str = Field(..., min_length=1, max_length=255)
    option_id: str = Field(..., min_length=1, max_length=128)


@router.post("/send")
async def chat_send(req: ChatRequest):
    """Create a durable Agent task and stream its events to this subscriber."""
    logger.info(f"[AssetAgent] Message: {req.message}")
    conversation_id = req.conversation_id or f"conversation-{uuid4().hex}"
    task_id = req.task_id or f"task-{uuid4().hex}"
    history = [item.model_dump(exclude_none=True) for item in req.history]
    try:
        _, assistant_message_id = await chat_store.prepare_task(
            conversation_id=conversation_id,
            task_id=task_id,
            message=req.message,
            history=history,
            strategy=req.strategy,
            asset_type=req.asset_type.value if req.asset_type else None,
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


@router.post("/tasks/{task_id}/respond")
async def respond_chat_task(task_id: str, req: ChatInteractionResponse):
    """Answer a persisted Agent interaction and resume the same task."""
    try:
        return await chat_task_manager.respond(task_id, req.interaction_id, req.option_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
    result = await chat_store.get_task(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="聊天任务不存在")
    return result


@router.get("/tasks/{task_id}/stream")
async def reconnect_chat_task(request: Request, task_id: str, last_event_id: int = 0):
    """Reconnect a browser to an existing server-owned chat SSE task."""
    record = await chat_store.get_task(task_id)
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
async def list_chat_conversations(q: str | None = None):
    """List durable chat history for the current local installation."""
    return {"conversations": await chat_store.list_conversations(query=q)}


@router.get("/conversations/{conversation_id}")
async def get_chat_conversation(conversation_id: str):
    """Load one durable conversation, including partial or cancelled messages."""
    conversation = await chat_store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conversation


@router.patch("/conversations/{conversation_id}")
async def rename_chat_conversation(conversation_id: str, req: ConversationUpdate):
    if not await chat_store.rename_conversation(conversation_id, req.title):
        raise HTTPException(status_code=404, detail="会话不存在")
    return await chat_store.get_conversation(conversation_id)


@router.delete("/conversations/{conversation_id}")
async def delete_chat_conversation(conversation_id: str):
    if await chat_store.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if not await chat_store.delete_conversation(conversation_id):
        raise HTTPException(status_code=409, detail="会话仍有任务运行，无法删除")
    return {"conversation_id": conversation_id, "deleted": True}


@router.get("/widgets/strategies")
async def get_strategy_widget():
    """Get a complete A2UI strategy surface."""
    return await chat_widget_service.strategies()


@router.get("/widgets/breakers")
async def get_breaker_widget():
    """Get a complete A2UI breaker-status surface."""
    return await chat_widget_service.breakers()


@router.get("/widgets/mini_chart/{ticker}")
async def get_mini_chart_widget(ticker: str):
    """Get a native A2UI sparkline surface for a stock."""
    return await chat_widget_service.mini_chart(ticker)


@router.get("/a2ui/catalog")
async def get_a2ui_catalog():
    """Publish the catalog understood by the native frontend renderer."""
    return chat_widget_service.catalog()


@router.post("/a2ui/actions")
async def handle_a2ui_action(req: A2UIActionRequest):
    """Receive native A2UI actions without executing agent-provided code."""
    logger.info("[A2UI] action={} surface={} context_keys={}", req.name, req.surface_id, list(req.context))
    return {"accepted": True, "name": req.name, "surfaceId": req.surface_id}
