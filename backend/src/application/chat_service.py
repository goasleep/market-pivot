"""Durable chat history and server-owned cancellable chat tasks."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

from loguru import logger

from agents.asset_agent import asset_agent
from application.chat_store import ChatStore
from config import resolve_llm_profile
from llm_runtime import use_llm_profile
from widgets.a2ui import render_activity, render_markdown, render_tool_result


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


@dataclass(frozen=True)
class ChatTaskInput:
    task_id: str
    conversation_id: str
    message: str
    history: list[dict[str, Any]]
    strategy: str | None
    asset_type: Any
    assistant_message_id: str
    llm_profile_id: str | None = None
    llm_model: str | None = None
    llm_auto: bool = False


def _llm_route(intent: Any, message: str = "") -> str:
    if isinstance(intent, dict):
        message = str(intent.get("message", message))
        value = str(intent.get("intent", ""))
    else:
        value = getattr(intent, "value", str(intent))
    research_terms = (
        "分析", "回测", "策略", "对比", "行情", "历史", "新闻", "风险", "买入", "卖出",
        "基金", "股票", "etf", "lof",
    )
    return "analysis" if value in {"analyze", "backtest", "compare", "strategies"} or any(
        term in message.lower() for term in research_terms
    ) or any(char.isdigit() for char in message) else "chat"


class ChatTaskManager:
    """Run Agent tasks independently from any one SSE subscriber."""

    def __init__(self, store: ChatStore):
        self.store = store
        self._tasks: dict[str, asyncio.Task] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict | None]]] = {}
        self._task_locks: dict[str, asyncio.Lock] = {}
        self._worker: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    async def start_worker(self) -> None:
        if self._worker and not self._worker.done():
            return
        self._stopping = asyncio.Event()
        self._worker = asyncio.create_task(self._poll_runnable_tasks(), name="chat-task-worker")

    async def stop_worker(self) -> None:
        self._stopping.set()
        if self._worker:
            await self._worker
        self._worker = None
        task_ids = tuple(self._tasks)
        for task_id in task_ids:
            await self.store.update_task(task_id, "interrupted", "节点正在关闭，任务等待其他节点接管")
        tasks = tuple(self._tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _poll_runnable_tasks(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.store.recover_stale_tasks()
                for task_id in await self.store.list_runnable_tasks():
                    state = await self.store.get_task_state(task_id)
                    if state is None:
                        continue
                    await self.start(self._task_input_from_state(state))
            except Exception as exc:
                logger.warning("Chat task worker poll failed: {}", exc)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=2)
            except asyncio.TimeoutError:
                continue

    async def start(
        self,
        task_input: ChatTaskInput,
        resume_interaction: dict[str, Any] | None = None,
    ) -> None:
        if task_input.task_id in self._tasks:
            return
        record = await self.store.get_task(task_input.task_id)
        if record and record["status"] == "cancel_requested":
            await self.store.update_task(task_input.task_id, "cancelled")
            return
        if record and record["status"] in {
            "completed", "failed", "cancelled", "waiting_user", "superseded"
        }:
            return
        if not await self.store.begin_task(task_input.task_id):
            return
        self._subscribers.setdefault(task_input.task_id, set())
        self._task_locks.setdefault(task_input.task_id, asyncio.Lock())
        runner = self._run_resume(task_input, resume_interaction) if resume_interaction else self._run(task_input)
        self._tasks[task_input.task_id] = asyncio.create_task(
            runner,
            name=f"chat-agent{'-resume' if resume_interaction else ''}-{task_input.task_id}",
        )

    async def subscribe(self, task_id: str, after_sequence: int = 0) -> AsyncIterator[dict]:
        """Replay and poll durable events so any node can serve the SSE stream."""
        cursor = max(0, after_sequence)
        while True:
            events = await self.store.list_events(task_id, cursor)
            for event in events:
                cursor = max(cursor, int(event["id"]))
                yield event
            record = await self.store.get_task(task_id)
            terminal = bool(
                record
                and record["status"]
                in {"completed", "failed", "cancelled", "interrupted", "waiting_user", "superseded"}
            )
            if terminal and not events:
                if cursor == 0:
                    yield {"event": "done", "data": "{}"}
                return
            await asyncio.sleep(0.25)

    async def cancel(self, task_id: str) -> dict[str, Any]:
        task = self._tasks.get(task_id)
        status = await self.store.request_cancel(task_id)
        if status is None:
            return {"task_id": task_id, "status": "not_found"}
        if status != "cancel_requested":
            return {"task_id": task_id, "status": status}
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        else:
            await self.store.mark_cancelled(task_id)
        final_record = await self.store.get_task(task_id)
        return {"task_id": task_id, "status": final_record["status"] if final_record else "not_found"}

    async def _broadcast(self, task_id: str, event: dict | None) -> None:
        if event is None:
            for queue in tuple(self._subscribers.get(task_id, set())):
                queue.put_nowait(event)
            return
        lock = self._task_locks.setdefault(task_id, asyncio.Lock())
        async with lock:
            record = await self.store.get_task(task_id)
            if event.get("event") != "done" and record and record["status"] in {
                "cancel_requested", "cancelled", "completed", "failed", "interrupted", "superseded"
            }:
                return
            persisted = await self.store.append_event(task_id, event["event"], event["data"])
            for queue in tuple(self._subscribers.get(task_id, set())):
                queue.put_nowait(persisted)

    async def _emit_text(self, task_input: ChatTaskInput, text: str) -> None:
        """Emit all assistant prose as a native A2UI Markdown surface."""
        await self._emit_a2ui(task_input, render_markdown(text))

    async def _emit_a2ui(self, task_input: ChatTaskInput, messages: list[dict[str, Any]]) -> None:
        """Persist and stream one or more protocol-compliant A2UI messages."""
        for message in messages:
            if not await self.store.append_part(
                task_input.assistant_message_id,
                {"type": "a2ui", "content": message},
                task_input.task_id,
            ):
                raise asyncio.CancelledError
            await self._broadcast(
                task_input.task_id,
                {"event": "a2ui", "data": _json({"a2ui": message})},
            )

    async def _emit_artifact(self, task_input: ChatTaskInput, artifact: dict[str, Any]) -> None:
        """Persist and stream one generated file artifact."""
        if not await self.store.append_part(
            task_input.assistant_message_id,
            {"type": "artifact", "content": artifact},
            task_input.task_id,
        ):
            raise asyncio.CancelledError
        await self._broadcast(
            task_input.task_id,
            {"event": "artifact", "data": _json({"artifact": artifact})},
        )

    async def _emit_interaction(self, task_input: ChatTaskInput, event: dict[str, Any]) -> None:
        interaction = await self.store.create_interaction(
            task_id=task_input.task_id,
            kind=str(event.get("kind", "intent_clarification")),
            question=str(event.get("question", "请确认下一步操作")),
            options=list(event.get("options") or []),
            payload=dict(event.get("resume") or {}),
        )
        if event.get("tool"):
            interaction["tool"] = event["tool"]
        if not await self.store.append_part(
            task_input.assistant_message_id,
            {"type": "interaction", "content": interaction},
            task_input.task_id,
        ):
            raise asyncio.CancelledError
        await self.store.update_task(task_input.task_id, "waiting_user")
        await self._broadcast(
            task_input.task_id,
            {"event": "interaction_required", "data": _json({"interaction": interaction})},
        )

    @staticmethod
    def _tool_references(name: str, payload: Any) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []

        def collect(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    collect(item)
                return
            if not isinstance(value, dict):
                return
            link = value.get("link") or value.get("url")
            if isinstance(link, str) and link.startswith(("http://", "https://")):
                references.append(
                    {
                        "title": str(value.get("title") or value.get("name") or link),
                        "url": link,
                        "snippet": str(value["snippet"]) if value.get("snippet") else None,
                        "date": str(value["date"]) if value.get("date") else None,
                    }
                )
            for child in value.values():
                collect(child)

        collect(payload)
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for reference in references:
            key = str(reference.get("url"))
            if key in seen:
                continue
            seen.add(key)
            unique.append({key: value for key, value in reference.items() if value is not None})
        return unique

    async def _emit_tool_event(
        self,
        task_input: ChatTaskInput,
        event: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Render tool progress and, when possible, its structured result."""
        name = str(event.get("name", "unknown"))
        status = str(event.get("status", "completed"))
        result = event.get("result")
        error_message = ""
        if status == "failed" and result:
            try:
                payload = json.loads(str(result))
                error = payload.get("error", {}) if isinstance(payload, dict) else {}
                if isinstance(error, dict):
                    error_message = f"{error.get('code', 'tool_error')}: {error.get('message', '工具执行失败')}"
                else:
                    error_message = str(error)
            except (TypeError, json.JSONDecodeError):
                error_message = str(result)
        await self._emit_a2ui(task_input, render_activity(name, status, error=error_message))
        artifacts: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        if result:
            try:
                tool_payload = json.loads(str(result))
            except (TypeError, json.JSONDecodeError):
                tool_payload = {}
            for artifact in tool_payload.get("artifacts", []) if isinstance(tool_payload, dict) else []:
                if isinstance(artifact, dict):
                    artifacts.append(artifact)
            surface = render_tool_result(name, str(result))
            if surface:
                await self._emit_a2ui(task_input, surface)
            if status != "failed":
                references = self._tool_references(name, tool_payload)
        return artifacts, references

    @staticmethod
    def _queue_unique_artifacts(target: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> None:
        existing = {(item.get("name"), item.get("ticker"), item.get("asset_type")) for item in target}
        for artifact in artifacts:
            key = (artifact.get("name"), artifact.get("ticker"), artifact.get("asset_type"))
            if key in existing and any(key):
                continue
            target.append(artifact)
            existing.add(key)

    @staticmethod
    def _queue_unique_references(target: list[dict[str, Any]], references: list[dict[str, Any]]) -> None:
        existing = {str(item.get("url") or f"{item.get('title')}:{item.get('source')}") for item in target}
        for reference in references:
            key = str(reference.get("url") or f"{reference.get('title')}:{reference.get('source')}")
            if key in existing:
                continue
            target.append(reference)
            existing.add(key)

    async def _consume_agent_events(
        self,
        task_input: ChatTaskInput,
        events: AsyncIterator[dict[str, Any]],
    ) -> bool:
        """Consume one Agent stream; return True when it paused for a user."""
        pending_artifacts: list[dict[str, Any]] = []
        pending_references: list[dict[str, Any]] = []
        seen_tool_events: set[str] = set()
        async for event in events:
            if event.get("type") == "interaction_required":
                for artifact in pending_artifacts:
                    await self._emit_artifact(task_input, artifact)
                if pending_references:
                    await self.store.set_references(task_input.assistant_message_id, pending_references)
                    await self._broadcast(
                        task_input.task_id,
                        {"event": "references", "data": _json({"references": pending_references})},
                    )
                await self._emit_interaction(task_input, event)
                return True
            if event.get("type") == "tool":
                event_key = f"{event.get('name', 'unknown')}:{event.get('result', '')}"
                if event_key in seen_tool_events:
                    continue
                seen_tool_events.add(event_key)
                artifacts, references = await self._emit_tool_event(task_input, event)
                self._queue_unique_artifacts(pending_artifacts, artifacts)
                self._queue_unique_references(pending_references, references)
                continue
            text = event.get("text", "")
            if text:
                prefix = "分析摘要：" if event.get("type") == "reasoning" else ""
                await self._emit_text(task_input, f"{prefix}{text}")

        for artifact in pending_artifacts:
            await self._emit_artifact(task_input, artifact)
        if pending_references:
            await self.store.set_references(task_input.assistant_message_id, pending_references)
            await self._broadcast(
                task_input.task_id,
                {"event": "references", "data": _json({"references": pending_references})},
            )
        return False

    async def _run(self, task_input: ChatTaskInput) -> None:
        task_id = task_input.task_id
        heartbeat = asyncio.create_task(self._heartbeat(task_id), name=f"chat-heartbeat-{task_id}")
        try:
            await self._emit_text(task_input, "A-Share Agent：正在让模型判断任务并选择数据工具。")
            orchestration_surface = f"orchestration-{task_id}"
            await self._emit_a2ui(
                task_input,
                render_activity("Agent 正在规划任务并选择数据工具", "running", orchestration_surface),
            )
            request = asset_agent.prepare(
                message=task_input.message,
                history=task_input.history,
                strategy=task_input.strategy,
                conversation_id=task_input.conversation_id,
                task_id=task_input.task_id,
                asset_type=task_input.asset_type,
                llm_profile_id=task_input.llm_profile_id,
                llm_model=task_input.llm_model,
                llm_auto=task_input.llm_auto,
            )
            profile = resolve_llm_profile(
                task_input.llm_profile_id,
                task_input.llm_model,
                route=_llm_route(
                    request.get("intent") if isinstance(request, dict) else request.intent,
                    request.get("message", "") if isinstance(request, dict) else request.message,
                ),
                auto=task_input.llm_auto,
            )
            with use_llm_profile(profile):
                paused = await self._consume_agent_events(task_input, asset_agent.chat(request))
            if paused:
                return
            await self._emit_a2ui(
                task_input,
                render_activity(
                    "Agent 正在规划任务并选择数据工具",
                    "completed",
                    orchestration_surface,
                    include_create=False,
                ),
            )
            record = await self.store.get_task(task_id)
            if record is None or record["status"] == "cancel_requested":
                raise asyncio.CancelledError
            if not await self.store.complete_task(task_id):
                raise asyncio.CancelledError
            await self._broadcast(task_id, {"event": "done", "data": "{}"})
        except asyncio.CancelledError:
            logger.info("[AssetAgent] Chat task cancelled: {}", task_id)
            if await self.store.mark_cancelled(task_id):
                await self._broadcast(task_id, {"event": "done", "data": "{}"})
        except Exception as exc:
            logger.exception("[AssetAgent] Chat task failed: {}", exc)
            await self._emit_text(task_input, f"请求失败：{exc}")
            await self.store.update_task(task_id, "failed", str(exc))
            await self._broadcast(task_id, {"event": "done", "data": "{}"})
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            await self._broadcast(task_id, None)
            self._tasks.pop(task_id, None)
            self._subscribers.pop(task_id, None)
            self._task_locks.pop(task_id, None)

    async def respond(self, task_id: str, interaction_id: str, option_id: str) -> dict[str, Any]:
        """Answer one pending interaction and resume the durable Agent task."""
        record = await self.store.get_task(task_id)
        if record is None:
            raise KeyError("聊天任务不存在")
        if record["status"] != "waiting_user":
            raise ValueError("该聊天任务当前不在等待用户输入")
        interaction = await self.store.get_interaction(interaction_id)
        if interaction is None or interaction["task_id"] != task_id:
            raise KeyError("交互请求不属于该聊天任务")
        answered = await self.store.answer_interaction(interaction_id, option_id)
        state = await self.store.get_task_state(task_id)
        if state is None:
            raise ValueError("聊天任务缺少恢复上下文")
        await self.store.update_task(task_id, "pending")
        task_input = ChatTaskInput(
            task_id=task_id,
            conversation_id=str(state["conversation_id"]),
            message=str(state["message"]),
            history=list(state.get("history") or []),
            strategy=state.get("strategy"),
            asset_type=state.get("asset_type"),
            assistant_message_id=str(state["assistant_message_id"]),
            llm_profile_id=state.get("llm_profile_id"),
            llm_model=state.get("llm_model"),
            llm_auto=bool(state.get("llm_auto", False)),
        )
        await self.start(task_input, answered)
        return {"task_id": task_id, "status": "running", "interaction": answered}

    @staticmethod
    def _task_input_from_state(state: dict[str, Any]) -> ChatTaskInput:
        return ChatTaskInput(
            task_id=str(state["task_id"]),
            conversation_id=str(state["conversation_id"]),
            message=str(state["message"]),
            history=list(state.get("history") or []),
            strategy=state.get("strategy"),
            asset_type=state.get("asset_type"),
            assistant_message_id=str(state["assistant_message_id"]),
            llm_profile_id=state.get("llm_profile_id"),
            llm_model=state.get("llm_model"),
            llm_auto=bool(state.get("llm_auto", False)),
        )

    async def _run_resume(self, task_input: ChatTaskInput, interaction: dict[str, Any]) -> None:
        task_id = task_input.task_id
        heartbeat = asyncio.create_task(self._heartbeat(task_id), name=f"chat-heartbeat-{task_id}")
        try:
            await self._emit_text(task_input, "已收到你的选择，Agent 继续执行。")
            request_payload = (interaction.get("payload") or {}).get("request") or {}
            if hasattr(asset_agent, "request_from_payload"):
                request = asset_agent.request_from_payload(request_payload)
            else:
                request = {"intent": "chat"}
            profile = resolve_llm_profile(
                task_input.llm_profile_id,
                task_input.llm_model,
                route=_llm_route(
                    request.get("intent") if isinstance(request, dict) else request.intent,
                    request.get("message", "") if isinstance(request, dict) else request.message,
                ),
                auto=task_input.llm_auto,
            )
            with use_llm_profile(profile):
                paused = await self._consume_agent_events(
                    task_input,
                    asset_agent.resume_chat(interaction, str(interaction["selected_option"])),
                )
            if paused:
                return
            record = await self.store.get_task(task_id)
            if record is None or record["status"] == "cancel_requested":
                raise asyncio.CancelledError
            if not await self.store.complete_task(task_id):
                raise asyncio.CancelledError
            await self._broadcast(task_id, {"event": "done", "data": "{}"})
        except asyncio.CancelledError:
            logger.info("[AssetAgent] Chat task resume cancelled: {}", task_id)
            if await self.store.mark_cancelled(task_id):
                await self._broadcast(task_id, {"event": "done", "data": "{}"})
        except Exception as exc:
            logger.exception("[AssetAgent] Chat task resume failed: {}", task_id)
            await self._emit_text(task_input, f"请求失败：{exc}")
            await self.store.update_task(task_id, "failed", str(exc))
            await self._broadcast(task_id, {"event": "done", "data": "{}"})
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            await self._broadcast(task_id, None)
            self._tasks.pop(task_id, None)
            self._subscribers.pop(task_id, None)
            self._task_locks.pop(task_id, None)

    async def _heartbeat(self, task_id: str) -> None:
        while True:
            await asyncio.sleep(20)
            await self.store.touch_task(task_id)


chat_store = ChatStore()
chat_task_manager = ChatTaskManager(chat_store)
