"""Compatibility implementation for the asset-focused conversational agent.

The public ``AssetAgent`` alias handles stocks, ETFs, and LOFs while
delegating analysis to the existing multi-agent LangGraph workflow.
"""

import json
import re
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import StructuredTool
from loguru import logger

from agents.asset_requests import AssetAgentRequest, AssetIntent, AssetRequestResolver, RequestMode
from application.fund_response import execute_direct_fund_task
from application.fund_task_compiler import compile_fund_task, uses_direct_fund_executor
from application.research import research_service
from application.research_plan import research_plan_service
from graph.agent_loop import (
    get_agent_loop,
    resume_agent_loop,
    resume_checkpoint_agent_loop,
    resume_native_agent_loop,
    stream_agent_loop,
)
from graph.checkpointing import checkpoint_manager
from llm.context import select_conversation_history
from models.fund_task import FundTaskKind
from models.schemas import AssetType
from observability import build_trace_config
from tools.registry import build_artifact_tools, build_chat_tools, build_task_tools

_HTML_SOURCE_BLOCK = re.compile(
    r"```(?:html|xhtml)?\s*(?:<!doctype\s+html|<html\b).*?```",
    flags=re.IGNORECASE | re.DOTALL,
)
_RAW_HTML_SOURCE = re.compile(r"(?:<!doctype\s+html|<html\b).*", flags=re.IGNORECASE | re.DOTALL)


def _compact_generated_report(text: str) -> str:
    """Keep the chat concise when a report file has already been generated."""
    match = _HTML_SOURCE_BLOCK.search(text) or _RAW_HTML_SOURCE.search(text)
    notice = "完整 HTML 报告已生成文件产物，请点击下方卡片预览或下载。"
    if match is not None:
        lead = text[: match.start()].strip()
        tail = text[match.end() :].strip()
        return "\n\n".join(part for part in (lead, notice, tail) if part)
    if len(text) > 1600 or text.count("\n") > 18:
        return notice
    return text


class AssetAgent(AssetRequestResolver):
    """Route conversational requests to common asset research capabilities."""

    def _analysis_tool(
        self,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        *,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> StructuredTool:
        async def run_analysis(
            ticker: str,
            config: RunnableConfig,
            asset_type: Literal["stock", "etf", "lof"],
            strategy: str | None = None,
        ) -> str:
            """运行综合研究分析，适合用户要求趋势、买卖、风险或交易建议时使用。"""
            normalized_tickers = self.extract_tickers(ticker)
            if len(normalized_tickers) != 1:
                raise ValueError("ticker 必须是单个六位 A 股代码，例如 600519 或 510300")
            try:
                normalized_asset_type = AssetType(asset_type)
            except ValueError as exc:
                raise ValueError("asset_type 必须是 stock、etf 或 lof") from exc
            request = self.prepare(
                f"分析 {normalized_tickers[0]}",
                strategy=strategy,
                asset_type=normalized_asset_type.value,
                conversation_id=conversation_id,
                task_id=task_id,
            )
            if progress_callback is None:
                _, result = await self.analyze(request, config=config)
            else:
                result: dict[str, Any] = {}
                async for update in self.analyze_stream(
                    request,
                    config=config,
                    progress_callback=progress_callback,
                ):
                    result = update.get("state", result)
            decision = result.get("final_decision")
            if decision is None:
                return "{}"
            market_context = result.get("market_context")
            try:
                artifacts = await research_service.create_artifacts(
                    decision,
                    market_context,
                    source="chat-tool-analysis",
                    conversation_id=request.conversation_id,
                    task_id=request.task_id,
                    execution_key=f"{request.task_id}:comprehensive-report" if request.task_id else None,
                )
            except Exception as exc:
                logger.warning("Analysis report artifact generation failed; returning decision only: {}", exc)
                artifacts = []
            payload = research_service.decision_payload(decision, market_context, artifacts=artifacts)
            return json.dumps(payload, ensure_ascii=False)

        return StructuredTool.from_function(
            coroutine=run_analysis,
            name="run_fund_or_stock_analysis",
            description=(
                "运行短中期股票、ETF或LOF研究分析。只有用户明确需要分析、判断、策略或风险建议时调用。"
                "必须同时传入 ticker 和 asset_type；asset_type 只能是 stock、etf 或 lof。"
            ),
        )

    async def chat(
        self,
        request: AssetAgentRequest,
        *,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the LangGraph LLM/tool loop and stream its trace to the UI."""
        request, clarification = self.resolve_intent(request)
        if clarification:
            yield {
                "type": "interaction_required",
                **clarification,
                "resume": {"request": self.request_payload(request)},
            }
            return
        task_spec = compile_fund_task(
            request.message,
            tickers=request.tickers,
            asset_type=request.asset_type.value,
            mutation_requested=request.allow_mutating_tools,
        )
        if task_spec is not None and uses_direct_fund_executor(task_spec):
            yield {
                "type": "execution_metadata",
                "execution_version": 3,
                "graph_name": "fund-task-orchestrator",
                "thread_id": request.task_id,
                "task_spec": task_spec.model_dump(mode="json"),
            }
            yield {
                "type": "reasoning",
                "text": f"已识别为基金任务：{task_spec.task_kind.value}；本题不需要调用市场数据工具。",
            }
            answer, acceptance = await execute_direct_fund_task(request.message, task_spec)
            yield {
                "type": "task_outcome",
                "task_spec": task_spec.model_dump(mode="json"),
                "acceptance": acceptance.model_dump(mode="json"),
            }
            yield {"type": "text", "text": answer}
            return
        analysis_tool = self._analysis_tool(
            progress_callback,
            conversation_id=request.conversation_id,
            task_id=request.task_id,
        )
        artifact_tools = build_artifact_tools(
            conversation_id=request.conversation_id,
            task_id=request.task_id,
        )
        tools = (
            build_task_tools(
                task_spec,
                analysis_tool,
                artifact_tools=artifact_tools,
                allow_mutating_tools=request.allow_mutating_tools,
                conversation_id=request.conversation_id,
                task_id=request.task_id,
            )
            if task_spec is not None
            else build_chat_tools(
                analysis_tool,
                artifact_tools=artifact_tools,
                allow_mutating_tools=request.allow_mutating_tools,
                conversation_id=request.conversation_id,
                task_id=request.task_id,
            )
        )
        if (
            request.mode == RequestMode.FINANCIAL_RESEARCH
            and request.intent != AssetIntent.PORTFOLIO
            and (task_spec is None or task_spec.task_kind != FundTaskKind.UNIVERSE_RESEARCH)
            and not request.allow_mutating_tools
            and request.task_id
            and checkpoint_manager.saver is not None
        ):
            yield {
                "type": "execution_metadata",
                "execution_version": 2,
                "graph_name": "market-research-plan",
                "thread_id": request.task_id,
            }
            research_config = build_trace_config(
                "market-research-plan",
                tags=["asset-agent", "research-plan", request.intent.value],
                metadata={"intent": request.intent.value, "task_id": request.task_id},
                session_id=request.conversation_id,
            )
            async for event in research_plan_service.stream(
                self.research_request_payload(request),
                tools,
                config=research_config,
            ):
                yield event
            return
        system = (
            "你是 Fund Agent 的对话入口。用户意图已经通过系统闸门确认，你只执行该意图范围内的任务；"
            "禁止根据记忆编造行情、历史价格或新闻。行情、历史、新闻、对比和策略都必须通过工具获取。"
            "用户明确指定历史区间时，调用 get_historical_prices 必须传入对应的 start_date 和 end_date。"
            "当前价格、历史价格、成交量、净值、折溢价、技术指标和候选筛选属于结构化市场数据，"
            "必须使用行情、历史或筛选工具，不能用网页摘要代替。"
            "全市场筛选、排序、聚合和跨年度财务问题先调用 search_market_data_catalog 解析数据集与字段，"
            "仅当目录返回 available=true 时，才可使用其返回的 dataset_id 调用 query_market_data，"
            "执行结构化查询、受限变换和业务验收；目录返回 available=false 时必须停止并说明数据缺口。"
            "不能为新问题臆造数据集 ID 或工具名，"
            "也不能用 search_web 拼接精确数值表。"
            "当用户要求走势、对比或可视化时，优先获取结构化历史/行情数据；聊天界面会把已知工具结果自动渲染为图表或数据表，"
            "不要在文本中伪造数据，也不要输出 SVG/HTML 源码。"
            "需要网页正文时调用 fetch_web_content；需要财务或基金基础数据时调用 "
            "get_fundamentals 或 get_fund_nav_history；"
            "需要技术指标、风险计算、交易计划或回测时调用对应的原子工具，不要凭记忆计算。"
            "当用户询问投资理念、投资经验、市场观点、论文方法、策略依据或历史复盘方法时，"
            "调用 search_methodology 检索本地方法论库；方法论只能用于形成、解释和比较可验证假设，"
            "不能替代当前行情、结构化指标、风险计算或回测，也不能单独作为买卖结论。"
            "最新新闻、公告、行业事件、政策、国家队动向和基金催化属于资讯数据，统一调用 search_web；"
            "search_web 会在内部选择可用搜索来源并合并去重，Agent 不需要选择具体来源；"
            "搜索结果必须注明来源、数据日期和链接。股票、ETF、LOF 都可以使用网页搜索。"
            "如果用户要综合分析，调用 run_fund_or_stock_analysis，并且必须传入正确的 asset_type（stock、etf 或 lof）。"
            "artifact 是可独立预览、下载或留档的生成产物；长文、HTML、Markdown、PDF、"
            "JSON、CSV、图片和视频都属于 artifact。"
            "当用户要求报告、保存、下载，或内容已经适合独立阅读时，自行调用 save_artifacts；可以一次保存多个不同文件。"
            "保存文本时使用 content，保存 PDF、图片或视频时使用 content_base64；不要把完整 HTML 源码直接放进聊天回复。"
            "已有产物可用 list_artifacts 查看，文本内容可用 read_artifact 读取，"
            "结构化价格序列可用 create_chart_artifact 生成使用 ECharts canvas 的 HTML 图表文件。"
            "每次调用工具前可以先给出一句简短的公开分析摘要，说明接下来要核对什么；不要输出详细内部思维链。"
            "如果工具返回失败，先读取错误代码和消息：临时网络错误可原参数重试一次；参数、格式或能力错误必须调整参数或改用"
            "同一意图范围内的替代工具，不能原样重复失败调用；权限不足、用户拒绝或无法安全修复时停止并明确说明。"
            "完成工具调用后，用中文简洁回答；需要判断时直接给出首选建议、证据和适用条件，最终选择交给用户，"
            "不要用‘不存在唯一最好方案’、‘不同指标代表不同取舍’等常识性段落代替建议。"
            "如果工具结果包含 artifacts，说明报告文件已经生成；禁止再次输出 HTML 或 Markdown 源码，只需给出简短结论。"
            "明确数据日期、来源和数据缺失。产品只服务于小散户的短中期基金交易研究和模拟交易，不承诺收益，"
            "股票分析不能冒充基金建议。若只是闲聊或询问能力，可以直接回答。"
            "系统支持查询模拟盘账户、持仓和订单；只有用户明确要求时才可创建或取消模拟盘订单，"
            "所有模拟盘操作都必须明确说明是纸面交易，禁止声称已经进行实盘交易。"
        )
        system_message = {"role": "system", "content": system}
        current_message = {"role": "user", "content": request.message}
        history_selection = select_conversation_history(
            request.history,
            p0_messages=[system_message, current_message],
            tools=tools,
        )
        messages: list[Any] = [system_message, *history_selection.messages, current_message]
        final_response = ""
        artifacts_generated = False
        chat_config = build_trace_config(
            "asset-agent-chat",
            tags=["asset-agent", "chat", request.intent.value],
            metadata={"intent": request.intent.value},
            session_id=request.conversation_id,
        )
        native_checkpoints = bool(request.task_id and checkpoint_manager.saver is not None)
        if native_checkpoints:
            chat_config = checkpoint_manager.graph_config(request.task_id or "", chat_config)
        stream = (
            stream_agent_loop(
                messages,
                tools,
                max_steps=12 if task_spec is not None else 100,
                config=chat_config,
                native_interrupts=True,
                task_id=request.task_id,
            )
            if native_checkpoints
            else stream_agent_loop(messages, tools, max_steps=12 if task_spec is not None else 100, config=chat_config)
        )
        async for update in stream:
            native_interrupt = update.get("__interrupt__")
            if native_interrupt:
                value = getattr(native_interrupt[0], "value", {})
                if not isinstance(value, dict):
                    value = {}
                yield {
                    "type": "interaction_required",
                    "kind": "tool_confirmation",
                    "question": str(value.get("question") or "Agent 准备执行一个需要用户确认的工具操作，是否继续？"),
                    "options": [
                        {"id": "approve", "label": "确认执行"},
                        {"id": "reject", "label": "取消执行"},
                    ],
                    "resume": {
                        "native_checkpoint": True,
                        "graph_name": "asset-agent-chat",
                        "thread_id": request.task_id,
                        "interrupt_id": getattr(native_interrupt[0], "id", ""),
                    },
                    "tool": value,
                }
                return
            for node_update in update.values():
                if not isinstance(node_update, dict):
                    continue
                if node_update.get("pending_tool_confirmation"):
                    yield {
                        "type": "interaction_required",
                        "kind": "tool_confirmation",
                        "question": "Agent 准备执行一个需要用户确认的工具操作，是否继续？",
                        "options": [
                            {"id": "approve", "label": "确认执行"},
                            {"id": "reject", "label": "取消执行"},
                        ],
                        "resume": {
                            "request": self.request_payload(request),
                            "checkpoint_messages": node_update.get("checkpoint_messages", []),
                            "pending_tool_call": node_update["pending_tool_confirmation"],
                        },
                        "tool": node_update["pending_tool_confirmation"],
                    }
                    return
                for event in node_update.get("tool_events", []):
                    if event.get("name") in {"run_fund_or_stock_analysis", "save_artifacts"}:
                        try:
                            tool_payload = json.loads(str(event.get("result", "")))
                            artifacts_generated = artifacts_generated or bool(tool_payload.get("artifacts"))
                        except (TypeError, json.JSONDecodeError):
                            pass
                    yield {
                        "type": "tool",
                        "name": event.get("name", "unknown"),
                        "status": event.get("status", "completed"),
                        "result": event.get("result", ""),
                    }
                for event in node_update.get("reasoning_events", []):
                    if isinstance(event, dict) and event.get("text"):
                        yield {"type": "reasoning", "text": str(event["text"])}
                if node_update.get("final_response"):
                    final_response = node_update["final_response"]
        if final_response:
            text = _compact_generated_report(final_response) if artifacts_generated else final_response
            yield {"type": "text", "text": text}

    async def resume_checkpoint(self, request: AssetAgentRequest) -> AsyncIterator[dict[str, Any]]:
        """Continue a v2 task reclaimed after its worker lease expired."""
        if not request.task_id or checkpoint_manager.saver is None:
            async for event in self.chat(request):
                yield event
            return
        tools = build_chat_tools(
            self._analysis_tool(conversation_id=request.conversation_id, task_id=request.task_id),
            artifact_tools=build_artifact_tools(
                conversation_id=request.conversation_id,
                task_id=request.task_id,
            ),
            allow_mutating_tools=request.allow_mutating_tools,
            conversation_id=request.conversation_id,
            task_id=request.task_id,
        )
        config = checkpoint_manager.graph_config(
            request.task_id,
            build_trace_config(
                "asset-agent-chat-recover",
                tags=["asset-agent", "chat", request.intent.value],
                metadata={"intent": request.intent.value, "task_id": request.task_id},
                session_id=request.conversation_id,
            ),
        )
        snapshot = await get_agent_loop().aget_state(config)
        if not snapshot.values:
            async for event in self.chat(request):
                yield event
            return
        if not snapshot.next:
            final_response = str(snapshot.values.get("final_response") or "")
            if final_response:
                yield {"type": "text", "text": final_response}
            return
        async for update in resume_checkpoint_agent_loop(tools, config=config, task_id=request.task_id):
            native_interrupt = update.get("__interrupt__")
            if native_interrupt:
                value = getattr(native_interrupt[0], "value", {})
                if not isinstance(value, dict):
                    value = {}
                yield {
                    "type": "interaction_required",
                    "kind": "tool_confirmation",
                    "question": str(value.get("question") or "Agent 准备执行一个需要用户确认的工具操作，是否继续？"),
                    "options": [
                        {"id": "approve", "label": "确认执行"},
                        {"id": "reject", "label": "取消执行"},
                    ],
                    "resume": {
                        "native_checkpoint": True,
                        "graph_name": "asset-agent-chat",
                        "thread_id": request.task_id,
                        "interrupt_id": getattr(native_interrupt[0], "id", ""),
                    },
                    "tool": value,
                }
                return
            for node_update in update.values():
                if not isinstance(node_update, dict):
                    continue
                for event in node_update.get("tool_events", []):
                    yield {
                        "type": "tool",
                        "name": event.get("name", "unknown"),
                        "status": event.get("status", "completed"),
                        "result": event.get("result", ""),
                    }
                for event in node_update.get("reasoning_events", []):
                    if isinstance(event, dict) and event.get("text"):
                        yield {"type": "reasoning", "text": str(event["text"])}
                if node_update.get("final_response"):
                    yield {"type": "text", "text": str(node_update["final_response"])}

    async def resume_research_checkpoint(self, request: AssetAgentRequest) -> AsyncIterator[dict[str, Any]]:
        """Continue a reclaimed v2 Research Plan from its latest native checkpoint."""
        if not request.task_id or checkpoint_manager.saver is None:
            async for event in self.chat(request):
                yield event
            return
        tools = build_chat_tools(
            self._analysis_tool(conversation_id=request.conversation_id, task_id=request.task_id),
            artifact_tools=build_artifact_tools(
                conversation_id=request.conversation_id,
                task_id=request.task_id,
            ),
            allow_mutating_tools=False,
            conversation_id=request.conversation_id,
            task_id=request.task_id,
        )
        config = build_trace_config(
            "market-research-plan-recover",
            tags=["asset-agent", "research-plan", "recover"],
            metadata={"task_id": request.task_id},
            session_id=request.conversation_id,
        )
        async for event in research_plan_service.resume(
            self.research_request_payload(request),
            tools,
            config=config,
        ):
            yield event

    async def resume_chat(
        self,
        interaction: dict[str, Any],
        option_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Resume a persisted clarification or tool confirmation."""
        payload = interaction.get("payload") or {}
        request_payload = payload.get("request") or {}
        request = self.request_from_payload(request_payload)
        if interaction.get("kind") == "intent_clarification":
            request = request.with_intent(AssetIntent(option_id))
            async for event in self.chat(request):
                yield event
            return

        if interaction.get("kind") != "tool_confirmation":
            raise ValueError(f"不支持的交互类型: {interaction.get('kind')}")
        tools = build_chat_tools(
            self._analysis_tool(
                conversation_id=request.conversation_id,
                task_id=request.task_id,
            ),
            artifact_tools=build_artifact_tools(
                conversation_id=request.conversation_id,
                task_id=request.task_id,
            ),
            allow_mutating_tools=request.allow_mutating_tools,
            conversation_id=request.conversation_id,
            task_id=request.task_id,
        )
        approved = option_id == "approve"
        if payload.get("native_checkpoint"):
            thread_id = str(payload.get("thread_id") or request.task_id or "")
            if not thread_id or checkpoint_manager.saver is None:
                raise ValueError("原生 checkpoint 不可用，无法恢复该任务")
            resume_config = checkpoint_manager.graph_config(
                thread_id,
                build_trace_config(
                    "asset-agent-chat-resume",
                    tags=["asset-agent", "chat", request.intent.value],
                    metadata={"intent": request.intent.value, "task_id": request.task_id or ""},
                    session_id=request.conversation_id,
                ),
            )
            async for update in resume_native_agent_loop(
                tools,
                approved=approved,
                config=resume_config,
                task_id=request.task_id,
            ):
                native_interrupt = update.get("__interrupt__")
                if native_interrupt:
                    value = getattr(native_interrupt[0], "value", {})
                    if not isinstance(value, dict):
                        value = {}
                    yield {
                        "type": "interaction_required",
                        "kind": "tool_confirmation",
                        "question": str(
                            value.get("question") or "Agent 准备执行一个需要用户确认的工具操作，是否继续？"
                        ),
                        "options": [
                            {"id": "approve", "label": "确认执行"},
                            {"id": "reject", "label": "取消执行"},
                        ],
                        "resume": {
                            "native_checkpoint": True,
                            "graph_name": "asset-agent-chat",
                            "thread_id": thread_id,
                            "interrupt_id": getattr(native_interrupt[0], "id", ""),
                        },
                        "tool": value,
                    }
                    return
                for node_update in update.values():
                    if not isinstance(node_update, dict):
                        continue
                    for event in node_update.get("tool_events", []):
                        yield {
                            "type": "tool",
                            "name": event.get("name", "unknown"),
                            "status": event.get("status", "completed"),
                            "result": event.get("result", ""),
                        }
                    for event in node_update.get("reasoning_events", []):
                        if isinstance(event, dict) and event.get("text"):
                            yield {"type": "reasoning", "text": str(event["text"])}
                    if node_update.get("final_response"):
                        yield {"type": "text", "text": str(node_update["final_response"])}
            return
        if not approved:
            yield {
                "type": "tool",
                "name": str((payload.get("pending_tool_call") or {}).get("tool_name", "unknown")),
                "status": "failed",
                "result": json.dumps(
                    {"ok": False, "error": {"code": "user_denied", "message": "用户拒绝执行该工具"}},
                    ensure_ascii=False,
                ),
            }
            yield {"type": "text", "text": "已取消该工具操作，未执行任何订单或外部副作用。"}
            return
        async for update in resume_agent_loop(
            payload.get("checkpoint_messages") or [],
            tools,
            payload.get("pending_tool_call") or {},
            approved=approved,
            max_steps=100,
            config=build_trace_config(
                "asset-agent-chat-resume",
                tags=["asset-agent", "chat", request.intent.value],
                metadata={"intent": request.intent.value, "task_id": request.task_id or ""},
                session_id=request.conversation_id,
            ),
            task_id=request.task_id,
        ):
            for node_update in update.values():
                if not isinstance(node_update, dict):
                    continue
                if node_update.get("pending_tool_confirmation"):
                    yield {
                        "type": "interaction_required",
                        "kind": "tool_confirmation",
                        "question": "Agent 准备执行一个需要用户确认的工具操作，是否继续？",
                        "options": [
                            {"id": "approve", "label": "确认执行"},
                            {"id": "reject", "label": "取消执行"},
                        ],
                        "resume": {
                            "request": self.request_payload(request),
                            "checkpoint_messages": node_update.get("checkpoint_messages", []),
                            "pending_tool_call": node_update["pending_tool_confirmation"],
                        },
                        "tool": node_update["pending_tool_confirmation"],
                    }
                    return
                for event in node_update.get("tool_events", []):
                    yield {
                        "type": "tool",
                        "name": event.get("name", "unknown"),
                        "status": event.get("status", "completed"),
                        "result": event.get("result", ""),
                    }
                for event in node_update.get("reasoning_events", []):
                    if isinstance(event, dict) and event.get("text"):
                        yield {"type": "reasoning", "text": str(event["text"])}
                if node_update.get("final_response"):
                    yield {"type": "text", "text": str(node_update["final_response"])}

    async def analyze(
        self,
        request: AssetAgentRequest,
        *,
        config: RunnableConfig | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the multi-agent stock analysis workflow with tracing metadata."""
        if not request.ticker:
            raise ValueError("A stock code is required for analysis")

        ticker = request.ticker
        run_config = self._analysis_trace_config(request, config)
        result = await research_service.run(
            ticker,
            strategy=request.strategy,
            asset_type=request.asset_type,
            conversation_history=request.history,
            trace_config=run_config,
        )
        context = result.get("market_context")
        return context.realtime if context else {}, result

    async def analyze_stream(
        self,
        request: AssetAgentRequest,
        *,
        config: RunnableConfig | None = None,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream actual LangGraph node updates for the chat UI."""
        if not request.ticker:
            raise ValueError("A stock code is required for analysis")

        ticker = request.ticker
        run_config = self._analysis_trace_config(request, config)
        async for update in research_service.stream(
            ticker,
            strategy=request.strategy,
            asset_type=request.asset_type,
            conversation_history=request.history,
            trace_config=run_config,
        ):
            context = update.get("state", {}).get("market_context")
            event = {
                **update,
                "realtime": context.realtime if context else {},
                "data_status": context.data_status if context else {},
            }
            if progress_callback is not None:
                await progress_callback(event)
            yield event

    @staticmethod
    def _analysis_trace_config(
        request: AssetAgentRequest,
        config: RunnableConfig | None,
    ) -> RunnableConfig:
        """Reuse the parent chat trace when analysis is invoked as a tool."""
        if config is None:
            trace_config = build_trace_config(
                "asset-agent-analysis",
                tags=["asset-agent", "chat", request.intent.value],
                metadata={
                    "ticker": request.ticker or "",
                    "intent": request.intent.value,
                    "strategy": request.strategy or "auto",
                    "conversation_id": request.conversation_id or "",
                },
                session_id=request.conversation_id,
            )
        else:
            trace_config = {
                **config,
                "run_name": "asset-agent-analysis",
                "tags": [*config.get("tags", []), "asset-agent", "analysis"],
                "metadata": {
                    **config.get("metadata", {}),
                    "ticker": request.ticker or "",
                    "intent": request.intent.value,
                    "strategy": request.strategy or "auto",
                    "conversation_id": request.conversation_id or "",
                },
            }
        if request.task_id and checkpoint_manager.saver is not None:
            return checkpoint_manager.graph_config(
                f"{request.task_id}:research:comprehensive",
                trace_config,
            )
        return trace_config


# Backward-compatible names for persisted callers and existing integrations.
StockIntent = AssetIntent
StockAgentRequest = AssetAgentRequest
StockAgent = AssetAgent
asset_agent = AssetAgent()
stock_agent = asset_agent


def capabilities_text() -> str:
    """Return the user-facing capability summary."""
    return (
        "我是资产研究 Agent，专注于 A 股股票、ETF 和 LOF 的短中期研究与模拟交易辅助。你可以让我：\n\n"
        "- 分析股票或基金：`分析 ETF 510300`\n"
        "- 查询实时行情：`查询 510300 行情`\n"
        "- 查看历史走势：`查看 510300 K 线`\n"
        "- 查看基金净值与折溢价：`看看 510300 基金数据`\n"
        "- 回测历史表现：`回测 ETF 510300 最近三年表现`\n"
        "- 设计并回测策略：`设计一个控制回撤的策略并回测 510300`\n"
        "- 查看可用策略：`有哪些选股策略？`\n"
        "- 继续追问上一只股票：例如 `为什么这样判断？`\n\n"
        "回测结果会直接显示在当前对话中；组合管理仍可在 Portfolio 页面进行。"
        "我不会把股票研究结论冒充基金建议。"
    )
