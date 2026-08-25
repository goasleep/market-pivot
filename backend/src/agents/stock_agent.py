"""Compatibility implementation for the asset-focused conversational agent.

The public ``AssetAgent`` alias handles stocks, ETFs, and LOFs while
delegating analysis to the existing multi-agent LangGraph workflow.
"""

import json
import re
from dataclasses import replace
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import StructuredTool
from loguru import logger

from agents.asset_requests import AssetAgentRequest, AssetIntent, AssetRequestResolver
from application.research import research_service
from application.research_plan import research_plan_service
from application.task_contract import classify_task_execution, compile_task_contract
from graph.agent_loop import (
    get_agent_loop,
    resume_agent_loop,
    resume_checkpoint_agent_loop,
    resume_native_agent_loop,
    stream_agent_loop,
)
from graph.checkpointing import checkpoint_manager
from llm.context import select_conversation_history
from models.schemas import AssetType
from models.supervisor import ExecutionMode, TaskRoutingDecision
from observability import build_trace_config
from tools.registry import build_artifact_tools, build_chat_tools

_HTML_SOURCE_BLOCK = re.compile(
    r"```(?:html|xhtml)?\s*(?:<!doctype\s+html|<html\b).*?```",
    flags=re.IGNORECASE | re.DOTALL,
)
_RAW_HTML_SOURCE = re.compile(r"(?:<!doctype\s+html|<html\b).*", flags=re.IGNORECASE | re.DOTALL)


_GENERIC_ARTIFACT_NOTICE = re.compile(r"完整\s*HTML\s*报告已生成文件产物", flags=re.IGNORECASE)


def _artifact_label(artifact: dict[str, Any]) -> str:
    mime_type = str(artifact.get("mime_type") or "")
    return {
        "text/markdown": "Markdown",
        "text/html": "HTML",
        "application/pdf": "PDF",
        "text/csv": "CSV",
        "application/json": "JSON",
    }.get(mime_type, "文件")


def _compact_generated_report(text: str, artifacts: list[dict[str, Any]] | None = None) -> str:
    """Remove embedded source while retaining a useful chat-native conclusion."""
    artifacts = artifacts or []
    artifact = artifacts[0] if artifacts else {}
    name = str(artifact.get("name") or "完整报告")
    label = _artifact_label(artifact)
    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    description = str(metadata.get("description") or "").strip()
    match = _HTML_SOURCE_BLOCK.search(text) or _RAW_HTML_SOURCE.search(text)
    if match is not None:
        lead = text[: match.start()].strip()
        tail = text[match.end() :].strip()
        text = "\n\n".join(part for part in (lead, tail) if part)
    if _GENERIC_ARTIFACT_NOTICE.search(text) and len(text.strip()) < 200:
        text = description
    if len(text) > 1800:
        prefix = text[:1600]
        boundary = max(prefix.rfind("\n\n"), prefix.rfind("。"), prefix.rfind("；"))
        text = prefix[: boundary + 1].strip() if boundary >= 400 else prefix.rstrip()
        text = f"{text}\n\n（正文已节选，完整内容保存在附件中。）"
    if not text.strip():
        text = description or "报告已经生成，核心结论和完整明细请查看附件。"
    notice = f"完整内容见下方附件：{name}（{label}），可直接预览或下载。"
    return "\n\n".join(part for part in (text.strip(), notice) if part)


def _select_tools_for_routing(
    all_tools: list[StructuredTool],
    artifact_tools: list[StructuredTool],
    routing: TaskRoutingDecision,
) -> list[StructuredTool]:
    """Enforce the execution surface chosen by the routing model."""
    if routing.mode == ExecutionMode.DIRECT_RESPONSE:
        return []
    if routing.mode == ExecutionMode.ARTIFACT_GENERATION:
        return list(artifact_tools)
    if routing.mode in {ExecutionMode.BACKTEST_EXECUTION, ExecutionMode.MIXED_WORKFLOW}:
        return [tool for tool in all_tools if tool.name != "read_artifact"]
    return list(all_tools)


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
                report_artifacts = await research_service.create_artifacts(
                    decision,
                    market_context,
                    source="chat-tool-analysis",
                    conversation_id=request.conversation_id,
                    task_id=request.task_id,
                    execution_key=f"{request.task_id}:comprehensive-report" if request.task_id else None,
                )
            except Exception as exc:
                logger.warning("Analysis report artifact generation failed; returning decision only: {}", exc)
                report_artifacts = []
            artifacts = [*(result.get("visual_artifacts") or []), *report_artifacts]
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

    def _research_plan_tool(
        self,
        request: AssetAgentRequest,
        tools: list[StructuredTool],
    ) -> StructuredTool:
        async def run_research_plan(
            objective: str,
            config: RunnableConfig,
            ticker: str | None = None,
            asset_type: Literal["stock", "etf", "lof"] | None = None,
        ) -> str:
            """把需要多步骤取证、校验和综合的复杂研究交给 ResearchPlan 子能力。"""
            child_request = replace(
                request,
                message=objective.strip() or request.message,
                tickers=(ticker,) if ticker else request.tickers,
                asset_type=AssetType(asset_type) if asset_type else request.asset_type,
                task_id=f"{request.task_id}:research" if request.task_id else "supervisor-research",
                allow_mutating_tools=False,
            )
            final_text = ""
            plan: dict[str, Any] = {}
            observations: list[dict[str, Any]] = []
            async for event in research_plan_service.stream(
                self.research_request_payload(child_request),
                tools,
                config=config,
            ):
                if event.get("type") == "text":
                    final_text = str(event.get("text") or "")
                elif event.get("type") == "plan_update" and isinstance(event.get("plan"), dict):
                    plan = event["plan"]
                elif event.get("type") == "tool":
                    observations.append(
                        {
                            "name": event.get("name"),
                            "status": event.get("status"),
                            "result": str(event.get("result") or "")[:4000],
                        }
                    )
            return json.dumps(
                {"final_response": final_text, "plan": plan, "observations": observations[-20:]},
                ensure_ascii=False,
            )

        return StructuredTool.from_function(
            coroutine=run_research_plan,
            name="run_research_plan",
            description=(
                "执行需要多步骤数据获取、依赖排序、验证和综合的研究子任务。"
                "简单查询优先直接调用原子工具；复杂比较或多来源研究可调用本工具。结果必须返回 Supervisor 再判断。"
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
        routing = await classify_task_execution(
            request.message,
            tickers=request.tickers,
            asset_type=request.asset_type.value,
            mutation_requested=request.allow_mutating_tools,
        )
        task_contract = compile_task_contract(
            request.message,
            tickers=request.tickers,
            asset_type=request.asset_type.value,
            mutation_requested=request.allow_mutating_tools,
            routing_decision=routing,
        )
        yield {
            "type": "execution_metadata",
            "graph_name": "supervisor-agent",
            "thread_id": request.task_id,
            "task_contract": task_contract.model_dump(mode="json"),
        }
        deliverable_text = "、".join(task_contract.deliverables[:4]) or "直接回答请求"
        yield {
            "type": "progress",
            "text": (
                f"阶段性结果：模型已将任务判定为 {routing.mode.value}；"
                f"本轮优先交付：{deliverable_text}。"
            ),
        }
        analysis_tool = self._analysis_tool(
            progress_callback,
            conversation_id=request.conversation_id,
            task_id=request.task_id,
        )
        artifact_tools = build_artifact_tools(
            conversation_id=request.conversation_id,
            task_id=request.task_id,
        )
        all_tools = build_chat_tools(
            analysis_tool,
            artifact_tools=artifact_tools,
            allow_mutating_tools=request.allow_mutating_tools,
            conversation_id=request.conversation_id,
            task_id=request.task_id,
        )
        tools = _select_tools_for_routing(all_tools, artifact_tools, routing)
        if routing.allow_research_plan and tools:
            tools.append(self._research_plan_tool(request, list(tools)))
        routing_payload = json.dumps(routing.model_dump(mode="json"), ensure_ascii=False)
        system = (
            "你是系统中唯一的 Supervisor Agent。你负责理解任务、选择行动、执行原子工具、必要时委派研究子能力、"
            "整合证据并完成回答。当前模型路由决定是：" + routing_payload + "。"
            "必须采用交付优先：先完成用户明确要求的最小交付项；一旦已有足够证据形成诚实、有用、可执行的回答，立即交付。"
            "可选证据缺失时明确披露，不要为了追求完美而重复查询、重新运行已完成步骤或阻塞最终回答。"
            "只有关键交付项仍缺失且现有工具可以取得时才继续执行。"
            "简单任务由你直接完成；复杂、多步骤研究可调用 run_research_plan，综合趋势与风险分析可调用 "
            "run_fund_or_stock_analysis；所有子能力结果都必须回到你这里再综合。"
            "用户只给出宽泛产品类别时，主动选择可验证且有代表性的样本，明确披露样本及选择依据；"
            "只有无法可靠找到候选时才向用户追问。用户意图已经通过系统闸门确认，你只执行该意图范围内的任务；"
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
            "回测工具会直接返回 Supervisor 所需的核心指标、成本情景、区间、数据口径、验收结论和限制；"
            "必须直接使用这些结构化字段解释结果。回测附件只供用户下载审计，禁止为了形成聊天正文调用 "
            "read_artifact 或 list_artifacts 读取完整回测文件。"
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
            "非回测任务的已有产物可用 list_artifacts 查看，文本内容可用 read_artifact 读取，"
            "结构化价格序列可用 create_chart_artifact 生成使用 ECharts canvas 的 HTML 图表文件。"
            "每次调用工具前可以先给出一句简短的公开分析摘要，说明接下来要核对什么；不要输出详细内部思维链。"
            "如果工具返回失败，先读取错误代码和消息：临时网络错误可原参数重试一次；参数、格式或能力错误必须调整参数或改用"
            "同一意图范围内的替代工具，不能原样重复失败调用；权限不足、用户拒绝或无法安全修复时停止并明确说明。"
            "完成工具调用后，用中文简洁回答；需要判断时直接给出首选建议、证据和适用条件，最终选择交给用户，"
            "不要用‘不存在唯一最好方案’、‘不同指标代表不同取舍’等常识性段落代替建议。"
            "如果工具结果包含 artifacts，正文仍必须给出结论、关键依据、限制和下一步的简短摘要；"
            "附件只能承载完整明细，不能代替聊天正文。禁止再次输出 HTML 或 Markdown 源码。"
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
        completion_result: dict[str, Any] = {}
        generated_artifacts: list[dict[str, Any]] = []
        chat_config = build_trace_config(
            "supervisor-agent",
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
                max_steps=16,
                config=chat_config,
                native_interrupts=True,
                task_id=request.task_id,
                task_contract=task_contract.model_dump(mode="json"),
            )
            if native_checkpoints
            else stream_agent_loop(
                messages,
                tools,
                max_steps=16,
                config=chat_config,
                task_contract=task_contract.model_dump(mode="json"),
            )
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
                        "graph_name": "supervisor-agent",
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
                            generated_artifacts.extend(
                                item for item in tool_payload.get("artifacts", []) if isinstance(item, dict)
                            )
                        except (TypeError, json.JSONDecodeError):
                            pass
                    yield {
                        "type": "tool",
                        "name": event.get("name", "unknown"),
                        "status": event.get("status", "completed"),
                        "result": event.get("result", ""),
                    }
                    yield {
                        "type": "progress",
                        "text": (
                            f"阶段性结果：工具 {event.get('name', 'unknown')} 已"
                            f"{'完成' if event.get('status', 'completed') == 'completed' else '结束但未取得有效结果'}；"
                            "已获得的结构化结果已保留，Supervisor 正在优先整理可交付结论。"
                        ),
                    }
                for event in node_update.get("reasoning_events", []):
                    if isinstance(event, dict) and event.get("text"):
                        yield {"type": "reasoning", "text": str(event["text"])}
                if isinstance(node_update.get("completion_result"), dict):
                    completion_result = node_update["completion_result"]
                if node_update.get("final_response"):
                    final_response = node_update["final_response"]
        if completion_result:
            yield {
                "type": "task_outcome",
                "task_contract": task_contract.model_dump(mode="json"),
                "acceptance": completion_result,
            }
        if final_response:
            text = (
                _compact_generated_report(final_response, generated_artifacts)
                if generated_artifacts
                else final_response
            )
            yield {"type": "text", "text": text}

    async def resume_checkpoint(self, request: AssetAgentRequest) -> AsyncIterator[dict[str, Any]]:
        """Continue a Supervisor task reclaimed after its worker lease expired."""
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
        tools.append(self._research_plan_tool(request, tools))
        config = checkpoint_manager.graph_config(
            request.task_id,
            build_trace_config(
                "supervisor-agent-recover",
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
            completion_result = snapshot.values.get("completion_result")
            if isinstance(completion_result, dict):
                yield {
                    "type": "task_outcome",
                    "task_contract": snapshot.values.get("task_contract") or {},
                    "acceptance": completion_result,
                }
            if final_response:
                yield {"type": "text", "text": final_response}
            return
        completion_result: dict[str, Any] = {}
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
                        "graph_name": "supervisor-agent",
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
                if isinstance(node_update.get("completion_result"), dict):
                    completion_result = node_update["completion_result"]
                if node_update.get("final_response"):
                    yield {"type": "text", "text": str(node_update["final_response"])}
        if completion_result:
            yield {
                "type": "task_outcome",
                "task_contract": compile_task_contract(
                    request.message,
                    tickers=request.tickers,
                    asset_type=request.asset_type.value,
                    mutation_requested=request.allow_mutating_tools,
                ).model_dump(mode="json"),
                "acceptance": completion_result,
            }

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
        tools.append(self._research_plan_tool(request, tools))
        approved = option_id == "approve"
        if payload.get("native_checkpoint"):
            thread_id = str(payload.get("thread_id") or request.task_id or "")
            if not thread_id or checkpoint_manager.saver is None:
                raise ValueError("原生 checkpoint 不可用，无法恢复该任务")
            resume_config = checkpoint_manager.graph_config(
                thread_id,
                build_trace_config(
                    "supervisor-agent-resume",
                    tags=["asset-agent", "chat", request.intent.value],
                    metadata={"intent": request.intent.value, "task_id": request.task_id or ""},
                    session_id=request.conversation_id,
                ),
            )
            completion_result: dict[str, Any] = {}
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
                            "graph_name": "supervisor-agent",
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
                    if isinstance(node_update.get("completion_result"), dict):
                        completion_result = node_update["completion_result"]
                    if node_update.get("final_response"):
                        yield {"type": "text", "text": str(node_update["final_response"])}
            if completion_result:
                yield {
                    "type": "task_outcome",
                    "task_contract": compile_task_contract(
                        request.message,
                        tickers=request.tickers,
                        asset_type=request.asset_type.value,
                        mutation_requested=request.allow_mutating_tools,
                    ).model_dump(mode="json"),
                    "acceptance": completion_result,
                }
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
            yield {
                "type": "task_outcome",
                "task_contract": compile_task_contract(
                    request.message,
                    tickers=request.tickers,
                    asset_type=request.asset_type.value,
                    mutation_requested=request.allow_mutating_tools,
                ).model_dump(mode="json"),
                "acceptance": {
                    "outcome": "partial",
                    "satisfied": False,
                    "terminal": True,
                    "missing": ["用户拒绝的工具操作"],
                    "reason": "尊重用户拒绝，未执行有副作用的操作",
                },
            }
            return
        completion_result = {}
        async for update in resume_agent_loop(
            payload.get("checkpoint_messages") or [],
            tools,
            payload.get("pending_tool_call") or {},
            approved=approved,
            max_steps=100,
            config=build_trace_config(
                "supervisor-agent-resume",
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
                if isinstance(node_update.get("completion_result"), dict):
                    completion_result = node_update["completion_result"]
                if node_update.get("final_response"):
                    yield {"type": "text", "text": str(node_update["final_response"])}
        if completion_result:
            yield {
                "type": "task_outcome",
                "task_contract": compile_task_contract(
                    request.message,
                    tickers=request.tickers,
                    asset_type=request.asset_type.value,
                    mutation_requested=request.allow_mutating_tools,
                ).model_dump(mode="json"),
                "acceptance": completion_result,
            }

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
            conversation_id=request.conversation_id,
            task_id=request.task_id,
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
            conversation_id=request.conversation_id,
            task_id=request.task_id,
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
