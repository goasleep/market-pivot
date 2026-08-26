"""Single public Financial Harness Agent for stock, exchange-fund and open-fund research."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any, AsyncIterator, Awaitable, Callable

from langchain_core.tools import StructuredTool

from agents.asset_requests import AssetAgentRequest, AssetRequestResolver
from agents.stock_agent import _compact_generated_report
from agents.stock_executor import stock_comprehensive_executor
from application.task_contract import classify_task_execution
from graph.agent_loop import resume_native_agent_loop, stream_agent_loop
from graph.checkpointing import checkpoint_manager
from harness.bootstrap import build_default_catalog, load_default_skills
from harness.context import context_assembler
from harness.evidence import EvidenceStore
from harness.graph import prepare_harness_plan
from harness.models import AcceptanceResult, HarnessPlan, HarnessTaskContract, SkillManifest, ValidatorResult
from harness.validators import covered_capabilities
from llm.context import select_conversation_history
from models.schemas import AssetType
from observability import build_trace_config
from tools.registry import build_artifact_tools, build_named_tools

HARNESS_GRAPH_NAME = "financial-harness"
HARNESS_RUNTIME_VERSION = "2.0.0"


async def _stream_with_deadline(stream: AsyncIterator[dict[str, Any]], seconds: int) -> AsyncIterator[dict[str, Any]]:
    try:
        async with asyncio.timeout(seconds):
            async for update in stream:
                yield update
    except TimeoutError:
        yield {"__harness_timeout__": True}


def _public_plan(plan: HarnessPlan, *, statuses: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    statuses = statuses or {}
    steps = []
    for step in plan.steps:
        state = statuses.get(step.id, {})
        steps.append(
            {
                "id": step.id,
                "kind": step.capability_id,
                "capability_id": step.capability_id,
                "skill_id": step.skill_id,
                "title": step.title,
                "status": state.get("status", "pending"),
                "evidence_status": state.get("evidence_status", "not_assessed"),
                "as_of": state.get("as_of"),
                "error": state.get("error"),
            }
        )
    completed = sum(item["status"] in {"completed", "failed", "skipped"} for item in steps)
    return {
        "plan_id": plan.plan_id,
        "objective": plan.objective,
        "depth": plan.budget_profile,
        "revision": plan.revision,
        "status": "completed" if steps and completed == len(steps) else "running",
        "progress": round(completed / len(steps) * 100) if steps else 100,
        "selected_skills": list(plan.selected_skills),
        "steps": steps,
    }


class FinancialHarnessAgent(AssetRequestResolver):
    """Compile, constrain, execute, verify and synthesize one financial task."""

    def _build_runtime_tools(
        self,
        request: AssetAgentRequest,
        tool_names: set[str],
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> list[StructuredTool]:
        analysis = stock_comprehensive_executor.build_tool(
            progress_callback,
            conversation_id=request.conversation_id,
            task_id=request.task_id,
        )
        artifacts = build_artifact_tools(conversation_id=request.conversation_id, task_id=request.task_id)
        return build_named_tools(
            tool_names,
            analysis,
            artifact_tools=artifacts,
            allow_mutating_tools=request.allow_mutating_tools,
            conversation_id=request.conversation_id,
            task_id=request.task_id,
        )

    async def resume_checkpoint(self, request: AssetAgentRequest) -> AsyncIterator[dict[str, Any]]:
        """Recover read-only work safely; never replay an uncertain write side effect."""
        if request.allow_mutating_tools:
            yield {
                "type": "task_outcome",
                "task_contract": {},
                "acceptance": {
                    "outcome": "partial",
                    "satisfied": False,
                    "terminal": True,
                    "missing": ["写操作 checkpoint 的明确审批状态"],
                    "reason": "为避免重复副作用，未自动重放中断的写操作",
                },
                "asset_type": request.asset_type.value,
            }
            yield {"type": "text", "text": "任务在写操作期间中断。为避免重复执行，未自动重放；请重新发起并确认。"}
            return
        async for event in self.chat(request):
            yield event

    async def chat(
        self,
        request: AssetAgentRequest,
        *,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
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
        kernel = await prepare_harness_plan(
            self.request_payload(request),
            routing,
            task_id=request.task_id,
            dynamic_planning=True,
        )
        contract = HarnessTaskContract.model_validate(kernel["contract"])
        catalog = build_default_catalog()
        registry = load_default_skills(catalog=catalog)
        skills = tuple(SkillManifest.model_validate(item) for item in kernel["selected_skills"])
        plan = HarnessPlan.model_validate(kernel["plan"])
        tool_names = {name for skill in skills for name in skill.tools}
        tools = self._build_runtime_tools(request, tool_names, progress_callback)
        catalog.bind(tools)
        yield {
            "type": "execution_metadata",
            "graph_name": HARNESS_GRAPH_NAME,
            "thread_id": request.task_id,
            "runtime_version": HARNESS_RUNTIME_VERSION,
            "contract_version": contract.contract_version,
            "skill_versions": {skill.id: skill.version for skill in skills},
            "task_contract": contract.model_dump(mode="json"),
        }
        statuses: dict[str, dict[str, Any]] = {}
        yield {"type": "plan_update", "plan": _public_plan(plan), "create": True}
        yield {
            "type": "progress",
            "text": f"Harness 已选择 {len(skills)} 个公开能力，工具面收敛为 {len(tools)} 个可信工具。",
        }

        system = context_assembler.assemble(
            contract,
            plan,
            skills,
            approval_status="required" if contract.allow_mutations else "not_required",
        )
        system += (
            "\n你是 Financial Harness 的综合执行器。严格按任务合同完成最小交付；只能调用当前提供的工具。"
            "精确金融数值不得凭记忆或网页摘要生成。工具失败或数据不足时返回部分结果并列出缺口，禁止无限重规划。"
            "回答必须包含数据日期、来源、限制；ETF 评分只用于筛选，不构成买入或实盘指令。不要输出内部思维链。"
        )
        selection = (contract.source_task_spec or {}).get("selection_requirements") or {}
        if selection.get("selection_mode") == "rank":
            comparison_tool = (
                "screen_compare_open_funds" if contract.asset_type == "open_fund" else "screen_compare_exchange_funds"
            )
            system += (
                f"\n本任务要求正式筛选：完成候选数据准备后必须调用 {comparison_tool}。"
                "只有该工具返回 ranking_is_formal=true 才能宣称正式排名或任务已满足；"
                "否则必须明确标为条件性比较/部分结果。"
            )
        if "strategy.sandbox_research" in contract.required_capabilities:
            system += (
                "\n本任务明确要求代码策略沙箱研究，必须调用 design_and_run_sandbox_strategy。"
                "必须展示生成源码、沙箱验证和可信交易引擎回放结果；不得改用普通回测工具，"
                "不得把研究回放描述为模拟盘部署或实盘执行。"
            )
        system_message = {"role": "system", "content": system}
        user_message = {"role": "user", "content": request.message}
        history = select_conversation_history(
            request.history,
            p0_messages=[system_message, user_message],
            tools=tools,
        )
        messages: list[Any] = [system_message, *history.messages, user_message]
        trace = build_trace_config(
            HARNESS_GRAPH_NAME,
            tags=["financial-harness", request.asset_type.value, request.intent.value],
            metadata={
                "contract_id": contract.contract_id,
                "runtime_version": HARNESS_RUNTIME_VERSION,
                "skill_ids": list(plan.selected_skills),
            },
            session_id=request.conversation_id,
        )
        native_checkpoints = bool(request.task_id and checkpoint_manager.saver is not None)
        if native_checkpoints:
            trace = checkpoint_manager.graph_config(request.task_id or "", trace)
        contract_payload = {
            **contract.model_dump(mode="json"),
            "requires_tools": bool(contract.evidence_requirements or contract.required_capabilities),
            "budget": contract.budget.model_dump(mode="json"),
            "compact_tool_results": True,
        }
        stream = stream_agent_loop(
            messages,
            tools,
            max_steps=contract.budget.max_steps,
            config=trace,
            native_interrupts=native_checkpoints,
            task_id=request.task_id,
            task_contract=contract_payload,
        )
        evidence = EvidenceStore()
        final_response = ""
        semantic_acceptance: dict[str, Any] = {}
        generated_artifacts: list[dict[str, Any]] = []
        tool_to_step = {tool_name: step for step in plan.steps for tool_name in step.tool_names}
        async for update in _stream_with_deadline(stream, contract.budget.deadline_seconds):
            if update.get("__harness_timeout__"):
                semantic_acceptance = {
                    "outcome": "partial",
                    "satisfied": False,
                    "terminal": True,
                    "missing": ["执行预算内未完成的步骤"],
                    "reason": f"Harness 达到 {contract.budget.deadline_seconds} 秒总时限",
                }
                final_response = final_response or "任务已达到执行时限；以下验收结果列出了已完成证据和缺失项。"
                break
            native_interrupt = update.get("__interrupt__")
            if native_interrupt:
                value = getattr(native_interrupt[0], "value", {})
                value = value if isinstance(value, dict) else {}
                yield {
                    "type": "interaction_required",
                    "kind": "tool_confirmation",
                    "question": str(value.get("question") or "Agent 准备执行模拟盘写操作，是否继续？"),
                    "options": [{"id": "approve", "label": "确认执行"}, {"id": "reject", "label": "取消执行"}],
                    "resume": {
                        "native_checkpoint": True,
                        "graph_name": HARNESS_GRAPH_NAME,
                        "thread_id": request.task_id,
                        "interrupt_id": getattr(native_interrupt[0], "id", ""),
                        "request": self.request_payload(request),
                    },
                    "tool": value,
                }
                return
            for node_update in update.values():
                if not isinstance(node_update, dict):
                    continue
                for event in node_update.get("tool_events", []):
                    tool_name = str(event.get("name") or "unknown")
                    raw_result = str(event.get("result") or "")
                    step = tool_to_step.get(tool_name)
                    if step and event.get("status", "completed") == "completed":
                        record = evidence.add_tool_result(step.capability_id, tool_name, raw_result)
                        statuses[step.id] = {
                            "status": "completed",
                            "evidence_status": record.status,
                            "as_of": record.as_of,
                        }
                    elif step:
                        statuses[step.id] = {
                            "status": "failed",
                            "evidence_status": "unavailable",
                            "error": "工具未取得有效结果",
                        }
                    try:
                        payload = json.loads(raw_result)
                        if isinstance(payload, dict):
                            generated_artifacts.extend(
                                item for item in payload.get("artifacts", []) if isinstance(item, dict)
                            )
                    except json.JSONDecodeError:
                        pass
                    yield {
                        "type": "tool",
                        "name": tool_name,
                        "status": event.get("status", "completed"),
                        "result": raw_result,
                    }
                    yield {"type": "plan_update", "plan": _public_plan(plan, statuses=statuses), "create": False}
                for event in node_update.get("reasoning_events", []):
                    if isinstance(event, dict) and event.get("text"):
                        yield {"type": "reasoning", "text": str(event["text"])}
                if isinstance(node_update.get("completion_result"), dict):
                    semantic_acceptance = node_update["completion_result"]
                if node_update.get("final_response"):
                    final_response = str(node_update["final_response"])

        validator_results: list[ValidatorResult] = []
        for validator_id in dict.fromkeys(item for skill in skills for item in skill.validators):
            result = registry.validators.get(validator_id)(contract, final_response, evidence.records())
            validator_results.append(ValidatorResult.model_validate(result))
        hard_satisfied = all(item.satisfied for item in validator_results)
        covered = covered_capabilities(skills, evidence.records())
        missing_capabilities = [item for item in contract.required_capabilities if item not in covered]
        semantic_satisfied = bool(semantic_acceptance.get("satisfied"))
        validator_missing = [missing for result in validator_results for missing in result.missing]
        missing = list(
            dict.fromkeys([*semantic_acceptance.get("missing", []), *validator_missing, *missing_capabilities])
        )
        outcome = str(semantic_acceptance.get("outcome") or "partial")
        if not hard_satisfied or missing_capabilities:
            outcome = "data_unavailable" if not evidence.records() and contract.evidence_requirements else "partial"
        acceptance = AcceptanceResult(
            outcome=outcome
            if outcome in {"satisfied", "partial", "needs_input", "data_unavailable", "failed"}
            else "partial",
            satisfied=hard_satisfied and semantic_satisfied and not missing_capabilities,
            terminal=True,
            validator_results=tuple(validator_results),
            evidence_coverage={
                "required": list(contract.required_capabilities),
                "covered": sorted(covered),
                "record_count": len(evidence.records()),
            },
            missing=tuple(missing),
            next_action=(
                str(semantic_acceptance.get("next_action") or "")
                if hard_satisfied
                else "补齐确定性验证器列出的证据后重新验收；不得用语义判断覆盖硬性失败。"
            ),
            reason=(
                str(semantic_acceptance.get("reason") or "Harness 验收完成")
                if hard_satisfied
                else "；".join(result.reason for result in validator_results if not result.satisfied)
            ),
        )
        latest_as_of = max((record.as_of for record in evidence.records() if record.as_of), default=None)
        for step in plan.steps:
            skill = next((item for item in skills if item.id == step.skill_id), None)
            if skill is not None and skill.composite:
                composite_complete = set(skill.capabilities) <= covered
                statuses[step.id] = {
                    "status": "completed" if composite_complete else "failed",
                    "evidence_status": "available" if composite_complete else "unavailable",
                    "as_of": latest_as_of,
                    "error": None if composite_complete else "组合能力的核心证据未齐备",
                }
                continue
            statuses.setdefault(
                step.id,
                {"status": "skipped", "evidence_status": "unavailable", "error": "执行预算内未调用"},
            )
        final_plan = _public_plan(plan, statuses=statuses)
        final_plan["outcome_status"] = acceptance.outcome
        final_plan["acceptance"] = acceptance.model_dump(mode="json")
        final_plan["missing"] = list(acceptance.missing)
        yield {"type": "plan_update", "plan": final_plan, "create": False}
        yield {
            "type": "task_outcome",
            "task_contract": contract.model_dump(mode="json"),
            "acceptance": acceptance.model_dump(mode="json"),
            "asset_type": contract.asset_type,
            "fund_domain": contract.fund_domain,
            "product_category": contract.product_category,
            "pricing_basis": contract.pricing_basis,
        }
        if final_response:
            yield {
                "type": "text",
                "text": _compact_generated_report(final_response, generated_artifacts)
                if generated_artifacts
                else final_response,
            }

    async def resume_chat(
        self,
        interaction: dict[str, Any],
        option_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Resume product-type clarification or a checkpointed tool approval."""
        payload = dict(interaction.get("payload") or {})
        request = self.request_from_payload(payload.get("request") or {})
        kind = str(interaction.get("kind") or "")
        if kind == "asset_type_clarification":
            request = replace(
                request,
                asset_type=AssetType(option_id),
                asset_type_explicit=True,
                asset_type_ambiguous=False,
                asset_type_candidates=(),
                intent_confirmed=True,
            )
            async for event in self.chat(request):
                yield event
            return
        if kind != "tool_confirmation":
            raise ValueError(f"不支持的交互类型: {kind}")

        thread_id = str(payload.get("thread_id") or request.task_id or "")
        if not payload.get("native_checkpoint") or not thread_id or checkpoint_manager.saver is None:
            if option_id != "approve":
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
                    "task_contract": {},
                    "acceptance": {
                        "outcome": "partial",
                        "satisfied": False,
                        "terminal": True,
                        "missing": ["用户拒绝的工具操作"],
                        "reason": "尊重用户拒绝，未执行副作用",
                    },
                    "asset_type": request.asset_type.value,
                }
                return
            raise ValueError("Harness 原生 checkpoint 不可用，不能安全重放写操作")

        routing = await classify_task_execution(
            request.message,
            tickers=request.tickers,
            asset_type=request.asset_type.value,
            mutation_requested=request.allow_mutating_tools,
        )
        kernel = await prepare_harness_plan(
            self.request_payload(request),
            routing,
            task_id=None,
            dynamic_planning=False,
        )
        contract = HarnessTaskContract.model_validate(kernel["contract"])
        skills = tuple(SkillManifest.model_validate(item) for item in kernel["selected_skills"])
        tool_names = {name for skill in skills for name in skill.tools}
        tools = self._build_runtime_tools(request, tool_names)
        resume_config = checkpoint_manager.graph_config(
            thread_id,
            build_trace_config(
                "financial-harness-resume",
                tags=["financial-harness", "resume", request.asset_type.value],
                metadata={"contract_id": contract.contract_id, "task_id": request.task_id or ""},
                session_id=request.conversation_id,
            ),
        )
        completion: dict[str, Any] = {}
        final_response = ""
        async for update in resume_native_agent_loop(
            tools,
            approved=option_id == "approve",
            config=resume_config,
            task_id=request.task_id,
        ):
            native_interrupt = update.get("__interrupt__")
            if native_interrupt:
                value = getattr(native_interrupt[0], "value", {})
                value = value if isinstance(value, dict) else {}
                yield {
                    "type": "interaction_required",
                    "kind": "tool_confirmation",
                    "question": str(value.get("question") or "Agent 准备执行模拟盘写操作，是否继续？"),
                    "options": [{"id": "approve", "label": "确认执行"}, {"id": "reject", "label": "取消执行"}],
                    "resume": {
                        "native_checkpoint": True,
                        "graph_name": HARNESS_GRAPH_NAME,
                        "thread_id": thread_id,
                        "interrupt_id": getattr(native_interrupt[0], "id", ""),
                        "request": self.request_payload(request),
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
                if isinstance(node_update.get("completion_result"), dict):
                    completion = node_update["completion_result"]
                if node_update.get("final_response"):
                    final_response = str(node_update["final_response"])
        acceptance = {
            "outcome": str(completion.get("outcome") or "partial"),
            "satisfied": bool(completion.get("satisfied")),
            "terminal": True,
            "missing": list(completion.get("missing") or []),
            "reason": str(completion.get("reason") or "Harness 恢复执行完成"),
        }
        yield {
            "type": "task_outcome",
            "task_contract": contract.model_dump(mode="json"),
            "acceptance": acceptance,
            "asset_type": contract.asset_type,
            "fund_domain": contract.fund_domain,
            "product_category": contract.product_category,
            "pricing_basis": contract.pricing_basis,
        }
        if final_response:
            yield {"type": "text", "text": final_response}


financial_harness_agent = FinancialHarnessAgent()
