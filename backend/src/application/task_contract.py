"""Compile one request into a completion contract without choosing an executor."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from loguru import logger

from application.fund_task_compiler import compile_fund_task
from llm.service import get_llm_service
from models.supervisor import ExecutionMode, TaskContract, TaskRoutingDecision

_REPRESENTATIVE_WORDS = re.compile(r"代表产品|自动选择|场内.*场外|ETF.*联接|联接.*ETF", re.IGNORECASE)


async def classify_task_execution(
    message: str,
    *,
    tickers: Sequence[str] = (),
    asset_type: str = "stock",
    mutation_requested: bool = False,
) -> TaskRoutingDecision:
    """Ask the configured model how the request should be executed.

    This intentionally has no keyword-based task fallback. If the routing call
    is unavailable, the existing Supervisor receives the full tool surface and
    decides from the original request.
    """
    prompt = json.dumps(
        {
            "message": message,
            "tickers": list(tickers),
            "asset_type": asset_type,
            "mutation_requested": mutation_requested,
            "modes": [
                "direct_response",
                "artifact_generation",
                "evidence_research",
                "backtest_execution",
                "mixed_workflow",
            ],
        },
        ensure_ascii=False,
    )
    try:
        payload = await get_llm_service().chat_json(
            prompt,
            system=(
                "你是请求执行分类器，只决定如何完成任务，不回答用户问题。"
                "根据语义区分：解释、代码审查、已有材料判断或仅设计方案通常可直接回答；"
                "查询当前/历史事实需要证据研究；只有用户明确要求实际运行回测时才选择backtest_execution；"
                "同时需要数据、计算和综合交付时选择mixed_workflow；明确要求生成独立文件时可选择artifact_generation。"
                "不要因为文本出现证券代码、‘回测’或‘分析’字样就自动要求工具，必须判断用户是在讨论/审查，还是要求实际执行。"
                "返回字段：mode、requires_tools、allow_research_plan、deliverables、reason、confidence。"
                "deliverables只列用户明确要求的最小交付项；reason只给简短公开理由，不输出思维链。"
            ),
        )
        decision = TaskRoutingDecision.model_validate(payload)
        if mutation_requested and not decision.requires_tools:
            decision = decision.model_copy(
                update={
                    "mode": ExecutionMode.MIXED_WORKFLOW,
                    "requires_tools": True,
                    "allow_research_plan": False,
                    "reason": "用户明确请求模拟盘变更，必须通过受控工具执行",
                }
            )
        return decision
    except Exception as exc:
        logger.warning("Task execution classification failed; deferring to Supervisor: {}", exc)
        return TaskRoutingDecision.supervisor_fallback()


def compile_task_contract(
    message: str,
    *,
    tickers: list[str] | None = None,
    asset_type: str = "stock",
    mutation_requested: bool = False,
    routing_decision: TaskRoutingDecision | None = None,
) -> TaskContract:
    """Describe what done means; execution remains owned by the Supervisor."""
    routing = routing_decision or TaskRoutingDecision.supervisor_fallback()
    task_spec = (
        compile_fund_task(
            message,
            tickers=tuple(tickers or []),
            asset_type=asset_type,
            mutation_requested=mutation_requested,
        )
        if routing.requires_tools
        else None
    )
    deliverables = list(routing.deliverables)
    if not deliverables and task_spec is not None:
        deliverables = list(task_spec.required_outputs)
    evidence = []
    if task_spec is not None and task_spec.evidence_mode.value != "none":
        evidence.append(task_spec.evidence_mode.value)
    if not deliverables:
        deliverables = ["直接回答用户的全部问题", "说明关键依据、适用条件和已知限制"]
    return TaskContract(
        objective=message.strip(),
        deliverables=deliverables,
        evidence_requirements=evidence,
        requires_tools=routing.requires_tools,
        resolve_representative_product=bool(_REPRESENTATIVE_WORDS.search(message)),
        missing_inputs=list(task_spec.missing_inputs) if task_spec is not None else [],
        source_task_spec=task_spec.model_dump(mode="json") if task_spec is not None else None,
        routing=routing,
    )
