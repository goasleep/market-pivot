"""Compile one request into a completion contract without choosing an executor."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from loguru import logger

from application.fund_task_compiler import compile_fund_task
from llm.service import get_llm_service
from models.fund_task import FundTaskKind
from models.supervisor import ExecutionMode, TaskContract, TaskRoutingDecision

_REPRESENTATIVE_WORDS = re.compile(r"代表产品|自动选择|场内.*场外|ETF.*联接|联接.*ETF", re.IGNORECASE)
_SANDBOX_CODE_SUBJECT = r"(?:python|代码策略|策略代码|代码|沙盒|自定义因子|仓位函数)"
_SANDBOX_CODE_ACTION = r"(?:执行|运行|生成|编写|写(?:一个|一段)?|设计|实现|创建)"
_SANDBOX_STRATEGY_CONTEXT = re.compile(r"(?:回测|backtest|策略|因子|仓位)", re.IGNORECASE)
_SANDBOX_EXECUTION_PATTERNS = (
    re.compile(rf"{_SANDBOX_CODE_ACTION}.{{0,24}}{_SANDBOX_CODE_SUBJECT}", re.IGNORECASE),
    re.compile(rf"{_SANDBOX_CODE_SUBJECT}.{{0,24}}(?:{_SANDBOX_CODE_ACTION}|回测|backtest)", re.IGNORECASE),
    re.compile(r"回测(?:一下|这段|以下|该|上面(?:的)?)\s*(?:python\s*)?代码", re.IGNORECASE),
)
_SANDBOX_NON_EXECUTION = re.compile(
    r"(?:不(?:要|用|必|要求).{0,12}(?:执行|运行|回测)|只(?:需|要|想)?(?:解释|审查|评审|检查|讨论))",
    re.IGNORECASE,
)
_OUTPUT_LABELS = {
    "candidate_pool": "候选基金池",
    "comparison": "候选对比",
    "primary_selection": "明确首选",
    "selection_rationale": "选择依据",
    "alternative_selection": "至少一个备选",
    "exclusions": "未选候选及排除原因",
    "data_as_of": "数据日期与来源",
}


def requests_sandbox_execution(message: str) -> bool:
    """Return whether the user explicitly requests executable strategy code research."""
    normalized = (message or "").strip()
    if (
        not normalized
        or not _SANDBOX_STRATEGY_CONTEXT.search(normalized)
        or _SANDBOX_NON_EXECUTION.search(normalized)
    ):
        return False
    return any(pattern.search(normalized) for pattern in _SANDBOX_EXECUTION_PATTERNS)


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
        compiled_fund_task = compile_fund_task(
            message,
            tickers=tuple(tickers),
            asset_type=asset_type,
            mutation_requested=mutation_requested,
        )
        if (
            compiled_fund_task is not None
            and compiled_fund_task.task_kind == FundTaskKind.UNIVERSE_RESEARCH
            and not decision.requires_tools
        ):
            decision = decision.model_copy(
                update={
                    "mode": ExecutionMode.EVIDENCE_RESEARCH,
                    "requires_tools": True,
                    "allow_research_plan": True,
                    "reason": "基金候选筛选需要全市场结构化数据",
                }
            )
        if requests_sandbox_execution(message) and (
            not decision.requires_tools
            or decision.mode not in {ExecutionMode.BACKTEST_EXECUTION, ExecutionMode.MIXED_WORKFLOW}
        ):
            decision = decision.model_copy(
                update={
                    "mode": ExecutionMode.BACKTEST_EXECUTION,
                    "requires_tools": True,
                    "allow_research_plan": False,
                    "deliverables": ["生成的策略源码", "沙箱验证结果", "可信交易引擎回测结果"],
                    "reason": "用户明确要求生成或执行代码策略，必须进入受控沙箱研究",
                }
            )
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
    required_outputs = list(task_spec.required_outputs) if task_spec is not None else []
    if not deliverables and required_outputs:
        deliverables = [_OUTPUT_LABELS.get(output, output) for output in required_outputs]
    evidence = []
    if task_spec is not None and task_spec.evidence_mode.value != "none":
        evidence.append(task_spec.evidence_mode.value)
    if not deliverables:
        deliverables = ["直接回答用户的全部问题", "说明关键依据、适用条件和已知限制"]
    return TaskContract(
        objective=message.strip(),
        deliverables=deliverables,
        required_outputs=required_outputs,
        evidence_requirements=evidence,
        requires_tools=routing.requires_tools,
        resolve_representative_product=bool(_REPRESENTATIVE_WORDS.search(message)),
        missing_inputs=list(task_spec.missing_inputs) if task_spec is not None else [],
        source_task_spec=task_spec.model_dump(mode="json") if task_spec is not None else None,
        routing=routing,
    )
