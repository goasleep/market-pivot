"""Compile one request into a completion contract without choosing an executor."""

from __future__ import annotations

import re

from application.fund_task_compiler import compile_fund_task
from models.supervisor import TaskContract

_DATA_WORDS = re.compile(
    r"最新|实时|行情|价格|净值|费率|费用|规模|成交|价差|流动性|跟踪|申赎|公告|新闻|"
    r"历史|走势|回测|筛选|比较|对比|分析|风险|收益"
)
_REPRESENTATIVE_WORDS = re.compile(r"代表产品|自动选择|场内.*场外|ETF.*联接|联接.*ETF", re.IGNORECASE)


def compile_task_contract(
    message: str,
    *,
    tickers: list[str] | None = None,
    asset_type: str = "stock",
    mutation_requested: bool = False,
) -> TaskContract:
    """Describe what done means; execution remains owned by the Supervisor."""
    task_spec = compile_fund_task(
        message,
        tickers=tickers or [],
        asset_type=asset_type,
        mutation_requested=mutation_requested,
    )
    deliverables = list(task_spec.required_outputs) if task_spec is not None else []
    evidence = []
    if task_spec is not None and task_spec.evidence_mode.value != "none":
        evidence.append(task_spec.evidence_mode.value)
    requires_tools = bool(_DATA_WORDS.search(message))
    if task_spec is not None:
        requires_tools = requires_tools or task_spec.requires_live_data
    if not deliverables:
        deliverables = ["直接回答用户的全部问题", "说明关键依据、适用条件和已知限制"]
    return TaskContract(
        objective=message.strip(),
        deliverables=deliverables,
        evidence_requirements=evidence,
        requires_tools=requires_tools,
        resolve_representative_product=bool(_REPRESENTATIVE_WORDS.search(message)),
        missing_inputs=list(task_spec.missing_inputs) if task_spec is not None else [],
        source_task_spec=task_spec.model_dump(mode="json") if task_spec is not None else None,
    )
