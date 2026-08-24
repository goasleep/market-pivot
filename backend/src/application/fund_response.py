"""Bounded direct executors for fund tasks that do not need market data."""

from __future__ import annotations

import json

from application.fund_completion import validate_fund_response
from domain.fund_calculations import calculate_from_question
from llm.service import get_llm_service
from models.fund_task import FundTaskAcceptance, FundTaskKind, FundTaskSpec, TaskOutcome

_SYSTEM = (
    "你是面向小额个人投资者的基金研究与模拟交易助手。"
    "回答必须直接完成任务，重点覆盖短中期趋势、入场退出、仓位、回撤、流动性、费用和持有期风险。"
    "区分已知事实、用户假设和模拟结果，不得把股票研究冒充基金结论。"
    "本任务不允许调用实时行情或网页数据；只能使用用户题设、稳定的基金常识和给出的确定性计算。"
    "不要写无来源的当前日期、最新数据、具体净值、基金排名或目标价。"
    "需要参数时可以给明确的示例规则，但必须说明它需要结合具体产品波动和用户风险预算校准。"
    "用中文回答，先给结论，再给可执行规则、条件和风险边界。普通回答控制在900个汉字内，"
    "筛选或组合方案最多1400个汉字；不要重复题目、免责声明或常识性铺垫。"
)


def _clarification_answer(spec: FundTaskSpec) -> str:
    missing = "、".join(spec.missing_inputs) or "基金代码或准确名称"
    return (
        f"要完成这项具体基金研究，还需要：{missing}。"
        "请同时提供计划持有期、当前是否持仓、最大可接受亏损金额，以及希望分析的重点。"
        "在标的未核验前，我不会调用行情或给出具体买卖点。"
    )


def _prompt(message: str, spec: FundTaskSpec, previous_answer: str = "", missing: list[str] | None = None) -> str:
    payload = {
        "question": message,
        "task_kind": spec.task_kind.value,
        "operation": spec.operation,
        "subject": spec.subject.model_dump(mode="json"),
        "user_inputs": spec.user_inputs,
        "required_outputs": spec.required_outputs,
        "previous_answer": previous_answer,
        "missing_outputs_to_repair": missing or [],
    }
    instruction = (
        "根据任务合同回答。required_outputs 每一项都必须有明确内容；不要描述工具、系统流程或说‘已完成’。"
        "如果题设不足以形成具体数值，给条件化规则并列出缺失数据。"
    )
    if previous_answer:
        instruction += "请保留原答案中正确内容，补齐缺项后输出一份完整替代答案，不要只输出补丁。"
    return f"{instruction}\n\n{json.dumps(payload, ensure_ascii=False)}"


async def execute_direct_fund_task(message: str, spec: FundTaskSpec) -> tuple[str, FundTaskAcceptance]:
    if spec.task_kind == FundTaskKind.CLARIFICATION:
        answer = _clarification_answer(spec)
        return answer, FundTaskAcceptance(
            outcome=TaskOutcome.NEEDS_INPUT,
            satisfied=False,
            checks={"missing_inputs_identified": True},
            missing=spec.missing_inputs,
        )
    if spec.task_kind == FundTaskKind.CALCULATION:
        calculation = calculate_from_question(message)
        if calculation is not None:
            answer = f"{calculation.render()}\n\n实际费用以基金合同、销售渠道和券商收费规则为准。"
            return answer, validate_fund_response(spec, answer)

    answer = await get_llm_service().chat(
        _prompt(message, spec),
        system=_SYSTEM,
        max_tokens=1200,
        route="analysis",
    )
    acceptance = validate_fund_response(spec, answer)
    if not acceptance.satisfied and acceptance.missing:
        answer = await get_llm_service().chat(
            _prompt(message, spec, previous_answer=answer, missing=acceptance.missing),
            system=_SYSTEM,
            max_tokens=1400,
            route="analysis",
        )
        acceptance = validate_fund_response(spec, answer)
    return answer, acceptance
