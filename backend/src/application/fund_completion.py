"""Business-level answer acceptance for fund tasks."""

from __future__ import annotations

import re

from models.fund_task import FundAnswerAcceptance, FundIntentSpec, TaskOutcome

_OUTPUT_TERMS = {
    "inputs": ("输入", "已知"),
    "formula": ("公式", "计算"),
    "result": ("结果", "元", "%"),
    "assumptions": ("假设", "前提"),
    "price_stop": ("价格止损",),
    "time_stop": ("时间止损",),
    "trend_stop": ("趋势止损",),
    "asset_weights": ("配置", "比例", "权重"),
    "amounts": ("金额", "万元", "元"),
    "rebalance_rules": ("再平衡", "调整"),
    "risk_limit": ("风险上限", "最大回撤", "最大亏损"),
    "initial_position": ("初始仓", "初始仓位", "首笔"),
    "position_triggers": ("触发", "加仓", "减仓"),
    "maximum_loss": ("最大亏损", "风险预算", "止损"),
    "eligibility": ("预筛选", "准入", "样本"),
    "metrics": ("指标", "收益率", "回撤", "波动"),
    "scoring": ("评分", "权重", "打分"),
    "exclusions": ("排除", "剔除"),
    "conditional_conclusion": ("不宜", "建议", "结论"),
    "succession": ("继任", "接任"),
    "strategy_stability": ("策略", "风格"),
    "holding_changes": ("持仓", "换手"),
    "missing_information": ("缺少", "需要提供"),
    "impact": ("影响", "无法"),
    "next_inputs": ("补充", "提供"),
    "stop_condition": ("停止", "不应", "不能"),
    "conditions": ("条件", "触发"),
    "actions": ("执行", "减仓", "买入", "卖出"),
    "risk_limits": ("风险", "上限", "止损"),
    "invalidations": ("失效", "停止", "重新评估"),
    "direct_answer": ("基金",),
    "key_differences": ("区别", "差异", "相比"),
    "risks": ("风险", "波动", "亏损"),
    "applicable_conditions": ("适合", "条件", "期限"),
    "candidate_pool": ("候选", "入围"),
    "comparison": ("对比", "比较", "排序"),
    "primary_selection": ("首选", "第一选择", "优先选择"),
    "selection_rationale": ("依据", "理由", "适合", "因为"),
    "alternative_selection": ("备选", "次选", "第二选择"),
    "data_as_of": ("截至", "数据日期", "交易日", "来源"),
}
_FUND_CODE = re.compile(r"(?<!\d)\d{6}(?!\d)")


def validate_fund_response(spec: FundIntentSpec, answer: str) -> FundAnswerAcceptance:
    if not answer.strip():
        return FundAnswerAcceptance(outcome=TaskOutcome.FAILED, satisfied=False, missing=["answer"])
    checks: dict[str, bool] = {}
    for output in spec.required_outputs:
        terms = _OUTPUT_TERMS.get(output, (output,))
        if output == "data_as_of":
            has_date = any(term in answer for term in ("截至", "数据日期", "交易日"))
            checks[output] = has_date and "来源" in answer
        else:
            checks[output] = any(term.lower() in answer.lower() for term in terms)
    selection = spec.selection_requirements
    if selection is not None:
        candidate_count = len(set(_FUND_CODE.findall(answer)))
        checks["minimum_candidates"] = candidate_count >= selection.minimum_candidates
    missing = [key for key, passed in checks.items() if not passed]
    if not missing:
        return FundAnswerAcceptance(outcome=TaskOutcome.SATISFIED, satisfied=True, checks=checks)
    passed = sum(checks.values())
    outcome = TaskOutcome.PARTIAL if passed else TaskOutcome.FAILED
    return FundAnswerAcceptance(outcome=outcome, satisfied=False, checks=checks, missing=missing)
