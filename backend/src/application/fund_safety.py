"""Deterministic safety gate for fund research and paper-trading requests."""

from __future__ import annotations

import re

from models.fund_task import RiskPolicyAction, RiskPolicyDecision

_GUARANTEE_PATTERN = re.compile(r"(?:保证|确保|承诺|稳赚|必赚|无风险).{0,18}(?:收益|盈利|上涨|回报)", re.IGNORECASE)
_REVERSE_GUARANTEE_PATTERN = re.compile(r"(?:收益|盈利|上涨|回报).{0,18}(?:保证|确保|承诺|稳赚|必赚)", re.IGNORECASE)
_FULL_POSITION_PATTERN = re.compile(r"(?:满仓|全仓|全部资金|梭哈)", re.IGNORECASE)
_UNVERIFIED_TIP_PATTERN = re.compile(r"(?:未经证实|内部消息|内幕消息|小道消息)", re.IGNORECASE)
_REAL_ACCOUNT_PATTERN = re.compile(r"(?:登录|代替).{0,12}(?:证券账户|基金账户|交易账户).{0,18}(?:买入|卖出|操作)")
_HIDE_RISK_PATTERN = re.compile(r"(?:隐瞒|隐藏|不展示).{0,18}(?:回撤|风险|手续费|费用)")
_BORROWED_FULL_POSITION_PATTERN = re.compile(r"(?:借款|贷款|融资).{0,18}(?:满仓|全仓|高波动)")
_NO_STOP_AVERAGING_PATTERN = re.compile(r"(?:不设置止损|不要止损).{0,18}(?:持续补仓|一直补仓|直到盈利)")
_STOCK_TO_FUND_CERTAINTY_PATTERN = re.compile(r"股票研究.{0,20}(?:直接转换|确定性).{0,20}(?:基金|买入)")
_NO_LOSS_HIGH_RETURN_PATTERN = re.compile(r"(?:没有亏损风险|零风险|完全无风险).{0,24}(?:收益|年化|回报)")
_CERTAIN_RISE_PATTERN = re.compile(r"(?:必然|一定|确定).{0,10}(?:上涨|盈利)")
_NEGATED_CERTAINTY_PATTERN = re.compile(r"(?:不要|不能|不可|避免).{0,30}(?:保证|必然|一定|确定)")


def evaluate_fund_safety(message: str, *, mutation_requested: bool = False) -> RiskPolicyDecision:
    """Return the non-negotiable safety decision before any data access."""
    text = message.strip()
    guarantee = bool(
        _GUARANTEE_PATTERN.search(text)
        or _REVERSE_GUARANTEE_PATTERN.search(text)
        or _NO_LOSS_HIGH_RETURN_PATTERN.search(text)
        or _CERTAIN_RISE_PATTERN.search(text)
    ) and not _NEGATED_CERTAINTY_PATTERN.search(text)
    full_position = bool(_FULL_POSITION_PATTERN.search(text))
    if guarantee:
        reasons = ["基金未来收益无法被保证，短期净值可能出现显著波动和亏损"]
        if full_position:
            reasons.append("满仓会放大单一时点、单一产品和流动性风险")
        return RiskPolicyDecision(
            action=RiskPolicyAction.REFUSE_GUARANTEE,
            reasons=reasons,
            prohibited_outputs=["收益保证", "无条件满仓方案", "确定性买卖结论"],
            required_disclosures=["仅用于基金研究和模拟交易，不构成收益承诺"],
            safe_alternatives=["设置最大仓位", "分批建仓", "设定最大亏损预算", "先在模拟盘验证"],
        )
    unsafe_cases = (
        (
            _UNVERIFIED_TIP_PATTERN,
            "不能根据未经证实的内部消息形成基金推荐",
            ["改用公开公告和可核验数据", "把消息作为待验证假设而不是买入依据"],
        ),
        (
            _REAL_ACCOUNT_PATTERN,
            "不能代替用户登录证券账户或执行实盘交易",
            ["提供研究方案", "在需要确认的纸面模拟账户中验证"],
        ),
        (
            _HIDE_RISK_PATTERN,
            "不能故意隐瞒最大回撤、费用等重要风险信息",
            ["同时展示收益、回撤、波动、费用和样本区间", "明确历史表现不代表未来收益"],
        ),
        (
            _BORROWED_FULL_POSITION_PATTERN,
            "借款资金满仓高波动基金可能造成超出本金承受能力的损失",
            ["只使用可承受亏损的自有闲置资金", "设置仓位和最大亏损上限"],
        ),
        (
            _NO_STOP_AVERAGING_PATTERN,
            "不设止损并持续补仓会让风险敞口在趋势恶化时不断扩大",
            ["设置组合回撤上限", "规定停止补仓和分批退出条件"],
        ),
        (
            _STOCK_TO_FUND_CERTAINTY_PATTERN,
            "股票研究不能直接转换成行业基金的确定性买入建议",
            ["核对基金持仓暴露、跟踪标的、费用和流动性", "给出条件性而非确定性结论"],
        ),
    )
    for pattern, reason, alternatives in unsafe_cases:
        if pattern.search(text):
            return RiskPolicyDecision(
                action=RiskPolicyAction.BLOCK,
                reasons=[reason],
                prohibited_outputs=["执行或美化该高风险请求"],
                required_disclosures=["仅用于基金研究和模拟交易"],
                safe_alternatives=alternatives,
            )
    if mutation_requested:
        return RiskPolicyDecision(
            action=RiskPolicyAction.REQUIRE_CONFIRMATION,
            reasons=["模拟盘写操作需要用户明确确认"],
            required_disclosures=["操作仅发生在纸面模拟账户"],
        )
    return RiskPolicyDecision(action=RiskPolicyAction.ALLOW)


def render_safe_alternative(decision: RiskPolicyDecision) -> str:
    """Render a deterministic refusal that still helps the user proceed safely."""
    reasons = "；".join(decision.reasons)
    alternatives = "\n".join(f"{index}. {item}" for index, item in enumerate(decision.safe_alternatives, 1))
    lead = (
        "我不能保证某只基金未来获得指定收益，也不建议据此满仓。"
        if decision.action == RiskPolicyAction.REFUSE_GUARANTEE
        else "我不能按这个要求执行或生成误导性方案。"
    )
    follow_up = (
        "仓位可以从风险预算反推：理论仓位金额 = 可承受亏损金额 ÷ 预设止损幅度，并应再为费用、滑点和跳空留出缓冲。\n\n"
        "如果你提供基金代码、计划持有期和最大可接受亏损金额，我可以进一步给出分批仓位、止损和退出方案。"
        if decision.action == RiskPolicyAction.REFUSE_GUARANTEE
        else "如果你说明真实的研究目标，我可以改用公开可核验信息、完整风险披露或纸面模拟方式完成合规分析。"
    )
    return lead + (
        f"原因是：{reasons}。\n\n"
        "可以改成以下可验证、风险受控的方案：\n"
        f"{alternatives}\n\n"
        f"{follow_up}\n\n"
        "以上仅用于基金研究和模拟交易，不构成收益承诺。"
    )
