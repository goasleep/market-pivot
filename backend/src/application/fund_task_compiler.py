"""Compile fund questions into a small, stable set of execution contracts."""

from __future__ import annotations

import re
from typing import Any

from application.fund_instruments import resolve_fund_instruments
from application.fund_safety import evaluate_fund_safety
from models.fund_task import (
    EvidenceMode,
    FundSubject,
    FundTaskKind,
    FundTaskSpec,
    InstrumentResolutionStatus,
    RiskPolicyAction,
)

_FUND_TERMS = (
    "基金",
    "etf",
    "lof",
    "净值",
    "申购",
    "赎回",
    "销售服务费",
    "跟踪误差",
    "跟踪差额",
    "场内",
    "场外",
)
_CALCULATION_TERMS = ("计算", "费用是多少", "成本是多少", "手续费是多少", "盈亏平衡")
_RULE_TERMS = ("规则", "方法", "指标", "如何判断", "怎么判断", "需要核查", "需要关注", "哪些信息")
_SCENARIO_TERMS = (
    "仓位",
    "止损",
    "止盈",
    "建仓",
    "加仓",
    "减仓",
    "退出",
    "入场",
    "再平衡",
    "组合",
    "最大亏损",
    "回撤上限",
)
_UNIVERSE_ACTIONS = ("筛选", "找出", "选出", "排名", "排行", "全市场")
_DESIGN_WORDS = ("设计一套", "设计一个", "设计筛选", "筛选方法", "筛选规则", "如何筛选")
_CURRENT_DATA_TERMS = ("最新", "当前", "今日", "现在", "实时", "近一周", "近一个月", "近一年")
_EVENT_TERMS = ("公告", "离任", "更换基金经理", "限购", "暂停申购", "分红公告", "清盘")
_SIMULATION_QUERY_TERMS = ("模拟盘账户", "模拟盘持仓", "模拟订单", "纸面账户")
_BACKTEST_TERMS = ("回测", "历史测试", "backtest")


def is_fund_request(message: str, *, asset_type: str = "stock") -> bool:
    text = message.lower()
    return asset_type in {"etf", "lof", "fund"} or any(term in text for term in _FUND_TERMS)


def _product_type(text: str) -> str:
    mappings = (
        (("货币基金", "货币型"), "money_market"),
        (("债券基金", "债券型", "债基"), "bond"),
        (("混合基金", "混合型"), "hybrid"),
        (("股票基金", "股票型"), "equity"),
        (("增强指数", "指数增强"), "enhanced_index"),
        (("qdii",), "qdii"),
        (("fof",), "fof"),
        (("reit",), "reit"),
        (("lof",), "lof"),
        (("etf",), "etf"),
        (("指数基金", "宽基", "行业基金"), "index"),
    )
    for terms, value in mappings:
        if any(term in text for term in terms):
            return value
    return "unknown"


def _extract_user_inputs(message: str) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    percentages = [float(item) for item in re.findall(r"(\d+(?:\.\d+)?)\s*%", message)]
    if percentages:
        inputs["percentages"] = percentages
    amounts = []
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(万元|万|元)", message):
        value = float(number) * (10_000 if unit in {"万元", "万"} else 1)
        amounts.append(value)
    if amounts:
        inputs["amounts"] = amounts
    months = re.findall(r"(\d+)\s*个?月", message)
    if months:
        inputs["months"] = [int(item) for item in months]
    return inputs


def _required_outputs(text: str, task_kind: FundTaskKind) -> list[str]:
    if task_kind == FundTaskKind.CALCULATION:
        return ["inputs", "formula", "result", "assumptions"]
    outputs: list[str] = []
    if "价格止损" in text:
        outputs.append("price_stop")
    if "时间止损" in text:
        outputs.append("time_stop")
    if "趋势止损" in text:
        outputs.append("trend_stop")
    if "组合" in text:
        outputs.extend(["asset_weights", "amounts", "rebalance_rules", "risk_limit"])
    if any(term in text for term in ("仓位", "建仓", "加仓", "减仓")):
        outputs.extend(["initial_position", "position_triggers", "maximum_loss"])
    if any(term in text for term in ("筛选", "筛选方法")) and task_kind == FundTaskKind.RULE_DESIGN:
        outputs.extend(["eligibility", "metrics", "scoring", "exclusions"])
    if any(term in text for term in ("离任", "基金经理")):
        outputs.extend(["conditional_conclusion", "succession", "strategy_stability", "holding_changes"])
    if any(term in text for term in ("数据不足", "没有提供", "缺少", "无法获得")):
        outputs.extend(["missing_information", "impact", "next_inputs", "stop_condition"])
    if not outputs and task_kind in {FundTaskKind.RULE_DESIGN, FundTaskKind.SCENARIO_PLAN}:
        outputs = ["conditions", "actions", "risk_limits", "invalidations"]
    if not outputs and task_kind == FundTaskKind.EDUCATION:
        outputs = ["direct_answer", "key_differences", "risks", "applicable_conditions"]
    return list(dict.fromkeys(outputs))


def compile_fund_task(
    message: str,
    *,
    tickers: tuple[str, ...] = (),
    asset_type: str = "stock",
    mutation_requested: bool = False,
) -> FundTaskSpec | None:
    """Return a fund task contract, or ``None`` for the legacy stock workflow."""
    if not is_fund_request(message, asset_type=asset_type):
        return None
    text = message.lower().strip()
    safety = evaluate_fund_safety(message, mutation_requested=mutation_requested)
    instruments = resolve_fund_instruments(message, tickers, asset_type=asset_type)
    verified = [item for item in instruments if item.status == InstrumentResolutionStatus.VERIFIED]
    product_type = _product_type(text)
    if product_type == "unknown" and asset_type in {"etf", "lof"}:
        product_type = asset_type
    subject = FundSubject(
        scope="fund_instrument" if instruments else "fund_concept",
        product_type=product_type,
    )

    if safety.action in {RiskPolicyAction.REFUSE_GUARANTEE, RiskPolicyAction.BLOCK}:
        kind = FundTaskKind.SAFETY_RESPONSE
        operation = "refuse_return_guarantee"
        evidence = EvidenceMode.NONE
    elif mutation_requested:
        kind = FundTaskKind.SIMULATION_MUTATION
        operation = "mutate_paper_trading"
        evidence = EvidenceMode.SIMULATION_STATE
        subject.scope = "account"
    elif any(term in text for term in _SIMULATION_QUERY_TERMS):
        kind = FundTaskKind.SIMULATION_QUERY
        operation = "query_paper_trading"
        evidence = EvidenceMode.SIMULATION_STATE
        subject.scope = "account"
    elif any(term in text for term in _BACKTEST_TERMS):
        kind = FundTaskKind.INSTRUMENT_RESEARCH
        operation = "backtest"
        evidence = EvidenceMode.NAV_HISTORY
    elif (
        any(term in text for term in _CALCULATION_TERMS)
        or (
            bool(re.search(r"\d", text))
            and any(term in text for term in ("申购费", "赎回费", "佣金", "销售服务费", "持有成本"))
        )
    ) and not any(term in text for term in _SCENARIO_TERMS):
        kind = FundTaskKind.CALCULATION
        operation = "calculate_fund_cost_or_risk"
        evidence = EvidenceMode.USER_PROVIDED
    elif any(term in text for term in _SCENARIO_TERMS):
        kind = FundTaskKind.SCENARIO_PLAN
        operation = "design_fund_trade_or_portfolio_plan"
        evidence = EvidenceMode.USER_PROVIDED
        subject.scope = "portfolio" if "组合" in text else subject.scope
    elif any(term in text for term in ("两只", "另一只", "分别")) and any(
        term in text for term in ("比较", "更好", "选出", "哪一只")
    ):
        kind = FundTaskKind.RULE_DESIGN
        operation = "compare_funds_conditionally"
        evidence = EvidenceMode.USER_PROVIDED
    elif any(term in text for term in _UNIVERSE_ACTIONS) and not any(term in text for term in _DESIGN_WORDS):
        kind = FundTaskKind.UNIVERSE_RESEARCH
        operation = "screen_or_rank_funds"
        evidence = EvidenceMode.UNIVERSE_DATA
        subject.scope = "fund_universe"
    elif any(term in text for term in _EVENT_TERMS) and verified and any(term in text for term in _CURRENT_DATA_TERMS):
        kind = FundTaskKind.EVENT_RESEARCH
        operation = "research_current_fund_event"
        evidence = EvidenceMode.ANNOUNCEMENTS
    elif any(term in text for term in _EVENT_TERMS) or any(term in text for term in _RULE_TERMS) or any(
        term in text for term in _DESIGN_WORDS
    ):
        kind = FundTaskKind.RULE_DESIGN
        operation = "design_or_explain_fund_rules"
        evidence = EvidenceMode.USER_PROVIDED
    elif verified and any(term in text for term in _CURRENT_DATA_TERMS + ("分析", "走势", "比较", "对比")):
        kind = FundTaskKind.INSTRUMENT_RESEARCH
        operation = "research_fund_instrument"
        evidence = (
            EvidenceMode.REALTIME_MARKET
            if any(term in text for term in _CURRENT_DATA_TERMS)
            else EvidenceMode.NAV_HISTORY
        )
    else:
        kind = FundTaskKind.EDUCATION
        operation = "explain_fund_concept"
        evidence = EvidenceMode.USER_PROVIDED if re.search(r"\d", text) else EvidenceMode.NONE

    requires_verified = kind in {FundTaskKind.INSTRUMENT_RESEARCH, FundTaskKind.EVENT_RESEARCH}
    missing = ["可验证的基金代码或准确名称"] if requires_verified and not verified else []
    if missing:
        kind = FundTaskKind.CLARIFICATION
    live_data = evidence in {
        EvidenceMode.FUND_PROFILE,
        EvidenceMode.NAV_HISTORY,
        EvidenceMode.REALTIME_MARKET,
        EvidenceMode.ANNOUNCEMENTS,
        EvidenceMode.UNIVERSE_DATA,
        EvidenceMode.SIMULATION_STATE,
    }
    capabilities = {
        FundTaskKind.CALCULATION: ["calculation.fund"],
        FundTaskKind.INSTRUMENT_RESEARCH: ["fund.profile", "fund.timeseries", "fund.liquidity", "fund.events"],
        FundTaskKind.UNIVERSE_RESEARCH: ["fund.universe"],
        FundTaskKind.EVENT_RESEARCH: ["fund.events", "fund.profile"],
        FundTaskKind.SIMULATION_QUERY: ["simulation.read"],
        FundTaskKind.SIMULATION_MUTATION: ["simulation.read", "simulation.write"],
    }.get(kind, [])
    return FundTaskSpec(
        task_kind=kind,
        operation=operation,
        subject=subject,
        evidence_mode=evidence,
        user_inputs=_extract_user_inputs(message),
        instruments=instruments,
        missing_inputs=missing,
        requires_live_data=live_data,
        requires_verified_instrument=requires_verified,
        required_outputs=_required_outputs(text, kind),
        allowed_capabilities=capabilities,
        forbidden_capabilities=(
            ["fund.realtime_quote", "fund.timeseries", "stock.fundamentals", "fund.universe"]
            if not live_data
            else []
        ),
        safety_decision=safety,
        confidence=0.98 if kind != FundTaskKind.EDUCATION else 0.9,
    )


def uses_direct_fund_executor(spec: FundTaskSpec) -> bool:
    return spec.task_kind in {
        FundTaskKind.EDUCATION,
        FundTaskKind.CALCULATION,
        FundTaskKind.RULE_DESIGN,
        FundTaskKind.SCENARIO_PLAN,
        FundTaskKind.CLARIFICATION,
        FundTaskKind.SAFETY_RESPONSE,
    }
