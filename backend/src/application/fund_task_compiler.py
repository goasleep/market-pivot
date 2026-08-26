"""Compile fund questions into a small, stable set of execution contracts."""

from __future__ import annotations

import re
from typing import Any

from application.fund_instruments import resolve_fund_instruments
from models.fund_task import (
    EvidenceMode,
    FundDomain,
    FundIntentSpec,
    FundProductCategory,
    FundSelectionRequirements,
    FundSubject,
    FundTaskKind,
    InstrumentResolutionStatus,
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
_UNIVERSE_ACTIONS = ("筛选", "挑选", "找出", "找一只", "选出", "排名", "排行", "全市场")
_DESIGN_WORDS = ("设计一套", "设计一个", "设计筛选", "筛选方法", "筛选规则", "如何筛选")
_CURRENT_DATA_TERMS = ("最新", "当前", "今日", "现在", "实时", "近一周", "近一个月", "近一年")
_EVENT_TERMS = ("公告", "离任", "更换基金经理", "限购", "暂停申购", "分红公告", "清盘")
_SIMULATION_QUERY_TERMS = ("模拟盘账户", "模拟盘持仓", "模拟订单", "纸面账户")
_BACKTEST_TERMS = ("回测", "历史测试", "backtest")
_EXPLICIT_SINGLE_RESULT = re.compile(
    r"(?:只|仅)(?:要|需|给(?:我)?|推荐|选择|保留)?(?:一只|一个|1只|1个)"
    r"|(?:找|选|推荐)(?:出)?一只"
    r"|(?:一只|一个)(?:即可|就行|足够)"
    r"|不要备选"
)
_REQUESTED_RESULT_COUNT = re.compile(r"(?:筛选|挑选|找出|选出|推荐)(?:给我)?\s*([1-9]\d?|[一二三四五六七八九十])\s*只")
_CHINESE_COUNTS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def is_fund_request(message: str, *, asset_type: str = "stock") -> bool:
    text = message.lower()
    return asset_type in {"etf", "lof", "open_fund"} or any(term in text for term in _FUND_TERMS)


def _is_universe_request(text: str) -> bool:
    if any(term in text for term in _UNIVERSE_ACTIONS):
        return True
    return "推荐" in text and any(
        term in text for term in ("适合", "几只", "哪些", "候选", "比较", "对比", "短线", "中线")
    )


def _product_category(text: str) -> FundProductCategory:
    mappings = (
        (("qdii",), "qdii"),
        (("fof",), "fof"),
        (("货币基金", "货币型"), "money_market"),
        (("债券基金", "债券型", "债基"), "bond"),
        (("混合基金", "混合型"), "hybrid"),
        (("股票基金", "股票型"), "equity"),
        (("增强指数", "指数增强"), "enhanced_index"),
        (("指数基金", "宽基", "行业基金"), "index"),
    )
    for terms, value in mappings:
        if any(term in text for term in terms):
            return FundProductCategory(value)
    return FundProductCategory.UNKNOWN


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


def _selection_requirements(
    text: str,
    *,
    fund_domain: FundDomain,
    product_category: FundProductCategory,
) -> FundSelectionRequirements:
    count_match = _REQUESTED_RESULT_COUNT.search(text)
    requested_count = 3
    if count_match:
        raw_count = count_match.group(1)
        requested_count = int(raw_count) if raw_count.isdigit() else _CHINESE_COUNTS[raw_count]
    if _EXPLICIT_SINGLE_RESULT.search(text):
        requested_count = 1

    if fund_domain == FundDomain.OPEN_FUND and product_category == FundProductCategory.MONEY_MARKET:
        holding_horizon = "short_term" if any(term in text for term in ("短线", "短期")) else "unspecified"
        comparison_dimensions = ["seven_day_yield", "yield_stability", "scale", "redemption", "fees"]
    elif fund_domain == FundDomain.OPEN_FUND and product_category == FundProductCategory.BOND:
        holding_horizon = "medium_term" if any(term in text for term in ("中线", "中期")) else "unspecified"
        comparison_dimensions = [
            "drawdown",
            "nav_stability",
            "credit_exposure",
            "rate_sensitivity",
            "scale_liquidity",
            "fees",
        ]
    elif fund_domain == FundDomain.OPEN_FUND:
        holding_horizon = (
            "short_term"
            if any(term in text for term in ("短线", "短期"))
            else "medium_term"
            if any(term in text for term in ("中线", "中期"))
            else "unspecified"
        )
        comparison_dimensions = ["trend", "drawdown", "volatility", "scale_liquidity", "fees", "exposure"]
    elif any(term in text for term in ("短线", "日内", "波段", "短期")):
        holding_horizon = "short_term"
        comparison_dimensions = [
            "liquidity",
            "bid_ask_spread",
            "intraday_volatility",
            "tracking_purity",
            "fund_size",
            "fees",
        ]
    elif any(term in text for term in ("中线", "中期")):
        holding_horizon = "medium_term"
        comparison_dimensions = ["liquidity", "trend_strength", "tracking_quality", "fund_size", "fees", "drawdown"]
    else:
        holding_horizon = "unspecified"
        comparison_dimensions = ["liquidity", "tracking_quality", "fund_size", "fees", "risk"]

    single = requested_count == 1
    return FundSelectionRequirements(
        selection_mode="single" if single else "rank",
        holding_horizon=holding_horizon,
        minimum_candidates=requested_count,
        require_comparison=not single,
        require_primary=True,
        require_alternative=not single,
        require_exclusions=not single,
        require_data_as_of=True,
        comparison_dimensions=comparison_dimensions,
    )


def _required_outputs(
    text: str,
    task_kind: FundTaskKind,
    selection: FundSelectionRequirements | None = None,
) -> list[str]:
    if task_kind == FundTaskKind.CALCULATION:
        return ["inputs", "formula", "result", "assumptions"]
    if task_kind == FundTaskKind.UNIVERSE_RESEARCH and selection is not None:
        outputs = ["primary_selection", "selection_rationale"]
        if selection.selection_mode == "rank":
            outputs.insert(0, "candidate_pool")
        if selection.require_comparison:
            outputs.append("comparison")
        if selection.require_alternative:
            outputs.append("alternative_selection")
        if selection.require_exclusions:
            outputs.append("exclusions")
        if selection.require_data_as_of:
            outputs.append("data_as_of")
        return outputs
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
) -> FundIntentSpec | None:
    """Return a fund task contract, or ``None`` for the legacy stock workflow."""
    if not is_fund_request(message, asset_type=asset_type):
        return None
    text = message.lower().strip()
    instruments = resolve_fund_instruments(message, tickers, asset_type=asset_type)
    resolvable = [
        item
        for item in instruments
        if item.status in {InstrumentResolutionStatus.CANDIDATE, InstrumentResolutionStatus.VERIFIED}
    ]
    product_category = _product_category(text)
    fund_domain = FundDomain.EXCHANGE_FUND if asset_type in {"etf", "lof"} else FundDomain.OPEN_FUND
    if product_category == FundProductCategory.UNKNOWN and fund_domain == FundDomain.EXCHANGE_FUND:
        product_category = FundProductCategory.INDEX
    normalized_asset_type = asset_type if asset_type in {"etf", "lof", "open_fund"} else "open_fund"
    pricing_basis = "money_yield" if product_category == FundProductCategory.MONEY_MARKET else (
        "market_price" if fund_domain == FundDomain.EXCHANGE_FUND else "nav"
    )
    subject = FundSubject(
        scope="fund_instrument" if instruments else "fund_concept",
        fund_domain=fund_domain,
        asset_type=normalized_asset_type,
        product_category=product_category,
        pricing_basis=pricing_basis,
    )

    if mutation_requested:
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
    elif _is_universe_request(text) and not any(term in text for term in _DESIGN_WORDS):
        kind = FundTaskKind.UNIVERSE_RESEARCH
        operation = "screen_or_rank_funds"
        evidence = EvidenceMode.UNIVERSE_DATA
        subject.scope = "fund_universe"
    elif (
        any(term in text for term in _EVENT_TERMS)
        and resolvable
        and any(term in text for term in _CURRENT_DATA_TERMS)
    ):
        kind = FundTaskKind.EVENT_RESEARCH
        operation = "research_current_fund_event"
        evidence = EvidenceMode.ANNOUNCEMENTS
    elif (
        any(term in text for term in _EVENT_TERMS)
        or any(term in text for term in _RULE_TERMS)
        or any(term in text for term in _DESIGN_WORDS)
    ):
        kind = FundTaskKind.RULE_DESIGN
        operation = "design_or_explain_fund_rules"
        evidence = EvidenceMode.USER_PROVIDED
    elif resolvable and any(term in text for term in _CURRENT_DATA_TERMS + ("分析", "走势", "比较", "对比")):
        kind = FundTaskKind.INSTRUMENT_RESEARCH
        operation = "research_fund_instrument"
        evidence = (
            EvidenceMode.REALTIME_MARKET
            if fund_domain == FundDomain.EXCHANGE_FUND and any(term in text for term in _CURRENT_DATA_TERMS)
            else EvidenceMode.NAV_HISTORY
        )
    else:
        kind = FundTaskKind.EDUCATION
        operation = "explain_fund_concept"
        evidence = EvidenceMode.USER_PROVIDED if re.search(r"\d", text) else EvidenceMode.NONE

    requires_verified = kind in {FundTaskKind.INSTRUMENT_RESEARCH, FundTaskKind.EVENT_RESEARCH}
    missing = ["可供Provider核验的基金代码或准确名称"] if requires_verified and not resolvable else []
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
        FundTaskKind.INSTRUMENT_RESEARCH: (
            ["exchange_fund.comprehensive_analysis"]
            if fund_domain == FundDomain.EXCHANGE_FUND
            else ["open_fund.comprehensive_analysis"]
        ),
        FundTaskKind.UNIVERSE_RESEARCH: (
            ["exchange_fund.screen_compare"]
            if fund_domain == FundDomain.EXCHANGE_FUND
            else ["open_fund.screen_compare"]
        ),
        FundTaskKind.EVENT_RESEARCH: (
            ["exchange_fund.event_risk"]
            if fund_domain == FundDomain.EXCHANGE_FUND
            else ["open_fund.event_risk"]
        ),
        FundTaskKind.SIMULATION_QUERY: ["simulation.read"],
        FundTaskKind.SIMULATION_MUTATION: ["simulation.read", "simulation.write"],
    }.get(kind, [])
    selection = (
        _selection_requirements(text, fund_domain=fund_domain, product_category=product_category)
        if kind == FundTaskKind.UNIVERSE_RESEARCH
        else None
    )
    if (
        kind == FundTaskKind.UNIVERSE_RESEARCH
        and fund_domain == FundDomain.OPEN_FUND
        and product_category in {FundProductCategory.UNKNOWN, FundProductCategory.QDII, FundProductCategory.FOF}
    ):
        missing.append("可正式筛选的单一场外基金类别")
    return FundIntentSpec(
        task_kind=kind,
        operation=operation,
        subject=subject,
        evidence_mode=evidence,
        user_inputs=_extract_user_inputs(message),
        instruments=instruments,
        missing_inputs=missing,
        requires_live_data=live_data,
        requires_verified_instrument=requires_verified,
        required_outputs=_required_outputs(text, kind, selection),
        selection_requirements=selection,
        allowed_capabilities=capabilities,
        forbidden_capabilities=(
            ["market.quote", "market.history", "exchange_fund.nav", "stock.comprehensive_analysis"]
            if fund_domain == FundDomain.OPEN_FUND
            else ["open_fund.nav", "open_fund.money_yield"]
        ),
        confidence=0.98 if kind != FundTaskKind.EDUCATION else 0.9,
    )


def uses_direct_fund_executor(spec: FundIntentSpec) -> bool:
    return spec.task_kind in {
        FundTaskKind.EDUCATION,
        FundTaskKind.CALCULATION,
        FundTaskKind.RULE_DESIGN,
        FundTaskKind.SCENARIO_PLAN,
        FundTaskKind.CLARIFICATION,
    }
