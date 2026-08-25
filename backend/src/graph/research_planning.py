"""Deterministic planning policy for the market research graph.

This module contains plan classification, fallback construction, and contract
validation.  Runtime graph wiring and tool execution remain in
``graph.research_plan``.
"""

from __future__ import annotations

from typing import Any

from models.research_plan import ResearchBudget, ResearchPlan, ResearchStep
from models.schemas import AssetType
from models.strategy_research import TaskContract, strategy_comparison_contract

DEPTH_BUDGETS = {
    "quick": ResearchBudget(max_steps=3, max_tool_calls=4, max_replans=1, deadline_seconds=1800),
    "standard": ResearchBudget(max_steps=8, max_tool_calls=16, max_replans=1, deadline_seconds=1800),
    "deep": ResearchBudget(max_steps=16, max_tool_calls=32, max_replans=2, deadline_seconds=1800),
}
DEPTH_STEP_RANGES = {"quick": (1, 3), "standard": (4, 8), "deep": (9, 16)}

REFLECTABLE_LONG_STEP_KINDS = {"backtest", "comprehensive_analysis"}
SINGLE_ATTEMPT_STEP_KINDS = {"report"}
DEEP_TERMS = ("深度", "全面", "系统", "多源", "调研报告", "deep research")


def effective_max_attempts(step: ResearchStep) -> int:
    """Upgrade legacy read-only checkpoints without changing write-side steps."""
    return 2 if step.kind in REFLECTABLE_LONG_STEP_KINDS else step.max_attempts


def classify_depth(request: dict[str, Any]) -> str:
    message = str(request.get("message", "")).lower()
    intent = str(request.get("intent", "analyze"))
    if any(term in message for term in DEEP_TERMS):
        return "deep"
    if intent in {"quote", "history", "strategies"}:
        return "quick"
    return "standard"


def derive_task_contract(request: dict[str, Any]) -> TaskContract:
    """Translate user wording into terminal acceptance criteria before planning."""
    message = str(request.get("message", ""))
    intent = str(request.get("intent", "analyze"))
    compares_strategies = (
        intent == "backtest"
        and "策略" in message
        and any(term in message for term in ("不同", "多个", "几个", "多种", "对比", "比较"))
    )
    if compares_strategies:
        return strategy_comparison_contract()
    if intent == "backtest" and any(term in message.lower() for term in ("python", "代码", "沙盒", "自定义因子")):
        return TaskContract(
            operation="sandbox_research",
            comparison_axis="none",
            minimum_strategy_count=1,
            required_benchmark="buy_hold",
            minimum_history_years=5.0,
            required_metrics=("total_return", "max_drawdown", "sharpe_ratio"),
            required_outputs=("sandbox_validation", "equity_curves", "deterministic_replay"),
        )
    return TaskContract(
        operation="backtest" if intent == "backtest" else "research",
        comparison_axis="asset" if intent == "compare" else "none",
        minimum_strategy_count=1,
        required_benchmark="buy_hold" if intent == "backtest" else None,
        minimum_history_years=1.0,
        required_metrics=("total_return", "max_drawdown") if intent == "backtest" else (),
        required_outputs=("equity_curves",) if intent == "backtest" else (),
    )


def _step(kind: str, title: str, depends_on: list[str] | None = None, *, attempts: int = 2) -> dict[str, Any]:
    return {
        "id": kind,
        "kind": kind,
        "title": title,
        "depends_on": depends_on or [],
        "inputs": {},
        "success_criteria": ["返回带来源和数据状态的可审计结果"],
        "max_attempts": attempts,
    }


def fallback_steps(request: dict[str, Any], depth: str) -> list[dict[str, Any]]:
    intent = str(request.get("intent", "analyze"))
    asset_type = str(request.get("asset_type", "stock"))
    if intent == "quote" and depth == "quick":
        return [_step("market_snapshot", "获取最新结构化行情")]
    if intent == "history" and depth == "quick":
        return [_step("price_history", "获取历史价格与走势数据")]
    if intent == "strategies" and depth == "quick":
        return [_step("methodology", "整理可用策略与方法论")]

    steps: list[dict[str, Any]] = []
    if depth == "deep" or intent == "news":
        steps.append(_step("instrument_profile", "核对标的类型与基础资料"))
    steps.append(_step("market_snapshot", "获取最新行情快照"))
    if intent in {"analyze", "compare", "backtest"} or depth == "deep":
        steps.append(_step("price_history", "获取可验证的历史价格序列"))
    is_fund = asset_type in {"etf", "lof"}
    if is_fund and depth != "quick":
        steps.append(_step("fund_nav", "核对基金净值、折溢价和历史表现"))
    if depth == "deep" or (is_fund and depth == "standard"):
        steps.append(_step("liquidity", "评估成交量、成交额和流动性风险", ["market_snapshot"]))
    if intent == "news":
        steps.append(_step("news", "检索最新资讯、公告与催化"))
    elif intent == "compare":
        steps.append(_step("comparison", "对比候选标的行情与差异", ["market_snapshot"]))
        if not (is_fund and depth == "standard"):
            steps.append(_step("technical", "比较趋势与技术指标", ["price_history"]))
        steps.append(_step("news", "比较近期资讯与事件风险"))
    elif intent == "backtest":
        steps.extend(
            [
                _step("methodology", "确定可复现的策略假设"),
                _step("backtest", "执行历史回测并保存实验结果", ["price_history", "methodology"]),
            ]
        )
    else:
        if not (is_fund and depth == "standard"):
            steps.append(_step("technical", "计算趋势、动量和量价指标", ["price_history"]))
        if asset_type == "stock":
            steps.append(_step("fundamentals", "核对股票基本面数据"))
        if not (is_fund and depth == "standard"):
            steps.append(_step("news", "检索最新资讯、公告和风险事件"))
        dependencies = ["market_snapshot"]
        if not (is_fund and depth == "standard"):
            dependencies.extend(["technical", "news"])
        dependencies.append("fundamentals" if asset_type == "stock" else "fund_nav")
        steps.append(_step("comprehensive_analysis", "运行多角色综合分析、辩论与风控", dependencies))
    if depth == "deep" and not any(item["kind"] == "technical" for item in steps):
        steps.append(_step("technical", "核对趋势、动量和量价指标", ["price_history"]))
    if depth == "deep" and not any(item["kind"] == "news" for item in steps):
        steps.append(_step("news", "检索最新资讯、公告和风险事件"))
    if depth == "deep" and not any(item["kind"] == "methodology" for item in steps):
        steps.append(_step("methodology", "检索并验证适用的投资方法论"))
    if intent in {"analyze", "compare", "backtest", "news"} or depth == "deep":
        dependencies = [
            item["id"] for item in steps if item["kind"] in {"comprehensive_analysis", "backtest", "comparison"}
        ] or [item["id"] for item in steps if item["kind"] in {"market_snapshot", "news"}]
        steps.append(_step("risk", "汇总回撤、流动性、仓位和持有期风险", dependencies))
    steps.append(_step("synthesis", "综合证据并形成短中期研究结论", [item["id"] for item in steps]))
    if "报告" in str(request.get("message", "")) or "保存" in str(request.get("message", "")):
        steps.append(_step("report", "生成可预览和下载的研究报告", ["synthesis"], attempts=1))

    protected = {"risk", "synthesis", "report"}
    protected.add(
        {
            "quote": "market_snapshot",
            "history": "price_history",
            "strategies": "methodology",
            "news": "news",
            "compare": "comparison",
            "backtest": "backtest",
            "analyze": "comprehensive_analysis",
        }.get(intent, "synthesis")
    )
    if is_fund and depth != "quick":
        protected.update({"fund_nav", "liquidity"})
    for kind in ("fundamentals", "technical", "news", "instrument_profile", "methodology", "market_snapshot"):
        if len(steps) <= DEPTH_STEP_RANGES[depth][1]:
            break
        if kind in protected:
            continue
        removed_ids = {item["id"] for item in steps if item["kind"] == kind}
        if not removed_ids:
            continue
        steps = [item for item in steps if item["id"] not in removed_ids]
        for item in steps:
            item["depends_on"] = [dependency for dependency in item["depends_on"] if dependency not in removed_ids]
    return steps


def normalize_steps(raw: Any, request: dict[str, Any], depth: str) -> list[dict[str, Any]]:
    candidate = raw.get("steps") if isinstance(raw, dict) else None
    if not isinstance(candidate, list) or not candidate:
        return fallback_steps(request, depth)
    normalized = []
    for index, item in enumerate(candidate):
        if not isinstance(item, dict):
            continue
        value = dict(item)
        value.setdefault("id", f"step-{index + 1}")
        value.setdefault("title", str(value.get("kind", "研究步骤")))
        value.setdefault("depends_on", [])
        value.setdefault("inputs", {})
        value.setdefault("success_criteria", ["返回带来源和数据状态的可审计结果"])
        if isinstance(value.get("success_criteria"), str):
            value["success_criteria"] = [value["success_criteria"]]
        if value.get("kind") in SINGLE_ATTEMPT_STEP_KINDS:
            value["max_attempts"] = 1
        elif value.get("kind") in REFLECTABLE_LONG_STEP_KINDS:
            value["max_attempts"] = 2
        else:
            value.setdefault("max_attempts", 2)
            try:
                value["max_attempts"] = max(1, min(int(value["max_attempts"]), 2))
            except (TypeError, ValueError):
                value["max_attempts"] = 2
        normalized.append(value)
    return normalized or fallback_steps(request, depth)


def validate_plan_contract(plan: ResearchPlan, request: dict[str, Any], budget: ResearchBudget) -> None:
    minimum, maximum = DEPTH_STEP_RANGES[plan.depth]
    if not minimum <= len(plan.steps) <= min(maximum, budget.max_steps):
        raise ValueError(f"研究计划不符合 {plan.depth} 深度的步骤数预算")
    if any(step.kind in SINGLE_ATTEMPT_STEP_KINDS and step.max_attempts != 1 for step in plan.steps):
        raise ValueError("带外部写入副作用的报告步骤只允许执行一次")
    if plan.asset_type in {AssetType.ETF, AssetType.LOF} and any(
        step.kind in {"technical", "comprehensive_analysis", "backtest", "comparison", "news"} for step in plan.steps
    ):
        kinds = {step.kind for step in plan.steps}
        if not {"fund_nav", "liquidity"} <= kinds:
            raise ValueError("ETF/LOF 标准或深度研究必须包含 fund_nav 和 liquidity 步骤")
    kinds = {step.kind for step in plan.steps}
    if "data_query" in kinds:
        if not {"data_catalog", "data_query", "data_validation", "synthesis"} <= kinds:
            raise ValueError("结构化数据研究必须包含目录、查询、验收与综合步骤")
        if next(step for step in plan.steps if step.kind == "data_validation").id not in next(
            step for step in plan.steps if step.kind == "synthesis"
        ).depends_on:
            raise ValueError("结构化数据综合必须依赖业务验收步骤")
        return
    if str(request.get("intent")) in {"analyze", "compare", "backtest", "news"}:
        by_kind = {step.kind: step for step in plan.steps}
        if not {"risk", "synthesis"} <= set(by_kind):
            raise ValueError("研究结论计划必须包含 risk 和 synthesis 步骤")
        if by_kind["risk"].id not in by_kind["synthesis"].depends_on:
            raise ValueError("synthesis 必须依赖 risk 步骤")
