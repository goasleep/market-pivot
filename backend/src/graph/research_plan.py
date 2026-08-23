"""Checkpointed orchestrator-worker graph for market Deep Research."""

from __future__ import annotations

import asyncio
import hashlib
import json
import operator
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, TypedDict, get_args
from uuid import uuid4

from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from loguru import logger

from graph.agent_loop import tool_timeout_seconds
from llm.context import (
    TokenCounter,
    compact_json_value,
    context_safe_error,
    get_context_budget,
    is_context_overflow_error,
)
from llm.service import get_llm_service
from models.research_plan import (
    EvidenceRef,
    ResearchBudget,
    ResearchPlan,
    ResearchStep,
    ResearchStepKind,
    StepRecovery,
    StepResult,
)
from models.schemas import AssetType
from models.strategy_research import TaskContract, strategy_comparison_contract


def merge_step_results(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {**(left or {}), **(right or {})}


class ResearchPlanState(TypedDict, total=False):
    request: dict[str, Any]
    depth: str
    budget: dict[str, Any]
    deadline_at: str
    plan: dict[str, Any]
    step: dict[str, Any]
    step_results: Annotated[dict[str, dict[str, Any]], merge_step_results]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    replan_count: int
    tool_calls: Annotated[int, operator.add]
    needs_replan: bool
    final_response: str
    task_contract: dict[str, Any]


@dataclass(frozen=True)
class ResearchPlanContext:
    tools: dict[str, StructuredTool]


class ResearchToolExecutionError(RuntimeError):
    """A tool observation that retains safe call context for recovery."""

    def __init__(self, tool_name: str, args: dict[str, Any], message: str):
        super().__init__(message)
        self.tool_name = tool_name
        self.tool_args = _compact(args)


DEPTH_BUDGETS = {
    "quick": ResearchBudget(max_steps=3, max_tool_calls=4, max_replans=1, deadline_seconds=300),
    "standard": ResearchBudget(max_steps=8, max_tool_calls=16, max_replans=1, deadline_seconds=900),
    "deep": ResearchBudget(max_steps=16, max_tool_calls=32, max_replans=2, deadline_seconds=1800),
}
DEPTH_STEP_RANGES = {"quick": (1, 3), "standard": (4, 8), "deep": (9, 16)}

REFLECTABLE_LONG_STEP_KINDS = {"backtest", "comprehensive_analysis"}
SINGLE_ATTEMPT_STEP_KINDS = {"report"}
DEEP_TERMS = ("深度", "全面", "系统", "多源", "调研报告", "deep research")

RECOVERY_INPUT_RULES: dict[str, dict[str, str]] = {
    "price_history": {"limit": "20 到 500 的整数"},
    "fund_nav": {"limit": "20 到 500 的整数"},
    "technical": {"limit": "20 到 500 的整数"},
    "news": {
        "query": "不超过 500 字的检索词",
        "num_results": "1 到 20 的整数",
        "freshness": "搜索工具支持的时间过滤表达式",
    },
    "methodology": {"query": "不超过 500 字的检索词", "limit": "1 到 10 的整数"},
    "backtest": {
        "start_date": "YYYY-MM-DD，必须早于 end_date",
        "end_date": "YYYY-MM-DD，不能晚于研究截止日",
        "objective": "保留用户原目标，并补充解决错误所需的约束，不超过 2000 字",
        "execution_mode": "agent、comparison 或 sandbox",
        "initial_capital": "1000 到 1000000000 的数值",
        "decision_interval": "1 到 250 的整数",
    },
    "risk": {
        "stop_loss_pct": "0 到 0.5 的小数",
        "take_profit_pct": "0 到 2 的小数",
        "position_size_pct": "0 到 1 的小数",
    },
}


def _effective_max_attempts(step: ResearchStep) -> int:
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


async def scope_research(state: ResearchPlanState) -> dict[str, Any]:
    depth = classify_depth(state["request"])
    budget = DEPTH_BUDGETS[depth]
    deadline = datetime.now(timezone.utc) + timedelta(seconds=budget.deadline_seconds)
    contract = derive_task_contract(state["request"])
    return {
        "depth": depth,
        "budget": budget.model_dump(mode="json"),
        "deadline_at": deadline.isoformat(),
        "step_results": {},
        "evidence": [],
        "replan_count": 0,
        "tool_calls": 0,
        "needs_replan": False,
        "task_contract": contract.model_dump(mode="json"),
    }


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


def _fallback_steps(request: dict[str, Any], depth: str) -> list[dict[str, Any]]:
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
        comprehensive_dependencies = ["market_snapshot"]
        if not (is_fund and depth == "standard"):
            comprehensive_dependencies.extend(["technical", "news"])
        comprehensive_dependencies.append("fundamentals" if asset_type == "stock" else "fund_nav")
        steps.append(
            _step(
                "comprehensive_analysis",
                "运行多角色综合分析、辩论与风控",
                comprehensive_dependencies,
                attempts=2,
            )
        )
    if depth == "deep" and not any(step["kind"] == "technical" for step in steps):
        steps.append(_step("technical", "核对趋势、动量和量价指标", ["price_history"]))
    if depth == "deep" and not any(step["kind"] == "news" for step in steps):
        steps.append(_step("news", "检索最新资讯、公告和风险事件"))
    if depth == "deep" and not any(step["kind"] == "methodology" for step in steps):
        steps.append(_step("methodology", "检索并验证适用的投资方法论"))
    if intent in {"analyze", "compare", "backtest", "news"} or depth == "deep":
        risk_dependencies = [
            step["id"] for step in steps if step["kind"] in {"comprehensive_analysis", "backtest", "comparison"}
        ] or [step["id"] for step in steps if step["kind"] in {"market_snapshot", "news"}]
        steps.append(_step("risk", "汇总回撤、流动性、仓位和持有期风险", risk_dependencies))
    steps.append(_step("synthesis", "综合证据并形成短中期研究结论", [step["id"] for step in steps]))
    if "报告" in str(request.get("message", "")) or "保存" in str(request.get("message", "")):
        steps.append(_step("report", "生成可预览和下载的研究报告", ["synthesis"], attempts=1))
    max_steps = DEPTH_STEP_RANGES[depth][1]
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
    removable = ["fundamentals", "technical", "news", "instrument_profile", "methodology", "market_snapshot"]
    for kind in removable:
        if len(steps) <= max_steps:
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


def _normalize_steps(raw: Any, request: dict[str, Any], depth: str) -> list[dict[str, Any]]:
    candidate = raw.get("steps") if isinstance(raw, dict) else None
    if not isinstance(candidate, list) or not candidate:
        return _fallback_steps(request, depth)
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
        if value.get("kind") in SINGLE_ATTEMPT_STEP_KINDS:
            value["max_attempts"] = 1
        elif value.get("kind") in REFLECTABLE_LONG_STEP_KINDS:
            value["max_attempts"] = 2
        else:
            value.setdefault("max_attempts", 2)
        normalized.append(value)
    return normalized or _fallback_steps(request, depth)


def _validate_plan_contract(
    plan: ResearchPlan,
    request: dict[str, Any],
    budget: ResearchBudget,
) -> None:
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
    if str(request.get("intent")) in {"analyze", "compare", "backtest", "news"}:
        by_kind = {step.kind: step for step in plan.steps}
        if not {"risk", "synthesis"} <= set(by_kind):
            raise ValueError("研究结论计划必须包含 risk 和 synthesis 步骤")
        if by_kind["risk"].id not in by_kind["synthesis"].depends_on:
            raise ValueError("synthesis 必须依赖 risk 步骤")


async def plan_research(state: ResearchPlanState) -> dict[str, Any]:
    request = state["request"]
    depth = state["depth"]
    budget = ResearchBudget.model_validate(state["budget"])
    prompt = json.dumps(
        {
            "objective": request.get("message"),
            "intent": request.get("intent"),
            "tickers": request.get("tickers", []),
            "asset_type": request.get("asset_type", "stock"),
            "depth": depth,
            "max_steps": budget.max_steps,
            "allowed_kinds": list(get_args(ResearchStepKind)),
            "rules": [
                "只输出公开的操作计划，不输出内部思维链",
                "步骤必须构成无环依赖图",
                "市场数值必须由结构化数据步骤提供",
                "最终结论必须依赖证据步骤",
                "回测和综合分析步骤 max_attempts=2，报告步骤 max_attempts=1",
            ],
        },
        ensure_ascii=False,
    )
    try:
        raw = await get_llm_service().chat_json(
            prompt,
            system=(
                "你是市场研究 Planner。仅返回 JSON 对象："
                "{steps:[{id,kind,title,depends_on,inputs,success_criteria,max_attempts}]}。"
            ),
        )
        steps = _normalize_steps(raw, request, depth)
    except Exception as exc:
        logger.warning("Research planner fell back to deterministic template: {}", exc)
        steps = _fallback_steps(request, depth)

    plan_kwargs = {
        "plan_id": f"research-{uuid4().hex[:16]}",
        "objective": str(request.get("message", "市场研究")),
        "asset_type": AssetType(str(request.get("asset_type", "stock"))),
        "tickers": [str(item) for item in request.get("tickers", [])],
        "as_of_date": str(request.get("as_of_date") or date.today().isoformat()),
        "depth": depth,
        "task_contract": state.get("task_contract"),
    }
    try:
        plan = ResearchPlan(**plan_kwargs, steps=[ResearchStep.model_validate(item) for item in steps])
        _validate_plan_contract(plan, request, budget)
    except (TypeError, ValueError) as exc:
        logger.warning("Research planner output was invalid; using deterministic template: {}", exc)
        plan = ResearchPlan(
            **plan_kwargs,
            steps=[ResearchStep.model_validate(item) for item in _fallback_steps(request, depth)],
        )
        _validate_plan_contract(plan, request, budget)
    return {"plan": plan.model_dump(mode="json")}


async def validate_plan(state: ResearchPlanState) -> dict[str, Any]:
    plan = ResearchPlan.model_validate(state["plan"])
    budget = ResearchBudget.model_validate(state["budget"])
    _validate_plan_contract(plan, state["request"], budget)
    return {"plan": plan.model_dump(mode="json")}


async def dispatch_ready(state: ResearchPlanState) -> dict[str, Any]:
    return {}


def route_dispatch(state: ResearchPlanState) -> list[Send] | str:
    plan = ResearchPlan.model_validate(state["plan"])
    results = {key: StepResult.model_validate(value) for key, value in (state.get("step_results") or {}).items()}
    completed = {step_id for step_id, result in results.items() if result.status == "completed"}
    ready = [
        step
        for step in plan.steps
        if (step.id not in results or results[step.id].status == "pending") and set(step.depends_on) <= completed
    ]
    if not ready:
        return "verify"
    budget = ResearchBudget.model_validate(state["budget"])
    remaining_calls = max(0, budget.max_tool_calls - state.get("tool_calls", 0))
    if remaining_calls == 0:
        return "verify"
    return [
        Send(
            "worker",
            {
                "request": state["request"],
                "depth": state["depth"],
                "budget": state["budget"],
                "deadline_at": state["deadline_at"],
                "plan": state["plan"],
                "step": step.model_dump(mode="json"),
                "step_results": state.get("step_results", {}),
                "replan_count": state.get("replan_count", 0),
                "tool_calls": state.get("tool_calls", 0),
                "task_contract": state.get("task_contract", {}),
            },
        )
        for step in ready[: min(budget.max_parallel, remaining_calls)]
    ]


def _compact(value: Any, *, depth: int = 0) -> Any:
    if depth == 0 and isinstance(value, dict) and value.get("data_type") == "strategy_backtest_comparison":
        return _compact_strategy_comparison(value)
    if depth >= 5:
        return str(value)[:500]
    if isinstance(value, dict):
        return {str(key): _compact(item, depth=depth + 1) for key, item in list(value.items())[:30]}
    if isinstance(value, list):
        return [_compact(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, str):
        return value[:4000]
    return value


def _sample_curve(points: Any, maximum: int = 240) -> list[dict[str, Any]]:
    if not isinstance(points, list):
        return []
    rows = [item for item in points if isinstance(item, dict)]
    if len(rows) <= maximum:
        return rows
    step = max(1, (len(rows) - 1) // (maximum - 1) + 1)
    sampled = rows[::step]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled[: maximum - 1] + [rows[-1]] if len(sampled) > maximum else sampled


def _compact_strategy_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep A2UI/audit metadata while bounding checkpoint and SSE payload size."""
    comparison_fields = {
        "strategy_name",
        "display_name",
        "description",
        "strategy_spec",
        "entry_rules",
        "exit_rules",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "win_rate",
        "profit_factor",
        "exposure",
        "turnover",
        "total_fees",
        "excess_return",
        "final_value",
        "total_trades",
        "metrics",
        "diagnostics",
        "error",
    }
    comparisons = []
    for row in payload.get("comparisons", []):
        if not isinstance(row, dict):
            continue
        compact_row = {key: value for key, value in row.items() if key in comparison_fields}
        compact_row["equity_curve"] = _sample_curve(row.get("equity_curve"))
        compact_row["drawdown_curve"] = _sample_curve(row.get("drawdown_curve"))
        compact_row["signal_curve"] = _sample_curve(row.get("signal_curve"))
        comparisons.append(compact_row)
    keep = {
        "comparison_id",
        "data_type",
        "_tool_name",
        "ticker",
        "asset_type",
        "start_date",
        "end_date",
        "requested_start_date",
        "warmup_start_date",
        "evaluation_start_date",
        "evaluation_end_date",
        "warmup_bars",
        "actual_start_date",
        "actual_end_date",
        "history_years",
        "initial_capital",
        "strategy_count",
        "benchmark",
        "ranking_metric",
        "ranking_label",
        "task_contract",
        "data_snapshot",
        "execution",
        "ranking",
        "cost_scenarios",
        "cost_consistency",
        "parameter_sensitivity",
        "acceptance",
        "conclusion",
        "artifacts",
        "artifact_error",
        "provenance",
        "decision_interval",
    }
    compact = {key: value for key, value in payload.items() if key in keep}
    compact["comparisons"] = comparisons
    validation = dict(payload.get("data_validation") or {})
    validation["differences"] = (validation.get("differences") or [])[:20]
    compact["data_validation"] = validation
    return compact


def _evidence(payload: dict[str, Any], kind: str) -> list[EvidenceRef]:
    now = datetime.now(timezone.utc).isoformat()

    def collect_provenance(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [row for item in value for row in collect_provenance(item)]
        if not isinstance(value, dict):
            return []
        found: list[dict[str, Any]] = []
        provenance = value.get("provenance")
        if isinstance(provenance, list):
            found.extend(item for item in provenance if isinstance(item, dict))
        elif isinstance(provenance, dict):
            found.append(provenance)
        for key, child in value.items():
            if key != "provenance":
                found.extend(collect_provenance(child))
        return found

    rows = collect_provenance(payload)
    source_type = (
        "web"
        if kind == "news"
        else "methodology"
        if kind == "methodology"
        else "backtest"
        if kind == "backtest"
        else "derived"
        if kind in {"risk", "synthesis"}
        else "market_data"
    )
    result = []
    for row in rows or [{}]:
        result.append(
            EvidenceRef(
                source=str(
                    row.get("name")
                    or row.get("source")
                    or row.get("source_id")
                    or row.get("provider")
                    or payload.get("data_type")
                    or kind
                ),
                source_type=source_type,
                as_of=str(
                    row.get("as_of")
                    or payload.get("searched_at")
                    or (row.get("fetched_at") if row.get("freshness") in {"realtime", "latest_available"} else "")
                    or ""
                )
                or None,
                retrieved_at=str(row.get("fetched_at") or now),
                data_status=str(row.get("status") or "available"),
                url=row.get("url"),
                content_hash=hashlib.sha256(
                    json.dumps(_compact(payload), sort_keys=True, default=str).encode()
                ).hexdigest(),
            )
        )
    return result


def _result_summaries(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step_id": item.get("step_id"),
            "status": item.get("status"),
            "summary": item.get("summary"),
            "output": _synthesis_output(item.get("output", {})),
        }
        for item in results.values()
    ]


def _synthesis_request(request: dict[str, Any]) -> dict[str, Any]:
    """Keep only current-task fields; cross-turn history is not synthesis evidence."""
    keys = (
        "message",
        "intent",
        "tickers",
        "asset_type",
        "strategy",
        "as_of_date",
    )
    return {key: request.get(key) for key in keys if request.get(key) is not None}


def _deterministic_synthesis_fallback(evidence: list[dict[str, Any]]) -> str:
    completed = [
        f"- {item.get('step_id')}: {item.get('summary') or '已完成'}"
        for item in evidence
        if item.get("status") == "completed"
    ]
    failed = [str(item.get("step_id")) for item in evidence if item.get("status") in {"failed", "skipped"}]
    lines = ["研究证据已完成汇总。", *(completed[:12] or ["- 暂无足够的已完成证据步骤。"])]
    if failed:
        lines.append("数据不足或未完成步骤：" + "、".join(failed[:8]) + "。")
    lines.append("请结合证据日期、数据缺失和风险约束审慎判断；以上仅用于短中期研究与模拟交易，不承诺收益。")
    return "\n".join(lines)


def _synthesis_output(output: Any) -> Any:
    """Bound the evidence sent to synthesis without weakening the public A2UI payload."""
    if not isinstance(output, dict) or output.get("data_type") != "strategy_backtest_comparison":
        return _compact(output)
    metric_fields = (
        "strategy_name",
        "display_name",
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "total_fees",
        "final_value",
    )
    return {
        "data_type": output.get("data_type"),
        "ticker": output.get("ticker"),
        "evaluation_start_date": output.get("evaluation_start_date"),
        "evaluation_end_date": output.get("evaluation_end_date"),
        "warmup_bars": output.get("warmup_bars"),
        "strategy_count": output.get("strategy_count"),
        "comparisons": [
            {key: row.get(key) for key in metric_fields}
            for row in (output.get("comparisons") or [])
            if isinstance(row, dict)
        ],
        "conclusion": output.get("conclusion"),
        "data_validation": {
            key: (output.get("data_validation") or {}).get(key)
            for key in ("status", "selected_source", "selection_reason", "rule_version")
        },
        "execution": output.get("execution"),
        "acceptance": output.get("acceptance"),
        "artifacts": [
            {key: artifact.get(key) for key in ("name", "mime_type", "size_bytes")}
            for artifact in (output.get("artifacts") or [])
            if isinstance(artifact, dict)
        ],
    }


def _format_winner_metric(metric: str, value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "指标缺失"
    if metric in {"total_return", "max_drawdown", "out_of_sample_return", "stress_total_return"}:
        sign = "" if metric == "max_drawdown" else "+"
        return f"{sign}{number:.2%}"
    return f"{number:.3f}"


def _comparison_synthesis_text(payload: dict[str, Any]) -> str:
    """Render frozen comparison facts without asking an LLM to re-rank strategies."""
    conclusion = payload.get("conclusion") or {}
    validation = payload.get("data_validation") or {}
    execution = payload.get("execution") or {}
    snapshot = payload.get("data_snapshot") or {}
    lines = [
        f"{payload.get('ticker') or '该标的'} 多策略回测已完成。统一评价区间为 "
        f"{payload.get('evaluation_start_date', '—')} 至 "
        f"{payload.get('evaluation_end_date', '—')}，热身期 {payload.get('warmup_bars', '—')} 个交易日，"
        f"共比较 {payload.get('strategy_count') or len(payload.get('comparisons') or [])} 个策略。",
        (
            f"行情采用 {snapshot.get('adjustment') or '前复权/数据源声明方式'}；基准成交成本为买入佣金 "
            f"{float(execution.get('buy_commission_rate') or 0):.3%}、卖出佣金 "
            f"{float(execution.get('sell_commission_rate') or 0):.3%}、滑点 "
            f"{float(execution.get('slippage_bps') or 0):g} bps，并按下一交易日开盘执行。"
        ),
        "",
        "程序确定的五类优胜者：",
    ]
    winner_labels = (
        ("absolute_return_winner", "绝对收益"),
        ("risk_adjusted_winner", "风险收益"),
        ("drawdown_winner", "回撤控制"),
        ("out_of_sample_winner", "样本外"),
        ("robustness_winner", "稳健性"),
    )
    winner_count = 0
    for key, label in winner_labels:
        winner = conclusion.get(key)
        if not isinstance(winner, dict):
            continue
        winner_count += 1
        name = winner.get("display_name") or winner.get("strategy_name") or "—"
        lines.append(
            f"- {label}：{name}（{winner.get('metric', 'metric')} "
            f"{_format_winner_metric(str(winner.get('metric') or ''), winner.get('value'))}）"
        )
    if not winner_count:
        lines.append("- 当前数据覆盖或核验状态不足以形成正式优胜者，结果仅作探索性比较。")
    lines.extend(["", "为什么不存在唯一“最好策略”："])
    lines.extend(f"- {item}" for item in (conclusion.get("tradeoffs") or ["不同评价维度对应不同交易取舍。"]))
    lines.extend(
        [
            "",
            f"数据核验状态：{validation.get('status', 'unknown')}；选定数据源："
            f"{validation.get('selected_source') or '未记录'}。{validation.get('selection_reason') or ''}",
        ]
    )
    warnings = conclusion.get("data_warnings") or []
    limitations = conclusion.get("limitations") or []
    if warnings:
        lines.append("数据警告：" + "；".join(str(item) for item in warnings))
    if limitations:
        lines.append("局限：" + "；".join(str(item) for item in limitations))
    artifacts = payload.get("artifacts") or []
    lines.append(
        f"完整可审计成果包已生成，共 {len(artifacts)} 个文件，包含 HTML、XLSX、JSON 与 CSV；可在上方成果区预览或下载。"
    )
    lines.append("以上仅用于研究与模拟盘，不构成收益承诺或直接交易建议。")
    return "\n".join(lines)


def _find_price(results: dict[str, dict[str, Any]]) -> float | None:
    def walk(value: Any) -> float | None:
        if isinstance(value, dict):
            quote = value.get("quote")
            if isinstance(quote, dict):
                try:
                    price = float(quote.get("price") or quote.get("最新价") or 0)
                    if price > 0:
                        return price
                except (TypeError, ValueError):
                    pass
            for child in value.values():
                found = walk(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = walk(child)
                if found:
                    return found
        return None

    return walk(results)


def _classify_failure(error: str) -> str:
    """Classify a public tool error before deciding whether reflection is useful."""
    text = error.lower()
    if any(
        token in text
        for token in (
            "user_denied",
            "unauthorized",
            "forbidden",
            "permission",
            "用户拒绝",
            "没有权限",
            "权限不足",
            "工具不可用",
            "不支持的研究步骤",
        )
    ):
        return "terminal"
    if any(
        token in text
        for token in (
            "timeout",
            "timed out",
            "rate limit",
            "429",
            "connection",
            "temporarily",
            "service unavailable",
            "超时",
            "限流",
            "网络",
            "连接失败",
            "暂时不可用",
            "服务繁忙",
        )
    ):
        return "transient"
    if any(
        token in text
        for token in (
            "invalid",
            "unsupported",
            "validation",
            "schema",
            "required",
            "missing",
            "argument",
            "parameter",
            "校验",
            "验证",
            "参数",
            "缺少",
            "不能为空",
            "不受支持",
            "指标",
            "格式",
        )
    ):
        return "correctable"
    return "unknown"


def _bounded_number(value: Any, minimum: float, maximum: float, *, integer: bool = False) -> int | float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not minimum <= number <= maximum:
        return None
    return int(number) if integer else number


def _sanitize_recovery_patch(
    step: ResearchStep,
    raw_patch: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Keep LLM recovery changes inside the public, read-only research contract."""
    if not isinstance(raw_patch, dict):
        return {}
    allowed = RECOVERY_INPUT_RULES.get(step.kind, {})
    patch = {key: value for key, value in raw_patch.items() if key in allowed}
    sanitized: dict[str, Any] = {}

    limit_ranges = {
        "price_history": (20, 500),
        "fund_nav": (20, 500),
        "technical": (20, 500),
        "methodology": (1, 10),
    }
    if "limit" in patch and step.kind in limit_ranges:
        value = _bounded_number(patch["limit"], *limit_ranges[step.kind], integer=True)
        if value is not None:
            sanitized["limit"] = value
    if step.kind == "news":
        if isinstance(patch.get("query"), str) and patch["query"].strip():
            sanitized["query"] = patch["query"].strip()[:500]
        value = _bounded_number(patch.get("num_results"), 1, 20, integer=True)
        if value is not None:
            sanitized["num_results"] = value
        if isinstance(patch.get("freshness"), str) and patch["freshness"].strip():
            sanitized["freshness"] = patch["freshness"].strip()[:100]
    if step.kind == "methodology" and isinstance(patch.get("query"), str) and patch["query"].strip():
        sanitized["query"] = patch["query"].strip()[:500]
    if step.kind == "backtest":
        original_objective = str(request.get("message", "")).strip()
        if isinstance(patch.get("objective"), str) and patch["objective"].strip():
            proposed = patch["objective"].strip()
            if original_objective and original_objective not in proposed:
                proposed = f"{original_objective}\n工具修复约束：{proposed}"
            sanitized["objective"] = proposed[:2000]
        mode = str(patch.get("execution_mode", "")).strip().lower()
        tickers = [str(item) for item in request.get("tickers", [])]
        compares_strategies = "策略" in original_objective and any(
            term in original_objective for term in ("不同", "多个", "几个", "多种", "对比", "比较")
        )
        if mode in {"agent", "comparison", "sandbox"}:
            if len(tickers) != 1 and mode != "agent":
                mode = ""
            elif compares_strategies and mode != "comparison":
                mode = ""
        if mode:
            sanitized["execution_mode"] = mode
        capital = _bounded_number(patch.get("initial_capital"), 1_000, 1_000_000_000)
        if capital is not None:
            sanitized["initial_capital"] = capital
        interval = _bounded_number(patch.get("decision_interval"), 1, 250, integer=True)
        if interval is not None:
            sanitized["decision_interval"] = interval

        cutoff_text = str(request.get("as_of_date") or date.today().isoformat())
        current_end = str(step.inputs.get("end_date") or cutoff_text)
        proposed_end = str(patch.get("end_date") or current_end)
        proposed_start = str(patch.get("start_date") or step.inputs.get("start_date") or "")
        try:
            cutoff = date.fromisoformat(cutoff_text)
            end = date.fromisoformat(proposed_end)
            start = date.fromisoformat(proposed_start) if proposed_start else None
        except ValueError:
            start = None
            end = None
            cutoff = None
        if end is not None and cutoff is not None and end <= cutoff:
            if "end_date" in patch:
                sanitized["end_date"] = end.isoformat()
            if start is not None and start < end and "start_date" in patch:
                sanitized["start_date"] = start.isoformat()
    if step.kind == "risk":
        for key, minimum, maximum in (
            ("stop_loss_pct", 0, 0.5),
            ("take_profit_pct", 0, 2),
            ("position_size_pct", 0, 1),
        ):
            value = _bounded_number(patch.get(key), minimum, maximum)
            if value is not None:
                sanitized[key] = value
    return sanitized


async def _reflect_on_failure(
    step: ResearchStep,
    result: StepResult,
    state: ResearchPlanState,
) -> StepRecovery:
    error = str(result.error or "工具执行失败")[:500]
    classification = _classify_failure(error)
    if classification == "terminal":
        return StepRecovery(
            attempt=result.attempt,
            classification="terminal",
            action="abort",
            summary="该错误涉及权限、用户拒绝或不可用能力，已停止自动重试。",
            error=error,
        )
    if classification == "transient":
        return StepRecovery(
            attempt=result.attempt,
            classification="transient",
            action="retry",
            summary="检测到临时网络或服务错误，将在预算内使用原参数重试一次。",
            error=error,
        )

    allowed_inputs = RECOVERY_INPUT_RULES.get(step.kind, {})
    prompt = json.dumps(
        {
            "user_objective": state.get("request", {}).get("message", ""),
            "asset_type": state.get("request", {}).get("asset_type", "stock"),
            "tickers": state.get("request", {}).get("tickers", []),
            "failed_step": step.model_dump(mode="json"),
            "failed_tool_call": result.failure_context,
            "attempt": result.attempt,
            "error": error,
            "classification": classification,
            "allowed_input_patch": allowed_inputs,
        },
        ensure_ascii=False,
        default=str,
    )
    try:
        raw = await get_llm_service().chat_json(
            prompt,
            system=(
                "你是工具失败恢复控制器。只返回 JSON："
                "{action:'retry|adjust|abort',summary:'公开的简短调整说明',input_patch:{}}。"
                "不要输出内部思维链。必须保持用户标的、资产类型和研究目标，不得请求实盘交易或扩大权限；"
                "input_patch 只能使用 allowed_input_patch 中的字段。参数可修正时选 adjust；"
                "原样重试可能成功时选 retry；无法安全恢复时选 abort。"
            ),
        )
    except Exception as exc:
        logger.warning("Research recovery reflection failed; using bounded retry: {}", exc)
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    action = str(raw.get("action") or "retry").lower()
    if action not in {"retry", "adjust", "abort"}:
        action = "retry"
    patch = _sanitize_recovery_patch(step, raw.get("input_patch"), state.get("request", {}))
    if action == "adjust" and not patch:
        action = "retry"
    summary = str(raw.get("summary") or "").strip()[:500] or "已检查失败原因，将在安全预算内重试一次。"
    return StepRecovery(
        attempt=result.attempt,
        classification=classification,
        action=action,
        summary=summary,
        error=error,
        input_patch=patch if action == "adjust" else {},
    )


async def _call_tool(context: ResearchPlanContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    tool = context.tools.get(name)
    if tool is None:
        raise ResearchToolExecutionError(name, args, f"研究步骤需要的工具不可用: {name}")
    timeout = tool_timeout_seconds(name)
    try:
        raw = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError as exc:
        raise ResearchToolExecutionError(name, args, f"工具 {name} 执行超时") from exc
    except Exception as exc:
        raise ResearchToolExecutionError(name, args, str(exc)[:500] or type(exc).__name__) from exc
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError:
        payload = {"value": str(raw)}
    if isinstance(payload, dict) and payload.get("ok") is False:
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        raise ResearchToolExecutionError(name, args, str(message or "工具返回失败状态"))
    return payload if isinstance(payload, dict) else {"value": payload}


async def _execute_step(
    step: ResearchStep,
    state: ResearchPlanState,
    context: ResearchPlanContext,
) -> dict[str, Any]:
    request = state["request"]
    tickers = [str(item) for item in request.get("tickers", [])]
    ticker = tickers[0] if tickers else ""
    asset_type = str(request.get("asset_type", "stock"))
    common = {"ticker": ticker, "asset_type": asset_type}
    if step.kind in {"instrument_profile", "fundamentals"}:
        return await _call_tool(context, "get_fundamentals", common)
    if step.kind in {"market_snapshot", "liquidity"}:
        if len(tickers) > 1:
            return await _call_tool(context, "compare_quotes", {"tickers": tickers, "asset_type": asset_type})
        return await _call_tool(context, "get_realtime_quote", common)
    if step.kind == "price_history":
        limit = int(step.inputs.get("limit") or 120)
        payloads = await asyncio.gather(
            *(
                _call_tool(
                    context,
                    "get_historical_prices",
                    {"ticker": item, "asset_type": asset_type, "limit": limit},
                )
                for item in tickers[:10]
            )
        )
        return {"data_type": "price_history_collection", "items": payloads}
    if step.kind == "fund_nav":
        limit = int(step.inputs.get("limit") or 120)
        payloads = await asyncio.gather(
            *(
                _call_tool(
                    context,
                    "get_fund_nav_history",
                    {"ticker": item, "asset_type": asset_type, "limit": limit},
                )
                for item in tickers[:10]
            )
        )
        return {"data_type": "fund_nav_collection", "items": payloads}
    if step.kind == "technical":
        limit = int(step.inputs.get("limit") or 120)
        payloads = await asyncio.gather(
            *(
                _call_tool(
                    context,
                    "compute_technical_indicators",
                    {"ticker": item, "asset_type": asset_type, "limit": limit},
                )
                for item in tickers[:10]
            )
        )
        return {"data_type": "technical_collection", "items": payloads}
    if step.kind == "news":
        query = str(
            step.inputs.get("query")
            or f"{' '.join(tickers)} {asset_type} 最新新闻 公告 风险 催化 {request.get('message', '')}"
        )
        return await _call_tool(
            context,
            "search_web",
            {
                "query": query,
                "num_results": int(step.inputs.get("num_results") or 10),
                "freshness": str(step.inputs.get("freshness") or "qdr:m"),
            },
        )
    if step.kind == "methodology":
        if str(request.get("intent")) == "strategies":
            return await _call_tool(context, "list_trading_strategies", {})
        return await _call_tool(
            context,
            "search_methodology",
            {
                "query": str(step.inputs.get("query") or request.get("message", "")),
                "asset_type": asset_type,
                "limit": int(step.inputs.get("limit") or 5),
            },
        )
    if step.kind == "comparison":
        return await _call_tool(context, "compare_quotes", {"tickers": tickers, "asset_type": asset_type})
    if step.kind == "backtest":
        original_objective = str(request.get("message", ""))
        explicit_dates = re.findall(r"\d{4}-\d{2}-\d{2}", original_objective)
        explicit_start = explicit_end = None
        if len(explicit_dates) >= 2:
            try:
                first_date = date.fromisoformat(explicit_dates[0])
                second_date = date.fromisoformat(explicit_dates[1])
                if first_date < second_date:
                    explicit_start, explicit_end = first_date.isoformat(), second_date.isoformat()
            except ValueError:
                pass
        end_date = str(
            explicit_end or step.inputs.get("end_date") or request.get("as_of_date") or date.today().isoformat()
        )
        objective = str(step.inputs.get("objective") or request.get("message", ""))
        contract_operation = str((state.get("task_contract") or {}).get("operation") or "")
        compares_strategies = contract_operation == "strategy_comparison" or (
            "策略" in original_objective
            and any(term in original_objective for term in ("不同", "多个", "几个", "多种", "对比", "比较"))
        )
        sandbox_requested = contract_operation == "sandbox_research" or any(
            term in original_objective.lower() for term in ("python", "代码", "沙盒", "自定义因子")
        )
        execution_mode = str(step.inputs.get("execution_mode") or "").lower()
        if contract_operation not in {"strategy_comparison", "sandbox_research"}:
            if execution_mode == "comparison":
                compares_strategies = True
                sandbox_requested = False
            elif execution_mode == "sandbox":
                sandbox_requested = True
                compares_strategies = False
            elif execution_mode == "agent":
                sandbox_requested = False
                compares_strategies = False
        default_days = 3652 if compares_strategies else 1826 if sandbox_requested else 365
        start_date = str(
            explicit_start
            or step.inputs.get("start_date")
            or (date.fromisoformat(end_date) - timedelta(days=default_days)).isoformat()
        )
        shared_backtest_args = {
            "start_date": start_date,
            "end_date": end_date,
            "asset_type": asset_type,
        }
        if "initial_capital" in step.inputs:
            shared_backtest_args["initial_capital"] = float(step.inputs["initial_capital"])
        interval_args = (
            {"decision_interval": int(step.inputs["decision_interval"])} if "decision_interval" in step.inputs else {}
        )
        if len(tickers) == 1 and sandbox_requested:
            return await _call_tool(
                context,
                "design_and_run_sandbox_strategy",
                {
                    "objective": objective,
                    "ticker": ticker,
                    **shared_backtest_args,
                },
            )
        if len(tickers) == 1 and compares_strategies:
            return await _call_tool(
                context,
                "compare_strategy_backtests",
                {
                    "ticker": ticker,
                    "objective": objective,
                    **interval_args,
                    **shared_backtest_args,
                },
            )
        if len(tickers) > 1:
            return await _call_tool(
                context,
                "design_and_run_backtest",
                {
                    "objective": objective,
                    "tickers": tickers,
                    "mode": "pool",
                    **interval_args,
                    **shared_backtest_args,
                },
            )
        return await _call_tool(
            context,
            "design_and_run_backtest",
            {
                "objective": objective,
                "ticker": ticker,
                **interval_args,
                **shared_backtest_args,
            },
        )
    if step.kind == "comprehensive_analysis":
        return await _call_tool(context, "run_fund_or_stock_analysis", common)
    if step.kind == "risk":
        price = _find_price(state.get("step_results", {}))
        if not price:
            return {"data_type": "risk", "status": "partial", "message": "缺少有效当前价格，无法计算价格型风险指标"}
        return await _call_tool(
            context,
            "calculate_risk_metrics",
            {
                "current_price": price,
                **{
                    key: step.inputs[key]
                    for key in ("stop_loss_pct", "take_profit_pct", "position_size_pct")
                    if key in step.inputs
                },
            },
        )
    if step.kind == "synthesis":
        comparison = next(
            (
                item.get("output")
                for item in state.get("step_results", {}).values()
                if isinstance(item, dict)
                and isinstance(item.get("output"), dict)
                and item["output"].get("data_type") == "strategy_backtest_comparison"
            ),
            None,
        )
        if comparison:
            return {
                "data_type": "synthesis",
                "text": _comparison_synthesis_text(comparison),
                "provenance": {"source": "deterministic_comparison_conclusion"},
            }
        evidence = _result_summaries(state.get("step_results", {}))
        system = (
            "基于给定证据生成简洁中文市场研究结论。明确数据日期、缺失和风险；"
            "面向短中期基金交易研究，不承诺收益；股票分析只能表述为底层资产研究。"
        )
        try:
            context_budget = get_context_budget()
            counter = TokenCounter(context_budget.model)
            compact_request = _synthesis_request(request)
            empty_prompt = json.dumps(
                {"request": compact_request, "evidence": []},
                ensure_ascii=False,
                default=str,
            )
            fixed_tokens = counter.count_messages(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": empty_prompt},
                ]
            )
            evidence_budget = max(256, context_budget.input_limit - fixed_tokens - 512)
            compacted_evidence = compact_json_value(evidence, evidence_budget, counter=counter)
            prompt = json.dumps(
                {"request": compact_request, "evidence": compacted_evidence},
                ensure_ascii=False,
                default=str,
            )
            text = await get_llm_service().chat(prompt, system=system)
            source = "derived"
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise
            logger.warning("Synthesis context overflow; using deterministic evidence summary: {}", exc)
            text = _deterministic_synthesis_fallback(evidence)
            source = "deterministic_context_fallback"
        return {"data_type": "synthesis", "text": text, "provenance": {"source": source}}
    if step.kind == "report":
        summaries = _result_summaries(state.get("step_results", {}))
        content = "# 市场研究报告\n\n" + "\n\n".join(
            f"## {item['step_id']}\n{item.get('summary') or ''}" for item in summaries
        )
        return await _call_tool(
            context,
            "save_artifacts",
            {
                "artifacts": [
                    {
                        "name": f"{ResearchPlan.model_validate(state['plan']).plan_id}-研究报告.md",
                        "format": "md",
                        "content": content,
                        "description": "可审计市场研究计划报告",
                        "asset_type": asset_type,
                        "ticker": ticker or None,
                    }
                ],
                "execution_key": f"{request.get('task_id')}:{step.id}",
            },
        )
    raise ValueError(f"不支持的研究步骤: {step.kind}")


async def run_worker(
    state: ResearchPlanState,
    runtime: Runtime[ResearchPlanContext],
) -> dict[str, Any]:
    step = ResearchStep.model_validate(state["step"])
    prior = StepResult.model_validate(
        (state.get("step_results") or {}).get(step.id)
        or {
            "step_id": step.id,
            "status": "pending",
        }
    )
    attempt = prior.attempt + 1
    budget = ResearchBudget.model_validate(state["budget"])
    if state.get("tool_calls", 0) >= budget.max_tool_calls:
        result = StepResult(step_id=step.id, status="skipped", attempt=attempt, error="研究任务已耗尽工具调用预算")
        return {"step_results": {step.id: result.model_dump(mode="json")}}
    if datetime.now(timezone.utc) > datetime.fromisoformat(state["deadline_at"]):
        result = StepResult(step_id=step.id, status="skipped", attempt=attempt, error="研究任务已超过执行截止时间")
        return {"step_results": {step.id: result.model_dump(mode="json")}}
    try:
        payload = await _execute_step(step, state, runtime.context)
        evidence = _evidence(payload, step.kind)
        artifacts = [
            str(item.get("artifact_id"))
            for item in payload.get("artifacts", [])
            if isinstance(item, dict) and item.get("artifact_id")
        ]
        summary = str(payload.get("text") or payload.get("message") or payload.get("data_type") or step.title)[:1000]
        result = StepResult(
            step_id=step.id,
            status="completed",
            attempt=attempt,
            summary=summary,
            evidence=evidence,
            artifact_ids=artifacts,
            output=_compact(payload),
            failure_context=prior.failure_context,
            recovery_history=prior.recovery_history,
        )
        return {
            "step_results": {step.id: result.model_dump(mode="json")},
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "tool_calls": 1,
        }
    except Exception as exc:
        failure_context = (
            {"tool_name": exc.tool_name, "args": exc.tool_args} if isinstance(exc, ResearchToolExecutionError) else {}
        )
        _, error_message = context_safe_error(exc, str(exc)[:500])
        result = StepResult(
            step_id=step.id,
            status="failed",
            attempt=attempt,
            summary=f"{step.title}执行失败",
            error=error_message,
            failure_context=failure_context,
            recovery_history=prior.recovery_history,
        )
        return {
            "step_results": {step.id: result.model_dump(mode="json")},
            "tool_calls": 1,
        }


async def verify_evidence(state: ResearchPlanState) -> dict[str, Any]:
    plan = ResearchPlan.model_validate(state["plan"])
    results = {key: StepResult.model_validate(value) for key, value in (state.get("step_results") or {}).items()}
    evidence_kinds = {
        "instrument_profile",
        "market_snapshot",
        "price_history",
        "fund_nav",
        "liquidity",
        "technical",
        "fundamentals",
        "news",
        "methodology",
        "backtest",
        "comparison",
        "comprehensive_analysis",
    }
    updates: dict[str, dict[str, Any]] = {}
    for step in plan.steps:
        result = results.get(step.id)
        if result is None or result.status != "completed" or step.kind not in evidence_kinds:
            continue
        evidence_is_valid = bool(result.evidence) and all(
            item.source and item.retrieved_at and item.data_status not in {"unavailable", "failed"}
            for item in result.evidence
        )
        output = result.output
        has_coverage = output.get("available") is not False
        if step.kind == "news":
            has_coverage = has_coverage and bool(output.get("results") or output.get("news"))
        elif step.kind == "price_history":
            items = output.get("items") or []
            has_coverage = has_coverage and bool(items) and all(item.get("history") for item in items)
        elif step.kind == "fund_nav":
            items = output.get("items") or []
            has_coverage = has_coverage and bool(items) and all(item.get("history") for item in items)
        elif step.kind == "market_snapshot":
            has_coverage = has_coverage and bool(output.get("quote") or output.get("quotes") or output.get("results"))
        elif step.kind == "comparison":
            quotes = output.get("quotes") or []
            has_coverage = has_coverage and bool(quotes) and all(item.get("quote") for item in quotes)
        elif step.kind == "backtest" and output.get("data_type") == "strategy_backtest_comparison":
            acceptance = output.get("acceptance") or {}
            has_coverage = has_coverage and acceptance.get("satisfied") is True
        elif step.kind == "backtest" and output.get("data_type") == "sandbox_strategy_candidate":
            sandbox_backtest = output.get("result", {}).get("backtest", {})
            has_coverage = (
                has_coverage
                and bool(output.get("candidate_id"))
                and output.get("validation", {}).get("passed") is True
                and sandbox_backtest.get("final_value") is not None
            )
        requires_as_of = step.kind in {"market_snapshot", "price_history", "fund_nav", "news", "comparison"}
        has_as_of = not requires_as_of or all(item.as_of for item in result.evidence)
        if not evidence_is_valid or not has_coverage or not has_as_of:
            updates[step.id] = result.model_copy(
                update={
                    "status": "failed",
                    "error": (
                        "任务验收契约未满足: " + ", ".join(output.get("acceptance", {}).get("missing", []))
                        if step.kind == "backtest" and output.get("acceptance", {}).get("satisfied") is False
                        else "沙盒验证或可信回测未完成"
                        if step.kind == "backtest" and output.get("data_type") == "sandbox_strategy_candidate"
                        else "证据缺少来源、检索时间或所需数据覆盖范围"
                    ),
                }
            ).model_dump(mode="json")
    results.update({key: StepResult.model_validate(value) for key, value in updates.items()})
    failed = [step for step in plan.steps if step.id in results and results[step.id].status == "failed"]
    retryable = [step for step in failed if results[step.id].attempt < _effective_max_attempts(step)]
    budget = ResearchBudget.model_validate(state["budget"])
    exhausted = state.get("tool_calls", 0) >= budget.max_tool_calls
    expired = datetime.now(timezone.utc) > datetime.fromisoformat(state["deadline_at"])
    if exhausted or expired:
        reason = "研究任务已耗尽工具调用预算" if exhausted else "研究任务已超过执行截止时间"
        for step in plan.steps:
            result = results.get(step.id)
            if result is None or result.status == "pending":
                updates[step.id] = StepResult(
                    step_id=step.id,
                    status="skipped",
                    attempt=result.attempt if result else 0,
                    error=reason,
                ).model_dump(mode="json")
        results.update({key: StepResult.model_validate(value) for key, value in updates.items()})
        failed = [step for step in plan.steps if step.id in results and results[step.id].status == "failed"]
        retryable = []
    return {
        "step_results": updates,
        "needs_replan": bool(retryable and state.get("replan_count", 0) < budget.max_replans),
    }


def route_after_verify(state: ResearchPlanState) -> str:
    if state.get("needs_replan"):
        return "replan"
    plan = ResearchPlan.model_validate(state["plan"])
    results = {key: StepResult.model_validate(value) for key, value in (state.get("step_results") or {}).items()}
    terminal = {"completed", "failed", "skipped"}
    if all(step.id in results and results[step.id].status in terminal for step in plan.steps):
        return "finish"
    completed = {step_id for step_id, result in results.items() if result.status == "completed"}
    if any(
        (step.id not in results or results[step.id].status == "pending") and set(step.depends_on) <= completed
        for step in plan.steps
    ):
        return "dispatch"
    return "skip_blocked"


async def replan(state: ResearchPlanState) -> dict[str, Any]:
    plan = ResearchPlan.model_validate(state["plan"])
    results = {key: StepResult.model_validate(value) for key, value in (state.get("step_results") or {}).items()}
    updates: dict[str, dict[str, Any]] = {}
    revised_steps: list[ResearchStep] = []
    for step in plan.steps:
        result = results.get(step.id)
        max_attempts = _effective_max_attempts(step)
        if result and result.status == "failed" and result.attempt < max_attempts:
            recovery = await _reflect_on_failure(step, result, state)
            history = [*result.recovery_history, recovery]
            if recovery.action == "abort":
                updates[step.id] = result.model_copy(
                    update={
                        "attempt": max_attempts,
                        "summary": recovery.summary,
                        "recovery_history": history,
                    }
                ).model_dump(mode="json")
            else:
                if recovery.action == "adjust":
                    step = step.model_copy(update={"inputs": {**step.inputs, **recovery.input_patch}})
                updates[step.id] = result.model_copy(
                    update={
                        "status": "pending",
                        "summary": recovery.summary,
                        "error": None,
                        "recovery_history": history,
                    }
                ).model_dump(mode="json")
        revised_steps.append(step)
    plan = plan.model_copy(update={"steps": tuple(revised_steps), "revision": plan.revision + 1})
    _validate_plan_contract(plan, state["request"], ResearchBudget.model_validate(state["budget"]))
    return {
        "plan": plan.model_dump(mode="json"),
        "step_results": updates,
        "replan_count": state.get("replan_count", 0) + 1,
        "needs_replan": False,
    }


async def skip_blocked(state: ResearchPlanState) -> dict[str, Any]:
    plan = ResearchPlan.model_validate(state["plan"])
    results = {key: StepResult.model_validate(value) for key, value in (state.get("step_results") or {}).items()}
    updates = {}
    for step in plan.steps:
        if step.id not in results or results[step.id].status == "pending":
            updates[step.id] = StepResult(
                step_id=step.id,
                status="skipped",
                error="依赖步骤未成功，已跳过",
            ).model_dump(mode="json")
    return {"step_results": updates}


async def finish_research(state: ResearchPlanState) -> dict[str, Any]:
    plan = ResearchPlan.model_validate(state["plan"])
    results = {key: StepResult.model_validate(value) for key, value in (state.get("step_results") or {}).items()}
    synthesis = next(
        (
            result.output.get("text")
            for step in plan.steps
            if step.kind == "synthesis"
            and (result := results.get(step.id)) is not None
            and result.status == "completed"
        ),
        None,
    )
    if synthesis:
        response = str(synthesis)
    else:
        completed = [
            step.title for step in plan.steps if results.get(step.id) and results[step.id].status == "completed"
        ]
        failed = [
            step.title for step in plan.steps if not results.get(step.id) or results[step.id].status != "completed"
        ]
        response = f"已完成：{'、'.join(completed) or '无'}。"
        if failed:
            response += f" 数据不足或失败：{'、'.join(failed)}。"
        response += "以上仅用于短中期研究与模拟交易，不构成收益承诺。"
    return {"final_response": response}


def build_research_plan_graph(checkpointer: Any | None = None):
    graph = StateGraph(ResearchPlanState, context_schema=ResearchPlanContext)
    graph.add_node("scope", scope_research)
    graph.add_node("planner", plan_research)
    graph.add_node("validate", validate_plan)
    graph.add_node("dispatch", dispatch_ready)
    graph.add_node("worker", run_worker)
    graph.add_node("verify", verify_evidence)
    graph.add_node("replan", replan)
    graph.add_node("skip_blocked", skip_blocked)
    graph.add_node("finish", finish_research)
    graph.add_edge(START, "scope")
    graph.add_edge("scope", "planner")
    graph.add_edge("planner", "validate")
    graph.add_edge("validate", "dispatch")
    graph.add_conditional_edges("dispatch", route_dispatch, ["worker", "verify"])
    graph.add_edge("worker", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"replan": "replan", "dispatch": "dispatch", "skip_blocked": "skip_blocked", "finish": "finish"},
    )
    graph.add_edge("replan", "dispatch")
    graph.add_edge("skip_blocked", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


research_plan_graph = build_research_plan_graph()


def configure_research_plan_graph(checkpointer: Any | None) -> None:
    global research_plan_graph
    research_plan_graph = build_research_plan_graph(checkpointer)


def get_research_plan_graph():
    return research_plan_graph
