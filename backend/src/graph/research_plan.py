"""Checkpointed orchestrator-worker graph for market Deep Research."""

from __future__ import annotations

import asyncio
import hashlib
import json
import operator
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, TypedDict, get_args
from uuid import uuid4

from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from loguru import logger

from llm.service import get_llm_service
from models.research_plan import (
    EvidenceRef,
    ResearchBudget,
    ResearchPlan,
    ResearchStep,
    ResearchStepKind,
    StepResult,
)
from models.schemas import AssetType


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


@dataclass(frozen=True)
class ResearchPlanContext:
    tools: dict[str, StructuredTool]


DEPTH_BUDGETS = {
    "quick": ResearchBudget(max_steps=3, max_tool_calls=4, max_replans=0, deadline_seconds=300),
    "standard": ResearchBudget(max_steps=8, max_tool_calls=16, max_replans=1, deadline_seconds=900),
    "deep": ResearchBudget(max_steps=16, max_tool_calls=32, max_replans=2, deadline_seconds=1800),
}
DEPTH_STEP_RANGES = {"quick": (1, 3), "standard": (4, 8), "deep": (9, 16)}

LONG_STEP_KINDS = {"backtest", "comprehensive_analysis", "report"}
DEEP_TERMS = ("深度", "全面", "系统", "多源", "调研报告", "deep research")


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
    return {
        "depth": depth,
        "budget": budget.model_dump(mode="json"),
        "deadline_at": deadline.isoformat(),
        "step_results": {},
        "evidence": [],
        "replan_count": 0,
        "tool_calls": 0,
        "needs_replan": False,
    }


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
                _step("backtest", "执行历史回测并保存实验结果", ["price_history", "methodology"], attempts=1),
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
                attempts=1,
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
        value.setdefault("max_attempts", 1 if value.get("kind") in LONG_STEP_KINDS else 2)
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
    if any(step.kind in LONG_STEP_KINDS and step.max_attempts != 1 for step in plan.steps):
        raise ValueError("回测、综合分析和报告步骤只允许执行一次")
    if plan.asset_type in {AssetType.ETF, AssetType.LOF} and any(
        step.kind in {"technical", "comprehensive_analysis", "backtest", "comparison", "news"}
        for step in plan.steps
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
            },
        )
        for step in ready[: min(budget.max_parallel, remaining_calls)]
    ]


def _compact(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return str(value)[:500]
    if isinstance(value, dict):
        return {str(key): _compact(item, depth=depth + 1) for key, item in list(value.items())[:30]}
    if isinstance(value, list):
        return [_compact(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, str):
        return value[:4000]
    return value


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
            "output": item.get("output", {}),
        }
        for item in results.values()
    ]


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


async def _call_tool(context: ResearchPlanContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    tool = context.tools.get(name)
    if tool is None:
        raise ValueError(f"研究步骤需要的工具不可用: {name}")
    timeout = 300 if name in {"run_fund_or_stock_analysis", "run_backtest", "design_and_run_backtest"} else 60
    raw = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError:
        payload = {"value": str(raw)}
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
        payloads = await asyncio.gather(
            *(
                _call_tool(context, "get_historical_prices", {"ticker": item, "asset_type": asset_type, "limit": 120})
                for item in tickers[:10]
            )
        )
        return {"data_type": "price_history_collection", "items": payloads}
    if step.kind == "fund_nav":
        payloads = await asyncio.gather(
            *(
                _call_tool(context, "get_fund_nav_history", {"ticker": item, "asset_type": asset_type, "limit": 120})
                for item in tickers[:10]
            )
        )
        return {"data_type": "fund_nav_collection", "items": payloads}
    if step.kind == "technical":
        payloads = await asyncio.gather(
            *(
                _call_tool(context, "compute_technical_indicators", {"ticker": item, "asset_type": asset_type})
                for item in tickers[:10]
            )
        )
        return {"data_type": "technical_collection", "items": payloads}
    if step.kind == "news":
        query = f"{' '.join(tickers)} {asset_type} 最新新闻 公告 风险 催化 {request.get('message', '')}"
        return await _call_tool(context, "search_web", {"query": query, "num_results": 10, "freshness": "qdr:m"})
    if step.kind == "methodology":
        if str(request.get("intent")) == "strategies":
            return await _call_tool(context, "list_trading_strategies", {})
        return await _call_tool(
            context,
            "search_methodology",
            {"query": str(request.get("message", "")), "asset_type": asset_type, "limit": 5},
        )
    if step.kind == "comparison":
        return await _call_tool(context, "compare_quotes", {"tickers": tickers, "asset_type": asset_type})
    if step.kind == "backtest":
        end_date = str(request.get("as_of_date") or date.today().isoformat())
        start_date = str(
            step.inputs.get("start_date") or (date.fromisoformat(end_date) - timedelta(days=365)).isoformat()
        )
        if len(tickers) > 1:
            return await _call_tool(
                context,
                "design_and_run_backtest",
                {
                    "objective": str(request.get("message", "")),
                    "tickers": tickers,
                    "mode": "pool",
                    "start_date": start_date,
                    "end_date": end_date,
                    "asset_type": asset_type,
                },
            )
        return await _call_tool(
            context,
            "design_and_run_backtest",
            {
                "objective": str(request.get("message", "")),
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "asset_type": asset_type,
            },
        )
    if step.kind == "comprehensive_analysis":
        return await _call_tool(context, "run_fund_or_stock_analysis", common)
    if step.kind == "risk":
        price = _find_price(state.get("step_results", {}))
        if not price:
            return {"data_type": "risk", "status": "partial", "message": "缺少有效当前价格，无法计算价格型风险指标"}
        return await _call_tool(context, "calculate_risk_metrics", {"current_price": price})
    if step.kind == "synthesis":
        evidence = _result_summaries(state.get("step_results", {}))
        text = await get_llm_service().chat(
            json.dumps({"request": request, "evidence": evidence}, ensure_ascii=False, default=str),
            system=(
                "基于给定证据生成简洁中文市场研究结论。明确数据日期、缺失和风险；"
                "面向短中期基金交易研究，不承诺收益；股票分析只能表述为底层资产研究。"
            ),
        )
        return {"data_type": "synthesis", "text": text, "provenance": {"source": "derived"}}
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
    prior = StepResult.model_validate((state.get("step_results") or {}).get(step.id) or {
        "step_id": step.id,
        "status": "pending",
    })
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
        )
        return {
            "step_results": {step.id: result.model_dump(mode="json")},
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "tool_calls": 1,
        }
    except Exception as exc:
        result = StepResult(
            step_id=step.id,
            status="failed",
            attempt=attempt,
            summary=f"{step.title}执行失败",
            error=str(exc)[:500],
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
        requires_as_of = step.kind in {"market_snapshot", "price_history", "fund_nav", "news", "comparison"}
        has_as_of = not requires_as_of or all(item.as_of for item in result.evidence)
        if not evidence_is_valid or not has_coverage or not has_as_of:
            updates[step.id] = result.model_copy(
                update={
                    "status": "failed",
                    "error": "证据缺少来源、检索时间或所需数据覆盖范围",
                }
            ).model_dump(mode="json")
    results.update({key: StepResult.model_validate(value) for key, value in updates.items()})
    failed = [step for step in plan.steps if step.id in results and results[step.id].status == "failed"]
    retryable = [step for step in failed if results[step.id].attempt < step.max_attempts]
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
    for step in plan.steps:
        result = results.get(step.id)
        if result and result.status == "failed" and result.attempt < step.max_attempts:
            updates[step.id] = result.model_copy(update={"status": "pending", "error": None}).model_dump(mode="json")
    plan.revision += 1
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
