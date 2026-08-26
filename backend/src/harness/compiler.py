"""Compile legacy request metadata into one strict Harness task contract."""

from __future__ import annotations

from collections.abc import Iterable

from agents.asset_requests import AssetAgentRequest, AssetIntent, RequestMode
from application.task_contract import (
    compile_task_contract,
    requests_market_dataset,
    requests_sandbox_execution,
    requests_strategy_catalog,
)
from harness.models import HarnessTaskContract
from models.fund_task import FundTaskKind
from models.supervisor import ExecutionMode as LegacyExecutionMode
from models.supervisor import TaskRoutingDecision

_INTENT_CAPABILITIES: dict[AssetIntent, tuple[str, ...]] = {
    AssetIntent.QUOTE: ("market.quote",),
    AssetIntent.HISTORY: ("market.history",),
    AssetIntent.NEWS: ("news.events",),
    AssetIntent.STRATEGIES: ("methodology.search",),
    AssetIntent.PORTFOLIO: ("simulation.read",),
    AssetIntent.BACKTEST: ("market.history", "backtest.execute"),
    AssetIntent.COMPARE: ("market.quote", "market.history"),
    AssetIntent.HELP: (),
}

_COMMON_FUND_TASK_CAPABILITIES: dict[FundTaskKind, tuple[str, ...]] = {
    FundTaskKind.EDUCATION: ("methodology.search",),
    FundTaskKind.CALCULATION: (),
    FundTaskKind.RULE_DESIGN: ("methodology.search",),
    FundTaskKind.SCENARIO_PLAN: ("risk.metrics",),
    FundTaskKind.SIMULATION_QUERY: ("simulation.read",),
    FundTaskKind.SIMULATION_MUTATION: ("simulation.read", "simulation.write"),
    FundTaskKind.CLARIFICATION: (),
}

_LEGACY_MODE_MAP = {
    LegacyExecutionMode.DIRECT_RESPONSE: "direct_response",
    LegacyExecutionMode.ARTIFACT_GENERATION: "artifact_generation",
    LegacyExecutionMode.EVIDENCE_RESEARCH: "evidence_research",
    LegacyExecutionMode.BACKTEST_EXECUTION: "backtest_execution",
    LegacyExecutionMode.MIXED_WORKFLOW: "mixed_workflow",
    LegacyExecutionMode.SUPERVISOR_DECIDES: "evidence_research",
}

def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _requests_sandbox_research(request: AssetAgentRequest, *, needs_tools: bool) -> bool:
    """Recognize explicit executable code-strategy backtests, never explanations."""
    if not needs_tools:
        return False
    if request.asset_type.value not in {"stock", "etf", "lof"}:
        return False
    return requests_sandbox_execution(request.message)


class HarnessTaskCompiler:
    """Deterministic safety/compiler layer; the LLM may classify but cannot widen it."""

    def compile(
        self,
        request: AssetAgentRequest,
        routing: TaskRoutingDecision,
    ) -> HarnessTaskContract:
        legacy = compile_task_contract(
            request.message,
            tickers=list(request.tickers),
            asset_type=request.asset_type.value,
            mutation_requested=request.allow_mutating_tools,
            routing_decision=routing,
        )
        task_spec = legacy.source_task_spec or {}
        raw_kind = task_spec.get("task_kind")
        try:
            task_kind = FundTaskKind(raw_kind) if raw_kind else None
        except ValueError:
            task_kind = None

        needs_tools = routing.requires_tools or routing.mode == LegacyExecutionMode.SUPERVISOR_DECIDES
        fund_domain = task_spec.get("subject", {}).get("fund_domain")
        product_category = str(task_spec.get("subject", {}).get("product_category") or "unknown")
        pricing_basis = str(task_spec.get("subject", {}).get("pricing_basis") or "market_price")
        required = list(_COMMON_FUND_TASK_CAPABILITIES.get(task_kind, ())) if needs_tools else []
        sandbox_research = _requests_sandbox_research(request, needs_tools=needs_tools)
        market_dataset_research = needs_tools and requests_market_dataset(
            request.message,
            asset_type=request.asset_type.value,
        )
        strategy_catalog_research = needs_tools and requests_strategy_catalog(request.message)
        if sandbox_research:
            required = [
                "market.history",
                "strategy.sandbox_research",
            ]
        elif market_dataset_research:
            required = ["market.dataset"]
        elif strategy_catalog_research:
            required = ["strategy.list"]
        elif needs_tools and request.intent == AssetIntent.BACKTEST and request.asset_type.value != "open_fund":
            required = ["market.history", "backtest.execute"]
        elif needs_tools and task_kind == FundTaskKind.INSTRUMENT_RESEARCH:
            if fund_domain == "exchange_fund":
                required = ["exchange_fund.comprehensive_analysis"]
            elif fund_domain == "open_fund":
                required = ["open_fund.comprehensive_analysis"]
        elif needs_tools and task_kind == FundTaskKind.UNIVERSE_RESEARCH:
            required = [f"{fund_domain}.screen_compare"] if fund_domain else []
        elif needs_tools and task_kind == FundTaskKind.EVENT_RESEARCH:
            required = [f"{fund_domain}.event_risk"] if fund_domain else ["news.events"]
        optional_fund_branches: set[str] = set()
        if needs_tools:
            if not required:
                required.extend(_INTENT_CAPABILITIES.get(request.intent, ()))
            if (
                request.intent == AssetIntent.ANALYZE
                and task_kind != FundTaskKind.UNIVERSE_RESEARCH
                and not market_dataset_research
                and not strategy_catalog_research
            ):
                if request.asset_type.value in {"etf", "lof"}:
                    required = ["exchange_fund.comprehensive_analysis"]
                elif request.asset_type.value == "open_fund":
                    required = ["open_fund.comprehensive_analysis"]
                else:
                    required.extend(("market.quote", "market.history", "technical.indicators", "risk.metrics"))
                    required.append("stock.comprehensive_analysis")
            if request.asset_type.value in {"etf", "lof"}:
                normalized_message = request.message.lower()
                if task_kind == FundTaskKind.UNIVERSE_RESEARCH:
                    required = [item for item in required if item != "exchange_fund.screen_compare"]
                    required.append("exchange_fund.screen_compare")
                elif task_kind == FundTaskKind.EVENT_RESEARCH:
                    required.append("exchange_fund.event_risk")
                    optional_fund_branches.add("exchange_fund.event_risk")
                elif request.intent == AssetIntent.PORTFOLIO:
                    required.append("exchange_fund.portfolio_fit")
                    optional_fund_branches.add("exchange_fund.portfolio_fit")
                if any(token in normalized_message for token in ("筛选", "候选", "哪只", "首选")):
                    required.append("exchange_fund.screen_compare")
                if any(token in normalized_message for token in ("组合", "相关性", "集中度", "重复配置")):
                    required.append("exchange_fund.portfolio_fit")
                    optional_fund_branches.add("exchange_fund.portfolio_fit")
                if any(token in normalized_message for token in ("公告", "限购", "申赎", "暂停", "调仓")):
                    required.append("exchange_fund.event_risk")
                    optional_fund_branches.add("exchange_fund.event_risk")
                if any(token in normalized_message for token in ("折价", "溢价", "折溢价", "iopv", "qdii")):
                    required.append("exchange_fund.premium_discount")
                    optional_fund_branches.add("exchange_fund.premium_discount")
                if any(token in normalized_message for token in ("跟踪误差", "跟踪差额", "跟踪质量", "基准")):
                    required.append("exchange_fund.tracking_quality")
                    optional_fund_branches.add("exchange_fund.tracking_quality")
                if any(token in normalized_message for token in ("持仓", "行业暴露", "资产暴露", "底层资产")):
                    required.append("exchange_fund.exposure")
                    optional_fund_branches.add("exchange_fund.exposure")
            elif request.asset_type.value == "open_fund":
                normalized_message = request.message.lower()
                if request.intent == AssetIntent.BACKTEST or "回测" in normalized_message:
                    required = ["open_fund.nav_backtest"]
                elif task_kind == FundTaskKind.UNIVERSE_RESEARCH:
                    required = ["open_fund.screen_compare"]
                elif task_kind == FundTaskKind.EVENT_RESEARCH:
                    required = ["open_fund.event_risk"]
                elif request.intent == AssetIntent.ANALYZE:
                    required = ["open_fund.comprehensive_analysis"]
                supported_nav_categories = {"equity", "hybrid", "bond", "index", "enhanced_index"}
                if task_kind == FundTaskKind.INSTRUMENT_RESEARCH and product_category == "money_market" and any(
                    token in normalized_message for token in ("收益", "七日年化", "万份收益", "比较")
                ):
                    required.append("open_fund.money_yield")
                elif (
                    task_kind == FundTaskKind.INSTRUMENT_RESEARCH
                    and product_category in supported_nav_categories
                    and any(token in normalized_message for token in ("净值", "走势", "表现", "收益"))
                ):
                    required.append("open_fund.nav")
                if task_kind == FundTaskKind.INSTRUMENT_RESEARCH and any(
                    token in normalized_message for token in ("费率", "申购费", "赎回费", "销售服务费")
                ):
                    required.append("open_fund.fees")
                if (
                    task_kind == FundTaskKind.INSTRUMENT_RESEARCH
                    and product_category in supported_nav_categories
                    and any(token in normalized_message for token in ("持仓", "行业", "信用暴露", "利率敏感"))
                ):
                    required.append("open_fund.exposure")
                if task_kind == FundTaskKind.SCENARIO_PLAN:
                    required = ["methodology.search"]
        if routing.mode == LegacyExecutionMode.ARTIFACT_GENERATION:
            required.append("artifact.manage")
        if request.mode == RequestMode.SIMULATION_MUTATION or request.allow_mutating_tools:
            required.extend(("simulation.read", "simulation.write"))

        forbidden = list(task_spec.get("forbidden_capabilities") or [])
        if request.asset_type.value in {"etf", "lof"}:
            forbidden.append("stock.comprehensive_analysis")
            forbidden.extend(("open_fund.nav", "open_fund.money_yield", "open_fund.screen_compare"))
        elif request.asset_type.value == "open_fund":
            forbidden.extend(
                (
                    "market.quote",
                    "market.history",
                    "exchange_fund.nav",
                    "exchange_fund.premium_discount",
                    "exchange_fund.liquidity_cost",
                    "stock.comprehensive_analysis",
                )
            )
            if request.mode == RequestMode.SIMULATION_MUTATION:
                forbidden.append("simulation.write")
        if not request.allow_mutating_tools:
            forbidden.append("simulation.write")
        required = [item for item in _dedupe(required) if item not in forbidden]
        missing_inputs = list(legacy.missing_inputs)
        unsupported_open_fund_mutation = (
            request.asset_type.value == "open_fund" and request.mode == RequestMode.SIMULATION_MUTATION
        )
        if unsupported_open_fund_mutation:
            required = []
            missing_inputs.append("场外基金模拟申购赎回、份额确认和到账周期首期不支持")

        evidence = list(legacy.evidence_requirements)
        if routing.requires_tools and not evidence:
            evidence.append("structured_tool_evidence")
        allowed = _dedupe(required)
        execution_mode = _LEGACY_MODE_MAP[routing.mode]
        if request.mode == RequestMode.SIMULATION_MUTATION:
            execution_mode = "direct_response" if unsupported_open_fund_mutation else "simulation_write"
        elif task_kind == FundTaskKind.SIMULATION_QUERY:
            execution_mode = "simulation_read"

        return HarnessTaskContract(
            objective=legacy.objective,
            asset_type=request.asset_type.value,
            fund_domain=fund_domain,
            product_category=product_category,
            pricing_basis=pricing_basis,
            tickers=request.tickers,
            intent=request.intent.value,
            execution_mode=execution_mode,
            deliverables=tuple(legacy.deliverables),
            required_outputs=tuple(legacy.required_outputs),
            required_capabilities=tuple(required),
            allowed_capabilities=allowed,
            forbidden_capabilities=_dedupe(forbidden),
            evidence_requirements=_dedupe(evidence),
            missing_inputs=tuple(dict.fromkeys(missing_inputs)),
            resolve_representative_product=legacy.resolve_representative_product,
            allow_mutations=request.allow_mutating_tools and request.asset_type.value != "open_fund",
            budget_profile=(
                "deep"
                if sandbox_research or request.intent == AssetIntent.BACKTEST or len(optional_fund_branches) >= 3
                else "standard"
            ),
            acceptance_profile=(
                "exchange_fund.screening"
                if task_kind == FundTaskKind.UNIVERSE_RESEARCH and fund_domain == "exchange_fund"
                else "open_fund.screening"
                if task_kind == FundTaskKind.UNIVERSE_RESEARCH and fund_domain == "open_fund"
                else "exchange_fund.comprehensive"
                if "exchange_fund.comprehensive_analysis" in required
                else "open_fund.comprehensive"
                if "open_fund.comprehensive_analysis" in required
                else "open_fund.nav_backtest"
                if "open_fund.nav_backtest" in required
                else "default"
            ),
            source_task_spec=legacy.source_task_spec,
            routing=routing.model_dump(mode="json"),
        )


harness_task_compiler = HarnessTaskCompiler()
