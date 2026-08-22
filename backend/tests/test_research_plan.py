import asyncio
import json

import pytest
from langchain_core.tools import StructuredTool, tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

import agents.stock_agent as stock_agent_module
import graph.research_plan as research_graph
from agents.stock_agent import AssetAgentRequest, AssetIntent, StockAgent
from application.research_plan import _plan_snapshot
from graph.research_plan import ResearchPlanContext, _fallback_steps, build_research_plan_graph, classify_depth
from models.research_plan import ResearchPlan
from models.schemas import AssetType


def _plan(steps):
    return {
        "plan_id": "plan-1",
        "objective": "研究 600519",
        "asset_type": "stock",
        "tickers": ["600519"],
        "as_of_date": "2026-08-22",
        "depth": "standard",
        "steps": steps,
    }


def _step(step_id, kind="market_snapshot", depends_on=None):
    return {
        "id": step_id,
        "kind": kind,
        "title": step_id,
        "depends_on": depends_on or [],
        "success_criteria": ["有来源"],
    }


def test_depth_classifier_and_fallback_budgets():
    assert classify_depth({"intent": "quote", "message": "查询行情"}) == "quick"
    assert classify_depth({"intent": "analyze", "message": "分析 600519"}) == "standard"
    assert classify_depth({"intent": "quote", "message": "全面深度调研 510300"}) == "deep"

    quick = _fallback_steps({"intent": "quote", "asset_type": "stock", "message": ""}, "quick")
    standard = _fallback_steps({"intent": "analyze", "asset_type": "stock", "message": ""}, "standard")
    deep_etf = _fallback_steps({"intent": "analyze", "asset_type": "etf", "message": ""}, "deep")
    assert 1 <= len(quick) <= 3
    assert 4 <= len(standard) <= 8
    assert 9 <= len(deep_etf) <= 16
    assert "fund_nav" in {step["kind"] for step in deep_etf}


@pytest.mark.parametrize(
    "steps,match",
    [
        ([_step("same"), _step("same", "news")], "重复"),
        ([_step("one", depends_on=["missing"])], "不存在"),
        ([_step("one", depends_on=["two"]), _step("two", depends_on=["one"])], "循环"),
        ([_step("one") | {"success_criteria": []}], "成功标准"),
    ],
)
def test_plan_rejects_invalid_dag_and_success_criteria(steps, match):
    with pytest.raises(ValidationError, match=match):
        ResearchPlan.model_validate(_plan(steps))


def test_plan_rejects_uncontrolled_step_kind():
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(_plan([_step("one", "arbitrary_tool")]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent",
    [
        AssetIntent.QUOTE,
        AssetIntent.HISTORY,
        AssetIntent.NEWS,
        AssetIntent.STRATEGIES,
        AssetIntent.ANALYZE,
        AssetIntent.COMPARE,
        AssetIntent.BACKTEST,
    ],
)
async def test_all_readonly_research_intents_use_plan_graph(monkeypatch, intent):
    calls = []

    class FakePlanService:
        async def stream(self, request, tools, *, config):
            calls.append((request["intent"], {item.name for item in tools}, config))
            yield {"type": "text", "text": "planned"}

    monkeypatch.setattr(stock_agent_module, "research_plan_service", FakePlanService())
    monkeypatch.setattr(stock_agent_module.checkpoint_manager, "saver", object())
    request = AssetAgentRequest(
        message=f"执行 {intent.value} 研究",
        history=[],
        intent=intent,
        tickers=() if intent == AssetIntent.STRATEGIES else ("600519",),
        asset_type=AssetType.STOCK,
        task_id=f"task-{intent.value}",
        intent_confirmed=True,
    )

    events = [event async for event in StockAgent().chat(request)]

    assert events[0]["type"] == "execution_metadata"
    assert events[-1] == {"type": "text", "text": "planned"}
    assert calls[0][0] == intent.value


@pytest.mark.asyncio
async def test_standard_research_graph_runs_ready_steps_in_parallel(monkeypatch):
    active = 0
    max_active = 0

    async def payload(data):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return json.dumps(data, ensure_ascii=False)

    provenance = [
        {
            "name": "测试源",
            "fetched_at": "2026-08-22T00:00:00+00:00",
            "as_of": "2026-08-22",
            "status": "available",
        }
    ]

    @tool
    async def get_realtime_quote(ticker: str, asset_type: str = "stock") -> str:
        """Get quote."""
        return await payload({"quote": {"price": 10}, "provenance": provenance})

    @tool
    async def get_historical_prices(ticker: str, asset_type: str = "stock", limit: int = 120) -> str:
        """Get history."""
        return await payload({"history": [{"date": "2026-08-22", "close": 10}], "provenance": provenance})

    @tool
    async def compute_technical_indicators(ticker: str, asset_type: str = "stock") -> str:
        """Compute indicators."""
        return await payload({"indicators": {"trend": "up"}, "provenance": provenance})

    @tool
    async def get_fundamentals(ticker: str, asset_type: str = "stock") -> str:
        """Get fundamentals."""
        return await payload({"data": {"pe": 10}, "provenance": provenance})

    @tool
    async def search_web(query: str, num_results: int = 10, freshness: str | None = None) -> str:
        """Search news."""
        return await payload(
            {
                "available": True,
                "searched_at": "2026-08-22T00:00:00+00:00",
                "results": [{"title": "公告", "link": "https://example.com"}],
                "provenance": provenance,
            }
        )

    @tool
    async def run_fund_or_stock_analysis(ticker: str, asset_type: str = "stock") -> str:
        """Run comprehensive analysis."""
        return await payload({"decision": "hold", "provenance": provenance})

    @tool
    async def calculate_risk_metrics(current_price: float) -> str:
        """Calculate risk."""
        return await payload({"metrics": {"stop_loss": current_price * 0.92}, "provenance": provenance})

    class OfflinePlanner:
        async def chat_json(self, *args, **kwargs):
            raise RuntimeError("use deterministic plan")

        async def chat(self, *args, **kwargs):
            return "证据已综合；保持小仓位并设置止损。"

    monkeypatch.setattr(research_graph, "get_llm_service", lambda: OfflinePlanner())
    tools = [
        get_realtime_quote,
        get_historical_prices,
        compute_technical_indicators,
        get_fundamentals,
        search_web,
        run_fund_or_stock_analysis,
        calculate_risk_metrics,
    ]
    graph = build_research_plan_graph(MemorySaver())
    result = await graph.ainvoke(
        {
            "request": {
                "message": "分析 600519",
                "intent": "analyze",
                "tickers": ["600519"],
                "asset_type": "stock",
                "task_id": "task-plan",
            }
        },
        config={"configurable": {"thread_id": "task-plan"}, "recursion_limit": 120},
        context=ResearchPlanContext(tools={item.name: item for item in tools}),
    )

    assert result["plan"]["depth"] == "standard"
    assert len(result["plan"]["steps"]) == 8
    assert {item["status"] for item in result["step_results"].values()} == {"completed"}
    assert 2 <= max_active <= 4
    assert result["tool_calls"] <= 16
    stack = [result]
    while stack:
        value = stack.pop()
        assert not isinstance(value, StructuredTool)
        assert not type(value).__module__.startswith(("pandas", "polars"))
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    snapshot = _plan_snapshot(result, status="completed")
    assert snapshot is not None
    assert snapshot["progress"] == 100
    assert snapshot["status"] == "completed"
