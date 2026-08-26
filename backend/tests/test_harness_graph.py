import pytest

from agents.asset_requests import AssetAgentRequest, AssetIntent, AssetRequestResolver
from harness.graph import prepare_harness_plan
from models.schemas import AssetType
from models.supervisor import ExecutionMode, TaskRoutingDecision


@pytest.mark.asyncio
async def test_harness_kernel_runs_declared_lifecycle_and_blocks_stock_graph_for_etf():
    request = AssetAgentRequest(
        message="分析 510300 ETF 的趋势和风险",
        history=[],
        intent=AssetIntent.ANALYZE,
        tickers=("510300",),
        asset_type=AssetType.ETF,
        intent_confirmed=True,
    )
    state = await prepare_harness_plan(
        AssetRequestResolver.request_payload(request),
        TaskRoutingDecision(
            mode=ExecutionMode.EVIDENCE_RESEARCH,
            requires_tools=True,
            allow_research_plan=True,
        ),
        task_id=None,
    )
    assert state["lifecycle"] == [
        "compile",
        "select_skills",
        "plan",
        "dispatch",
        "execute",
        "verify",
        "replan",
        "synthesize",
        "judge",
    ]
    assert "stock.comprehensive_analysis" not in state["plan"]["selected_skills"]
