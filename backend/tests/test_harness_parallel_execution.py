import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from graph.agent_loop import AgentLoopContext, execute_tool_calls


@pytest.mark.asyncio
async def test_matching_backtest_evidence_skips_redundant_runtime_call():
    called = 0

    async def run_fixture(
        ticker: str,
        start_date: str,
        end_date: str,
        asset_type: str = "stock",
    ) -> str:
        del ticker, start_date, end_date, asset_type
        nonlocal called
        called += 1
        return "should not run"

    tool = StructuredTool.from_function(coroutine=run_fixture, name="run_backtest", description="fixture")
    args = {
        "ticker": "518880",
        "start_date": "2023-01-01",
        "end_date": "2025-12-31",
        "asset_type": "etf",
    }
    prior_event = {
        "name": "design_and_run_backtest",
        "status": "completed",
        "args": args,
        "result": json.dumps({"data_type": "backtest_experiment", "result": {"total_return": 0.21}}),
    }

    result = await execute_tool_calls(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "run_backtest", "args": args, "id": "call-1", "type": "tool_call"}],
                )
            ],
            "step": 2,
            "tool_events": [prior_event],
            "task_contract": {"budget": {"max_tool_calls": 10}},
        },
        SimpleNamespace(context=AgentLoopContext(tools=[tool], tool_map={tool.name: tool})),
        {},
    )

    assert called == 0
    assert result["tool_events"] == []
    assert json.loads(result["messages"][-1].content)["reused"] is True


@pytest.mark.asyncio
async def test_independent_read_tools_run_with_parallelism_capped_at_four():
    active = 0
    maximum = 0

    def build(name: str):
        async def read_fixture() -> str:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1
            return name

        return StructuredTool.from_function(coroutine=read_fixture, name=name, description="fixture")

    tools = [build(f"read_{index}") for index in range(6)]
    calls = [
        {"name": tool.name, "args": {}, "id": f"call-{index}", "type": "tool_call"}
        for index, tool in enumerate(tools)
    ]
    result = await execute_tool_calls(
        {
            "messages": [AIMessage(content="", tool_calls=calls)],
            "step": 1,
            "tool_events": [],
            "task_contract": {"budget": {"max_tool_calls": 10}},
        },
        SimpleNamespace(context=AgentLoopContext(tools=tools, tool_map={tool.name: tool for tool in tools})),
        {},
    )
    assert maximum == 4
    assert [event["name"] for event in result["tool_events"]] == [tool.name for tool in tools]


@pytest.mark.asyncio
async def test_tool_call_budget_rejects_excess_calls_without_execution():
    called = 0

    async def read_fixture() -> str:
        nonlocal called
        called += 1
        return "ok"

    tool = StructuredTool.from_function(coroutine=read_fixture, name="read_fixture", description="fixture")
    calls = [
        {"name": tool.name, "args": {}, "id": f"call-{index}", "type": "tool_call"}
        for index in range(3)
    ]
    result = await execute_tool_calls(
        {
            "messages": [AIMessage(content="", tool_calls=calls)],
            "step": 1,
            "tool_events": [],
            "task_contract": {"budget": {"max_tool_calls": 2}},
        },
        SimpleNamespace(context=AgentLoopContext(tools=[tool], tool_map={tool.name: tool})),
        {},
    )
    assert called == 2
    assert result["tool_events"][-1]["status"] == "failed"
    assert "tool_budget_exhausted" in result["tool_events"][-1]["result"]


@pytest.mark.asyncio
async def test_harness_compacts_large_tool_observation_but_retains_full_event():
    full_payload = '{"history":[' + ",".join(f'{{"date":"day-{index}","close":{index}}}' for index in range(500)) + "]}"

    async def read_fixture() -> str:
        return full_payload

    tool = StructuredTool.from_function(coroutine=read_fixture, name="read_fixture", description="fixture")
    result = await execute_tool_calls(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": tool.name, "args": {}, "id": "call-1", "type": "tool_call"}],
                )
            ],
            "step": 1,
            "tool_events": [],
            "task_contract": {"compact_tool_results": True, "budget": {"max_tool_calls": 2}},
        },
        SimpleNamespace(context=AgentLoopContext(tools=[tool], tool_map={tool.name: tool})),
        {},
    )

    assert result["tool_events"][0]["result"] == full_payload
    observation = result["messages"][-1]
    assert isinstance(observation, ToolMessage)
    assert len(str(observation.content)) < len(full_payload)
    assert "day-499" in str(observation.content)
