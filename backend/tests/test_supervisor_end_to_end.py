"""One deterministic complex-flow acceptance test for the single Supervisor Agent."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from agents import stock_agent as stock_agent_module
from application.chat_service import ChatStore, ChatTaskInput, ChatTaskManager
from graph import agent_loop as agent_loop_module
from graph.agent_loop import tool_attempts, tool_timeout_seconds

COMPLEX_QUESTION = (
    "我计划每月投入3000元持续六个月，请在场内沪深300ETF和场外沪深300ETF联接C之间比较。"
    "自动选择代表产品，查询可获得的最新费率、成交活跃度、价差、基金规模、跟踪情况和申赎规则，"
    "比较费用、资金利用率、操作便利性和适用条件；缺失的公开数据请继续查找，无法获得时说明原因。"
)


def test_research_plan_tool_has_thirty_minute_single_attempt_budget():
    assert tool_timeout_seconds("run_research_plan") == 1800
    assert tool_attempts("run_research_plan") == 1


class FakeSupervisorModel:
    def __init__(self) -> None:
        self.decisions = 0

    async def chat_with_tools(self, messages, tools, **kwargs):
        del kwargs
        self.decisions += 1
        names = {tool.name for tool in tools}
        assert "lookup_representative_funds" in names
        assert "run_research_plan" in names
        if not any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(
                content="先自动选择代表产品并核对比较所需的公开数据。",
                tool_calls=[
                    {
                        "name": "lookup_representative_funds",
                        "args": {"query": "沪深300ETF及ETF联接C"},
                        "id": "representative-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(
            content=(
                "代表样本选择为场内甲沪深300ETF与场外乙沪深300ETF联接C，选择依据是产品类型已核验且"
                "公开规模与交易数据可获得。场内样本按成交活跃度、价差、规模、跟踪情况和交易佣金评估；"
                "场外样本按管理费、托管费、销售服务费、申赎规则、确认到账时间和跟踪情况评估。"
                "对六个月每月投入的计划，重视盘中成交与资金利用率可选场内样本，重视自动定投和操作便利可选"
                "场外联接C。工具未提供的实时券商佣金属于账户私有条件，最终费用需按用户券商规则复核。"
            )
        )

    async def chat_json(self, prompt, **kwargs):
        del kwargs
        payload = json.loads(prompt)
        assert payload["task_contract"]["resolve_representative_product"] is True
        assert payload["tool_observations"]
        return {
            "outcome": "satisfied",
            "satisfied": True,
            "terminal": True,
            "missing": [],
            "next_action": "",
            "reason": "代表产品、公开数据、比较维度、适用条件和不可得原因均已覆盖",
        }


@pytest.mark.asyncio
async def test_complex_fund_comparison_runs_supervisor_to_terminal_outcome(tmp_path, monkeypatch):
    async def lookup_representative_funds(query: str) -> str:
        assert "沪深300" in query
        return json.dumps(
            {
                "representatives": [
                    {"name": "甲沪深300ETF", "venue": "exchange", "verified": True},
                    {"name": "乙沪深300ETF联接C", "venue": "otc", "verified": True},
                ],
                "fee_fields": ["commission", "management", "custody", "sales_service"],
                "market_fields": ["turnover", "spread", "size", "tracking"],
                "rules": ["exchange_t_plus_1", "otc_nav_confirmation"],
            },
            ensure_ascii=False,
        )

    representative_tool = StructuredTool.from_function(
        coroutine=lookup_representative_funds,
        name="lookup_representative_funds",
        description="查找并核验代表性场内与场外基金产品及公开比较数据。",
    )
    monkeypatch.setattr(
        stock_agent_module,
        "build_chat_tools",
        lambda *args, **kwargs: [representative_tool],
    )
    fake_model = FakeSupervisorModel()
    monkeypatch.setattr(agent_loop_module, "get_llm_service", lambda: fake_model)

    store = ChatStore(tmp_path / "supervisor.db")
    await store.init()
    try:
        _, assistant_id = await store.prepare_task(
            conversation_id="complex-conversation",
            task_id="complex-task",
            message=COMPLEX_QUESTION,
        )
        manager = ChatTaskManager(store)
        await manager.start(
            ChatTaskInput(
                task_id="complex-task",
                conversation_id="complex-conversation",
                message=COMPLEX_QUESTION,
                strategy=None,
                asset_type="etf",
                assistant_message_id=assistant_id,
            )
        )
        events = [event async for event in manager.subscribe("complex-task")]

        assert fake_model.decisions >= 2
        assert any(event and event.get("event") == "task_outcome" for event in events)
        assert any(event and event.get("event") == "done" for event in events)
        task = await store.get_task("complex-task")
        assert task is not None
        assert task["status"] == "completed"
        assert task["outcome_status"] == "satisfied"
        assert task["task_acceptance"]["terminal"] is True
        assert task["task_contract"]["resolve_representative_product"] is True
        conversation = await store.get_conversation("complex-conversation")
        assert conversation is not None
        answer_parts = conversation["messages"][-1]["parts"]
        assert answer_parts
        assert "代表样本" in json.dumps(answer_parts, ensure_ascii=False)
    finally:
        await store.close()
