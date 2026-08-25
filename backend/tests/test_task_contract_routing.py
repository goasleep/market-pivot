from __future__ import annotations

import pytest

from application import task_contract as task_contract_module
from application.task_contract import classify_task_execution, compile_task_contract
from models.supervisor import ExecutionMode, TaskRoutingDecision


class FakeRoutingService:
    def __init__(self, payload):
        self.payload = payload
        self.prompts: list[str] = []

    async def chat_json(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        assert "关键词" not in kwargs.get("system", "")
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


@pytest.mark.asyncio
async def test_model_classifies_code_review_as_direct_response(monkeypatch):
    service = FakeRoutingService(
        {
            "mode": "direct_response",
            "requires_tools": False,
            "allow_research_plan": False,
            "deliverables": ["指出代码错误", "说明修正方法"],
            "reason": "用户要求审查已有伪代码，不要求执行真实回测",
            "confidence": 0.99,
        }
    )
    monkeypatch.setattr(task_contract_module, "get_llm_service", lambda: service)

    decision = await classify_task_execution(
        "以下510300回测伪代码存在至少3处问题，请找出并说明。",
        tickers=("510300",),
        asset_type="etf",
    )
    contract = compile_task_contract(
        "以下510300回测伪代码存在至少3处问题，请找出并说明。",
        tickers=("510300",),
        asset_type="etf",
        routing_decision=decision,
    )

    assert decision.mode == ExecutionMode.DIRECT_RESPONSE
    assert contract.requires_tools is False
    assert contract.evidence_requirements == []
    assert contract.source_task_spec is None
    assert contract.deliverables == ["指出代码错误", "说明修正方法"]
    assert contract.routing is not None
    assert service.prompts


@pytest.mark.asyncio
async def test_model_classifies_real_backtest_as_backtest_execution(monkeypatch):
    service = FakeRoutingService(
        {
            "mode": "backtest_execution",
            "requires_tools": True,
            "allow_research_plan": True,
            "deliverables": ["回测结果", "成本口径", "限制说明"],
            "reason": "用户明确要求运行历史回测",
            "confidence": 0.98,
        }
    )
    monkeypatch.setattr(task_contract_module, "get_llm_service", lambda: service)

    decision = await classify_task_execution(
        "请实际运行510300从2019年至2025年的回测并给出结果。",
        tickers=("510300",),
        asset_type="etf",
    )
    contract = compile_task_contract(
        "请实际运行510300从2019年至2025年的回测并给出结果。",
        tickers=("510300",),
        asset_type="etf",
        routing_decision=decision,
    )

    assert contract.requires_tools is True
    assert contract.source_task_spec is not None
    assert contract.source_task_spec["operation"] == "backtest"
    assert contract.evidence_requirements == ["nav_history"]


@pytest.mark.asyncio
async def test_routing_failure_defers_to_supervisor_without_keyword_fallback(monkeypatch):
    service = FakeRoutingService(RuntimeError("routing unavailable"))
    monkeypatch.setattr(task_contract_module, "get_llm_service", lambda: service)

    decision = await classify_task_execution(
        "请分析510300的风险和收益",
        tickers=("510300",),
        asset_type="etf",
    )
    contract = compile_task_contract(
        "请分析510300的风险和收益",
        tickers=("510300",),
        asset_type="etf",
        routing_decision=decision,
    )

    assert decision == TaskRoutingDecision.supervisor_fallback()
    assert contract.requires_tools is False
    assert contract.source_task_spec is None
    assert contract.routing is not None
    assert contract.routing.mode == ExecutionMode.SUPERVISOR_DECIDES
