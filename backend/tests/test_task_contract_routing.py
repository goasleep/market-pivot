from __future__ import annotations

import pytest

from application import task_contract as task_contract_module
from application.task_contract import classify_task_execution, compile_task_contract
from models.fund_task import FundTaskKind
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
async def test_explicit_sandbox_execution_cannot_be_downgraded_to_direct_response(monkeypatch):
    service = FakeRoutingService(
        {
            "mode": "direct_response",
            "requires_tools": False,
            "allow_research_plan": False,
            "deliverables": ["给出代码示例"],
            "reason": "误判为代码示例",
            "confidence": 0.7,
        }
    )
    monkeypatch.setattr(task_contract_module, "get_llm_service", lambda: service)

    decision = await classify_task_execution(
        "用 Python 生成代码策略并回测 510300",
        tickers=("510300",),
        asset_type="etf",
    )

    assert decision.mode == ExecutionMode.BACKTEST_EXECUTION
    assert decision.requires_tools is True
    assert decision.allow_research_plan is False
    assert decision.deliverables == ["生成的策略源码", "沙箱验证结果", "可信交易引擎回测结果"]


@pytest.mark.asyncio
async def test_explicit_non_execution_language_keeps_code_explanation_direct(monkeypatch):
    service = FakeRoutingService(
        {
            "mode": "direct_response",
            "requires_tools": False,
            "allow_research_plan": False,
            "deliverables": ["解释代码"],
            "reason": "用户明确要求不执行",
            "confidence": 0.99,
        }
    )
    monkeypatch.setattr(task_contract_module, "get_llm_service", lambda: service)

    decision = await classify_task_execution(
        "只解释这段 Python 回测代码，不要执行",
        tickers=("510300",),
        asset_type="etf",
    )

    assert decision.mode == ExecutionMode.DIRECT_RESPONSE
    assert decision.requires_tools is False


@pytest.mark.asyncio
async def test_non_strategy_python_generation_does_not_trigger_sandbox(monkeypatch):
    service = FakeRoutingService(
        {
            "mode": "direct_response",
            "requires_tools": False,
            "allow_research_plan": False,
            "deliverables": ["给出图表代码"],
            "reason": "不涉及策略执行",
            "confidence": 0.99,
        }
    )
    monkeypatch.setattr(task_contract_module, "get_llm_service", lambda: service)

    decision = await classify_task_execution("生成一段 Python 图表代码", asset_type="etf")

    assert decision.mode == ExecutionMode.DIRECT_RESPONSE
    assert decision.requires_tools is False


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


@pytest.mark.asyncio
async def test_fund_universe_screening_cannot_be_downgraded_to_a_tool_free_response(monkeypatch):
    service = FakeRoutingService(
        {
            "mode": "direct_response",
            "requires_tools": False,
            "allow_research_plan": False,
            "deliverables": ["给出首选"],
            "reason": "误判为普通建议",
            "confidence": 0.7,
        }
    )
    monkeypatch.setattr(task_contract_module, "get_llm_service", lambda: service)

    decision = await classify_task_execution("挑选适合短线的半导体ETF", asset_type="etf")
    contract = compile_task_contract(
        "挑选适合短线的半导体ETF",
        asset_type="etf",
        routing_decision=decision,
    )

    assert decision.mode == ExecutionMode.EVIDENCE_RESEARCH
    assert decision.requires_tools is True
    assert contract.requires_tools is True
    assert contract.source_task_spec is not None
    assert contract.source_task_spec["task_kind"] == FundTaskKind.UNIVERSE_RESEARCH.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "asset_type"),
    [
        ("筛选连续5年都有现金分红的全部A股，并按累计分红排序", "stock"),
        ("系统目前支持哪些交易策略？", "stock"),
    ],
)
async def test_p0_skill_requests_cannot_be_downgraded_to_tool_free_responses(monkeypatch, message, asset_type):
    service = FakeRoutingService(
        {
            "mode": "direct_response",
            "requires_tools": False,
            "allow_research_plan": False,
            "deliverables": ["直接回答"],
            "reason": "误判为静态知识",
            "confidence": 0.7,
        }
    )
    monkeypatch.setattr(task_contract_module, "get_llm_service", lambda: service)

    decision = await classify_task_execution(message, asset_type=asset_type)

    assert decision.mode == ExecutionMode.EVIDENCE_RESEARCH
    assert decision.requires_tools is True
    assert decision.allow_research_plan is False
