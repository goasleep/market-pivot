from __future__ import annotations

import json

import pytest

from application.fund_completion import validate_fund_response
from application.fund_task_compiler import compile_fund_task
from application.task_contract import compile_task_contract
from graph import agent_loop as agent_loop_module
from graph.agent_loop import judge_completion
from models.fund_task import FundTaskKind
from models.supervisor import ExecutionMode, TaskRoutingDecision


def _screening_spec(message: str):
    spec = compile_fund_task(message, asset_type="etf")
    assert spec is not None
    assert spec.task_kind == FundTaskKind.UNIVERSE_RESEARCH
    assert spec.selection_requirements is not None
    return spec


def test_universe_screening_defaults_to_a_ranked_comparison_contract():
    spec = _screening_spec("帮我挑选适合短线交易的半导体ETF")

    requirements = spec.selection_requirements
    assert requirements is not None
    assert requirements.selection_mode == "rank"
    assert requirements.holding_horizon == "short_term"
    assert requirements.minimum_candidates == 3
    assert requirements.require_alternative is True
    assert requirements.require_exclusions is True
    assert "liquidity" in requirements.comparison_dimensions
    assert "intraday_volatility" in requirements.comparison_dimensions
    assert {
        "candidate_pool",
        "comparison",
        "primary_selection",
        "alternative_selection",
        "exclusions",
        "data_as_of",
    } <= set(spec.required_outputs)


def test_screening_contract_generalizes_to_recommendation_paraphrases_and_requested_counts():
    spec = _screening_spec("请推荐5只适合中线交易的消费ETF")

    requirements = spec.selection_requirements
    assert requirements is not None
    assert requirements.selection_mode == "rank"
    assert requirements.holding_horizon == "medium_term"
    assert requirements.minimum_candidates == 5
    assert "drawdown" in requirements.comparison_dimensions


@pytest.mark.parametrize("message", ["筛选消费ETF，只给我一个结果", "找一只最适合短线的ETF，不要备选"])
def test_explicit_single_result_request_relaxes_only_the_cardinality_requirements(message):
    spec = _screening_spec(message)

    requirements = spec.selection_requirements
    assert requirements is not None
    assert requirements.selection_mode == "single"
    assert requirements.minimum_candidates == 1
    assert requirements.require_comparison is False
    assert requirements.require_alternative is False
    assert "primary_selection" in spec.required_outputs
    assert "selection_rationale" in spec.required_outputs
    assert "alternative_selection" not in spec.required_outputs


def test_screening_contract_keeps_business_requirements_when_router_only_requests_a_primary():
    routing = TaskRoutingDecision(
        mode=ExecutionMode.EVIDENCE_RESEARCH,
        requires_tools=True,
        allow_research_plan=True,
        deliverables=["给出一个首选"],
        reason="需要最新数据",
        confidence=0.9,
    )

    contract = compile_task_contract(
        "筛选适合短线交易的半导体ETF",
        asset_type="etf",
        routing_decision=routing,
    )

    assert contract.deliverables == ["给出一个首选"]
    assert "candidate_pool" in contract.required_outputs
    assert "alternative_selection" in contract.required_outputs
    assert contract.source_task_spec is not None
    assert contract.source_task_spec["selection_requirements"]["minimum_candidates"] == 3


def test_ranked_screening_response_cannot_pass_with_only_one_primary():
    spec = _screening_spec("筛选适合短线交易的半导体ETF")

    acceptance = validate_fund_response(
        spec,
        "首选：588170。依据是成交活跃，数据截至2026-08-25。",
    )

    assert acceptance.satisfied is False
    assert "minimum_candidates" in acceptance.missing
    assert "alternative_selection" in acceptance.missing


def test_ranked_screening_response_passes_with_candidates_comparison_and_backup():
    spec = _screening_spec("筛选适合短线交易的半导体ETF")

    acceptance = validate_fund_response(
        spec,
        (
            "候选池（数据截至2026-08-25，来源：结构化市场行情）：588170、159516、512480。\n"
            "对比：588170成交活跃度最高；159516日内波动更强；512480费率较低。\n"
            "首选：588170，适合短线执行。备选：159516。\n"
            "排除说明：512480因流动性相对较弱暂不选。"
        ),
    )

    assert acceptance.satisfied is True


def test_single_instrument_research_does_not_gain_screening_requirements():
    spec = compile_fund_task("分析ETF 510300的最新走势", tickers=("510300",), asset_type="etf")

    assert spec is not None
    assert spec.task_kind == FundTaskKind.INSTRUMENT_RESEARCH
    assert spec.selection_requirements is None
    assert "candidate_pool" not in spec.required_outputs


class AlwaysSatisfiedJudge:
    async def chat_json(self, prompt: str, **kwargs):
        del prompt, kwargs
        return {
            "outcome": "satisfied",
            "satisfied": True,
            "terminal": True,
            "missing": [],
            "next_action": "",
            "reason": "模型认为已完成",
        }


@pytest.mark.asyncio
async def test_deterministic_screening_acceptance_overrides_a_false_positive_llm_judge(monkeypatch):
    spec = _screening_spec("筛选适合短线交易的半导体ETF")
    monkeypatch.setattr(agent_loop_module, "get_llm_service", lambda: AlwaysSatisfiedJudge())

    completion = await judge_completion(
        {
            "messages": [],
            "candidate_response": "首选：588170。数据截至2026-08-25。",
            "task_contract": {
                "requires_tools": True,
                "source_task_spec": spec.model_dump(mode="json"),
            },
            "tool_events": [
                {
                    "name": "screen_assets",
                    "status": "completed",
                    "result": json.dumps({"count": 3, "results": [{}, {}, {}]}),
                }
            ],
            "step": 2,
            "max_steps": 4,
        }
    )

    assert completion["completion_result"]["satisfied"] is False
    assert completion["completion_result"]["terminal"] is False
    assert "minimum_candidates" in completion["completion_result"]["missing"]
    assert completion["final_response"] == ""


@pytest.mark.asyncio
async def test_disclosed_candidate_shortage_finishes_as_terminal_partial_without_looping(monkeypatch):
    spec = _screening_spec("筛选适合短线交易的半导体ETF")
    monkeypatch.setattr(agent_loop_module, "get_llm_service", lambda: AlwaysSatisfiedJudge())

    completion = await judge_completion(
        {
            "messages": [],
            "candidate_response": (
                "按当前筛选条件只有2只符合：588170、159516。首选588170，备选159516；"
                "两者已做成交活跃度对比。数据截至2026-08-25，暂无更多合格候选。"
            ),
            "task_contract": {
                "requires_tools": True,
                "source_task_spec": spec.model_dump(mode="json"),
            },
            "tool_events": [
                {
                    "name": "screen_assets",
                    "status": "completed",
                    "result": json.dumps({"count": 2, "results": [{}, {}]}),
                }
            ],
            "step": 2,
            "max_steps": 4,
        }
    )

    assert completion["completion_result"]["outcome"] == "partial"
    assert completion["completion_result"]["satisfied"] is False
    assert completion["completion_result"]["terminal"] is True
    assert completion["final_response"].startswith("按当前筛选条件只有2只符合")
