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
from graph.research_plan import (
    ResearchPlanContext,
    _call_tool,
    _classify_failure,
    _compact,
    _comparison_synthesis_text,
    _execute_step,
    _fallback_steps,
    build_research_plan_graph,
    classify_depth,
    derive_task_contract,
    replan,
    verify_evidence,
)
from llm.context import ContextBudget, ContextWindowExceededError, TokenCounter
from models.research_plan import ResearchPlan, ResearchStep
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


def test_asset_request_extracts_and_serializes_explicit_history_range():
    request = StockAgent().prepare("查询 600000 从 2025年1月1日 到 2025-01-10 的历史走势")
    payload = StockAgent.request_payload(request)
    restored = StockAgent.request_from_payload(payload)

    assert request.start_date == "2025-01-01"
    assert request.end_date == "2025-01-10"
    assert payload["start_date"] == "2025-01-01"
    assert restored.start_date == request.start_date
    assert restored.end_date == request.end_date


@pytest.mark.asyncio
async def test_price_history_step_forwards_request_date_range_to_tool():
    captured = []

    @tool
    async def get_historical_prices(
        ticker: str,
        asset_type: str = "stock",
        limit: int = 120,
        start_date: str = "",
        end_date: str = "",
    ) -> str:
        """Get history for an explicit range."""
        captured.append(
            {
                "ticker": ticker,
                "asset_type": asset_type,
                "limit": limit,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        return json.dumps({"history": [{"date": start_date, "close": 10}]})

    result = await _execute_step(
        ResearchStep(
            id="history",
            kind="price_history",
            title="获取指定区间历史价格",
            inputs={"start_date": "2024-01-01", "end_date": "2024-01-10"},
            success_criteria=["有历史价格"],
        ),
        {
            "request": {
                "tickers": ["600000"],
                "asset_type": "stock",
                "start_date": "2025-01-01",
                "end_date": "2025-01-10",
            }
        },
        ResearchPlanContext(tools={"get_historical_prices": get_historical_prices}),
    )

    assert captured == [
        {
            "ticker": "600000",
            "asset_type": "stock",
            "limit": 120,
            "start_date": "2025-01-01",
            "end_date": "2025-01-10",
        }
    ]
    assert result["items"][0]["history"][0]["date"] == "2025-01-01"


def test_multi_strategy_prompt_gets_machine_checkable_completion_contract():
    contract = derive_task_contract(
        {
            "intent": "backtest",
            "message": "请给510300执行不同的几个量化策略并回测，对比盈利情况",
        }
    )
    assert contract.operation == "strategy_comparison"
    assert contract.comparison_axis == "strategy"
    assert contract.minimum_strategy_count == 7
    assert contract.required_benchmark == "buy_hold"
    assert contract.minimum_history_years == 5
    assert {"equity_curves", "drawdown_curves", "out_of_sample", "stability"} <= set(contract.required_outputs)


def test_comparison_checkpoint_payload_preserves_a2ui_contract_and_downsamples_curves():
    curve = [{"date": f"2026-01-{index + 1:02d}", "value": index} for index in range(300)]
    payload = {
        "data_type": "strategy_backtest_comparison",
        "_tool_name": "compare_strategy_backtests",
        "ticker": "510300",
        "comparisons": [
            {
                "strategy_name": "buy_hold",
                "total_return": 0.1,
                "equity_curve": curve,
                "drawdown_curve": curve,
                "signal_curve": curve,
                "trades": [{"ignored": True}],
            }
        ],
        "conclusion": {"official": True},
        "acceptance": {"satisfied": True},
        "artifacts": [{"artifact_id": "artifact-demo"}],
        "data_validation": {"status": "verified", "differences": list(range(100))},
    }

    compact = _compact(payload)

    assert compact["_tool_name"] == "compare_strategy_backtests"
    assert compact["conclusion"]["official"] is True
    assert compact["artifacts"][0]["artifact_id"] == "artifact-demo"
    assert compact["comparisons"][0]["equity_curve"][-1] == curve[-1]
    assert len(compact["comparisons"][0]["equity_curve"]) <= 240
    assert "trades" not in compact["comparisons"][0]
    assert len(compact["data_validation"]["differences"]) == 20


def test_comparison_synthesis_uses_frozen_winners_and_artifact_count():
    text = _comparison_synthesis_text(
        {
            "evaluation_start_date": "2016-08-22",
            "evaluation_end_date": "2026-08-21",
            "warmup_bars": 252,
            "strategy_count": 11,
            "execution": {"buy_commission_rate": 0.0003, "sell_commission_rate": 0.0003, "slippage_bps": 5},
            "data_snapshot": {"adjustment": "qfq"},
            "data_validation": {"status": "degraded", "selected_source": "tencent", "selection_reason": "质量分最高"},
            "conclusion": {
                "absolute_return_winner": {
                    "strategy_name": "buy_hold",
                    "display_name": "买入持有",
                    "metric": "total_return",
                    "value": 0.7,
                },
                "tradeoffs": ["不存在唯一最好策略。"],
                "data_warnings": ["一个候选源不可用。"],
                "limitations": ["历史表现不代表未来。"],
            },
            "market_benchmark": {
                "status": "available",
                "ticker": "000300",
                "name": "沪深300",
                "comparisons": [
                    {
                        "strategy_name": "buy_hold",
                        "display_name": "买入持有",
                        "asset_total_return": 0.7,
                        "market_total_return": 0.4,
                        "excess_return": 0.3,
                    }
                ],
            },
            "artifacts": [{"name": str(index)} for index in range(7)],
        }
    )

    assert "2016-08-22 至 2026-08-21" in text
    assert "绝对收益：买入持有" in text
    assert "数据核验状态：degraded" in text
    assert "同期大盘对比" in text
    assert "超额为 +30.00%" in text
    assert "共 7 个文件" in text
    assert "Agent 建议（最终由你判断）" in text
    assert "我的建议：优先把买入持有作为下一轮模拟验证候选" in text
    assert "为什么不存在唯一" not in text
    assert "不存在唯一最好策略" not in text


def test_comparison_synthesis_treats_no_data_as_observation_not_tool_failure():
    text = _comparison_synthesis_text(
        {
            "ticker": "159999",
            "available": False,
            "message": "所有历史行情源均不可用",
            "comparisons": [],
            "conclusion": {
                "official": False,
                "limitations": ["未产生回测结果，不能比较策略盈利情况。"],
            },
        }
    )

    assert "工具已正常返回" in text
    assert "所有历史行情源均不可用" in text
    assert "没有产生回测结果" in text
    assert "成果包已生成" not in text


@pytest.mark.asyncio
async def test_synthesis_step_uses_deterministic_comparison_conclusion_without_llm():
    step = ResearchStep.model_validate(_step("synthesis", "synthesis"))
    state = {
        "request": {"message": "比较 510300 的多个策略", "asset_type": "etf", "tickers": ["510300"]},
        "plan": _plan([_step("synthesis", "synthesis")]),
        "step_results": {
            "backtest": {
                "step_id": "backtest",
                "status": "completed",
                "output": {
                    "data_type": "strategy_backtest_comparison",
                    "ticker": "510300",
                    "evaluation_start_date": "2016-08-22",
                    "evaluation_end_date": "2026-08-21",
                    "warmup_bars": 252,
                    "strategy_count": 11,
                    "conclusion": {
                        "absolute_return_winner": {
                            "strategy_name": "buy_hold",
                            "display_name": "买入持有",
                            "metric": "total_return",
                            "value": 0.7,
                        }
                    },
                },
            }
        },
    }

    result = await _execute_step(step, state, ResearchPlanContext(tools={}))

    assert result["provenance"]["source"] == "deterministic_comparison_conclusion"
    assert "绝对收益：买入持有" in result["text"]


@pytest.mark.asyncio
async def test_synthesis_omits_cross_turn_history_and_compacts_evidence(monkeypatch):
    captured: dict[str, str] = {}

    class FakeLLM:
        async def chat(self, prompt, *, system):
            captured["prompt"] = prompt
            captured["system"] = system
            return "已生成结论"

    budget = ContextBudget(
        model="gpt-4o-mini",
        context_window=4096,
        output_reserve=1024,
        safety_margin=1024,
        input_limit=2048,
    )
    monkeypatch.setattr(research_graph, "get_context_budget", lambda: budget)
    monkeypatch.setattr(research_graph, "get_llm_service", lambda: FakeLLM())
    step = ResearchStep.model_validate(_step("synthesis", "synthesis"))
    state = {
        "request": {
            "message": "分析 600519",
            "asset_type": "stock",
            "tickers": ["600519"],
            "history": [{"role": "user", "content": "HISTORY_SECRET" * 10000}],
        },
        "plan": _plan([_step("synthesis", "synthesis")]),
        "step_results": {
            "analysis": {
                "step_id": "analysis",
                "status": "completed",
                "summary": "综合分析完成",
                "output": {
                    "data_type": "analysis",
                    "results": [{"reasoning": "大量研究证据" * 1000} for _ in range(20)],
                },
            }
        },
    }

    result = await _execute_step(step, state, ResearchPlanContext(tools={}))

    assert result["text"] == "已生成结论"
    assert "HISTORY_SECRET" not in captured["prompt"]
    counter = TokenCounter(budget.model)
    assert (
        counter.count_messages(
            [
                {"role": "system", "content": captured["system"]},
                {"role": "user", "content": captured["prompt"]},
            ]
        )
        <= budget.input_limit
    )


@pytest.mark.asyncio
async def test_synthesis_context_overflow_falls_back_without_failed_step(monkeypatch):
    class OverflowLLM:
        async def chat(self, *_args, **_kwargs):
            raise ContextWindowExceededError("context overflow")

    monkeypatch.setattr(research_graph, "get_llm_service", lambda: OverflowLLM())
    step = ResearchStep.model_validate(_step("synthesis", "synthesis"))
    state = {
        "request": {"message": "分析 600519", "asset_type": "stock", "tickers": ["600519"]},
        "plan": _plan([_step("synthesis", "synthesis")]),
        "step_results": {
            "risk": {
                "step_id": "risk",
                "status": "completed",
                "summary": "风险测算完成",
                "output": {"data_type": "risk"},
            }
        },
    }

    result = await _execute_step(step, state, ResearchPlanContext(tools={}))

    assert result["provenance"]["source"] == "deterministic_context_fallback"
    assert "风险测算完成" in result["text"]


@pytest.mark.asyncio
async def test_draft_sandbox_candidate_with_valid_backtest_is_completed_research_evidence():
    result = await verify_evidence(
        {
            "plan": _plan([_step("sandbox", "backtest")]),
            "step_results": {
                "sandbox": {
                    "step_id": "sandbox",
                    "status": "completed",
                    "evidence": [
                        {
                            "source": "受限策略研究沙盒",
                            "source_type": "backtest",
                            "retrieved_at": "2026-08-22T12:00:00+00:00",
                            "data_status": "available",
                        }
                    ],
                    "output": {
                        "data_type": "sandbox_strategy_candidate",
                        "candidate_id": "candidate-draft",
                        "status": "draft",
                        "validation": {"passed": True},
                        "result": {
                            "promotion_eligible": False,
                            "backtest": {"final_value": 940_000, "total_return": -0.06},
                        },
                    },
                }
            },
            "budget": {
                "max_steps": 8,
                "max_tool_calls": 12,
                "max_replans": 1,
                "deadline_seconds": 900,
            },
            "tool_calls": 1,
            "deadline_at": "2099-01-01T00:00:00+00:00",
        }
    )

    update = result["step_results"]["sandbox"]
    assert update["status"] == "completed"
    assert update["evidence_status"] == "sufficient"
    assert update["evidence_issues"] == []


@pytest.mark.asyncio
async def test_unmet_strategy_contract_is_a_completed_limited_observation():
    result = await verify_evidence(
        {
            "plan": _plan([_step("backtest", "backtest")]),
            "step_results": {
                "backtest": {
                    "step_id": "backtest",
                    "status": "completed",
                    "attempt": 1,
                    "evidence": [
                        {
                            "source": "tencent",
                            "source_type": "backtest",
                            "retrieved_at": "2026-08-23T00:00:00+00:00",
                            "data_status": "available",
                        }
                    ],
                    "output": {
                        "data_type": "strategy_backtest_comparison",
                        "comparisons": [{"strategy_name": "buy_hold"}],
                        "acceptance": {
                            "satisfied": False,
                            "missing": ["minimum_history_years"],
                        },
                    },
                }
            },
            "budget": {
                "max_steps": 8,
                "max_tool_calls": 12,
                "max_replans": 1,
                "deadline_seconds": 900,
            },
            "tool_calls": 1,
            "deadline_at": "2099-01-01T00:00:00+00:00",
        }
    )

    update = result["step_results"]["backtest"]
    assert update["status"] == "completed"
    assert update["evidence_status"] == "limited"
    assert update["error"] is None
    assert "minimum_history_years" in update["evidence_issues"][0]
    assert result["needs_replan"] is False
    snapshot = _plan_snapshot(
        {
            "plan": _plan([_step("backtest", "backtest")]),
            "step_results": result["step_results"],
        },
        status="completed",
    )
    assert snapshot is not None
    assert snapshot["status"] == "completed_with_gaps"
    assert snapshot["steps"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_empty_market_data_is_a_completed_unavailable_observation():
    result = await verify_evidence(
        {
            "plan": _plan([_step("history", "price_history")]),
            "step_results": {
                "history": {
                    "step_id": "history",
                    "status": "completed",
                    "attempt": 1,
                    "evidence": [
                        {
                            "source": "akshare",
                            "source_type": "market_data",
                            "retrieved_at": "2026-08-23T00:00:00+00:00",
                            "data_status": "unavailable",
                        }
                    ],
                    "output": {
                        "data_type": "price_history_collection",
                        "items": [
                            {
                                "ticker": "159999",
                                "available": False,
                                "history": [],
                                "error": {"message": "历史价格数据不可用"},
                            }
                        ],
                    },
                }
            },
            "budget": {
                "max_steps": 8,
                "max_tool_calls": 12,
                "max_replans": 1,
                "deadline_seconds": 900,
            },
            "tool_calls": 1,
            "deadline_at": "2099-01-01T00:00:00+00:00",
        }
    )

    update = result["step_results"]["history"]
    assert update["status"] == "completed"
    assert update["evidence_status"] == "unavailable"
    assert update["error"] is None
    assert result["needs_replan"] is False


@pytest.mark.asyncio
async def test_research_plan_uses_shared_long_running_tool_timeout(monkeypatch):
    captured: dict[str, int] = {}

    @tool
    async def run_fund_or_stock_analysis(ticker: str, asset_type: str = "stock") -> str:
        """Run comprehensive analysis."""
        return json.dumps({"ticker": ticker, "asset_type": asset_type})

    original_wait_for = asyncio.wait_for

    async def capture_wait_for(awaitable, timeout):
        captured["timeout"] = timeout
        return await original_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(research_graph.asyncio, "wait_for", capture_wait_for)
    context = ResearchPlanContext(tools={run_fund_or_stock_analysis.name: run_fund_or_stock_analysis})

    await _call_tool(
        context,
        "run_fund_or_stock_analysis",
        {"ticker": "510300", "asset_type": "etf"},
    )

    assert captured["timeout"] == research_graph.tool_timeout_seconds("run_fund_or_stock_analysis") == 900


@pytest.mark.asyncio
async def test_research_plan_treats_structured_tool_failure_as_recoverable_error():
    @tool
    async def broken_tool() -> str:
        """Return a structured tool error."""
        return json.dumps({"ok": False, "error": {"code": "invalid_arguments", "message": "limit 参数无效"}})

    context = ResearchPlanContext(tools={broken_tool.name: broken_tool})

    with pytest.raises(RuntimeError, match="limit 参数无效"):
        await _call_tool(context, broken_tool.name, {})


@pytest.mark.asyncio
async def test_multi_strategy_request_uses_strategy_comparison_backtest_tool():
    captured = {}

    @tool
    async def compare_strategy_backtests(
        ticker: str,
        start_date: str,
        end_date: str,
        asset_type: str = "stock",
        objective: str = "",
    ) -> str:
        """Compare strategies."""
        captured.update(
            {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "asset_type": asset_type,
                "objective": objective,
            }
        )
        return json.dumps({"data_type": "strategy_backtest_comparison"})

    step = ResearchStep.model_validate(_step("backtest", "backtest"))
    state = {
        "request": {
            "message": "给510300执行不同的几个量化策略并回测，对比盈利情况",
            "intent": "backtest",
            "tickers": ["510300"],
            "asset_type": "etf",
            "as_of_date": "2026-08-21",
        },
        "plan": _plan([_step("backtest", "backtest")]),
        "step_results": {},
    }
    context = ResearchPlanContext(tools={compare_strategy_backtests.name: compare_strategy_backtests})

    result = await _execute_step(step, state, context)

    assert result["data_type"] == "strategy_backtest_comparison"
    assert captured == {
        "ticker": "510300",
        "start_date": "2016-08-21",
        "end_date": "2026-08-21",
        "asset_type": "etf",
        "objective": "给510300执行不同的几个量化策略并回测，对比盈利情况",
    }


@pytest.mark.asyncio
async def test_strategy_comparison_contract_overrides_planner_mode_and_preserves_user_dates():
    captured = {}

    @tool
    async def compare_strategy_backtests(
        ticker: str,
        start_date: str,
        end_date: str,
        asset_type: str = "stock",
        objective: str = "",
    ) -> str:
        """Compare strategies."""
        captured.update(locals())
        return json.dumps({"data_type": "strategy_backtest_comparison"})

    step = ResearchStep.model_validate(
        _step("backtest", "backtest")
        | {
            "inputs": {
                "objective": "运行一个回测实验",
                "execution_mode": "agent",
                "start_date": "2018-01-01",
                "end_date": "2026-08-23",
            }
        }
    )
    state = {
        "request": {
            "message": "给510300执行多个量化策略并对比，使用2016-08-22至2026-08-21。",
            "intent": "backtest",
            "tickers": ["510300"],
            "asset_type": "etf",
            "as_of_date": "2026-08-23",
        },
        "task_contract": {"operation": "strategy_comparison"},
        "plan": _plan([_step("backtest", "backtest")]),
        "step_results": {},
    }
    context = ResearchPlanContext(tools={compare_strategy_backtests.name: compare_strategy_backtests})

    result = await _execute_step(step, state, context)

    assert result["data_type"] == "strategy_backtest_comparison"
    assert captured["start_date"] == "2016-08-22"
    assert captured["end_date"] == "2026-08-21"


def test_tool_failure_classifier_separates_retry_adjust_and_terminal_errors():
    assert _classify_failure("connection timeout while reading response") == "transient"
    assert _classify_failure("策略包含不受支持的指标: fast_ma") == "correctable"
    assert _classify_failure("permission denied") == "terminal"
    assert _classify_failure("unexpected upstream response") == "unknown"


@pytest.mark.asyncio
async def test_replan_reflects_on_failure_and_applies_a_bounded_input_patch(monkeypatch):
    class RecoveryLLM:
        async def chat_json(self, prompt, system):
            assert "fast_ma" in prompt
            assert "allowed_input_patch" in prompt
            assert "不要输出内部思维链" in system
            return {
                "action": "adjust",
                "summary": "改用受控均线价差指标描述后重新调用回测工具。",
                "input_patch": {
                    "objective": "只使用 ma_spread_pct，并配置 fast_window=5、slow_window=20。",
                    "execution_mode": "agent",
                    "ticker": "600519",
                    "initial_capital": 200000,
                },
            }

    monkeypatch.setattr(research_graph, "get_llm_service", lambda: RecoveryLLM())
    steps = [
        _step("snapshot", "market_snapshot"),
        _step("history", "price_history"),
        _step("method", "methodology"),
        _step("backtest", "backtest") | {"max_attempts": 2},
    ]
    state = {
        "request": {
            "message": "为 510300 设计均线交叉策略并回测",
            "intent": "quote",
            "tickers": ["510300"],
            "asset_type": "etf",
            "as_of_date": "2026-08-22",
        },
        "plan": {
            **_plan(steps),
            "asset_type": "stock",
            "tickers": ["510300"],
        },
        "step_results": {
            "backtest": {
                "step_id": "backtest",
                "status": "failed",
                "attempt": 1,
                "error": "策略包含不受支持的指标: fast_ma; slow_ma",
            }
        },
        "budget": {
            "max_steps": 8,
            "max_tool_calls": 16,
            "max_replans": 1,
            "deadline_seconds": 900,
        },
        "replan_count": 0,
    }

    update = await replan(state)

    revised_step = next(item for item in update["plan"]["steps"] if item["id"] == "backtest")
    revised_result = update["step_results"]["backtest"]
    assert update["plan"]["revision"] == 2
    assert update["replan_count"] == 1
    assert revised_result["status"] == "pending"
    assert revised_result["recovery_history"][0]["action"] == "adjust"
    assert revised_step["inputs"]["execution_mode"] == "agent"
    assert revised_step["inputs"]["initial_capital"] == 200000
    assert "为 510300 设计均线交叉策略并回测" in revised_step["inputs"]["objective"]
    assert "ma_spread_pct" in revised_step["inputs"]["objective"]
    assert "ticker" not in revised_step["inputs"]

    snapshot = _plan_snapshot(
        {
            "plan": update["plan"],
            "step_results": update["step_results"],
        }
    )
    assert snapshot is not None
    recovery = next(item for item in snapshot["steps"] if item["id"] == "backtest")["recovery"]
    assert recovery["action"] == "adjust"
    assert "受控均线价差" in recovery["summary"]


@pytest.mark.asyncio
async def test_revised_backtest_step_uses_adjusted_arguments_without_changing_ticker():
    captured = {}

    @tool
    async def design_and_run_backtest(
        objective: str,
        ticker: str,
        start_date: str,
        end_date: str,
        asset_type: str,
        initial_capital: float = 1_000_000,
    ) -> str:
        """Run an adjusted backtest."""
        captured.update(locals())
        return json.dumps({"data_type": "backtest_experiment", "result": {"final_value": 210000}})

    step = ResearchStep.model_validate(
        _step("backtest", "backtest")
        | {
            "inputs": {
                "objective": "原目标；只使用 ma_spread_pct",
                "start_date": "2020-01-01",
                "end_date": "2026-08-20",
                "initial_capital": 200000,
                "execution_mode": "agent",
            }
        }
    )
    state = {
        "request": {
            "message": "原目标",
            "tickers": ["510300"],
            "asset_type": "etf",
            "as_of_date": "2026-08-22",
        },
        "plan": _plan([_step("backtest", "backtest")]),
        "step_results": {},
    }
    context = ResearchPlanContext(tools={design_and_run_backtest.name: design_and_run_backtest})

    result = await _execute_step(step, state, context)

    assert result["result"]["final_value"] == 210000
    assert captured["ticker"] == "510300"
    assert captured["objective"] == "原目标；只使用 ma_spread_pct"
    assert captured["initial_capital"] == 200000


@pytest.mark.asyncio
async def test_research_graph_reflects_after_failure_and_succeeds_on_second_tool_call(monkeypatch):
    calls = 0

    @tool
    async def get_realtime_quote(ticker: str, asset_type: str = "stock") -> str:
        """Get a quote, failing once to exercise recovery."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("invalid upstream response schema")
        return json.dumps(
            {
                "quote": {"ticker": ticker, "asset_type": asset_type, "price": 4.2},
                "provenance": {
                    "source": "test-market-data",
                    "as_of": "2026-08-22T09:30:00+08:00",
                    "fetched_at": "2026-08-22T09:30:01+08:00",
                    "status": "available",
                },
            }
        )

    class PlannerAndRecoveryLLM:
        async def chat_json(self, _prompt, system):
            if "市场研究 Planner" in system:
                raise RuntimeError("use deterministic plan")
            return {
                "action": "retry",
                "summary": "上游响应结构异常，保留原查询范围重试一次。",
                "input_patch": {},
            }

    monkeypatch.setattr(research_graph, "get_llm_service", lambda: PlannerAndRecoveryLLM())
    graph = build_research_plan_graph(MemorySaver())

    result = await graph.ainvoke(
        {
            "request": {
                "message": "查询 510300 行情",
                "intent": "quote",
                "tickers": ["510300"],
                "asset_type": "etf",
                "task_id": "task-recovery",
            }
        },
        config={"configurable": {"thread_id": "task-recovery"}, "recursion_limit": 40},
        context=ResearchPlanContext(tools={get_realtime_quote.name: get_realtime_quote}),
    )

    step_result = result["step_results"]["market_snapshot"]
    assert calls == 2
    assert result["plan"]["revision"] == 2
    assert result["replan_count"] == 1
    assert step_result["status"] == "completed"
    assert step_result["attempt"] == 2
    assert step_result["recovery_history"][0]["action"] == "retry"
    assert step_result["failure_context"]["tool_name"] == "get_realtime_quote"
    assert step_result["failure_context"]["args"]["ticker"] == "510300"
    assert step_result["output"]["quote"]["price"] == 4.2


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
async def test_all_readonly_research_intents_expose_plan_as_supervisor_tool(monkeypatch, intent):
    calls = []

    async def fake_supervisor(messages, tools, **kwargs):
        calls.append(({item.name for item in tools}, kwargs["task_contract"]))
        yield {
            "judge": {
                "final_response": "supervised",
                "completion_result": {
                    "outcome": "satisfied",
                    "satisfied": True,
                    "terminal": True,
                },
            }
        }

    monkeypatch.setattr(stock_agent_module, "stream_agent_loop", fake_supervisor)
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
    assert events[-1] == {"type": "text", "text": "supervised"}
    assert "run_research_plan" in calls[0][0]
    assert calls[0][1]["objective"] == request.message


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
