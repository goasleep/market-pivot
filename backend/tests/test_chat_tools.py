import asyncio
import json

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

import graph.agent_loop as agent_loop_module
from agents.sentiment_analyst import analyze as analyze_sentiment
from agents.stock_agent import StockAgent, _compact_generated_report, _select_tools_for_routing
from application import strategy_comparison
from application.research import research_service
from artifacts.service import ArtifactService
from artifacts.storage import LocalArtifactStorage
from data import serper_provider
from data.backtest_data import BacktestDataError
from data.web_content import extract_article_content
from engine.simulation_account import SimulationAccountService
from graph.agent_loop import (
    LONG_RUNNING_TOOL_TIMEOUT_SECONDS,
    TOOL_TIMEOUT_SECONDS,
    tool_attempts,
    tool_timeout_seconds,
)
from llm.context import ContextWindowExceededError
from models.schemas import AssetType, Decision, MarketContext, TradeDecision
from models.supervisor import ExecutionMode, TaskRoutingDecision
from tools import artifacts as artifact_tools
from tools import assets, data, research, simulation
from tools.policies import tool_requires_confirmation
from tools.registry import build_chat_tools
from widgets.a2ui import render_activity, render_tool_result


def test_chat_tools_expose_paper_portfolio_and_orders(monkeypatch, tmp_path):
    accounts = SimulationAccountService(tmp_path / "simulation.db")
    monkeypatch.setattr(simulation, "simulation_accounts", accounts)

    portfolio = asyncio.run(simulation.get_simulation_portfolio.ainvoke({"account_id": "default"}))
    portfolio_payload = json.loads(portfolio)
    assert portfolio_payload["paper_trading"] is True
    assert portfolio_payload["portfolio"]["account_id"] == "default"

    order = asyncio.run(
        simulation.submit_simulation_order.ainvoke(
            {
                "ticker": "510300",
                "side": "buy",
                "shares": 100,
                "account_id": "default",
                "asset_type": "etf",
            }
        )
    )
    order_payload = json.loads(order)
    assert order_payload["paper_trading"] is True
    assert order_payload["order"]["status"] == "pending"

    orders = asyncio.run(simulation.get_simulation_orders.ainvoke({"account_id": "default"}))
    assert len(json.loads(orders)["orders"]) == 1


def test_simulation_order_execution_key_is_idempotent(monkeypatch, tmp_path):
    accounts = SimulationAccountService(tmp_path / "simulation-idempotent.db")
    monkeypatch.setattr(simulation, "simulation_accounts", accounts)
    arguments = {
        "ticker": "510300",
        "side": "buy",
        "shares": 100,
        "account_id": "default",
        "asset_type": "etf",
        "execution_key": "task-1:call-1",
    }

    first = json.loads(asyncio.run(simulation.submit_simulation_order.ainvoke(arguments)))
    second = json.loads(asyncio.run(simulation.submit_simulation_order.ainvoke(arguments)))
    orders = asyncio.run(accounts.list_orders("default"))

    assert first["order"]["order_id"] == second["order"]["order_id"]
    assert len(orders) == 1


def test_analysis_tool_requires_and_validates_asset_type():
    tool = StockAgent()._analysis_tool()
    schema = tool.args_schema.model_json_schema()

    assert "asset_type" in schema["required"]
    assert schema["properties"]["asset_type"]["enum"] == ["stock", "etf", "lof"]

    with pytest.raises(Exception, match="asset_type"):
        asyncio.run(tool.ainvoke({"ticker": "510300", "asset_type": "invalid"}))


def test_chat_agent_exposes_only_provider_agnostic_web_search():
    names = {tool.name for tool in build_chat_tools(assets.get_realtime_quote)}
    assert "search_web" in names
    assert "search_web_anysearch" not in names
    assert "search_web_ddgs" not in names
    assert "get_latest_news" not in names


def test_chat_agent_hides_mutating_tools_without_explicit_execution_request():
    readonly_names = {
        tool.name
        for tool in build_chat_tools(assets.get_realtime_quote, allow_mutating_tools=False)
    }
    assert "submit_simulation_order" not in readonly_names
    assert "cancel_simulation_order" not in readonly_names
    assert "create_simulation_account" not in readonly_names
    assert "deploy_backtest_experiment" not in readonly_names
    assert "set_strategy_deployment_status" not in readonly_names
    assert "list_simulation_accounts" in readonly_names
    assert "list_strategy_deployments" in readonly_names

    execution_names = {
        tool.name
        for tool in build_chat_tools(assets.get_realtime_quote, allow_mutating_tools=True)
    }
    assert "submit_simulation_order" in execution_names
    assert "create_simulation_account" in execution_names
    assert "deploy_backtest_experiment" in execution_names
    assert "set_strategy_deployment_status" in execution_names


def test_deployment_chat_tools_require_explicit_language_and_confirmation():
    agent = StockAgent()
    assert agent._explicitly_requests_mutation("把这个回测部署到模拟盘") is True
    assert agent._explicitly_requests_mutation("暂停部署 deploy-123") is True
    assert agent._explicitly_requests_mutation("查看有哪些模拟盘") is False
    for name in (
        "create_simulation_account",
        "deploy_backtest_experiment",
        "set_strategy_deployment_status",
    ):
        assert tool_requires_confirmation(name) is True


@pytest.mark.asyncio
async def test_agent_loop_pauses_before_confirmed_tool_execution(monkeypatch):
    called = []

    async def submit_order(ticker: str) -> str:
        called.append(ticker)
        return "submitted"

    tool = StructuredTool.from_function(
        coroutine=submit_order,
        name="submit_simulation_order",
        description="创建模拟订单",
    )

    class FakeLLM:
        async def chat_with_tools(self, messages, tools, temperature=0.2):
            del messages, tools, temperature
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_simulation_order",
                        "args": {"ticker": "510300"},
                        "id": "call-confirm-1",
                        "type": "tool_call",
                    }
                ],
            )

    monkeypatch.setattr(agent_loop_module, "get_llm_service", lambda: FakeLLM())
    updates = [
        update
        async for update in agent_loop_module.stream_agent_loop(
            [{"role": "user", "content": "提交模拟订单"}],
            [tool],
            max_steps=2,
        )
    ]
    pending = [
        node_update.get("pending_tool_confirmation")
        for update in updates
        for node_update in update.values()
        if isinstance(node_update, dict) and node_update.get("pending_tool_confirmation")
    ]
    assert pending[0]["tool_call_id"] == "call-confirm-1"
    assert called == []


@pytest.mark.asyncio
async def test_agent_loop_context_exhaustion_finishes_with_deterministic_tool_result_handoff(monkeypatch):
    class OverflowLLM:
        async def chat_with_tools(self, messages, tools, temperature=0.2):
            del messages, tools, temperature
            raise ContextWindowExceededError("context overflow")

    monkeypatch.setattr(agent_loop_module, "get_llm_service", lambda: OverflowLLM())
    state = await agent_loop_module.run_agent_loop(
        [
            {"role": "user", "content": "分析 510300"},
            AIMessage(
                content="",
                tool_calls=[{"name": "quote", "args": {}, "id": "quote-1", "type": "tool_call"}],
            ),
            ToolMessage(content='{"price":4.2}', tool_call_id="quote-1"),
        ],
        [],
        max_steps=2,
    )

    assert "结构化结果" in state["final_response"]
    assert state["messages"][-1].tool_calls == []


def test_chat_agent_exposes_artifact_tool_and_persists_multiple_files(monkeypatch, tmp_path):
    service = ArtifactService(
        db_path=tmp_path / "artifacts.db",
        storage=LocalArtifactStorage(tmp_path / "objects"),
    )
    monkeypatch.setattr(artifact_tools, "artifact_service", service)

    names = {tool.name for tool in build_chat_tools(assets.get_realtime_quote)}
    assert "save_artifacts" in names

    result = asyncio.run(
        artifact_tools.save_artifacts.ainvoke(
            {
                "artifacts": [
                    {"name": "摘要", "format": "md", "content": "# 摘要"},
                    {"name": "数据", "format": "csv", "content": "name,value\nA,1"},
                ]
            }
        )
    )
    payload = json.loads(result)
    assert payload["ok"] is True
    assert len(payload["artifacts"]) == 2


def test_artifact_execution_key_reuses_stable_sequence_ids(monkeypatch, tmp_path):
    service = ArtifactService(
        db_path=tmp_path / "artifacts-idempotent.db",
        storage=LocalArtifactStorage(tmp_path / "idempotent-objects"),
    )
    monkeypatch.setattr(artifact_tools, "artifact_service", service)
    arguments = {
        "artifacts": [{"name": "研究摘要", "format": "md", "content": "# 摘要"}],
        "execution_key": "task-1:call-artifact",
    }

    first = json.loads(asyncio.run(artifact_tools.save_artifacts.ainvoke(arguments)))
    second = json.loads(asyncio.run(artifact_tools.save_artifacts.ainvoke(arguments)))

    assert first["artifacts"][0]["artifact_id"] == second["artifacts"][0]["artifact_id"]


def test_chat_agent_exposes_atomic_research_tools():
    names = {tool.name for tool in build_chat_tools(assets.get_realtime_quote)}
    assert {
        "fetch_web_content",
        "get_fund_nav_history",
        "get_fundamentals",
        "compute_technical_indicators",
        "calculate_risk_metrics",
        "build_trade_plan",
        "run_backtest",
        "save_artifacts",
    }.issubset(names)


def test_risk_and_trade_plan_tools_return_traceable_calculations():
    risk = json.loads(
        asyncio.run(
            research.calculate_risk_metrics.ainvoke(
                {
                    "current_price": 10,
                    "stop_loss_pct": 0.1,
                    "take_profit_pct": 0.2,
                    "available_capital": 100_000,
                }
            )
        )
    )
    assert risk["data_type"] == "risk_metrics"
    assert risk["metrics"]["stop_loss"] == 9.0
    assert risk["metrics"]["take_profit"] == 12.0

    plan = json.loads(
        asyncio.run(
            research.build_trade_plan.ainvoke(
                {"ticker": "510300", "current_price": 4, "asset_type": "etf"}
            )
        )
    )
    assert plan["data_type"] == "trade_plan"
    assert len(plan["plan"]["price_evidence"]) == 3


def test_legacy_latest_news_combines_akshare_and_web_for_stocks(monkeypatch):
    async def fake_akshare_news(ticker, *, limit=10):
        return [{"title": "AkShare 新闻", "source": "AkShare", "date": "2026-08-16"}]

    async def fake_web_search(query, *, num_results=8, tbs=None):
        return {
            "results": [{"title": "网页公告", "source": "DDGS", "link": "https://example.com"}],
            "providers": ["DDGS"],
            "errors": [],
        }

    monkeypatch.setattr(data, "async_get_stock_news", fake_akshare_news)
    monkeypatch.setattr(data, "async_search_web_parallel", fake_web_search)
    payload = json.loads(
        asyncio.run(data.get_latest_news.ainvoke({"ticker": "600519", "asset_type": "stock"}))
    )

    assert payload["akshare_news"][0]["source"] == "AkShare"
    assert payload["web_news"][0]["source"] == "DDGS"
    assert len(payload["news"]) == 2


def test_search_web_tool_returns_source_aware_results(monkeypatch):
    captured = {}

    async def fake_search(query, *, num_results=8, tbs=None):
        captured.update({"query": query, "num_results": num_results, "tbs": tbs})
        return {
            "available": True,
            "query": query,
            "results": [{"title": "公告", "link": "https://example.com", "snippet": "摘要"}],
        }

    monkeypatch.setattr(data, "async_search_web_parallel", fake_search)
    result = asyncio.run(
        data.search_web.ainvoke(
            {"query": "510300 最新公告", "num_results": 5, "freshness": "qdr:w"}
        )
    )

    assert json.loads(result)["results"][0]["link"] == "https://example.com"
    assert json.loads(result)["data_type"] == "news"
    assert captured == {"query": "510300 最新公告", "num_results": 5, "tbs": "qdr:w"}


def test_anysearch_tool_returns_normalised_results(monkeypatch):
    captured = {}

    async def fake_search(query, *, num_results=8):
        captured.update({"query": query, "num_results": num_results})
        return {
            "available": True,
            "source": "AnySearch",
            "results": [{"title": "公告", "link": "https://example.com", "snippet": "摘要"}],
        }

    monkeypatch.setattr(data, "async_search_web_anysearch", fake_search)
    result = asyncio.run(data.search_web_anysearch.ainvoke({"query": "510300 最新公告", "num_results": 5}))

    assert json.loads(result)["results"][0]["link"] == "https://example.com"
    assert captured == {"query": "510300 最新公告", "num_results": 5}


def test_ddgs_search_tool_maps_freshness(monkeypatch):
    captured = {}

    async def fake_search(query, *, num_results=8, timelimit=None):
        captured.update({"query": query, "num_results": num_results, "timelimit": timelimit})
        return {"available": True, "source": "DDGS metasearch", "results": []}

    monkeypatch.setattr(data, "async_search_web_ddgs", fake_search)
    result = asyncio.run(
        data.search_web_ddgs.ainvoke(
            {"query": "510300 公告", "num_results": 3, "freshness": "qdr:m"}
        )
    )

    assert json.loads(result)["source"] == "DDGS metasearch"
    assert captured == {"query": "510300 公告", "num_results": 3, "timelimit": "m"}


def test_compare_quotes_uses_one_market_snapshot(monkeypatch):
    calls = []

    async def fake_snapshot(asset_type, *, limit=1000):
        calls.append({"asset_type": asset_type, "limit": limit})
        return [
            {"ticker": "600519", "name": "贵州茅台", "price": 1500},
            {"ticker": "000001", "name": "平安银行", "price": 10},
        ]

    monkeypatch.setattr(assets, "async_get_asset_spot", fake_snapshot)
    result = asyncio.run(
        assets.compare_quotes.ainvoke(
            {"tickers": ["sh600519", "000001"], "asset_type": "stock"}
        )
    )

    payload = json.loads(result)
    assert calls == [{"asset_type": "stock", "limit": 5000}]
    assert [item["quote"]["name"] for item in payload["quotes"]] == ["贵州茅台", "平安银行"]
    assert payload["provenance"][0]["source_id"] == "akshare"


def test_historical_prices_passes_explicit_date_range_to_stock_provider(monkeypatch):
    captured = {}
    schema = assets.get_historical_prices.args_schema.model_json_schema()
    assert {"start_date", "end_date"} <= set(schema["properties"])

    async def history(ticker, *, start_date, end_date):
        captured.update({"ticker": ticker, "start_date": start_date, "end_date": end_date})
        return pd.DataFrame(
            [
                {"date": "2025-01-02", "close": 10.0},
                {"date": "2025-01-03", "close": 10.2},
            ]
        )

    monkeypatch.setattr(assets, "async_get_stock_history", history)
    result = asyncio.run(
        assets.get_historical_prices.ainvoke(
            {
                "ticker": "600000",
                "asset_type": "stock",
                "start_date": "2025-01-01",
                "end_date": "2025-01-10",
            }
        )
    )
    payload = json.loads(result)

    assert captured == {"ticker": "600000", "start_date": "20250101", "end_date": "20250110"}
    assert payload["requested_range"] == {"start_date": "20250101", "end_date": "20250110"}
    assert [item["date"] for item in payload["history"]] == ["2025-01-02", "2025-01-03"]


def test_historical_prices_rejects_reversed_or_invalid_date_range():
    with pytest.raises(ValueError, match="start_date 必须早于"):
        asyncio.run(
            assets.get_historical_prices.ainvoke(
                {
                    "ticker": "600000",
                    "start_date": "2025-01-10",
                    "end_date": "2025-01-01",
                }
            )
        )
    with pytest.raises(ValueError, match="end_date 必须是"):
        asyncio.run(
            assets.get_historical_prices.ainvoke(
                {"ticker": "600000", "start_date": "2025-01-01", "end_date": "not-a-date"}
            )
        )


def test_realtime_quote_falls_back_to_latest_history(monkeypatch):
    async def empty_realtime(ticker, *, asset_type):
        return {}

    async def history(ticker, *, asset_type):
        return pd.DataFrame(
            [
                {"date": "2026-08-20", "close": 4.5, "volume": 100},
                {"date": "2026-08-21", "close": 4.68, "volume": 120},
            ]
        )

    monkeypatch.setattr(assets, "async_get_fund_realtime", empty_realtime)
    monkeypatch.setattr(assets, "async_get_fund_history", history)

    result = asyncio.run(assets.get_realtime_quote.ainvoke({"ticker": "510300", "asset_type": "etf"}))
    payload = json.loads(result)

    assert payload["available"] is True
    assert payload["data_status"] == "degraded"
    assert payload["quote"]["price"] == 4.68
    assert payload["quote"]["data_date"] == "2026-08-21"
    assert payload["provenance"][0]["status"] == "degraded"
    assert payload["provenance"][0]["freshness"] == "historical_fallback"


def test_realtime_stock_quote_uses_retrieval_time_when_provider_has_no_market_date(monkeypatch):
    async def realtime(ticker):
        return {"ticker": ticker, "price": 10.5}

    monkeypatch.setattr(assets, "async_get_stock_realtime", realtime)

    result = asyncio.run(assets.get_realtime_quote.ainvoke({"ticker": "600519", "asset_type": "stock"}))
    payload = json.loads(result)

    assert payload["data_status"] == "available"
    assert payload["quote"]["updated_at"]
    assert payload["provenance"][0]["as_of"] == payload["quote"]["updated_at"]


def test_empty_fund_nav_is_returned_as_unavailable_observation(monkeypatch):
    async def empty_nav(*_args, **_kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(assets, "async_get_fund_nav_history", empty_nav)

    result = asyncio.run(
        assets.get_fund_nav_history.ainvoke({"ticker": "159999", "asset_type": "etf"})
    )
    payload = json.loads(result)

    assert payload["available"] is False
    assert payload["data_status"] == "unavailable"
    assert payload["history"] == []
    assert payload["error"]["code"] == "fund_nav_unavailable"


def test_empty_technical_history_is_returned_as_unavailable_observation(monkeypatch):
    async def empty_history(*_args, **_kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(research, "async_get_fund_history", empty_history)

    result = asyncio.run(
        research.compute_technical_indicators.ainvoke({"ticker": "159999", "asset_type": "etf"})
    )
    payload = json.loads(result)

    assert payload["available"] is False
    assert payload["data_status"] == "unavailable"
    assert payload["indicators"] == {}


def test_compare_strategy_backtests_runs_same_assumptions_for_builtin_strategies(monkeypatch):
    import application.strategy_comparison as comparison_module

    captured = {}

    async def fake_compare(spec, **kwargs):
        captured["spec"] = spec
        captured["options"] = kwargs
        return {
            "data_type": "strategy_backtest_comparison",
            "strategy_count": len(spec.strategies),
            "ranking": [item.name for item in spec.strategies],
            "comparisons": [],
            "data_snapshot": {"sha256": "a" * 64},
            "acceptance": {"satisfied": True, "missing": []},
        }

    monkeypatch.setattr(comparison_module, "compare_strategies", fake_compare)
    result = asyncio.run(
        research.compare_strategy_backtests.ainvoke(
            {
                "ticker": "510300",
                "asset_type": "etf",
                "start_date": "2025-08-22",
                "end_date": "2026-08-21",
                "objective": "对比盈利情况",
            }
        )
    )
    payload = json.loads(result)

    assert payload["strategy_count"] == 11
    assert payload["ranking"][0] == "buy_hold"
    assert captured["spec"].asset_type == AssetType.ETF
    assert captured["spec"].initial_capital == 1_000_000
    assert captured["spec"].ranking_metric == "total_return"
    assert captured["spec"].task_contract.minimum_strategy_count == 7
    assert captured["options"] == {"publish_artifacts": True, "generate_explanation": True}


def test_strategy_comparison_returns_core_metrics_without_full_curves_or_trades():
    curve = [{"date": f"2026-01-{index % 28 + 1:02d}", "value": index} for index in range(10_000)]
    payload = {
        "data_type": "strategy_backtest_comparison",
        "ticker": "510300",
        "asset_type": "etf",
        "evaluation_start_date": "2018-01-01",
        "evaluation_end_date": "2026-08-21",
        "strategy_count": 1,
        "ranking": ["trend_pullback"],
        "comparisons": [
            {
                "strategy_name": "trend_pullback",
                "display_name": "趋势回踩",
                "annualized_return": 0.081,
                "max_drawdown": 0.173,
                "calmar_ratio": 0.468,
                "win_rate": 0.55,
                "total_trades": 28,
                "total_fees": 1234.5,
                "equity_curve": curve,
                "drawdown_curve": curve,
                "signal_curve": curve,
                "trades": [{"date": "2026-01-01", "action": "buy"}] * 100,
            }
        ],
        "cost_scenarios": {
            "base": [
                {
                    "strategy_name": "trend_pullback",
                    "total_return": 0.42,
                    "max_drawdown": 0.173,
                    "total_fees": 1234.5,
                    "total_trades": 28,
                }
            ]
        },
        "acceptance": {"satisfied": True, "missing": []},
        "conclusion": {"official": True, "limitations": ["历史回测不代表未来"]},
        "price_curve": curve,
        "artifacts": [
            {
                "artifact_id": "artifact-full",
                "name": "完整结果.json",
                "mime_type": "application/json",
                "size_bytes": 12_000_000,
                "object_key": "private/full.json",
                "download_url": "/api/artifacts/artifact-full/download",
            }
        ],
    }

    compact = research._comparison_supervisor_payload(payload)
    encoded = json.dumps(compact, ensure_ascii=False)

    assert len(encoded) < 20_000
    assert "price_curve" not in compact
    assert "equity_curve" not in compact["comparisons"][0]
    assert "trades" not in compact["comparisons"][0]
    assert compact["supervisor_summary"]["core_metrics"][0]["annualized_return"] == 0.081
    assert compact["supervisor_summary"]["core_metrics"][0]["calmar_ratio"] == 0.468
    assert compact["supervisor_summary"]["cost_scenarios"]["base"][0]["total_fees"] == 1234.5
    assert compact["artifacts"][0]["artifact_id"] == "artifact-full"
    assert "object_key" not in compact["artifacts"][0]
    assert "不要调用 read_artifact" in compact["supervisor_summary"]["instruction"]


def test_backtest_routing_hides_read_artifact_from_supervisor():
    def placeholder():
        return "ok"

    read_tool = StructuredTool.from_function(
        func=placeholder,
        name="read_artifact",
        description="Read artifact",
    )
    backtest_tool = StructuredTool.from_function(
        func=placeholder,
        name="compare_strategy_backtests",
        description="Run comparison",
    )
    tools = [read_tool, backtest_tool]

    for mode in (ExecutionMode.BACKTEST_EXECUTION, ExecutionMode.MIXED_WORKFLOW):
        selected = _select_tools_for_routing(
            tools,
            [read_tool],
            TaskRoutingDecision(mode=mode, requires_tools=True),
        )
        assert [tool.name for tool in selected] == ["compare_strategy_backtests"]

    research_selected = _select_tools_for_routing(
        tools,
        [read_tool],
        TaskRoutingDecision(mode=ExecutionMode.EVIDENCE_RESEARCH, requires_tools=True),
    )
    assert [tool.name for tool in research_selected] == ["read_artifact", "compare_strategy_backtests"]


def test_strategy_comparison_without_history_returns_completed_observation(monkeypatch):
    async def no_history(*_args, **_kwargs):
        raise BacktestDataError("159999 所有历史行情源均不可用")

    monkeypatch.setattr(strategy_comparison, "compare_strategies", no_history)
    result = asyncio.run(
        research.compare_strategy_backtests.ainvoke(
            {
                "ticker": "159999",
                "asset_type": "etf",
                "start_date": "2016-08-22",
                "end_date": "2026-08-21",
            }
        )
    )
    payload = json.loads(result)

    assert payload["available"] is False
    assert payload["data_status"] == "unavailable"
    assert payload["comparisons"] == []
    assert payload["acceptance"]["satisfied"] is False
    assert payload["error"]["code"] == "backtest_data_unavailable"


def test_sandbox_strategy_tool_keeps_source_code_for_a2ui(monkeypatch):
    import application.strategy_candidates as candidate_module

    class FakeCandidate:
        def model_dump(self, mode="json"):
            assert mode == "json"
            return {
                "candidate_id": "candidate-demo",
                "status": "validated",
                "name": "demo",
                "ticker": "510300",
                "asset_type": "etf",
                "source_code": "def generate_target_positions(frame):\n    return [0] * len(frame)",
                "strategy_spec": {"name": "demo"},
                "validation": {"passed": True},
                "result": {"promotion_eligible": True, "backtest": {}},
            }

    async def fake_generate(**_kwargs):
        return FakeCandidate()

    monkeypatch.setattr(candidate_module.strategy_candidates, "generate", fake_generate)
    result = asyncio.run(
        research.design_and_run_sandbox_strategy.ainvoke(
            {
                "objective": "生成一个测试策略",
                "ticker": "510300",
                "start_date": "2020-01-01",
                "end_date": "2026-08-21",
                "asset_type": "etf",
            }
        )
    )
    payload = json.loads(result)

    assert "generate_target_positions" in payload["source_code"]
    assert payload["data_type"] == "sandbox_strategy_candidate"


def test_parallel_search_merges_serper_and_ddgs_results(monkeypatch):
    monkeypatch.setattr(serper_provider.settings, "anysearch_api_key", "configured")
    monkeypatch.setattr(serper_provider.settings, "serper_api_key", "configured")

    async def fake_anysearch(query, *, num_results=8):
        return {
            "available": True,
            "source": "AnySearch",
            "results": [{"title": "AnySearch 结果", "link": "https://any.example", "snippet": "anysearch"}],
        }

    async def fake_serper(query, *, num_results=8, tbs=None):
        return {
            "available": True,
            "source": "Serper / Google Search",
            "results": [{"title": "共同结果", "link": "https://same.example", "snippet": "serper"}],
        }

    async def fake_ddgs(query, *, num_results=8, timelimit=None):
        return {
            "available": True,
            "source": "DDGS metasearch",
            "results": [
                {"title": "共同结果", "link": "https://same.example", "snippet": "ddgs"},
                {"title": "独立结果", "link": "https://ddgs.example", "snippet": "ddgs"},
            ],
        }

    monkeypatch.setattr(serper_provider, "async_search_web_anysearch", fake_anysearch)
    monkeypatch.setattr(serper_provider, "async_search_web", fake_serper)
    monkeypatch.setattr(serper_provider, "async_search_web_ddgs", fake_ddgs)
    result = asyncio.run(serper_provider.async_search_web_parallel("510300 公告", num_results=5))

    assert result["providers"] == ["AnySearch", "Serper / Google Search", "DDGS metasearch"]
    assert [item["link"] for item in result["results"]] == [
        "https://any.example",
        "https://same.example",
        "https://ddgs.example",
    ]


def test_web_content_extractor_removes_non_article_markup():
    result = extract_article_content(
        """
        <html><head><title>半导体行业快讯</title><script>恶意指令</script></head>
        <body><nav>导航菜单</nav><article><p>半导体设备订单在近期出现改善，相关公司披露了新的业务进展。</p>
        <p>市场参与者仍需关注估值、出口限制和需求波动等风险因素。</p>
        <p>该信息仅代表公开报道中的行业变化，不能直接视为具体标的的买卖建议。</p></article>
        <footer>版权信息</footer></body></html>
        """
    )

    assert result["content_status"] == "full_text"
    assert "半导体设备订单" in result["content"]
    assert "恶意指令" not in result["content"]
    assert "导航菜单" not in result["content"]
    assert result["page_title"] == "半导体行业快讯"


def test_web_content_extractor_removes_portal_boilerplate_and_prefers_article():
    html = """
    <html><head><title>沪深300ETF份额变化</title></head><body>
      <div><a href="/guba">股吧首页</a> | <a href="/hot">热门个股吧</a> |
      <a href="/topics">热门主题吧</a> | <a href="/more">更多</a></div>
      <div>欢迎扫码安装 - 行情和统计 相关公告 基本信息 公告申购赎回清单</div>
      <main><article>
        <h1>沪深300ETF份额变化</h1>
        <p>最新公开数据显示，该基金份额较上一交易日有所变化，成交额保持在近期正常区间。</p>
        <p>分析交易信号时仍应结合折溢价、成交量、跟踪误差以及市场整体趋势进行判断。</p>
      </article></main>
      <div>郑重声明：本网站所刊载的所有资料及图表仅供参考使用，投资者据此操作风险自担。</div>
      <div>扫一扫下载APP 东方财富Level-2 东方财富证券开户 东方财富在线交易</div>
      <div>信息网络传播视听节目许可证 0908328号 沪ICP备05006054号 版权所有</div>
    </body></html>
    """

    result = extract_article_content(html)
    second_result = extract_article_content(html)

    assert result["content_status"] == "full_text"
    assert result["content_filter_version"] == "v3"
    assert "基金份额较上一交易日" in result["content"]
    assert "股吧首页" not in result["content"]
    assert "欢迎扫码" not in result["content"]
    assert "郑重声明" not in result["content"]
    assert "东方财富Level-2" not in result["content"]
    assert "沪ICP备" not in result["content"]
    assert second_result["content"] == result["content"]


def test_web_content_extractor_rejects_dynamic_quote_page_shell():
    result = extract_article_content(
        """
        <html><head><title>沪深300ETF华泰柏瑞(510300)股票行情</title></head><body>
          <div>上海证券交易所服务热线：400-8888-400 APP下载 微博微信 English繁</div>
          <div>上交所APP“通办”栏目</div>
          <div>行情和统计 相关公告 基本信息 公告申购赎回清单</div>
          <div>上证：- - - - (涨:- 平:- 跌:-) 深证：- - - - (涨:- 平:- 跌:-)</div>
          <div>今开: - 最高: - 涨停: - 换手: - 成交量: - 振幅: - 外盘: -</div>
          <div>名称最新价涨跌幅 股票名称持仓占比涨跌幅</div>
          <div>数据加载中...</div>
          <div>关于我们 广告服务 联系我们 法律声明 隐私保护 友情链接</div>
        </body></html>
        """
    )

    assert result["content_status"] == "empty"
    assert result["content"] == ""


def test_sentiment_does_not_use_snippet_only_results_for_llm_signal():
    class ExplodingLLM:
        async def chat_json(self, *args, **kwargs):
            raise AssertionError("snippet-only evidence must not call the LLM")

    report = asyncio.run(
        analyze_sentiment(
            "512480",
            context=MarketContext(
                ticker="512480",
                asset_type=AssetType.ETF,
                web_results=[
                    {
                        "title": "半导体 ETF 上涨",
                        "snippet": "搜索摘要",
                        "link": "https://example.com/news",
                        "content_status": "snippet_only",
                    }
                ],
            ),
            llm=ExplodingLLM(),
        )
    )

    assert report.signal == Decision.HOLD
    assert report.confidence == 0.3
    assert report.key_data["evidence_level"] == "snippet_only"


def test_sentiment_prompt_contains_fetched_content(monkeypatch):
    captured = {}

    class FakeLLM:
        async def chat_json(self, prompt, *, system):
            captured["prompt"] = prompt
            return {
                "signal": "hold",
                "confidence": 0.7,
                "reasoning": "基于已抓取正文",
                "sentiment_score": 0.1,
                "key_themes": ["需求"]
            }

    report = asyncio.run(
        analyze_sentiment(
            "512480",
            context=MarketContext(
                ticker="512480",
                asset_type=AssetType.ETF,
                web_results=[
                    {
                        "title": "半导体行业报道",
                        "snippet": "摘要不应作为主要证据",
                        "content": "正文明确提到半导体设备订单出现改善，但行业仍面临需求波动风险。",
                        "content_status": "full_text",
                        "link": "https://example.com/news",
                    }
                ],
            ),
            llm=FakeLLM(),
        )
    )

    assert report.signal == Decision.HOLD
    assert "订单出现改善" in captured["prompt"]
    assert "证据等级：full_text" in captured["prompt"]
    assert "A-share stock" not in captured["prompt"]


def test_analysis_tool_passes_asset_type_to_workflow(monkeypatch):
    agent = StockAgent()
    captured = {}

    async def fake_analyze(request, *, config=None):
        captured["ticker"] = request.ticker
        captured["asset_type"] = request.asset_type
        return {}, {
            "final_decision": TradeDecision(
                ticker=request.ticker,
                asset_type=request.asset_type,
                decision=Decision.HOLD,
            )
        }

    async def no_artifacts(*args, **kwargs):
        return []

    monkeypatch.setattr(agent, "analyze", fake_analyze)
    monkeypatch.setattr(research_service, "create_artifacts", no_artifacts)
    result = asyncio.run(agent._analysis_tool().ainvoke({"ticker": "510300", "asset_type": "etf"}))

    assert json.loads(result)["asset_type"] == "etf"
    assert captured == {"ticker": "510300", "asset_type": AssetType.ETF}


def test_analysis_tool_returns_decision_when_report_generation_fails(monkeypatch):
    agent = StockAgent()

    async def fake_analyze(request, *, config=None):
        return {}, {
            "final_decision": TradeDecision(
                ticker=request.ticker,
                asset_type=request.asset_type,
                decision=Decision.HOLD,
                reasoning="结构化分析仍可用",
            )
        }

    async def failed_artifacts(*args, **kwargs):
        raise RuntimeError("sensitive_words_detected")

    monkeypatch.setattr(agent, "analyze", fake_analyze)
    monkeypatch.setattr(research_service, "create_artifacts", failed_artifacts)

    result = asyncio.run(agent._analysis_tool().ainvoke({"ticker": "510300", "asset_type": "etf"}))
    payload = json.loads(result)

    assert payload["asset_type"] == "etf"
    assert payload["reasoning"] == "结构化分析仍可用"
    assert payload["artifacts"] == []


def test_analysis_tool_has_dedicated_long_running_budget():
    assert tool_timeout_seconds("run_fund_or_stock_analysis") == LONG_RUNNING_TOOL_TIMEOUT_SECONDS
    assert tool_attempts("run_fund_or_stock_analysis") == 1
    assert tool_timeout_seconds("run_backtest") == LONG_RUNNING_TOOL_TIMEOUT_SECONDS
    assert tool_timeout_seconds("design_and_run_backtest") == LONG_RUNNING_TOOL_TIMEOUT_SECONDS
    assert tool_timeout_seconds("compare_strategy_backtests") == LONG_RUNNING_TOOL_TIMEOUT_SECONDS
    assert tool_attempts("run_backtest") == 1
    assert tool_timeout_seconds("get_latest_news") == TOOL_TIMEOUT_SECONDS
    assert tool_attempts("get_latest_news") == 2


def test_render_activity_exposes_error_reason():
    messages = render_activity("run_fund_or_stock_analysis", "failed", error="tool_timeout: 超过 300 秒")
    update = next(message["updateDataModel"] for message in messages if "updateDataModel" in message)

    assert update["value"] == {
        "name": "run_fund_or_stock_analysis",
        "status": "failed",
        "error": "tool_timeout: 超过 300 秒",
    }


def test_failed_analysis_payload_does_not_render_a_decision_card():
    messages = render_tool_result(
        "run_fund_or_stock_analysis",
        json.dumps({"error": "tool_timeout: 工具执行超过 300 秒"}, ensure_ascii=False),
    )

    assert messages is None


def test_chat_renders_analysis_result_as_inline_a2ui():
    messages = render_tool_result(
        "run_fund_or_stock_analysis",
        json.dumps(
            {
                "ticker": "510300",
                "asset_type": "etf",
                "decision": "buy",
                "confidence": 0.72,
                "plan": {
                    "entry_price": 3.8,
                    "stop_loss": 3.5,
                    "take_profit": 4.2,
                    "position_size": 0.2,
                },
                "reasoning": "趋势改善，但仍需控制仓位。",
                "dashboard": {
                    "core_conclusion": {
                        "signal": "buy",
                        "confidence": 0.72,
                        "one_line_summary": "趋势改善",
                        "position_advice": "分批建仓",
                    },
                    "battle_plan": {},
                },
            },
            ensure_ascii=False,
        ),
    )

    assert messages is not None
    components = [
        message["updateComponents"]["components"]
        for message in messages
        if "updateComponents" in message
    ]
    component_names = {component["component"] for group in components for component in group}
    assert "Badge" in component_names
    assert "Section" in component_names
    assert any(
        message.get("updateDataModel", {}).get("value", {}).get("decisionLabel") == "买入"
        for message in messages
    )


def test_chat_renders_backtest_result_with_curve_and_trades():
    messages = render_tool_result(
        "run_backtest",
        json.dumps(
            {
                "data_type": "backtest",
                "result": {
                    "ticker": "510300",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-03",
                    "initial_capital": 100000,
                    "final_value": 101500,
                    "total_return": 0.015,
                    "max_drawdown": 0.02,
                    "sharpe_ratio": 1.1,
                    "win_rate": 0.6,
                    "total_trades": 2,
                    "equity_curve": [
                        {"date": "2024-01-01", "value": 100000},
                        {"date": "2024-01-03", "value": 101500},
                    ],
                    "trades": [
                        {
                            "date": "2024-01-02",
                            "action": "buy",
                            "ticker": "510300",
                            "shares": 100,
                            "price": 3.8,
                            "amount": 380,
                        }
                    ],
                },
            }
        ),
    )

    assert messages is not None
    components = [
        component
        for message in messages
        if "updateComponents" in message
        for component in message["updateComponents"]["components"]
    ]
    assert {component["component"] for component in components} >= {"LineChart", "DataTable", "Collapsible"}
    model = next(message["updateDataModel"]["value"] for message in messages if "updateDataModel" in message)
    assert model["points"][-1] == {"label": "2024-01-03", "value": 101500.0}


def test_generated_html_source_is_compacted_when_a_file_artifact_exists():
    response = _compact_generated_report(
        "报告已生成。\n\n```html\n<!doctype html><html><body>完整报告</body></html>\n```",
        [{"name": "研究报告.html", "mime_type": "text/html", "metadata": {"description": "完整研究结论"}}],
    )

    assert "<!doctype html>" not in response
    assert "报告已生成" in response
    assert "HTML" in response


def test_long_generated_report_is_compacted_when_artifact_exists():
    response = _compact_generated_report(
        "标题\n" + ("很长的分析内容。\n" * 200),
        [{"name": "验证方案.md", "mime_type": "text/markdown", "metadata": {"description": "完整验证方案"}}],
    )

    assert response.startswith("标题")
    assert "Markdown" in response
    assert "验证方案.md" in response
    assert len(response) > 100


def test_generic_artifact_notice_is_replaced_with_description():
    response = _compact_generated_report(
        "完整 HTML 报告已生成文件产物，请点击下方卡片预览或下载。",
        [
            {
                "name": "A股宽基验证方案.md",
                "mime_type": "text/markdown",
                "metadata": {"description": "覆盖数据、策略、成本、指标和稳健性检验的预注册方案。"},
            }
        ],
    )

    assert "覆盖数据、策略、成本" in response
    assert "Markdown" in response
    assert "HTML 报告" not in response
