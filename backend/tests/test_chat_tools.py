import asyncio
import json

import pytest

from agents import chat_tools
from agents.sentiment_analyst import analyze as analyze_sentiment
from agents.stock_agent import StockAgent, _compact_generated_report
from application.research import research_service
from data import serper_provider
from data.web_content import extract_article_content
from engine.simulation_account import SimulationAccountService
from graph.agent_loop import (
    LONG_RUNNING_TOOL_TIMEOUT_SECONDS,
    TOOL_TIMEOUT_SECONDS,
    tool_attempts,
    tool_timeout_seconds,
)
from models.schemas import AssetType, Decision, MarketContext, TradeDecision
from widgets.a2ui import render_activity


def test_chat_tools_expose_paper_portfolio_and_orders(monkeypatch, tmp_path):
    accounts = SimulationAccountService(tmp_path / "simulation.db")
    monkeypatch.setattr(chat_tools, "simulation_accounts", accounts)

    portfolio = asyncio.run(chat_tools.get_simulation_portfolio.ainvoke({"account_id": "default"}))
    portfolio_payload = json.loads(portfolio)
    assert portfolio_payload["paper_trading"] is True
    assert portfolio_payload["portfolio"]["account_id"] == "default"

    order = asyncio.run(
        chat_tools.submit_simulation_order.ainvoke(
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

    orders = asyncio.run(chat_tools.get_simulation_orders.ainvoke({"account_id": "default"}))
    assert len(json.loads(orders)["orders"]) == 1


def test_analysis_tool_requires_and_validates_asset_type():
    tool = StockAgent()._analysis_tool()
    schema = tool.args_schema.model_json_schema()

    assert "asset_type" in schema["required"]
    assert schema["properties"]["asset_type"]["enum"] == ["stock", "etf", "lof"]

    with pytest.raises(Exception, match="asset_type"):
        asyncio.run(tool.ainvoke({"ticker": "510300", "asset_type": "invalid"}))


def test_search_web_tool_returns_source_aware_results(monkeypatch):
    captured = {}

    async def fake_search(query, *, num_results=8, tbs=None):
        captured.update({"query": query, "num_results": num_results, "tbs": tbs})
        return {
            "available": True,
            "query": query,
            "results": [{"title": "公告", "link": "https://example.com", "snippet": "摘要"}],
        }

    monkeypatch.setattr(chat_tools, "async_search_web_parallel", fake_search)
    result = asyncio.run(
        chat_tools.search_web.ainvoke(
            {"query": "510300 最新公告", "num_results": 5, "freshness": "qdr:w"}
        )
    )

    assert json.loads(result)["results"][0]["link"] == "https://example.com"
    assert captured == {"query": "510300 最新公告", "num_results": 5, "tbs": "qdr:w"}


def test_ddgs_search_tool_maps_freshness(monkeypatch):
    captured = {}

    async def fake_search(query, *, num_results=8, timelimit=None):
        captured.update({"query": query, "num_results": num_results, "timelimit": timelimit})
        return {"available": True, "source": "DDGS metasearch", "results": []}

    monkeypatch.setattr(chat_tools, "async_search_web_ddgs", fake_search)
    result = asyncio.run(
        chat_tools.search_web_ddgs.ainvoke(
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

    monkeypatch.setattr(chat_tools, "async_get_asset_spot", fake_snapshot)
    result = asyncio.run(
        chat_tools.compare_quotes.ainvoke(
            {"tickers": ["sh600519", "000001"], "asset_type": "stock"}
        )
    )

    payload = json.loads(result)
    assert calls == [{"asset_type": "stock", "limit": 5000}]
    assert [item["quote"]["name"] for item in payload["quotes"]] == ["贵州茅台", "平安银行"]


def test_parallel_search_merges_serper_and_ddgs_results(monkeypatch):
    monkeypatch.setattr(serper_provider.settings, "serper_api_key", "configured")

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

    monkeypatch.setattr(serper_provider, "async_search_web", fake_serper)
    monkeypatch.setattr(serper_provider, "async_search_web_ddgs", fake_ddgs)
    result = asyncio.run(serper_provider.async_search_web_parallel("510300 公告", num_results=5))

    assert result["providers"] == ["Serper / Google Search", "DDGS metasearch"]
    assert [item["link"] for item in result["results"]] == ["https://same.example", "https://ddgs.example"]


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


def test_analysis_tool_has_dedicated_long_running_budget():
    assert tool_timeout_seconds("run_fund_or_stock_analysis") == LONG_RUNNING_TOOL_TIMEOUT_SECONDS
    assert tool_attempts("run_fund_or_stock_analysis") == 1
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


def test_generated_html_source_is_compacted_when_a_file_artifact_exists():
    response = _compact_generated_report(
        "报告已生成。\n\n```html\n<!doctype html><html><body>完整报告</body></html>\n```"
    )

    assert "<!doctype html>" not in response
    assert "文件产物" in response


def test_long_generated_report_is_compacted_when_artifact_exists():
    response = _compact_generated_report("标题\n" + ("很长的分析内容。\n" * 20))

    assert response == "完整 HTML 报告已生成文件产物，请点击下方卡片预览或下载。"
