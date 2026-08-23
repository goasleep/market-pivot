import json

from widgets.a2ui import CATALOG_ID, render_agent_pipeline, render_research_plan, render_stock_card, render_tool_result


def test_stock_card_uses_a2ui_surface_and_data_model():
    messages = render_stock_card(
        {"ticker": "510300", "name": "沪深300ETF", "price": 3.92, "pct_chg": 1.25}
    )

    assert messages[0]["version"] == "v0.9"
    assert messages[0]["createSurface"]["catalogId"] == CATALOG_ID
    assert messages[1]["updateComponents"]["components"][0]["id"] == "root"
    assert messages[2]["updateDataModel"]["value"]["priceLabel"] == "¥3.92"


def test_stock_card_skips_missing_price_instead_of_fabricating_zero():
    assert render_stock_card({"ticker": "600519", "quote": {}}) == []


def test_search_tool_result_exposes_clickable_links_and_visible_urls():
    messages = render_tool_result(
        "search_web",
        json.dumps(
            {
                "results": [
                    {
                        "title": "公告",
                        "link": "https://example.com/article",
                        "snippet": "摘要",
                        "source": "Example",
                    }
                ]
            }
        ),
    )

    components = messages[1]["updateComponents"]["components"]
    result_list = next(component for component in components if component["id"] == "list")
    root = next(component for component in components if component["id"] == "root")
    item = next(component for component in components if component["id"] == "searchItem")
    data = messages[2]["updateDataModel"]["value"]
    assert root["component"] == "Collapsible"
    assert root["title"] == "搜索结果（1 条）"
    assert root["defaultExpanded"] is True
    assert result_list["component"] == "List"
    assert item["component"] == "SearchResultItem"
    assert data["items"][0]["link"] == "https://example.com/article"


def test_search_tool_result_fully_collapses_large_result_sets():
    messages = render_tool_result(
        "search_web",
        json.dumps(
            {
                "results": [
                    {"title": f"结果 {index}", "link": f"https://example.com/{index}"}
                    for index in range(5)
                ]
            }
        ),
    )

    root = next(
        component
        for component in messages[1]["updateComponents"]["components"]
        if component["id"] == "root"
    )
    assert root["defaultExpanded"] is False


def test_progressive_pipeline_update_does_not_recreate_surface():
    messages = render_agent_pipeline(
        [{"name": "technical", "label": "Technical", "status": "done"}],
        surface_id="pipeline-1",
        include_create=False,
    )

    assert "createSurface" not in messages[0]
    assert messages[0]["updateComponents"]["surfaceId"] == "pipeline-1"


def test_research_plan_card_uses_stable_statuses_and_surface_updates():
    messages = render_research_plan(
        {
            "objective": "比较 510300 与 510500",
            "depth": "standard",
            "revision": 2,
            "status": "completed_with_gaps",
            "progress": 100,
            "steps": [
                {"title": "获取行情", "status": "completed"},
                {"title": "检索新闻", "status": "failed", "error": "外部数据源暂时不可用"},
                {"title": "形成结论", "status": "skipped"},
            ],
        },
        "research-plan-task-1",
        include_create=False,
    )

    assert "createSurface" not in messages[0]
    components = messages[0]["updateComponents"]["components"]
    statuses = [item["status"] for item in components if item.get("component") == "PipelineStep"]
    assert statuses == ["completed", "failed", "skipped"]
    failed_step = next(item for item in components if item.get("status") == "failed")
    assert failed_step["detail"] == "外部数据源暂时不可用"
    data = messages[1]["updateDataModel"]["value"]
    assert data["progress"] == 100
    assert "Revision 2" in data["meta"]


def test_research_plan_card_displays_failure_recovery_adjustment():
    messages = render_research_plan(
        {
            "objective": "回测 510300",
            "depth": "standard",
            "revision": 2,
            "status": "running",
            "progress": 50,
            "steps": [
                {
                    "title": "执行回测",
                    "status": "pending",
                    "recovery": {
                        "action": "adjust",
                        "summary": "修正不受支持的指标名称后重新调用。",
                    },
                }
            ],
        },
        "research-plan-recovery",
        include_create=False,
    )

    components = messages[0]["updateComponents"]["components"]
    step = next(item for item in components if item.get("component") == "PipelineStep")
    assert step["status"] == "pending"
    assert step["detail"] == "反思后调整：修正不受支持的指标名称后重新调用。"


def test_historical_prices_render_as_inline_trend_chart_and_table():
    messages = render_tool_result(
        "get_historical_prices",
        json.dumps(
            {
                "ticker": "510300",
                "asset_type": "etf",
                "history": [
                    {"date": "2026-08-12", "close": 4.0, "pct_chg": 0.5, "volume": 100},
                    {"date": "2026-08-13", "close": 4.2, "pct_chg": 5.0, "volume": 120},
                ],
            }
        ),
    )

    components = messages[1]["updateComponents"]["components"]
    chart = next(component for component in components if component["id"] == "chart")
    data = messages[2]["updateDataModel"]["value"]
    assert chart["component"] == "LineChart"
    assert chart["points"] == {"path": "/points"}
    assert data["prices"] == [4.0, 4.2]
    assert data["points"] == [
        {"label": "2026-08-12", "value": 4.0},
        {"label": "2026-08-13", "value": 4.2},
    ]


def test_price_history_collection_unwraps_single_ticker_for_rendering():
    messages = render_tool_result(
        "get_historical_prices",
        json.dumps(
            {
                "data_type": "price_history_collection",
                "items": [
                    {
                        "ticker": "510300",
                        "history": [{"date": "2026-08-21", "close": 4.68, "volume": 120}],
                    }
                ],
            }
        ),
    )

    data = messages[2]["updateDataModel"]["value"]
    assert data["prices"] == [4.68]
    assert data["rows"][0]["date"] == "2026-08-21"


def test_sandbox_candidate_renders_source_validation_and_backtest_in_one_surface():
    messages = render_tool_result(
        "design_and_run_sandbox_strategy",
        json.dumps(
            {
                "candidate_id": "candidate-demo",
                "status": "validated",
                "name": "ma_rsi_demo",
                "version": "1.0.0",
                "ticker": "510300",
                "asset_type": "etf",
                "source_code": "def generate_target_positions(frame):\n    return (frame['close'] > 0).astype(int)",
                "source_sha256": "a" * 64,
                "strategy_spec": {
                    "name": "ma_rsi_demo",
                    "description": "均线与 RSI 趋势信号",
                    "components": [
                        {
                            "id": "rsi_entry",
                            "type": "dsl",
                            "role": "signal",
                            "expression": {
                                "type": "compare",
                                "left": {"type": "indicator", "indicator": "rsi", "window": 14},
                                "operator": "lt",
                                "right": {"type": "constant", "value": 30},
                            },
                        }
                    ],
                    "position_policy": {"mode": "continuous", "max_exposure": 0.95},
                },
                "validation": {
                    "passed": True,
                    "static_checks": {"ast_parse": True, "allowed_imports": True},
                    "output_checks": {"binary_positions": True, "dsl_signal_equivalent": True},
                    "deterministic": True,
                    "causal": True,
                    "errors": [],
                },
                "result": {
                    "promotion_eligible": True,
                    "backtest": {
                        "final_value": 1_120_000,
                        "total_return": 0.12,
                        "buy_hold_return": 0.08,
                        "max_drawdown": 0.04,
                        "sharpe_ratio": 1.2,
                        "total_trades": 2,
                        "equity_curve": [
                            {"date": "2026-01-01", "value": 1_000_000},
                            {"date": "2026-08-21", "value": 1_120_000},
                        ],
                        "trades": [
                            {
                                "date": "2026-01-02",
                                "action": "buy",
                                "shares": 100,
                                "price": 4.0,
                                "amount": 400,
                            }
                        ],
                    },
                },
            }
        ),
    )

    components = messages[1]["updateComponents"]["components"]
    data = messages[2]["updateDataModel"]["value"]
    root = next(component for component in components if component["id"] == "root")
    code = next(component for component in components if component["id"] == "source-code")
    chart = next(component for component in components if component["id"] == "chart")

    assert {"performance", "validation", "code"}.issubset(root["children"])
    assert code["component"] == "CodeBlock"
    assert chart["points"] == {"path": "/points"}
    assert "generate_target_positions" in data["sourceCode"]
    assert data["validationRows"][0]["status"] == "通过"
    assert data["points"][-1]["value"] == 1_120_000


def test_strategy_comparison_renders_auditable_research_sections_and_safe_artifacts():
    comparison = {
        "strategy_name": "buy_hold",
        "display_name": "买入持有",
        "total_return": 0.1,
        "annualized_return": 0.05,
        "max_drawdown": 0.08,
        "sharpe_ratio": 0.6,
        "calmar_ratio": 0.62,
        "exposure": 0.95,
        "turnover": 0.1,
        "total_fees": 120,
        "final_value": 1_100_000,
        "equity_curve": [
            {"date": "2020-01-02", "value": 1_000_000},
            {"date": "2026-08-21", "value": 1_100_000},
        ],
        "drawdown_curve": [
            {"date": "2020-01-02", "value": 0},
            {"date": "2026-08-21", "value": -0.08},
        ],
        "diagnostics": {
            "out_of_sample": {"out_of_sample_return": 0.03},
            "rolling": [{"total_return": 0.1}, {"total_return": -0.1}],
        },
        "strategy_spec": {"position_policy": {"mode": "continuous", "max_exposure": 0.95}},
        "entry_rules": [],
        "exit_rules": [],
    }
    payload = {
        "ticker": "510300",
        "initial_capital": 1_000_000,
        "evaluation_start_date": "2020-01-02",
        "evaluation_end_date": "2026-08-21",
        "warmup_bars": 252,
        "benchmark": "buy_hold",
        "ranking": ["buy_hold"],
        "comparisons": [comparison],
        "acceptance": {"satisfied": True, "checks": {"fair_evaluation_period": True}},
        "conclusion": {
            "official": True,
            "absolute_return_winner": {
                "strategy_name": "buy_hold",
                "display_name": "买入持有",
                "metric": "total_return",
                "value": 0.1,
            },
            "tradeoffs": ["不存在唯一最好策略。"],
            "limitations": ["历史表现不代表未来。"],
        },
        "data_validation": {
            "status": "verified",
            "selected_source": "eastmoney",
            "selection_reason": "质量分最高",
            "candidates": [],
        },
        "artifacts": [
            {
                "name": "报告.html",
                "mime_type": "text/html",
                "size_bytes": 1024,
                "preview_url": "/api/artifacts/artifact-demo/preview",
                "download_url": "/api/artifacts/artifact-demo/download",
            }
        ],
    }

    messages = render_tool_result("compare_strategy_backtests", json.dumps(payload))
    components = messages[1]["updateComponents"]["components"]
    model = messages[2]["updateDataModel"]["value"]

    assert sum(item.get("component") == "MultiLineChart" for item in components) == 3
    assert any(item.get("component") == "ArtifactLink" for item in components)
    assert model["winners"][0]["strategy"] == "买入持有"
    assert model["stabilityRows"][0]["rollingPositive"] == "+50.00%"
    assert model["artifacts"][0]["downloadUrl"].startswith("/api/artifacts/")
