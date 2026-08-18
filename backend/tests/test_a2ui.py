import json

from widgets.a2ui import CATALOG_ID, render_agent_pipeline, render_stock_card, render_tool_result


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
