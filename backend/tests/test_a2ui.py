from widgets.a2ui import CATALOG_ID, render_agent_pipeline, render_stock_card


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
    item = next(component for component in components if component["id"] == "searchItem")
    data = messages[2]["updateDataModel"]["value"]
    assert item["component"] == "SearchResultItem"
    assert data["items"][0]["link"] == "https://example.com/article"


def test_progressive_pipeline_update_does_not_recreate_surface():
    messages = render_agent_pipeline(
        [{"name": "technical", "label": "Technical", "status": "done"}],
        surface_id="pipeline-1",
        include_create=False,
    )

    assert "createSurface" not in messages[0]
    assert messages[0]["updateComponents"]["surfaceId"] == "pipeline-1"
