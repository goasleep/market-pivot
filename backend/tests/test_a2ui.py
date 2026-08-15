from widgets.a2ui import CATALOG_ID, render_agent_pipeline, render_stock_card


def test_stock_card_uses_a2ui_surface_and_data_model():
    messages = render_stock_card(
        {"ticker": "510300", "name": "沪深300ETF", "price": 3.92, "pct_chg": 1.25}
    )

    assert messages[0]["version"] == "v0.9"
    assert messages[0]["createSurface"]["catalogId"] == CATALOG_ID
    assert messages[1]["updateComponents"]["components"][0]["id"] == "root"
    assert messages[2]["updateDataModel"]["value"]["priceLabel"] == "¥3.92"


def test_progressive_pipeline_update_does_not_recreate_surface():
    messages = render_agent_pipeline(
        [{"name": "technical", "label": "Technical", "status": "done"}],
        surface_id="pipeline-1",
        include_create=False,
    )

    assert "createSurface" not in messages[0]
    assert messages[0]["updateComponents"]["surfaceId"] == "pipeline-1"
