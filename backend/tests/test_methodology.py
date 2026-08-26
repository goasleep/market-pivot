import json

import pytest

from methodology.library import MethodologyLibrary
from tools import assets, methodology
from tools.registry import build_chat_tools


def test_methodology_library_loads_seed_documents():
    library = MethodologyLibrary()

    documents = library.load_documents()

    assert len(documents) >= 5
    assert {document.methodology_type for document in documents} >= {
        "philosophy",
        "experience",
        "paper",
    }
    assert all(document.document_id for document in documents)
    assert all(document.content for document in documents)


def test_methodology_search_ranks_title_and_filters_asset_type():
    library = MethodologyLibrary()

    results = library.search("ETF 跟踪误差", asset_type="ETF", limit=3)

    assert results
    assert results[0]["id"] == "philosophy-etf-tracking-liquidity-001"
    assert "跟踪误差" in results[0]["excerpt"]
    assert all("ETF" in result["asset_types"] for result in results)


def test_methodology_search_rejects_empty_query():
    with pytest.raises(ValueError, match="query 不能为空"):
        MethodologyLibrary().search("  ")


def test_methodology_tool_returns_citations_and_usage_note():
    payload = json.loads(
        __import__("asyncio").run(
            methodology.search_methodology.ainvoke(
                {"query": "仓位 回撤", "asset_type": "ETF", "limit": 2}
            )
        )
    )

    assert payload["data_type"] == "methodology_search"
    assert payload["results"]
    assert payload["provenance"][0]["source_id"] == "methodology_library"
    assert "回测" in payload["usage_note"]


def test_chat_tools_register_methodology_search():
    names = {tool.name for tool in build_chat_tools(assets.get_realtime_quote)}

    assert "search_methodology" in names
