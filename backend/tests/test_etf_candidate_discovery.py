import json

import pytest

from data import exchange_fund_provider as fund_provider_module
from data.exchange_fund_provider import AkShareExchangeFundDataProvider
from harness.bootstrap import build_default_catalog, load_default_skills
from tools.exchange_fund import _theme_candidate_matches, discover_exchange_fund_candidates


def test_white_liquor_theme_expands_to_core_and_food_beverage_proxies_only():
    records = [
        {"ticker": "512690", "name": "酒ETF鹏华", "amount": 200_000_000},
        {"ticker": "515170", "name": "食品饮料ETF华夏", "amount": 40_000_000},
        {"ticker": "159736", "name": "食品饮料ETF天弘", "amount": 12_000_000},
        {"ticker": "159732", "name": "消费电子ETF华夏", "amount": 220_000_000},
        {"ticker": "513070", "name": "港股通消费ETF易方达", "amount": 60_000_000},
    ]
    matches = _theme_candidate_matches(records, "白酒")
    assert [item["ticker"] for item in matches] == ["512690", "515170", "159736"]
    assert matches[0]["theme_scope"] == "core"
    assert all(item["theme_scope"] == "related_proxy" for item in matches[1:])


def test_etf_screen_skill_exposes_provider_backed_candidate_discovery():
    registry = load_default_skills(catalog=build_default_catalog())
    skill = registry.get("exchange_fund.screen_compare")
    assert "discover_exchange_fund_candidates" in skill.tools
    assert "screen_assets" not in skill.tools


@pytest.mark.asyncio
async def test_fund_profile_provenance_is_a_flat_source_list(monkeypatch):
    async def fake_realtime(_ticker: str, *, asset_type: str):
        assert asset_type == "etf"
        return {"ticker": "512690", "name": "酒ETF鹏华", "price": 0.42}

    monkeypatch.setattr(fund_provider_module, "async_get_exchange_fund_quote", fake_realtime)
    result = await AkShareExchangeFundDataProvider().profile("512690")

    assert result.status == "available"
    assert result.sources[0]["source_id"] == "akshare"


@pytest.mark.asyncio
async def test_candidate_discovery_provenance_is_a_flat_source_list(monkeypatch):
    async def fake_spot(_asset_type: str, *, limit: int):
        assert limit == 5000
        return [
            {"ticker": "512690", "name": "酒ETF鹏华", "amount": 200_000_000},
            {"ticker": "515170", "name": "食品饮料ETF华夏", "amount": 40_000_000},
            {"ticker": "159736", "name": "食品饮料ETF天弘", "amount": 12_000_000},
            {"ticker": "515710", "name": "食品饮料ETF华宝", "amount": 10_000_000},
            {"ticker": "159843", "name": "食品饮料ETF招商", "amount": 3_000_000},
        ]

    monkeypatch.setattr("tools.exchange_fund.async_get_asset_spot", fake_spot)
    payload = json.loads(await discover_exchange_fund_candidates.ainvoke({"theme": "白酒"}))

    assert payload["sources"][0]["source_id"] == "akshare"
    assert payload["data"]["deep_analysis_shortlist"] == ["512690", "515170", "159736", "515710"]
