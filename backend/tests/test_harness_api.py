import pytest

from api.main import system_status
from api.routers.agent import list_agent_capabilities


@pytest.mark.asyncio
async def test_public_capability_catalog_contains_etf_pack_without_instructions():
    payload = await list_agent_capabilities()
    capabilities = payload["capabilities"]
    ids = {item["skill_id"] for item in capabilities}
    assert {
        "exchange_fund.profile",
        "exchange_fund.exposure",
        "exchange_fund.tracking_quality",
        "exchange_fund.liquidity_cost",
        "exchange_fund.premium_discount",
        "exchange_fund.relative_strength",
        "exchange_fund.screen_compare",
        "exchange_fund.portfolio_fit",
        "exchange_fund.event_risk",
    } <= ids
    assert all("instructions" not in item for item in capabilities)
    assert all(item["domain"] in {"stock", "exchange_fund", "open_fund", "shared"} for item in capabilities)
    assert not any(item["skill_id"].startswith(("fund.", "etf.")) for item in capabilities)
    assert {
        "open_fund.profile",
        "open_fund.nav",
        "open_fund.fees",
        "open_fund.exposure",
        "open_fund.money_yield",
        "open_fund.relative_strength",
        "open_fund.screen_compare",
        "open_fund.comprehensive_analysis",
        "open_fund.event_risk",
        "open_fund.nav_backtest",
    } <= ids


@pytest.mark.asyncio
async def test_system_status_exposes_harness_health(monkeypatch):
    monkeypatch.setattr("api.main.get_breaker_status", lambda: {})
    monkeypatch.setattr("api.main.get_history_cache_status", lambda: {})
    payload = await system_status()
    assert payload["harness"]["version"] == "2.0.0"
    assert payload["harness"]["registry"]["healthy"] is True
