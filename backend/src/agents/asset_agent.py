"""Canonical asset research chat entry point."""

from agents.fund_agent import (
    AssetAgentRequest,
    AssetIntent,
    FundAgent,
    capabilities_text,
    fund_agent,
)

AssetAgent = FundAgent
asset_agent = fund_agent

__all__ = ["AssetAgent", "AssetAgentRequest", "AssetIntent", "asset_agent", "capabilities_text"]
