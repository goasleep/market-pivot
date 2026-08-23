"""Canonical fund-focused conversational entry point."""

from agents.stock_agent import AssetAgent, AssetAgentRequest, AssetIntent, capabilities_text

FundAgent = AssetAgent
fund_agent = FundAgent()

__all__ = ["FundAgent", "AssetAgentRequest", "AssetIntent", "fund_agent", "capabilities_text"]
