"""Canonical Financial Harness entry point."""

from agents.asset_requests import AssetAgentRequest, AssetIntent
from agents.financial_harness_agent import FinancialHarnessAgent


class AssetAgent(FinancialHarnessAgent):
    """Single public agent entry point for all supported financial products."""


asset_agent = AssetAgent()


def capabilities_text() -> str:
    """Return the public product and capability boundary without legacy aliases."""
    return (
        "我是 Financial Harness，支持股票、场内 ETF、场内 LOF 和场外开放式公募基金研究。\n\n"
        "- 股票：行情、历史、技术与受控综合研究；结论只代表股票。\n"
        "- ETF/LOF：市场价格、流动性、折溢价、跟踪和场内筛选。\n"
        "- 场外基金：产品核验、净值、费率、持仓、同类筛选和 NAV 回测。\n"
        "- QDII/FOF：首期可识别并披露数据或能力缺口。\n\n"
        "场外基金没有盘口、买卖价差、换手率、IOPV 或实时折溢价；不同产品领域的结论不会互相替代。"
    )

__all__ = ["AssetAgent", "AssetAgentRequest", "AssetIntent", "asset_agent", "capabilities_text"]
