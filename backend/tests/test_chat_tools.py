import asyncio
import json

from agents import chat_tools
from engine.simulation_account import SimulationAccountService


def test_chat_tools_expose_paper_portfolio_and_orders(monkeypatch, tmp_path):
    accounts = SimulationAccountService(tmp_path / "simulation.db")
    monkeypatch.setattr(chat_tools, "simulation_accounts", accounts)

    portfolio = asyncio.run(chat_tools.get_simulation_portfolio.ainvoke({"account_id": "default"}))
    portfolio_payload = json.loads(portfolio)
    assert portfolio_payload["paper_trading"] is True
    assert portfolio_payload["portfolio"]["account_id"] == "default"

    order = asyncio.run(
        chat_tools.submit_simulation_order.ainvoke(
            {
                "ticker": "510300",
                "side": "buy",
                "shares": 100,
                "account_id": "default",
                "asset_type": "etf",
            }
        )
    )
    order_payload = json.loads(order)
    assert order_payload["paper_trading"] is True
    assert order_payload["order"]["status"] == "pending"

    orders = asyncio.run(chat_tools.get_simulation_orders.ainvoke({"account_id": "default"}))
    assert len(json.loads(orders)["orders"]) == 1
