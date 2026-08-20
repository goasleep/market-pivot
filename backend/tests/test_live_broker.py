import pytest

from engine import broker_adapters
from models.schemas import Decision, LiveOrderIntent, LiveTradingConfig


@pytest.mark.asyncio
async def test_custom_http_live_broker_uses_async_http_client(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        async def post(self, path, *, json):
            calls.append(("post", path, json))
            return FakeResponse(
                {
                    "broker_order_id": "broker-1",
                    "status": "submitted",
                    "message": "accepted",
                    "filled_shares": 0,
                    "fill_price": None,
                }
            )

        async def aclose(self):
            calls.append(("close",))

    monkeypatch.setattr(broker_adapters.httpx, "AsyncClient", FakeAsyncClient)
    config = LiveTradingConfig(
        enabled=True,
        provider="custom_http",
        endpoint="https://broker.example/api",
        account_id="account-1",
        token="secret",
    )
    broker = broker_adapters.CustomHttpLiveBroker(config)

    result = await broker.submit_order(
        LiveOrderIntent(
            client_order_id="client-1",
            account_id="account-1",
            ticker="510300",
            side=Decision.BUY,
            shares=100,
            submitted_date="2026-08-21",
        )
    )
    await broker.close()

    assert result.broker_order_id == "broker-1"
    assert calls[0][0] == "init"
    assert calls[1][0:2] == ("post", "/orders")
    assert calls[-1] == ("close",)
