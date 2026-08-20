import pytest

from engine.simulation_events import SimulationEventHub


@pytest.mark.asyncio
async def test_simulation_event_hub_fans_out_and_unsubscribes():
    hub = SimulationEventHub(queue_size=2)
    first = await hub.subscribe("account-a")
    second = await hub.subscribe("account-a")

    await hub.publish("account-a", "order.updated", {"order_id": "sim-1"})

    assert (await first.get())["data"] == {"order_id": "sim-1"}
    assert (await second.get())["event"] == "order.updated"

    await hub.unsubscribe("account-a", first)
    await hub.publish("account-a", "account.updated", {"total_value": 1_000_000})

    assert first.empty()
    assert (await second.get())["type"] == "account.updated"
