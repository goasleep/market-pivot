"""In-process event hub for the Web-native simulation account API."""

import asyncio
from typing import Any


class SimulationEventHub:
    """Fan out account events to WebSocket subscribers.

    The account state remains persisted by the simulation repository. This
    hub only carries live in-process notifications.
    """

    def __init__(self, queue_size: int = 100):
        self.queue_size = queue_size
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, account_id: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.queue_size)
        async with self._lock:
            self._subscribers.setdefault(account_id, set()).add(queue)
        return queue

    async def unsubscribe(self, account_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            subscribers = self._subscribers.get(account_id)
            if not subscribers:
                return
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(account_id, None)

    async def publish(self, account_id: str, event_type: str, data: dict[str, Any]) -> None:
        events = self._events.setdefault(account_id, [])
        event = {"event_id": len(events) + 1, "event": event_type, "type": event_type, "data": data}
        event["account_id"] = account_id
        events.append(event)
        async with self._lock:
            subscribers = tuple(self._subscribers.get(account_id, ()))

        for queue in subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)

    async def list_events(self, account_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        """Read events produced by this process after the supplied cursor."""
        return [event for event in self._events.get(account_id, []) if event["event_id"] > after_id]


simulation_events = SimulationEventHub()
