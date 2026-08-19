"""In-process event hub for the Web-native simulation account API."""

import asyncio
from datetime import datetime, timezone
from typing import Any

from data.runtime_store import runtime_store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SimulationEventHub:
    """Fan out account events to WebSocket subscribers.

    The account state remains persisted in SQLite. This hub only carries live
    notifications, so reconnecting clients can always recover by fetching the
    REST account endpoint.
    """

    def __init__(self, queue_size: int = 100):
        self.queue_size = queue_size
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
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
        event = await runtime_store.append_event(
            "simulation-account",
            account_id,
            event_type,
            data,
        )
        event["account_id"] = account_id
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
        """Read events produced by any node after the supplied cursor."""
        events = await runtime_store.list_events(
            "simulation-account",
            account_id,
            after_id,
        )
        return [{**event, "account_id": account_id} for event in events]


simulation_events = SimulationEventHub()
