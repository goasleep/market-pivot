"""Broker adapter boundary for internal and external paper trading.

Only the internal SQLite-backed simulator is active today. External providers
are represented by configuration and this narrow interface so an adapter can
be added later without changing Agent outputs or account persistence.
"""

from typing import Protocol

from models.schemas import ExternalSimulationConfig, SimulationOrder


class SimulationBroker(Protocol):
    """Minimal contract required by a paper-trading provider adapter."""

    def submit_order(self, order: SimulationOrder) -> SimulationOrder:
        ...

    def cancel_order(self, order_id: str) -> SimulationOrder:
        ...

    def sync(self) -> dict:
        ...


class ExternalSimulationBroker:
    """Placeholder that fails closed until a provider adapter is implemented."""

    def __init__(self, config: ExternalSimulationConfig):
        self.config = config

    def _not_implemented(self) -> None:
        raise NotImplementedError(
            f"外部模拟平台 {self.config.provider} 尚未接入；当前请使用 internal 模拟账户"
        )

    def submit_order(self, order: SimulationOrder) -> SimulationOrder:
        self._not_implemented()

    def cancel_order(self, order_id: str) -> SimulationOrder:
        self._not_implemented()

    def sync(self) -> dict:
        self._not_implemented()
