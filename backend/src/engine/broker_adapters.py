"""Broker adapter boundary for Web-native and external paper trading."""

import importlib.util
import platform
import sys
from typing import Any, Protocol

import httpx

from config import settings
from models.schemas import (
    ExternalSimulationConfig,
    LiveOrderIntent,
    LiveOrderResult,
    LiveTradingConfig,
    SimulationOrder,
)


class SimulationBroker(Protocol):
    """Minimal contract required by a paper-trading provider adapter."""

    def submit_order(self, order: SimulationOrder) -> SimulationOrder:
        ...

    def cancel_order(self, order_id: str) -> SimulationOrder:
        ...

    def sync(self) -> dict:
        ...


class LiveBroker(Protocol):
    """Provider-neutral contract for a reviewed live trading gateway."""

    def submit_order(self, intent: LiveOrderIntent) -> LiveOrderResult:
        ...

    def cancel_order(self, broker_order_id: str) -> LiveOrderResult:
        ...

    def sync(self) -> dict[str, Any]:
        ...


class LiveBrokerUnavailableError(RuntimeError):
    """Raised when live execution is not configured or implemented."""


class CustomHttpLiveBroker:
    """Adapter for a user-managed broker sidecar.

    The sidecar contract is intentionally small and provider-neutral:

    - ``POST /orders`` receives a serialized :class:`LiveOrderIntent` and
      returns ``{broker_order_id, status, message, filled_shares, fill_price}``.
    - ``DELETE /orders/{broker_order_id}`` cancels an order.
    - ``GET /sync?account_id=...`` returns provider reconciliation data.

    This adapter never becomes active unless the service-level safety gate,
    account configuration, and automation-level live arming are all enabled.
    """

    provider = "custom_http"

    def __init__(self, config: LiveTradingConfig):
        if not config.endpoint.strip():
            raise LiveBrokerUnavailableError("custom_http 实盘 Adapter 缺少 endpoint")
        if not config.account_id.strip():
            raise LiveBrokerUnavailableError("custom_http 实盘 Adapter 缺少 account_id")
        self.config = config
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if config.token:
            headers["Authorization"] = f"Bearer {config.token}"
        self.client = httpx.Client(
            base_url=config.endpoint.rstrip("/"),
            headers=headers,
            timeout=10.0,
        )

    def submit_order(self, intent: LiveOrderIntent) -> LiveOrderResult:
        response = self.client.post("/orders", json=intent.model_dump(mode="json"))
        response.raise_for_status()
        payload = response.json()
        return LiveOrderResult(
            client_order_id=intent.client_order_id,
            broker_order_id=payload.get("broker_order_id") or payload.get("order_id"),
            status=payload.get("status", "unknown"),
            message=payload.get("message", ""),
            filled_shares=int(payload.get("filled_shares", 0) or 0),
            fill_price=payload.get("fill_price"),
        )

    def cancel_order(self, broker_order_id: str) -> LiveOrderResult:
        response = self.client.delete(f"/orders/{broker_order_id}")
        response.raise_for_status()
        payload = response.json()
        return LiveOrderResult(
            client_order_id=payload.get("client_order_id", ""),
            broker_order_id=broker_order_id,
            status=payload.get("status", "cancelled"),
            message=payload.get("message", ""),
        )

    def sync(self) -> dict[str, Any]:
        response = self.client.get("/sync", params={"account_id": self.config.account_id})
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"data": payload}

    def close(self) -> None:
        self.client.close()


class FailClosedLiveBroker:
    """Default live broker that refuses every order."""

    def __init__(self, reason: str):
        self.reason = reason

    def _raise(self) -> None:
        raise LiveBrokerUnavailableError(self.reason)

    def submit_order(self, intent: LiveOrderIntent) -> LiveOrderResult:
        self._raise()

    def cancel_order(self, broker_order_id: str) -> LiveOrderResult:
        self._raise()

    def sync(self) -> dict[str, Any]:
        self._raise()


def get_live_broker(config: LiveTradingConfig) -> LiveBroker:
    """Build the configured live broker, failing closed for unknown providers."""
    if not config.enabled:
        return FailClosedLiveBroker("实盘账户未启用")
    if config.provider == "custom_http":
        return CustomHttpLiveBroker(config)
    return FailClosedLiveBroker(f"实盘 Adapter 尚未实现: {config.provider}")


def live_broker_status(config: LiveTradingConfig) -> dict[str, Any]:
    """Return safe, non-secret readiness information for the live adapter."""
    if not config.enabled:
        return {
            "provider": config.provider,
            "enabled": False,
            "configured": False,
            "service_gate": settings.live_trading_enabled,
            "can_submit_orders": False,
            "state": "disabled",
            "message": "实盘账户未启用",
        }
    if config.provider != "custom_http":
        return {
            "provider": config.provider,
            "enabled": True,
            "configured": False,
            "service_gate": settings.live_trading_enabled,
            "can_submit_orders": False,
            "state": "not_implemented",
            "message": f"实盘 Adapter 尚未实现: {config.provider}",
        }
    configured = bool(config.endpoint.strip() and config.account_id.strip())
    return {
        "provider": config.provider,
        "enabled": True,
        "configured": configured,
        "service_gate": settings.live_trading_enabled,
        "can_submit_orders": configured and settings.live_trading_enabled,
        "state": "ready" if configured else "not_configured",
        "message": (
            "custom_http 实盘网关配置就绪"
            if configured and settings.live_trading_enabled
            else "请配置 endpoint 和 account_id"
            if not configured
            else "Adapter 已配置，但服务端 LIVE_TRADING_ENABLED 未开启"
        ),
    }


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


class EastmoneyEmtBroker:
    """Safe readiness adapter for an Eastmoney EMT simulation account.

    The actual EMT gateway is a native client that normally runs in a Windows
    sidecar. Keeping the readiness check here lets the API and frontend expose
    the account state now while refusing to send orders until a supported EMT
    runtime and explicit paper-only configuration are present.
    """

    provider = "eastmoney_emt"

    def __init__(self, config: ExternalSimulationConfig):
        self.config = config

    @staticmethod
    def _has_module(module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None

    def status(self) -> dict[str, Any]:
        options = self.config.options
        account_id = self.config.account_id.strip()
        endpoint = self.config.endpoint.strip()
        has_vnpy = self._has_module("vnpy")
        has_emt = self._has_module("vnpy_emt")
        supported_platform = sys.platform == "win32" and (3, 7) <= sys.version_info[:2] <= (3, 10)

        if not self.config.enabled:
            state = "disabled"
            message = "东方财富 EMT 模拟账户未启用"
        elif not self.config.simulation_only:
            state = "blocked_live_mode"
            message = "已阻止：当前系统只允许 simulation_only=true，不能连接实盘账户"
        elif not account_id:
            state = "not_configured"
            message = "请填写东方财富 EMT 模拟账户 ID"
        elif not endpoint and not options.get("trade_host"):
            state = "not_configured"
            message = "请填写 EMT sidecar 地址或交易柜台地址"
        elif not has_vnpy or not has_emt:
            state = "not_installed"
            message = "当前运行环境未安装 vn.py/vnpy_emt；建议在 Windows EMT sidecar 中运行"
        elif not supported_platform:
            state = "unsupported_platform"
            message = "当前环境不是 vnpy_emt 官方适配的 Windows/Python 3.7-3.10 运行环境"
        else:
            state = "ready"
            message = "配置已就绪；点击连接测试后才会建立 EMT 连接"

        return {
            "provider": self.provider,
            "label": "东方财富 EMT",
            "enabled": self.config.enabled,
            "simulation_only": self.config.simulation_only,
            "account_id": account_id,
            "state": state,
            "connected": False,
            "can_submit_orders": False,
            "installed": has_vnpy and has_emt,
            "supported_platform": supported_platform,
            "runtime": f"Python {platform.python_version()} / {sys.platform}",
            "endpoint_configured": bool(endpoint or options.get("trade_host")),
            "message": message,
            "requirements": [
                "东方财富 EMT 模拟账户权限",
                "Windows + Python 3.7-3.10 的 EMT sidecar",
                "vnpy、vnpy_emt 和 EMT 行情/交易地址",
            ],
        }

    def validate(self) -> dict[str, Any]:
        """Validate configuration without opening a broker connection."""
        return self.status()


def broker_status(config: ExternalSimulationConfig) -> dict[str, Any]:
    """Return a normalized status for the selected external simulation broker."""
    if config.provider == "eastmoney_emt":
        return EastmoneyEmtBroker(config).status()
    if config.provider == "internal":
        return {
            "provider": "internal",
            "label": "FastAPI Web 日级模拟",
            "enabled": True,
            "simulation_only": True,
            "state": "connected",
            "connected": True,
            "can_submit_orders": True,
            "installed": True,
            "supported_platform": True,
            "runtime": "FastAPI REST + WebSocket / SQLite 模拟撮合",
            "endpoint_configured": True,
            "message": "当前订单由 Web-native 日级模拟引擎执行；前端可通过 REST 下单并通过 WebSocket 接收状态变化",
            "requirements": ["FastAPI 服务", "SQLite 持久化", "日线或手动价格输入"],
        }
    return {
        "provider": config.provider,
        "label": config.provider,
        "enabled": config.enabled,
        "simulation_only": config.simulation_only,
        "state": "not_implemented",
        "connected": False,
        "can_submit_orders": False,
        "installed": False,
        "supported_platform": False,
        "runtime": "",
        "endpoint_configured": bool(config.endpoint),
        "message": f"外部模拟平台 {config.provider} 尚未实现",
        "requirements": [],
    }
