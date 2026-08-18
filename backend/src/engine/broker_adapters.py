"""Broker adapter boundary for Web-native and external paper trading."""

import csv
import importlib.util
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

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

    def cancel_order(self, order_id: str) -> SimulationOrder | None:
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

    def cancel_order(self, order_id: str) -> SimulationOrder | None:
        self._not_implemented()

    def sync(self) -> dict:
        self._not_implemented()


class SimulationBrokerUnavailableError(RuntimeError):
    """Raised when an external simulation provider is not ready."""


def _row_value(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    """Read a CSV field while tolerating BOMs and whitespace in headers."""
    normalised = {
        str(key).strip().lstrip("\ufeff"): value
        for key, value in row.items()
        if key is not None
    }
    for name in names:
        if name in normalised:
            return normalised[name]
    return default


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return default


def _date_part(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    return text.replace("/", "-").replace("T", " ")[:10]


def _normalise_symbol(symbol: str) -> str:
    """Convert SHSE.600000/SZSE.000001 to the application's six-digit code."""
    value = symbol.strip().upper()
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    return value.removeprefix("SH").removeprefix("SZ").removeprefix("BJ").zfill(6)


def _eastmoney_symbol(ticker: str, options: dict[str, Any]) -> str:
    """Map the application's ticker to the Eastmoney market.code format."""
    code = _normalise_symbol(ticker)
    market_overrides = options.get("market_overrides", {})
    market = str(market_overrides.get(code, "")).strip().upper()
    if market in {"SH", "SHSE", "SSE"}:
        return f"SHSE.{code}"
    if market in {"SZ", "SZSE", "SZSE"}:
        return f"SZSE.{code}"
    # Shanghai stocks and the 5xxxx Shanghai ETF family use SHSE. The
    # remaining six-digit A-share/ETF codes are Shenzhen by default.
    return f"SHSE.{code}" if code.startswith(("5", "6", "68", "9")) else f"SZSE.{code}"


class EastmoneyFileBroker:
    """Eastmoney Quant file-order adapter for simulation accounts.

    The Eastmoney terminal watches an input directory for ``*.order.csv`` and
    ``*.cancel_order.csv`` files. It writes account snapshots and execution
    reports under ``output_dir/<account_id>/``. This adapter deliberately does
    not implement matching locally; Eastmoney remains the source of truth.
    """

    provider = "eastmoney_file"
    _ORDER_HEADERS = [
        "sid",
        "account_id",
        "symbol",
        "volume",
        "order_type",
        "order_business(order_biz)",
        "price",
        "comment",
    ]
    _CANCEL_HEADERS = ["sid", "comment"]

    def __init__(self, config: ExternalSimulationConfig):
        self.config = config

    @property
    def input_dir(self) -> Path:
        configured = self.config.input_dir or self.config.options.get("input_dir") or self.config.endpoint
        return Path(str(configured).strip()).expanduser()

    @property
    def output_dir(self) -> Path:
        configured = self.config.output_dir or self.config.options.get("output_dir")
        return Path(str(configured).strip()).expanduser()

    @property
    def encoding(self) -> str:
        return str(self.config.options.get("encoding") or "utf-8")

    def _output_account_dir(self) -> Path:
        root = self.output_dir
        account_dir = root / self.config.account_id.strip()
        return account_dir if account_dir.is_dir() else root

    def _validate_submission(self) -> None:
        if not self.config.enabled:
            raise SimulationBrokerUnavailableError("东方财富文件单未启用")
        if not self.config.simulation_only:
            raise SimulationBrokerUnavailableError("东方财富文件单只允许连接模拟账户")
        if not self.config.account_id.strip():
            raise SimulationBrokerUnavailableError("请填写东方财富量化模拟账户 ID")
        if not str(self.config.input_dir or self.config.options.get("input_dir") or self.config.endpoint).strip():
            raise SimulationBrokerUnavailableError("请填写东方财富文件单输入目录")
        if not self.input_dir.is_dir():
            raise SimulationBrokerUnavailableError(f"文件单输入目录不存在: {self.input_dir}")
        if not os.access(self.input_dir, os.W_OK):
            raise SimulationBrokerUnavailableError(f"文件单输入目录不可写: {self.input_dir}")

    def status(self) -> dict[str, Any]:
        input_dir = self.input_dir
        output_configured = bool(str(self.config.output_dir or self.config.options.get("output_dir") or "").strip())
        output_dir = self._output_account_dir() if output_configured else None
        input_ready = input_dir.is_dir() and os.access(input_dir, os.W_OK)
        output_ready = output_dir is not None and output_dir.is_dir() and os.access(output_dir, os.R_OK)

        if not self.config.enabled:
            state, message = "disabled", "东方财富文件单未启用"
        elif not self.config.simulation_only:
            state, message = "blocked_live_mode", "已阻止：文件单适配器只允许 simulation_only=true"
        elif not self.config.account_id.strip():
            state, message = "not_configured", "请填写东方财富量化模拟账户 ID"
        elif not str(self.config.input_dir or self.config.options.get("input_dir") or self.config.endpoint).strip():
            state, message = "not_configured", "请填写文件单输入目录"
        elif not input_ready:
            state, message = "not_ready", f"文件单输入目录不存在或不可写: {input_dir}"
        elif output_configured and not output_ready:
            state, message = "not_ready", f"文件单输出目录不存在或不可读: {self.output_dir}"
        else:
            state, message = "ready", "文件单目录已就绪；请确认东方财富量化终端已启动文件单输入/输出并连接仿真账户"

        return {
            "provider": self.provider,
            "label": "东方财富文件单仿真",
            "enabled": self.config.enabled,
            "simulation_only": self.config.simulation_only,
            "account_id": self.config.account_id.strip(),
            "state": state,
            "connected": output_ready,
            "can_submit_orders": state == "ready",
            "installed": input_ready,
            "supported_platform": True,
            "runtime": "CSV 文件单 + 东方财富量化终端",
            "endpoint_configured": bool(
                self.config.input_dir or self.config.options.get("input_dir") or self.config.endpoint
            ),
            "message": message,
            "requirements": [
                "东方财富量化仿真账户",
                "已启动文件单输入服务",
                "输入目录可写；如需同步，输出目录可读",
                "东方财富账户 ID 与文件单输出子目录一致",
            ],
        }

    def validate(self) -> dict[str, Any]:
        return self.status()

    def _write_command(self, suffix: str, headers: list[str], row: dict[str, Any]) -> Path:
        self._validate_submission()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        final_path = self.input_dir / f"{stamp}_{uuid4().hex[:8]}{suffix}.csv"
        temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        with temp_path.open("w", encoding=self.encoding, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(final_path)
        # The .fin marker tells the terminal that the file is complete and may
        # be removed after all rows have been scanned.
        final_path.with_name(final_path.name + ".fin").touch()
        return final_path

    def submit_order(self, order: SimulationOrder) -> SimulationOrder:
        order_type = 1 if order.order_type == "limit" else 2
        self._write_command(
            ".order",
            self._ORDER_HEADERS,
            {
                "sid": order.order_id,
                "account_id": self.config.account_id.strip(),
                "symbol": _eastmoney_symbol(order.ticker, self.config.options),
                "volume": order.shares,
                "order_type": order_type,
                "order_business(order_biz)": 1 if order.side.value == "buy" else 2,
                "price": order.limit_price or 0,
                "comment": f"a-share-agent:{order.source}",
            },
        )
        return order

    def cancel_order(self, order_id: str) -> None:
        self._write_command(
            ".cancel_order",
            self._CANCEL_HEADERS,
            {"sid": order_id, "comment": "a-share-agent cancel"},
        )

    @staticmethod
    def _read_csv(path: Path, encoding: str) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        for candidate in (encoding, "utf-8-sig", "gbk"):
            try:
                with path.open("r", encoding=candidate, newline="") as handle:
                    return list(csv.DictReader(handle))
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                raise SimulationBrokerUnavailableError(f"无法读取东方财富文件: {path}: {exc}") from exc
        raise SimulationBrokerUnavailableError(f"无法识别东方财富文件编码: {path}")

    def sync(self) -> dict[str, Any]:
        if not self.config.enabled:
            raise SimulationBrokerUnavailableError("东方财富文件单未启用")
        if not self.config.simulation_only:
            raise SimulationBrokerUnavailableError("东方财富文件单只允许连接模拟账户")
        if not self.config.account_id.strip():
            raise SimulationBrokerUnavailableError("请填写东方财富量化模拟账户 ID")
        if not str(self.config.output_dir or self.config.options.get("output_dir") or "").strip():
            raise SimulationBrokerUnavailableError("请填写东方财富文件单输出目录")

        root = self._output_account_dir()
        cash_rows = self._read_csv(root / "cash.csv", self.encoding)
        position_rows = self._read_csv(root / "position.csv", self.encoding)
        order_rows = self._read_csv(root / "order_status.csv", self.encoding)
        execution_rows = self._read_csv(root / "execution_report.csv", self.encoding)
        account_id = self.config.account_id.strip()

        cash_row = next(
            (
                row
                for row in reversed(cash_rows)
                if str(_row_value(row, "account_id", default="")).strip() in {"", account_id}
            ),
            None,
        )
        if cash_row is None:
            raise SimulationBrokerUnavailableError(f"尚未读取到东方财富资金文件: {root / 'cash.csv'}")

        positions = []
        for row in position_rows:
            volume = int(_number(_row_value(row, "volume", default=0)))
            if volume <= 0:
                continue
            ticker = _normalise_symbol(str(_row_value(row, "symbol", default="")))
            price = _number(_row_value(row, "price", default=0))
            avg_cost = _number(_row_value(row, "vwap", "vwap_dild", default=price))
            today = int(_number(_row_value(row, "vol_today", default=0)))
            available = int(_number(_row_value(row, "avl_now", "available", default=volume - today)))
            available = max(0, min(volume, available))
            positions.append(
                {
                    "ticker": ticker,
                    "shares": volume,
                    "avg_cost": avg_cost,
                    "current_price": price or avg_cost,
                    "available_shares": available,
                    "frozen_shares": max(0, volume - available),
                }
            )

        status_map = {
            1: "pending",
            2: "pending",
            3: "filled",
            5: "cancelled",
            8: "rejected",
            10: "pending",
            12: "cancelled",
        }
        orders = []
        for row in order_rows:
            sid = str(_row_value(row, "sid", default="")).strip()
            if not sid:
                continue
            status_code = int(_number(_row_value(row, "status", default=10)))
            orders.append(
                {
                    "sid": sid,
                    "broker_order_id": str(_row_value(row, "order_id", default="")).strip() or None,
                    "status": status_map.get(status_code, "pending"),
                    "filled_shares": int(_number(_row_value(row, "filled_vol", default=0))),
                    "fill_price": _number(_row_value(row, "filledvwap", "price", default=0)) or None,
                    "fill_date": _date_part(_row_value(row, "updated_at", "created_at", default="")),
                    "reject_reason": str(_row_value(row, "rej_detail", default="")).strip() or None,
                }
            )

        executions = []
        for row in execution_rows:
            exec_id = str(_row_value(row, "exec_id", default="")).strip()
            if not exec_id:
                continue
            executions.append(
                {
                    "external_id": exec_id,
                    "ticker": _normalise_symbol(str(_row_value(row, "symbol", default=""))),
                    "shares": int(_number(_row_value(row, "volume", default=0))),
                    "price": _number(_row_value(row, "price", default=0)),
                    "amount": _number(_row_value(row, "amount", default=0)),
                    "action": (
                        "buy"
                        if int(_number(_row_value(row, "order_biz", "order_business", default=1))) == 1
                        else "sell"
                    ),
                    "date": _date_part(_row_value(row, "created_at", "recv_at", default="")),
                }
            )

        return {
            "provider": self.provider,
            "account_id": account_id,
            "as_of": _date_part(_row_value(cash_row, "updated_at", "recv_at", default="")),
            "cash": _number(_row_value(cash_row, "available", "balance", default=0)),
            "positions": positions,
            "orders": orders,
            "trades": executions,
            "files": {
                "cash": str(root / "cash.csv"),
                "position": str(root / "position.csv"),
                "order_status": str(root / "order_status.csv"),
                "execution_report": str(root / "execution_report.csv"),
            },
        }


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
    if config.provider == "eastmoney_file":
        return EastmoneyFileBroker(config).status()
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


def get_simulation_broker(config: ExternalSimulationConfig) -> SimulationBroker:
    """Build the configured external simulation broker, failing closed by default."""
    if not config.enabled:
        raise SimulationBrokerUnavailableError("外部模拟账户未启用")
    if config.provider == "eastmoney_file":
        return EastmoneyFileBroker(config)
    raise SimulationBrokerUnavailableError(f"外部模拟平台 {config.provider} 尚未实现")
