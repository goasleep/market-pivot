import csv
from pathlib import Path

import pytest

from api.routers import portfolio as portfolio_router
from engine.broker_adapters import EastmoneyFileBroker
from engine.simulation_account import SimulationAccountService
from models.schemas import Decision, ExternalSimulationConfig, SimulationOrder


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _config(input_dir: Path, output_dir: Path | None = None) -> ExternalSimulationConfig:
    return ExternalSimulationConfig(
        provider="eastmoney_file",
        enabled=True,
        simulation_only=True,
        account_id="eastmoney-sim-1",
        input_dir=str(input_dir),
        output_dir=str(output_dir or input_dir / "push"),
    )


def test_file_broker_writes_order_and_cancel_commands(tmp_path):
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    broker = EastmoneyFileBroker(_config(scan_dir))
    order = SimulationOrder(
        order_id="sim-order-1",
        account_id="default",
        ticker="000001",
        side=Decision.BUY,
        shares=100,
        order_type="limit",
        limit_price=10.5,
    )

    broker.submit_order(order)
    order_files = list((tmp_path / "scan").glob("*.order.csv"))
    assert len(order_files) == 1
    with order_files[0].open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["sid"] == "sim-order-1"
    assert row["account_id"] == "eastmoney-sim-1"
    assert row["symbol"] == "SZSE.000001"
    assert row["order_type"] == "1"
    assert row["order_business(order_biz)"] == "1"
    assert (tmp_path / "scan" / f"{order_files[0].name}.fin").exists()

    broker.cancel_order(order.order_id)
    assert len(list((tmp_path / "scan").glob("*.cancel_order.csv"))) == 1


def test_file_broker_sync_reads_account_files(tmp_path):
    output = tmp_path / "push" / "eastmoney-sim-1"
    _write_csv(
        output / "cash.csv",
        [{"account_id": "eastmoney-sim-1", "available": "88000", "updated_at": "2026-08-18 15:00:00"}],
    )
    _write_csv(
        output / "position.csv",
        [
            {
                "account_id": "eastmoney-sim-1",
                "symbol": "SHSE.600000",
                "volume": "1000",
                "vol_today": "0",
                "vwap": "9.8",
                "price": "10.2",
                "avl_now": "1000",
            }
        ],
    )
    _write_csv(
        output / "order_status.csv",
        [
            {
                "sid": "sim-order-1",
                "order_id": "broker-1",
                "status": "3",
                "filled_vol": "1000",
                "filledvwap": "10.2",
                "updated_at": "2026-08-18 10:00:00",
            }
        ],
    )
    _write_csv(
        output / "execution_report.csv",
        [
            {
                "exec_id": "exec-1",
                "symbol": "SHSE.600000",
                "order_biz": "1",
                "volume": "1000",
                "price": "10.2",
                "amount": "10200",
                "created_at": "2026-08-18 10:00:01",
            }
        ],
    )

    snapshot = EastmoneyFileBroker(_config(tmp_path / "scan", tmp_path / "push")).sync()
    assert snapshot["cash"] == 88000
    assert snapshot["as_of"] == "2026-08-18"
    assert snapshot["positions"][0]["ticker"] == "600000"
    assert snapshot["positions"][0]["available_shares"] == 1000
    assert snapshot["orders"][0]["status"] == "filled"
    assert snapshot["trades"][0]["external_id"] == "exec-1"


@pytest.mark.asyncio
async def test_external_snapshot_updates_local_account_and_orders(tmp_path):
    service = SimulationAccountService(tmp_path / "simulation.db")
    await service.update_external_config("default", _config(tmp_path / "scan", tmp_path / "push"))
    order = await service.create_order("default", "600000", Decision.BUY, 1000, submitted_date="2026-08-18")

    await service.apply_external_snapshot(
        "default",
        {
            "as_of": "2026-08-18",
            "cash": 88000,
            "positions": [
                {
                    "ticker": "600000",
                    "shares": 1000,
                    "avg_cost": 9.8,
                    "current_price": 10.2,
                    "available_shares": 1000,
                    "frozen_shares": 0,
                }
            ],
            "orders": [
                {
                    "sid": order.order_id,
                    "status": "filled",
                    "fill_price": 10.2,
                    "fill_date": "2026-08-18",
                }
            ],
            "trades": [
                {
                    "external_id": "exec-1",
                    "ticker": "600000",
                    "action": "buy",
                    "shares": 1000,
                    "price": 10.2,
                    "amount": 10200,
                    "date": "2026-08-18",
                }
            ],
        },
    )

    account = await service.get_account("default")
    assert account.portfolio.cash == 88000
    assert account.portfolio.positions[0].ticker == "600000"
    assert (await service.list_orders("default"))[0].status == "filled"
    assert account.portfolio.trades[0].external_id == "exec-1"


@pytest.mark.asyncio
async def test_portfolio_order_routes_to_file_broker(monkeypatch, tmp_path):
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    service = SimulationAccountService(tmp_path / "simulation.db")
    await service.update_external_config("default", _config(scan_dir, tmp_path / "push"))
    monkeypatch.setattr(portfolio_router, "simulation_accounts", service)

    result = await portfolio_router.create_order(
        "default",
        portfolio_router.OrderRequest(
            ticker="000001",
            side=Decision.BUY,
            shares=100,
            fill_immediately=True,
        ),
    )

    assert result["status"] == "pending"
    assert (await service.list_orders("default"))[0].status == "pending"
    assert len(list(scan_dir.glob("*.order.csv"))) == 1
