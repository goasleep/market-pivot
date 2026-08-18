import pandas as pd
import pytest

from engine import backtester
from engine.simulation_account import SimulationAccountService
from graph import workflow as workflow_module
from models.schemas import AgentReport, Decision, MarketContext, TradeDecision


@pytest.mark.asyncio
async def test_workflow_parallel_nodes_merge_progress_without_conflict(monkeypatch):
    async def report(name: str, *args, **kwargs):
        return AgentReport(agent_name=name, reasoning=name)

    async def debate(*args, **kwargs):
        return AgentReport(agent_name="debate", reasoning="debate")

    async def risk(*args, **kwargs):
        return AgentReport(agent_name="risk_manager", reasoning="risk")

    async def decide(ticker, *args, **kwargs):
        return TradeDecision(ticker=ticker, decision=Decision.HOLD)

    monkeypatch.setattr(workflow_module, "tech_analyze", lambda *a, **k: report("technical"))
    monkeypatch.setattr(workflow_module, "fund_analyze", lambda *a, **k: report("fundamentals"))
    monkeypatch.setattr(workflow_module, "sent_analyze", lambda *a, **k: report("sentiment"))
    monkeypatch.setattr(workflow_module, "debate", debate)
    monkeypatch.setattr(workflow_module, "risk_assess", risk)
    monkeypatch.setattr(workflow_module, "pm_decide", decide)

    result = await workflow_module.build_workflow().ainvoke(
        {
            "ticker": "000001",
            "market_context": MarketContext(ticker="000001", current_price=10),
            "progress": [],
        }
    )

    stages = {item["stage"] for item in result["progress"]}
    assert stages == {"market_data", "technical", "fundamentals", "sentiment", "debate", "risk", "portfolio"}
    assert result["final_decision"].decision == Decision.HOLD


@pytest.mark.asyncio
async def test_backtest_builds_as_of_context_and_does_not_use_live_data(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=6, freq="D").strftime("%Y-%m-%d")
    history = pd.DataFrame(
        {
            "date": dates,
            "open": [10, 10, 11, 11, 12, 12],
            "close": [10, 11, 11, 12, 12, 13],
            "high": [10, 11, 12, 12, 13, 14],
            "low": [9, 10, 10, 11, 11, 12],
            "volume": [100] * 6,
            "pct_chg": [0] * 6,
        }
    )
    contexts = []

    async def history_provider(*args, **kwargs):
        return history.copy()

    async def fake_invoke(state):
        contexts.append(state["market_context"])
        return {"final_decision": TradeDecision(ticker="000001", decision=Decision.HOLD)}

    monkeypatch.setattr(backtester, "async_get_stock_history", history_provider)
    monkeypatch.setattr(backtester.workflow, "ainvoke", fake_invoke)

    result = await backtester.run_backtest(
        "000001",
        "2026-01-01",
        "2026-01-06",
        decision_interval=2,
    )

    assert result["final_value"] == result["initial_capital"]
    assert result["data_snapshot"]["sha256"]
    assert result["data_snapshot"]["quality"]["status"] == "valid"
    assert contexts
    for context in contexts:
        assert context.is_backtest is True
        assert not context.financial
        assert not context.news
        assert all(row["date"] <= context.as_of_date for row in context.history)


@pytest.mark.asyncio
async def test_pool_backtest_uses_one_agent_portfolio_without_live_enrichment(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=6, freq="D").strftime("%Y-%m-%d")

    def make_history(offset: float):
        return pd.DataFrame(
            {
                "date": dates,
                "open": [10 + offset] * 6,
                "close": [10 + offset, 11 + offset, 11 + offset, 12 + offset, 12 + offset, 13 + offset],
                "high": [14 + offset] * 6,
                "low": [9 + offset] * 6,
                "volume": [100] * 6,
                "pct_chg": [0] * 6,
            }
        )

    async def history_provider(ticker, *args, **kwargs):
        return make_history(0 if ticker == "000001" else 2)

    async def fake_invoke(state):
        assert state["market_context"].is_backtest is True
        return {"final_decision": TradeDecision(ticker=state["ticker"], decision=Decision.HOLD)}

    monkeypatch.setattr(backtester, "async_get_stock_history", history_provider)
    monkeypatch.setattr(backtester.workflow, "ainvoke", fake_invoke)
    result = await backtester.run_pool_backtest(
        ["000001", "600519"],
        "2026-01-01",
        "2026-01-06",
        decision_interval=2,
    )

    assert result["ticker"] == "pool"
    assert result["tickers"] == ["000001", "600519"]
    assert result["total_trades"] == 0
    assert len(result["equity_curve"]) == 6
    assert len(result["data_snapshots"]) == 2


def test_simulation_account_enforces_t_plus_one_and_persists(tmp_path):
    service = SimulationAccountService(tmp_path / "simulation.db")
    buy = service.create_order("default", "000001", Decision.BUY, 100, submitted_date="2026-08-01")
    buy = service.fill_order(buy.order_id, 10.0, "2026-08-01")
    assert buy.status == "filled"

    sell_same_day = service.create_order("default", "000001", Decision.SELL, 100, submitted_date="2026-08-01")
    sell_same_day = service.fill_order(sell_same_day.order_id, 11.0, "2026-08-01")
    assert sell_same_day.status == "rejected"

    service.advance_date("default", "2026-08-02")
    sell_next_day = service.create_order("default", "000001", Decision.SELL, 100, submitted_date="2026-08-02")
    sell_next_day = service.fill_order(sell_next_day.order_id, 11.0, "2026-08-02")
    assert sell_next_day.status == "filled"
    assert service.get_account("default").portfolio.total_value > 1_000_000


@pytest.mark.asyncio
async def test_backtest_job_manager_runs_in_background(monkeypatch):
    import application.backtest_jobs as jobs_module

    async def fake_backtest(**kwargs):
        callback = kwargs["progress_callback"]
        await callback("data_fetch", "ready")
        return {"final_value": 123}

    monkeypatch.setattr(jobs_module, "run_backtest", fake_backtest)
    job = await jobs_module.BacktestJobManager().submit({"ticker": "000001"})
    await job.task
    assert job.status == "completed"
    assert job.result == {"final_value": 123}
    assert job.progress == [{"stage": "data_fetch", "message": "ready"}]
