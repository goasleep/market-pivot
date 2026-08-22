"""Deterministic research and decision-support tools."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import tool

from agents.technical_analyst import calculate_technical_indicators
from data.fund_provider import async_get_fund_history
from data.source_registry import provenance
from data.stock_provider import async_get_stock_history
from domain.risk_tools import build_trade_plan as _build_trade_plan
from domain.risk_tools import calculate_risk_metrics as _calculate_risk_metrics
from models.schemas import AssetType
from strategies.skill_manager import list_strategies


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


@tool
async def compute_technical_indicators(ticker: str, asset_type: str = "stock", limit: int = 120) -> str:
    """获取历史价格并计算均线、MACD、RSI、KDJ、布林带和量能指标。"""
    kind = AssetType(asset_type)
    frame = (
        await async_get_stock_history(ticker)
        if kind == AssetType.STOCK
        else await async_get_fund_history(ticker, asset_type=kind.value)
    )
    frame = frame.tail(max(20, min(int(limit), 500)))
    return _dump(
        {
            "data_type": "technical_indicators",
            "ticker": ticker,
            "asset_type": kind.value,
            "history_count": len(frame),
            "indicators": calculate_technical_indicators(frame),
            "provenance": provenance("akshare", freshness="historical"),
        }
    )


@tool
async def calculate_risk_metrics(
    current_price: float,
    entry_price: float | None = None,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.16,
    position_size_pct: float = 0.2,
    available_capital: float | None = None,
    max_loss_pct: float | None = None,
) -> str:
    """按确定性公式计算止损、止盈、仓位、风险收益比和预估亏损。"""
    return _dump(
        {
            "data_type": "risk_metrics",
            "metrics": _calculate_risk_metrics(
                current_price,
                entry_price=entry_price,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                position_size_pct=position_size_pct,
                available_capital=available_capital,
                max_loss_pct=max_loss_pct,
            ),
            "provenance": provenance("derived", freshness="calculated"),
        }
    )


@tool
async def build_trade_plan(
    ticker: str,
    current_price: float,
    asset_type: str = "stock",
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.16,
    position_size_pct: float = 0.2,
    available_capital: float | None = None,
    max_loss_pct: float | None = None,
) -> str:
    """基于明确输入生成带计算依据的交易计划，不替代综合分析结论。"""
    return _dump(
        {
            "data_type": "trade_plan",
            "plan": _build_trade_plan(
                ticker,
                current_price,
                asset_type=asset_type,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                position_size_pct=position_size_pct,
                available_capital=available_capital,
                max_loss_pct=max_loss_pct,
            ),
            "provenance": provenance("derived", freshness="calculated"),
        }
    )


@tool
async def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    asset_type: str = "stock",
    initial_capital: float = 1_000_000,
    strategy_name: str | None = None,
    decision_interval: int = 1,
) -> str:
    """按指定区间和策略运行研究回测；结果仅用于模拟，不执行真实交易。"""
    from application.backtest_service import run_backtest as execute_backtest

    result = await execute_backtest(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        asset_type=asset_type,
        initial_capital=initial_capital,
        strategy_name=strategy_name,
        decision_interval=decision_interval,
    )
    return _dump(
        {
            "data_type": "backtest",
            "backtest": True,
            "paper_trading": False,
            "result": result,
            "provenance": provenance("akshare", freshness="historical"),
        }
    )


@tool
async def design_and_run_backtest(
    objective: str,
    start_date: str,
    end_date: str,
    ticker: str | None = None,
    tickers: list[str] | None = None,
    mode: str = "auto",
    asset_type: str = "stock",
    initial_capital: float = 1_000_000,
    decision_interval: int = 1,
    portfolio_spec: dict | None = None,
) -> str:
    """让 Agent 设计策略和组合规则，运行回测并保存完整实验报告附件。"""
    from application.backtest_experiment import run_backtest_experiment

    experiment = await run_backtest_experiment(
        objective=objective,
        ticker=ticker,
        tickers=tickers,
        mode=mode,
        start_date=start_date,
        end_date=end_date,
        asset_type=asset_type,
        initial_capital=initial_capital,
        decision_interval=decision_interval,
        portfolio_spec=portfolio_spec,
    )
    return _dump(
        {
            "data_type": "backtest_experiment",
            "backtest": True,
            "paper_trading": False,
            "experiment_id": experiment["experiment_id"],
            "strategy_spec": experiment["strategy_spec"],
            "result": experiment["result"],
            "artifacts": experiment["artifacts"],
            "provenance": provenance("akshare", freshness="historical"),
        }
    )


@tool
async def compare_strategy_backtests(
    ticker: str,
    start_date: str,
    end_date: str,
    asset_type: str = "stock",
    strategy_names: list[str] | None = None,
    initial_capital: float = 1_000_000,
    decision_interval: int = 1,
    objective: str = "",
) -> str:
    """在共享数据快照上比较标准策略，并完成成本、样本外和稳定性检验。"""
    from application.strategy_comparison import build_comparison_spec, compare_strategies, standard_strategy_suite

    kind = AssetType(asset_type)
    strategies = standard_strategy_suite(kind)
    if strategy_names:
        requested = set(strategy_names)
        selected = tuple(item for item in strategies if item.name in requested)
        if len(selected) >= 7:
            strategies = selected
    spec = build_comparison_spec(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        asset_type=kind,
        initial_capital=initial_capital,
        strategies=strategies,
        objective=objective,
    )
    payload = await compare_strategies(spec)
    payload.update(
        {
            "_tool_name": "compare_strategy_backtests",
            "decision_interval": decision_interval,
            "provenance": provenance("akshare", freshness="historical"),
        }
    )
    return _dump(payload)


@tool
async def design_and_run_sandbox_strategy(
    objective: str,
    ticker: str,
    start_date: str,
    end_date: str,
    asset_type: str = "etf",
    initial_capital: float = 1_000_000,
) -> str:
    """让代码 Agent 生成受限目标仓位函数，在隔离子进程验证后由可信核心回测。"""
    from application.strategy_candidates import strategy_candidates

    candidate = await strategy_candidates.generate(
        objective=objective,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        asset_type=AssetType(asset_type),
        initial_capital=initial_capital,
    )
    payload = candidate.model_dump(mode="json")
    payload.pop("source_code", None)
    payload.update(
        {
            "data_type": "sandbox_strategy_candidate",
            "_tool_name": "design_and_run_sandbox_strategy",
            "paper_trading": False,
            "live_trading": False,
            "provenance": provenance("sandbox", freshness="historical"),
        }
    )
    return _dump(payload)


@tool
async def list_trading_strategies() -> str:
    """列出系统支持的研究和交易策略。"""
    return _dump(
        {
            "data_type": "strategies",
            "strategies": await asyncio.to_thread(list_strategies),
            "provenance": provenance("derived", freshness="configuration"),
        }
    )


TOOLS = [
    compute_technical_indicators,
    calculate_risk_metrics,
    build_trade_plan,
    run_backtest,
    design_and_run_backtest,
    compare_strategy_backtests,
    design_and_run_sandbox_strategy,
    list_trading_strategies,
]
