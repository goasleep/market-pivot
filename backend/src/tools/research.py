"""Deterministic research and decision-support tools."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import tool

import application.strategy_comparison as strategy_comparison
from agents.technical_analyst import calculate_technical_indicators
from application.backtest_experiment import run_backtest_experiment
from application.backtest_service import run_backtest as execute_backtest
from application.strategy_candidates import strategy_candidates
from data.backtest_data import BacktestDataError
from data.exchange_fund_provider import async_get_exchange_fund_history
from data.source_registry import provenance
from data.stock_provider import async_get_stock_history
from domain.risk_tools import build_trade_plan as _build_trade_plan
from domain.risk_tools import calculate_risk_metrics as _calculate_risk_metrics
from models.schemas import AssetType
from strategies.strategy_registry import list_strategies


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


_CORE_BACKTEST_FIELDS = (
    "strategy_name",
    "display_name",
    "description",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "exposure",
    "turnover",
    "total_fees",
    "total_trades",
    "final_value",
    "excess_return",
    "error",
)


def _core_backtest_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    compact = {field: row.get(field) for field in _CORE_BACKTEST_FIELDS if field in row}
    diagnostics = row.get("diagnostics")
    out_of_sample = diagnostics.get("out_of_sample") if isinstance(diagnostics, dict) else None
    if isinstance(out_of_sample, dict):
        compact["out_of_sample"] = {
            key: out_of_sample.get(key)
            for key in ("start_date", "end_date", "out_of_sample_return", "max_drawdown", "sharpe_ratio")
            if key in out_of_sample
        }
    return compact


def _compact_cost_scenarios(value: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(name): [_core_backtest_row(row) for row in rows if isinstance(row, dict)]
        for name, rows in value.items()
        if isinstance(rows, list)
    }


def _compact_artifacts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    fields = (
        "artifact_id",
        "name",
        "artifact_type",
        "mime_type",
        "size_bytes",
        "preview_url",
        "download_url",
    )
    return [
        {field: artifact.get(field) for field in fields if artifact.get(field) is not None}
        for artifact in value
        if isinstance(artifact, dict)
    ]


def _compact_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    fields = (
        "source",
        "source_id",
        "provider",
        "adjustment",
        "requested_start_date",
        "requested_end_date",
        "actual_start_date",
        "actual_end_date",
        "row_count",
        "sha256",
        "fetched_at",
    )
    return {field: value.get(field) for field in fields if value.get(field) is not None}


def _comparison_supervisor_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return only explainable metrics; full curves and trades remain in artifacts."""
    raw_comparisons = payload.get("comparisons")
    comparisons = (
        [_core_backtest_row(row) for row in raw_comparisons if isinstance(row, dict)]
        if isinstance(raw_comparisons, list)
        else []
    )
    cost_scenarios = _compact_cost_scenarios(payload.get("cost_scenarios"))
    artifacts = _compact_artifacts(payload.get("artifacts"))
    conclusion = payload.get("conclusion") if isinstance(payload.get("conclusion"), dict) else {}
    acceptance = payload.get("acceptance") if isinstance(payload.get("acceptance"), dict) else {}
    data_validation = payload.get("data_validation") if isinstance(payload.get("data_validation"), dict) else {}
    validation_summary = {
        key: data_validation.get(key)
        for key in ("status", "selected_source", "selection_reason", "rule_version")
        if data_validation.get(key) is not None
    }
    compact = {
        "data_type": "strategy_backtest_comparison",
        "_tool_name": "compare_strategy_backtests",
        "available": payload.get("available", True),
        "data_status": payload.get("data_status", "available"),
        "ticker": payload.get("ticker"),
        "asset_type": payload.get("asset_type"),
        "requested_start_date": payload.get("requested_start_date"),
        "evaluation_start_date": payload.get("evaluation_start_date") or payload.get("start_date"),
        "evaluation_end_date": payload.get("evaluation_end_date") or payload.get("end_date"),
        "actual_start_date": payload.get("actual_start_date"),
        "actual_end_date": payload.get("actual_end_date"),
        "history_years": payload.get("history_years"),
        "initial_capital": payload.get("initial_capital"),
        "strategy_count": payload.get("strategy_count", len(comparisons)),
        "benchmark": payload.get("benchmark"),
        "ranking_metric": payload.get("ranking_metric"),
        "ranking_label": payload.get("ranking_label"),
        "ranking": list(payload.get("ranking") or []),
        "comparisons": comparisons,
        "cost_scenarios": cost_scenarios,
        "cost_consistency": payload.get("cost_consistency", {}),
        "data_snapshot": _compact_snapshot(payload.get("data_snapshot")),
        "data_validation": validation_summary,
        "execution": payload.get("execution", {}),
        "acceptance": acceptance,
        "conclusion": conclusion,
        "research_decision": payload.get("research_decision", {}),
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "artifact_usage": "完整曲线、逐日净值、信号和成交明细仅供用户下载审计；Supervisor 不应读取附件形成正文。",
        "provenance": payload.get("provenance", {}),
    }
    if payload.get("message") is not None:
        compact["message"] = payload.get("message")
    if payload.get("error") is not None:
        compact["error"] = payload.get("error")
    compact["supervisor_summary"] = {
        "instruction": (
            "核心指标已经完整包含在本对象中。请直接解释 comparisons、cost_scenarios、acceptance、"
            "conclusion 和数据口径；不要调用 read_artifact 或 list_artifacts 补取回测正文。"
        ),
        "period": {
            "start": compact["evaluation_start_date"],
            "end": compact["evaluation_end_date"],
        },
        "core_metrics": comparisons,
        "cost_scenarios": cost_scenarios,
        "ranking": compact["ranking"],
        "acceptance": acceptance,
        "conclusion": conclusion,
    }
    return compact


@tool
async def compute_technical_indicators(ticker: str, asset_type: str = "stock", limit: int = 120) -> str:
    """获取历史价格并计算均线、MACD、RSI、KDJ、布林带和量能指标。"""
    kind = AssetType(asset_type)
    frame = (
        await async_get_stock_history(ticker)
        if kind == AssetType.STOCK
        else await async_get_exchange_fund_history(ticker, asset_type=kind.value)
    )
    frame = frame.tail(max(20, min(int(limit), 500)))
    indicators = calculate_technical_indicators(frame)
    available = bool(indicators)
    return _dump(
        {
            "data_type": "technical_indicators",
            "ticker": ticker,
            "asset_type": kind.value,
            "available": available,
            "data_status": "available" if available else "unavailable",
            "message": None if available else "没有可用于计算技术指标的历史数据",
            "history_count": len(frame),
            "indicators": indicators,
            "provenance": provenance(
                "akshare",
                freshness="historical",
                status="available" if available else "unavailable",
            ),
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
    try:
        result = await execute_backtest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            asset_type=asset_type,
            initial_capital=initial_capital,
            strategy_name=strategy_name,
            decision_interval=decision_interval,
        )
    except BacktestDataError as exc:
        return _dump(
            {
                "data_type": "backtest",
                "backtest": True,
                "paper_trading": False,
                "available": False,
                "data_status": "unavailable",
                "result": None,
                "message": str(exc),
                "error": {"code": "backtest_data_unavailable", "message": str(exc)},
                "provenance": provenance("akshare", freshness="historical", status="unavailable"),
            }
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
    try:
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
    except BacktestDataError as exc:
        return _dump(
            {
                "data_type": "backtest_experiment",
                "backtest": True,
                "paper_trading": False,
                "available": False,
                "data_status": "unavailable",
                "result": None,
                "message": str(exc),
                "error": {"code": "backtest_data_unavailable", "message": str(exc)},
                "provenance": provenance("akshare", freshness="historical", status="unavailable"),
            }
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
    market_benchmark_ticker: str = "000300",
) -> str:
    """比较策略并直接返回核心指标；完整曲线和成交明细仅保存在审计附件中。"""
    kind = AssetType(asset_type)
    strategies = strategy_comparison.standard_strategy_suite(kind)
    if strategy_names:
        requested = set(strategy_names)
        selected = tuple(item for item in strategies if item.name in requested)
        if len(selected) >= 7:
            strategies = selected
    spec = strategy_comparison.build_comparison_spec(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        asset_type=kind,
        initial_capital=initial_capital,
        strategies=strategies,
        objective=objective,
        market_benchmark_ticker=market_benchmark_ticker,
        market_benchmark_name="沪深300" if market_benchmark_ticker == "000300" else market_benchmark_ticker,
    )
    try:
        payload = await strategy_comparison.compare_strategies(
            spec,
            publish_artifacts=True,
            generate_explanation=True,
        )
    except BacktestDataError as exc:
        payload = {
            "data_type": "strategy_backtest_comparison",
            "ticker": ticker,
            "asset_type": kind.value,
            "start_date": start_date,
            "end_date": end_date,
            "available": False,
            "data_status": "unavailable",
            "strategy_count": 0,
            "comparisons": [],
            "task_contract": spec.task_contract.model_dump(mode="json"),
            "acceptance": {
                "satisfied": False,
                "checks": {"data_available": False},
                "missing": ["data_available"],
            },
            "conclusion": {
                "official": False,
                "tradeoffs": ["没有可用历史数据，无法运行策略比较。"],
                "data_warnings": [str(exc)],
                "limitations": ["未产生回测结果，不能比较策略盈利情况。"],
            },
            "message": str(exc),
            "error": {"code": "backtest_data_unavailable", "message": str(exc)},
        }
    payload.update(
        {
            "_tool_name": "compare_strategy_backtests",
            "decision_interval": decision_interval,
            "provenance": provenance(
                str(payload.get("data_validation", {}).get("selected_source") or "akshare"),
                freshness="historical",
                status="unavailable" if payload.get("available") is False else "available",
            ),
        }
    )
    return _dump(_comparison_supervisor_payload(payload))


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
    try:
        candidate = await strategy_candidates.generate(
            objective=objective,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            asset_type=AssetType(asset_type),
            initial_capital=initial_capital,
        )
    except BacktestDataError as exc:
        return _dump(
            {
                "data_type": "sandbox_strategy_candidate",
                "paper_trading": False,
                "live_trading": False,
                "available": False,
                "data_status": "unavailable",
                "message": str(exc),
                "error": {"code": "backtest_data_unavailable", "message": str(exc)},
                "provenance": provenance("sandbox", freshness="historical", status="unavailable"),
            }
        )
    payload = candidate.model_dump(mode="json")
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
