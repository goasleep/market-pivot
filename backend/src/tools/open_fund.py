"""Trusted tools for off-exchange open-fund research."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from data.open_fund_provider import open_fund_data_provider
from harness.open_fund_metrics import nav_performance, score_open_fund_candidate


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@tool
async def get_open_fund_profile(fund_code: str) -> str:
    """通过 OpenFundDataProvider 核验场外开放式基金身份、类别和申赎状态。"""
    return (await open_fund_data_provider.profile(fund_code)).model_dump_json()


@tool
async def get_open_fund_nav(fund_code: str, product_category: str = "unknown") -> str:
    """获取场外基金历史净值；货币基金必须改用收益序列工具。"""
    if product_category == "money_market":
        return _dump(
            {
                "status": "data_unavailable",
                "data": None,
                "fetched_at": _now(),
                "freshness": "not_applicable",
                "sources": [],
                "errors": [{"code": "money_nav_not_applicable", "message": "货币基金应使用万份收益和七日年化序列"}],
            }
        )
    return (await open_fund_data_provider.nav_history(fund_code, product_category)).model_dump_json()


@tool
async def get_open_fund_money_yield(fund_code: str) -> str:
    """获取货币基金万份收益和七日年化历史，不使用固定单位净值计算收益。"""
    return (await open_fund_data_provider.money_yield_history(fund_code)).model_dump_json()


@tool
async def get_open_fund_fees(fund_code: str) -> str:
    """获取场外基金运作、申购、赎回和确认日规则。"""
    return (await open_fund_data_provider.fees(fund_code)).model_dump_json()


@tool
async def get_open_fund_exposure(fund_code: str, year: str = "") -> str:
    """获取场外基金股票、债券和行业配置，并保留报告期。"""
    return (await open_fund_data_provider.exposure(fund_code, year)).model_dump_json()


@tool
async def discover_open_fund_candidates(product_category: str, limit: int = 20) -> str:
    """从同一场外基金类别发现结构化候选；不允许跨类别直接排名。"""
    result = await open_fund_data_provider.universe(product_category)
    if result.data is not None:
        result.data = result.data[: max(1, min(int(limit), 50))]
    return result.model_dump_json()


@tool
async def calculate_open_fund_relative_strength(
    fund_code: str,
    nav_values: list[float],
    peer_returns: dict[str, float] | None = None,
    as_of: str = "",
) -> str:
    """使用净值而非场内价格计算区间表现、波动和最大回撤。"""
    performance = nav_performance(nav_values)
    fund_return = performance["return"]
    peers = peer_returns or {}
    rank = 1 + sum(float(value) > float(fund_return) for value in peers.values()) if fund_return is not None else None
    status = "available" if fund_return is not None else "data_unavailable"
    return _dump(
        {
            "status": status,
            "data": {
                "fund_code": fund_code,
                **performance,
                "peer_rank": rank,
                "peer_count": len(peers) + 1 if peers else 0,
                "pricing_basis": "nav",
            }
            if status == "available"
            else None,
            "as_of": as_of or None,
            "fetched_at": _now(),
            "freshness": "calculated",
            "sources": [],
            "errors": [] if status == "available" else [{"code": "nav_insufficient", "message": "净值序列不足"}],
        }
    )


@tool
async def calculate_money_fund_stability(
    fund_code: str,
    yield_per_10k: list[float],
    seven_day_annualized: list[float],
    as_of: str = "",
) -> str:
    """计算货币基金收益水平与稳定性，不把单位净值变化当作收益。"""
    clean_10k = [float(value) for value in yield_per_10k]
    clean_7d = [float(value) for value in seven_day_annualized]
    available = bool(clean_10k and clean_7d)
    data = None
    if available:
        data = {
            "fund_code": fund_code,
            "average_yield_per_10k": round(statistics.fmean(clean_10k), 8),
            "average_seven_day_annualized": round(statistics.fmean(clean_7d), 8),
            "seven_day_yield_stdev": round(statistics.pstdev(clean_7d), 8),
            "pricing_basis": "money_yield",
        }
    return _dump(
        {
            "status": "available" if available else "data_unavailable",
            "data": data,
            "as_of": as_of or None,
            "fetched_at": _now(),
            "freshness": "calculated",
            "sources": [],
            "errors": [] if available else [{"code": "money_yield_insufficient", "message": "货币基金收益序列不足"}],
        }
    )


@tool
async def screen_compare_open_funds(
    candidates: list[dict[str, Any]],
    product_category: str,
    allow_cross_category: bool = False,
) -> str:
    """按产品类别的固定权重评分；缺失维度为零且不重新归一化。"""
    categories = {str(item.get("product_category") or product_category) for item in candidates}
    cross_category = len(categories) > 1 or any(category != product_category for category in categories)
    if cross_category and not allow_cross_category:
        return _dump(
            {
                "status": "data_unavailable",
                "data": {"ranking": [], "candidates": candidates, "ranking_is_formal": False},
                "fetched_at": _now(),
                "freshness": "calculated",
                "sources": [],
                "errors": [{"code": "cross_category_ranking_forbidden", "message": "场外基金必须在同一类别内筛选"}],
            }
        )
    scored = [
        {
            **item,
            **score_open_fund_candidate(product_category, dict(item.get("dimensions") or {})),
        }
        for item in candidates
    ]
    ready = all(
        item.get("provider_verified") is True
        and item.get("as_of")
        and str(item.get("product_category")) == product_category
        for item in scored
    )
    scored.sort(key=lambda item: (-float(item["score"]), str(item.get("fund_code") or "")))
    return _dump(
        {
            "status": "available" if ready and len(scored) >= 2 else "limited",
            "data": {
                "ranking": scored if ready else [],
                "candidates": scored,
                "ranking_is_formal": ready and len(scored) >= 2,
                "selection_signal": False,
                "product_category": product_category,
            },
            "as_of": max((str(item.get("as_of")) for item in scored if item.get("as_of")), default=None),
            "fetched_at": _now(),
            "freshness": "calculated",
            "sources": [],
            "errors": []
            if ready and len(scored) >= 2
            else [{"code": "ranking_requirements_missing", "message": "缺少身份核验、数据日期、类别一致性或候选不足"}],
        }
    )


@tool
async def get_open_fund_event_risk(fund_code: str, announcements: list[dict[str, Any]], as_of: str = "") -> str:
    """整理暂停申购赎回、限购、分红、清盘和基金经理变更事件。"""
    keywords = ("暂停", "限购", "申购", "赎回", "分红", "清盘", "基金经理", "离任")
    events = [item for item in announcements if any(word in str(item) for word in keywords)]
    return _dump(
        {
            "status": "available" if events else "limited" if announcements else "data_unavailable",
            "data": {"fund_code": fund_code, "events": events, "announcement_count": len(announcements)},
            "as_of": as_of or None,
            "fetched_at": _now(),
            "freshness": "1h",
            "sources": [],
            "errors": [] if events else [{"code": "event_evidence_limited", "message": "未取得关键场外基金事件"}],
        }
    )


@tool
async def run_open_fund_nav_backtest(
    nav_points: list[dict[str, Any]],
    signal_dates: list[str],
    fee_rate: float | None = None,
    strategy_uses_volume: bool = False,
) -> str:
    """按信号后的下一可用累计净值执行场外基金回测，禁止成交量策略和零费用假设。"""
    if strategy_uses_volume:
        return _dump(
            {
                "status": "data_unavailable",
                "data": None,
                "fetched_at": _now(),
                "freshness": "calculated",
                "sources": [],
                "errors": [
                    {
                        "code": "intraday_or_volume_strategy_not_applicable",
                        "message": "场外基金 NAV 回测不支持成交量或盘中策略",
                    }
                ],
            }
        )
    if fee_rate is None:
        return _dump(
            {
                "status": "data_unavailable",
                "data": None,
                "fetched_at": _now(),
                "freshness": "calculated",
                "sources": [],
                "errors": [
                    {
                        "code": "fee_assumption_required",
                        "message": "缺少结构化费率或显式费用假设，不能默认为零",
                    }
                ],
            }
        )
    ordered = sorted(
        (point for point in nav_points if point.get("date")),
        key=lambda point: str(point["date"]),
    )
    executions: list[dict[str, Any]] = []
    for signal_date in sorted(set(signal_dates)):
        execution = next((point for point in ordered if str(point["date"]) > signal_date), None)
        if execution is None:
            continue
        nav_value = execution.get("cumulative_nav")
        basis = "cumulative_nav"
        if nav_value is None:
            nav_value = execution.get("unit_nav")
            basis = "unit_nav"
        if nav_value is not None:
            executions.append(
                {
                    "signal_date": signal_date,
                    "execution_date": str(execution["date"]),
                    "execution_nav": float(nav_value),
                    "nav_field": basis,
                }
            )
    if len(executions) < 2:
        status = "data_unavailable"
        data = None
        errors = [{"code": "nav_backtest_insufficient", "message": "至少需要两个可在下一净值日执行的信号"}]
    else:
        gross_return = executions[-1]["execution_nav"] / executions[0]["execution_nav"] - 1
        total_fee = max(0.0, float(fee_rate)) * 2
        status = "available"
        data = {
            "pricing_basis": "nav",
            "execution_rule": "next_available_nav",
            "nav_field": (
                "cumulative_nav"
                if all(item["nav_field"] == "cumulative_nav" for item in executions)
                else "unit_nav_fallback_disclosed"
            ),
            "executions": executions,
            "gross_return": round(gross_return, 8),
            "fee_rate": float(fee_rate),
            "net_return": round(gross_return - total_fee, 8),
            "uses_intraday_price": False,
            "uses_volume": False,
            "lookahead_safe": True,
        }
        errors = []
    return _dump(
        {
            "status": status,
            "data": data,
            "as_of": str(ordered[-1]["date"]) if ordered else None,
            "fetched_at": _now(),
            "freshness": "calculated",
            "sources": [],
            "errors": errors,
        }
    )


TOOLS = [
    get_open_fund_profile,
    get_open_fund_nav,
    get_open_fund_money_yield,
    get_open_fund_fees,
    get_open_fund_exposure,
    discover_open_fund_candidates,
    calculate_open_fund_relative_strength,
    calculate_money_fund_stability,
    screen_compare_open_funds,
    get_open_fund_event_risk,
    run_open_fund_nav_backtest,
]
