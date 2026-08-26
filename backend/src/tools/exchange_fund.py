"""Trusted deterministic tools for the exchange-fund domain Skill pack."""

from __future__ import annotations

import json
import math
import statistics
from datetime import date, datetime, timezone
from typing import Any

from langchain_core.tools import tool

from data.akshare_provider import async_get_asset_spot
from data.exchange_fund_provider import exchange_fund_data_provider
from data.source_registry import provenance
from harness.exchange_fund_metrics import liquidity_metrics, premium_discount, score_candidate, tracking_metrics


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _theme_candidate_matches(records: list[dict[str, Any]], theme: str) -> list[dict[str, Any]]:
    """Expand a theme into deterministic provider-backed ETF name matches."""
    normalized = theme.strip().lower()
    if any(token in normalized for token in ("白酒", "酒类", "酿酒")):
        core_tokens = ("白酒", "酒etf")
        related_tokens = ("食品饮料",)
    else:
        core_tokens = (normalized,)
        related_tokens = ()
    matches: list[dict[str, Any]] = []
    for record in records:
        name = str(record.get("name") or record.get("名称") or "")
        name_lower = name.lower()
        if any(token and token in name_lower for token in core_tokens):
            scope = "core"
            reason = f"基金名称与主题核心词匹配：{name}"
        elif any(token and token in name_lower for token in related_tokens):
            scope = "related_proxy"
            reason = f"相关代理产品，底层白酒权重仍需持仓证据核验：{name}"
        else:
            continue
        matches.append(
            {
                **record,
                "ticker": str(record.get("ticker") or record.get("代码") or "").zfill(6),
                "name": name,
                "provider_verified": True,
                "theme_scope": scope,
                "theme_match_reason": reason,
            }
        )
    matches.sort(
        key=lambda item: (
            item["theme_scope"] != "core",
            -float(item.get("amount") or 0),
            item["ticker"],
        )
    )
    return matches


@tool
async def discover_exchange_fund_candidates(theme: str, limit: int = 20) -> str:
    """从结构化全市场快照发现主题 ETF 候选，并区分核心与相关代理产品。"""
    fetched_at = _now()
    try:
        records = await async_get_asset_spot("etf", limit=5000)
        matches = _theme_candidate_matches(records, theme)[: max(1, min(int(limit), 50))]
    except Exception as exc:
        matches = []
        error = str(exc)[:500]
    else:
        error = ""
    core_count = sum(item["theme_scope"] == "core" for item in matches)
    eligible = len(matches) >= 3 and core_count > 0
    status = "available" if eligible else "limited" if matches else "data_unavailable"
    return _dump(
        {
            "data_type": "etf_candidate_universe",
            "status": status,
            "available": bool(matches),
            "theme": theme,
            "as_of": fetched_at,
            "fetched_at": fetched_at,
            "freshness": "60s",
            "data": {
                "candidates": matches,
                "candidate_count": len(matches),
                "core_count": core_count,
                "deep_analysis_shortlist": [item["ticker"] for item in matches[:4]],
                "screening_eligible": eligible,
                "formal_ranking_eligible": False,
            },
            "sources": provenance("akshare", as_of=fetched_at, freshness="60s", status=status),
            "errors": []
            if matches
            else [{"code": "theme_candidates_unavailable", "message": error or "结构化快照未匹配到主题 ETF"}],
        }
    )


@tool
async def get_exchange_fund_profile(ticker: str, asset_type: str = "etf") -> str:
    """通过结构化 Provider 核验 ETF/LOF 身份并返回产品资料；代码前缀不作为核验结论。"""
    return (await exchange_fund_data_provider.profile(ticker, asset_type)).model_dump_json()


@tool
async def get_exchange_fund_exposure(ticker: str, report_date: str = "") -> str:
    """返回 ETF 持仓暴露状态；免费结构化源无字段时明确返回 data_unavailable。"""
    return _dump(
        {
            "status": "data_unavailable",
            "data": None,
            "as_of": report_date or None,
            "fetched_at": _now(),
            "freshness": "report_period_required",
            "sources": [],
            "errors": [{"code": "exposure_unavailable", "message": f"{ticker} 缺少可验证的结构化持仓报告"}],
        }
    )


@tool
async def calculate_exchange_fund_tracking_quality(
    ticker: str,
    fund_returns: list[float],
    benchmark_returns: list[float],
    periods_per_year: int = 252,
    sample_start: str = "",
    sample_end: str = "",
    benchmark_source: str = "",
) -> str:
    """用对齐的基金与基准收益率计算年化跟踪差额和跟踪误差。"""
    result = tracking_metrics(
        ticker=ticker,
        fund_returns=fund_returns,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        sample_start=sample_start or None,
        sample_end=sample_end or None,
        benchmark_source=benchmark_source or None,
    )
    status = "available" if result.tracking_error is not None else "data_unavailable"
    return _dump(
        {
            "status": status,
            "data": result.model_dump(mode="json"),
            "as_of": sample_end or None,
            "fetched_at": _now(),
            "freshness": "calculated",
            "sources": ([{"source_id": benchmark_source}] if benchmark_source else []),
            "errors": [] if status == "available" else [{"code": "unaligned_returns", "message": "收益率不足或未对齐"}],
        }
    )


@tool
async def calculate_exchange_fund_liquidity(
    ticker: str,
    average_amount: float,
    bid: float | None = None,
    ask: float | None = None,
    turnover_rate: float | None = None,
    planned_amount: float | None = None,
    as_of: str = "",
) -> str:
    """计算价差、计划金额占比和冲击风险，不生成交易指令。"""
    data = liquidity_metrics(
        ticker=ticker,
        average_amount=average_amount,
        bid=bid,
        ask=ask,
        turnover_rate=turnover_rate,
        planned_amount=planned_amount,
    )
    return _dump(
        {
            "status": "available",
            "data": data.model_dump(mode="json"),
            "as_of": as_of or None,
            "fetched_at": _now(),
            "freshness": "calculated",
            "sources": [],
            "errors": [],
        }
    )


@tool
async def calculate_exchange_fund_premium_discount(
    ticker: str,
    price: float,
    nav_or_iopv: float,
    price_at: str,
    nav_at: str,
    is_qdii: bool = False,
    markets_comparable: bool = True,
) -> str:
    """在价格与 NAV/IOPV 可比时计算折溢价；QDII 时差不可比时禁止精确结论。"""
    comparable = markets_comparable and not (is_qdii and price_at != nav_at)
    reason = "QDII 价格与 NAV/IOPV 时点或交易日不可比" if not comparable else ""
    data = premium_discount(
        ticker=ticker,
        price=price,
        nav_or_iopv=nav_or_iopv,
        price_at=price_at,
        nav_at=nav_at,
        comparable=comparable,
        reason=reason,
    )
    return _dump(
        {
            "status": "available" if data.comparable else "limited",
            "data": data.model_dump(mode="json"),
            "as_of": price_at,
            "fetched_at": _now(),
            "freshness": "calculated",
            "sources": [],
            "errors": [] if data.comparable else [{"code": "nav_not_comparable", "message": reason}],
        }
    )


@tool
async def calculate_exchange_fund_relative_strength(
    ticker: str,
    prices: list[float],
    benchmark_prices: list[float],
    peer_returns: dict[str, float] | None = None,
    as_of: str = "",
) -> str:
    """计算区间收益和相对基准/同类强弱。"""
    if len(prices) < 2 or len(benchmark_prices) < 2 or min(prices[0], benchmark_prices[0]) <= 0:
        return _dump(
            {
                "status": "data_unavailable",
                "data": None,
                "as_of": as_of or None,
                "fetched_at": _now(),
                "freshness": "calculated",
                "sources": [],
                "errors": [{"code": "prices_insufficient", "message": "价格序列不足"}],
            }
        )
    fund_return = prices[-1] / prices[0] - 1
    benchmark_return = benchmark_prices[-1] / benchmark_prices[0] - 1
    peers = peer_returns or {}
    rank = 1 + sum(float(value) > fund_return for value in peers.values())
    return _dump(
        {
            "status": "available",
            "data": {
                "ticker": ticker,
                "return": round(fund_return, 8),
                "benchmark_return": round(benchmark_return, 8),
                "excess_return": round(fund_return - benchmark_return, 8),
                "peer_rank": rank if peers else None,
                "peer_count": len(peers) + 1 if peers else 0,
            },
            "as_of": as_of or None,
            "fetched_at": _now(),
            "freshness": "calculated",
            "sources": [],
            "errors": [],
        }
    )


@tool
async def screen_compare_exchange_funds(
    candidates: list[dict[str, Any]],
    horizon: str = "short_term",
) -> str:
    """按固定权重比较已核验候选；不重新归一化缺失维度，也不生成买入信号。"""
    if horizon not in {"short_term", "medium_term"}:
        raise ValueError("horizon 必须是 short_term 或 medium_term")
    required_ready = all(
        (item.get("verified") is True or item.get("provider_verified") is True)
        and any(item.get(field) is not None for field in ("liquidity", "average_amount", "amount"))
        and item.get("as_of")
        for item in candidates
    )
    scored = [{**item, **score_candidate(dict(item.get("dimensions") or {}), horizon=horizon)} for item in candidates]
    scored.sort(key=lambda item: (-float(item["score"]), str(item.get("ticker") or "")))
    return _dump(
        {
            "status": "available" if required_ready and len(scored) >= 2 else "limited",
            "data": {
                "ranking": scored if required_ready else [],
                "candidates": scored,
                "ranking_is_formal": required_ready and len(scored) >= 2,
                "selection_signal": False,
            },
            "as_of": max((str(item.get("as_of")) for item in candidates if item.get("as_of")), default=None),
            "fetched_at": _now(),
            "freshness": "calculated",
            "sources": [],
            "errors": (
                []
                if required_ready and len(scored) >= 2
                else [{"code": "ranking_requirements_missing", "message": "缺少标的核验、流动性、数据日期或候选不足"}]
            ),
        }
    )


@tool
async def calculate_exchange_fund_portfolio_fit(
    ticker: str,
    candidate_returns: list[float],
    portfolio_returns: list[float],
    current_concentration: float,
    proposed_weight: float,
) -> str:
    """计算候选与组合的相关性、加入后的集中度和边际风险提示。"""
    if len(candidate_returns) != len(portfolio_returns) or len(candidate_returns) < 2:
        correlation = None
    else:
        correlation = statistics.correlation(candidate_returns, portfolio_returns)
        correlation = correlation if math.isfinite(correlation) else None
    new_concentration = min(1.0, max(0.0, current_concentration + proposed_weight))
    return _dump(
        {
            "status": "available" if correlation is not None else "limited",
            "data": {
                "ticker": ticker,
                "correlation": round(correlation, 8) if correlation is not None else None,
                "current_concentration": current_concentration,
                "proposed_weight": proposed_weight,
                "new_concentration": round(new_concentration, 8),
                "concentration_breach": new_concentration > 0.5,
            },
            "as_of": None,
            "fetched_at": _now(),
            "freshness": "calculated",
            "sources": [],
            "errors": []
            if correlation is not None
            else [{"code": "returns_unaligned", "message": "收益序列不足或未对齐"}],
        }
    )


@tool
async def get_exchange_fund_event_risk(
    ticker: str,
    announcements: list[dict[str, Any]],
    as_of: str = "",
) -> str:
    """整理结构化公告中的限购、申赎、指数调仓、QDII 时差和异常事件。"""
    keywords = ("暂停", "限购", "申购", "赎回", "调仓", "溢价", "停牌", "QDII", "额度")
    events = [item for item in announcements if any(word.lower() in str(item).lower() for word in keywords)]
    stale = False
    if as_of:
        try:
            stale = (date.today() - date.fromisoformat(as_of[:10])).days > 1
        except ValueError:
            stale = True
    return _dump(
        {
            "status": "available" if events and not stale else "limited" if announcements else "data_unavailable",
            "data": {"ticker": ticker, "events": events, "announcement_count": len(announcements)},
            "as_of": as_of or None,
            "fetched_at": _now(),
            "freshness": "stale" if stale else "1h",
            "sources": [],
            "errors": []
            if events and not stale
            else [{"code": "event_evidence_limited", "message": "公告缺失、陈旧或未发现关键事件"}],
        }
    )


TOOLS = [
    discover_exchange_fund_candidates,
    get_exchange_fund_profile,
    get_exchange_fund_exposure,
    calculate_exchange_fund_tracking_quality,
    calculate_exchange_fund_liquidity,
    calculate_exchange_fund_premium_discount,
    calculate_exchange_fund_relative_strength,
    screen_compare_exchange_funds,
    calculate_exchange_fund_portfolio_fit,
    get_exchange_fund_event_risk,
]
