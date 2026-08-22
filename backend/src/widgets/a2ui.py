"""A2UI v0.9 message builders for the chat surface.

The agent sends declarative component definitions and a separate data model.
The browser owns the native React widget registry; no agent-provided HTML or
JavaScript is executed on the client.
"""

from __future__ import annotations

import json
import math
from typing import Any
from uuid import uuid4

CATALOG_ID = "https://a-share-agent.local/a2ui/catalog/v0.9"


def _message(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"version": "v0.9", kind: payload}


def _surface(
    surface_id: str,
    components: list[dict[str, Any]],
    data: dict[str, Any],
    include_create: bool = True,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if include_create:
        messages.append(
            _message(
                "createSurface",
                {
                    "surfaceId": surface_id,
                    "catalogId": CATALOG_ID,
                    "sendDataModel": True,
                },
            )
        )
    messages.extend(
        [
            _message(
                "updateComponents",
                {"surfaceId": surface_id, "components": components},
            ),
            _message(
                "updateDataModel",
                {"surfaceId": surface_id, "path": "/", "value": data},
            ),
        ]
    )
    return messages


def _ref(path: str) -> dict[str, str]:
    return {"path": path}


def _text(component_id: str, value: Any, variant: str = "body", **extra: Any) -> dict[str, Any]:
    return {
        "id": component_id,
        "component": "Text",
        "text": value,
        "variant": variant,
        **extra,
    }


def _card_children(title_id: str, title: str, body_ids: list[str]) -> list[dict[str, Any]]:
    return [
        _text(title_id, title, "caption"),
        {"id": "root", "component": "Card", "children": [title_id, *body_ids]},
    ]


def render_asset_card(
    asset_data: dict[str, Any], surface_id: str | None = None, include_create: bool = True
) -> list[dict[str, Any]]:
    surface_id = surface_id or f"asset-card-{uuid4().hex}"
    components = [
        {
            "id": "root",
            "component": "Card",
            "children": ["header", "price", "change", "meta"],
        },
        {"id": "header", "component": "Row", "children": ["ticker", "name"]},
        _text("ticker", _ref("/ticker"), "h3"),
        _text("name", _ref("/name"), "caption"),
        _text("price", _ref("/priceLabel"), "metric"),
        _text("change", _ref("/changeLabel"), "caption", tonePath="/changeTone"),
        {
            "id": "meta",
            "component": "Row",
            "children": ["pe", "pb", "volume"],
        },
        _text("pe", _ref("/peLabel"), "caption"),
        _text("pb", _ref("/pbLabel"), "caption"),
        _text("volume", _ref("/volumeLabel"), "caption"),
    ]
    try:
        price = float(asset_data["price"])
    except (KeyError, TypeError, ValueError):
        return []
    if not math.isfinite(price) or price <= 0:
        return []

    try:
        pct = float(asset_data["pct_chg"])
    except (KeyError, TypeError, ValueError):
        pct = None
    if pct is None or not math.isfinite(pct):
        change_label = "涨跌幅 暂无"
        change_tone = "secondary"
    else:
        change_label = f"{'▲' if pct >= 0 else '▼'} {abs(pct):.2f}%"
        change_tone = "positive" if pct >= 0 else "negative"
    data = {
        "ticker": asset_data.get("ticker", ""),
        "name": asset_data.get("name", ""),
        "priceLabel": f"¥{price:.2f}",
        "changeLabel": change_label,
        "changeTone": change_tone,
        "peLabel": f"PE {asset_data.get('pe', '-')}",
        "pbLabel": f"PB {asset_data.get('pb', '-')}",
        "volumeLabel": f"成交量 {asset_data.get('volume', '-')}",
    }
    return _surface(surface_id, components, data, include_create)


render_stock_card = render_asset_card


def render_agent_pipeline(
    stages: list[dict[str, str]], current_stage: str = "", surface_id: str | None = None, include_create: bool = True
) -> list[dict[str, Any]]:
    surface_id = surface_id or f"agent-pipeline-{uuid4().hex}"
    child_ids: list[str] = ["title", "progress"]
    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "children": child_ids},
        _text("title", "Agent 分析流程", "h4"),
        {"id": "progress", "component": "Progress", "value": _ref("/progress")},
    ]
    data = {
        "progress": 0,
        "current": current_stage,
        "stages": [],
    }
    done_count = 0
    for index, stage in enumerate(stages):
        status = "running" if stage.get("name") == current_stage else stage.get("status", "pending")
        if status in {"done", "complete", "completed"}:
            done_count += 1
        item_id = f"stage-{index}"
        child_ids.append(item_id)
        components.append(
            {
                "id": item_id,
                "component": "PipelineStep",
                "label": stage.get("label", stage.get("name", "")),
                "status": status,
            }
        )
        data["stages"].append({"label": stage.get("label", stage.get("name", "")), "status": status})
    data["progress"] = round(done_count / len(stages) * 100) if stages else 0
    components[0]["children"] = child_ids
    return _surface(surface_id, components, data, include_create)


def render_research_plan(
    plan: dict[str, Any],
    surface_id: str,
    include_create: bool = True,
) -> list[dict[str, Any]]:
    """Render one replaceable public plan snapshot without exposing chain of thought."""
    steps = [item for item in plan.get("steps", []) if isinstance(item, dict)]
    children = ["title", "meta", "objective", "progress"]
    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "children": children},
        _text("title", "市场研究执行计划", "h3"),
        _text("meta", _ref("/meta"), "caption"),
        _text("objective", _ref("/objective"), "body"),
        {"id": "progress", "component": "Progress", "value": _ref("/progress")},
    ]
    for index, step in enumerate(steps):
        component_id = f"step-{index}"
        children.append(component_id)
        recovery = step.get("recovery") if isinstance(step.get("recovery"), dict) else None
        detail = str(step.get("error") or "")
        if recovery:
            recovery_label = {
                "retry": "自动重试",
                "adjust": "反思后调整",
                "abort": "停止重试",
            }.get(str(recovery.get("action")), "失败恢复")
            detail = f"{recovery_label}：{recovery.get('summary') or detail}"
        components.append(
            {
                "id": component_id,
                "component": "PipelineStep",
                "label": str(step.get("title") or step.get("kind") or "研究步骤"),
                "status": str(step.get("status") or "pending"),
                "detail": detail,
            }
        )
    components[0]["children"] = children
    depth_labels = {"quick": "快速", "standard": "标准", "deep": "深度"}
    status_labels = {
        "running": "执行中",
        "completed": "已完成",
        "completed_with_gaps": "已完成（存在数据缺口）",
        "failed": "失败",
        "cancelled": "已取消",
    }
    return _surface(
        surface_id,
        components,
        {
            "objective": str(plan.get("objective") or "市场研究"),
            "progress": int(plan.get("progress") or 0),
            "meta": (
                f"{depth_labels.get(str(plan.get('depth')), str(plan.get('depth') or '标准'))} · "
                f"Revision {int(plan.get('revision') or 1)} · "
                f"{status_labels.get(str(plan.get('status')), str(plan.get('status') or '执行中'))} · "
                f"进度 {int(plan.get('progress') or 0)}%"
            ),
            "steps": steps,
        },
        include_create,
    )


def render_signal_gauge(
    scores: dict[str, float], surface_id: str | None = None, include_create: bool = True
) -> list[dict[str, Any]]:
    surface_id = surface_id or f"signal-gauge-{uuid4().hex}"
    fields = [
        ("technical", "技术面", "technical_score"),
        ("sentiment", "情绪面", "sentiment_score"),
        ("fundamental", "基本面", "fundamental_score"),
        ("market", "市场环境", "market_regime_score"),
    ]
    children = ["title", "overall", "bars"]
    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "children": children},
        _text("title", "信号归因", "h4"),
        _text("overall", _ref("/overallLabel"), "metric"),
        {"id": "bars", "component": "Column", "children": []},
    ]
    bars: list[dict[str, Any]] = []
    for component_id, label, key in fields:
        components[3]["children"].append(component_id)
        components.append(
            {
                "id": component_id,
                "component": "ScoreBar",
                "label": label,
                "value": _ref(f"/{key}"),
            }
        )
        bars.append({"label": label, "value": float(scores.get(key, 0) or 0)})
    overall = sum(item["value"] for item in bars) / len(bars) if bars else 0
    data = {**{key: float(scores.get(key, 0) or 0) for _, _, key in fields}, "overallLabel": f"综合信号 {overall:+.0f}"}
    return _surface(surface_id, components, data, include_create)


def render_decision_dashboard(
    dashboard: dict[str, Any] | None, surface_id: str | None = None, include_create: bool = True
) -> list[dict[str, Any]]:
    surface_id = surface_id or f"decision-dashboard-{uuid4().hex}"
    dashboard = dashboard or {}
    cc = dashboard.get("core_conclusion", {})
    dp = dashboard.get("data_perspective", {})
    intel = dashboard.get("intelligence", {})
    bp = dashboard.get("battle_plan", {})
    ph = dashboard.get("phase_decision", {})

    components: list[dict[str, Any]] = [
        {
            "id": "root",
            "component": "Card",
            "children": ["title", "signal", "summary", "perspective", "battle", "intelligence", "phase"],
        },
        _text("title", "综合决策面板", "h3"),
        {"id": "signal", "component": "Badge", "text": _ref("/signalLabel"), "tone": _ref("/signalTone")},
        _text("summary", _ref("/summary"), "body"),
        {
            "id": "perspective",
            "component": "Section",
            "title": "数据视角",
            "children": ["trend", "price", "volume", "chip"],
        },
        _text("trend", _ref("/trend"), "body"),
        _text("price", _ref("/price"), "body"),
        _text("volume", _ref("/volume"), "body"),
        _text("chip", _ref("/chip"), "body"),
        {
            "id": "battle",
            "component": "Section",
            "title": "交易计划",
            "children": ["entry", "stop", "target", "position", "actions"],
        },
        _text("entry", _ref("/entry"), "metric"),
        _text("stop", _ref("/stop"), "metric", tone="negative"),
        _text("target", _ref("/target"), "metric", tone="positive"),
        _text("position", _ref("/position"), "body"),
        {"id": "actions", "component": "List", "items": _ref("/actions")},
        {
            "id": "intelligence",
            "component": "Section",
            "title": "情报与风险",
            "children": ["news", "alerts", "catalysts"],
        },
        {"id": "news", "component": "List", "title": "最新新闻", "items": _ref("/news")},
        {"id": "alerts", "component": "List", "title": "风险警报", "items": _ref("/alerts")},
        {"id": "catalysts", "component": "List", "title": "积极催化", "items": _ref("/catalysts")},
        {
            "id": "phase",
            "component": "Section",
            "title": "盘前 / 盘中 / 盘后",
            "children": ["preMarket", "intraday", "postMarket"],
        },
        _text("preMarket", _ref("/preMarket"), "body"),
        _text("intraday", _ref("/intraday"), "body"),
        _text("postMarket", _ref("/postMarket"), "body"),
    ]
    confidence = float(cc.get("confidence", 0) or 0)
    actions = list(bp.get("action_items", []) or [])
    for label, key in (
        ("入场依据", "entry_explanation"),
        ("止损依据", "stop_loss_explanation"),
        ("止盈依据", "take_profit_explanation"),
    ):
        if bp.get(key):
            actions.append(f"{label}：{bp[key]}")
    data = {
        "signalLabel": {
            "strong_buy": "强烈买入",
            "buy": "买入",
            "watch": "观望",
            "reduce": "减仓",
            "sell": "卖出",
            "strong_sell": "强烈卖出",
        }.get(cc.get("signal", "watch"), cc.get("signal", "观望")),
        "signalTone": cc.get("signal", "watch"),
        "summary": (
            f"{cc.get('one_line_summary', '暂无结论')}（置信度 {confidence:.0%}）\n"
            f"仓位建议：{cc.get('position_advice', '暂无')}"
        ),
        "trend": f"趋势：{dp.get('trend_status', '暂无')}",
        "price": f"价格位置：{dp.get('price_position', '暂无')}",
        "volume": f"量能：{dp.get('volume_analysis', '暂无')}",
        "chip": f"筹码：{dp.get('chip_structure', '暂无')}",
        "entry": f"入场价：{bp.get('entry_price', '暂无')}",
        "stop": f"止损价：{bp.get('stop_loss', '暂无')}",
        "target": f"止盈价：{bp.get('take_profit', '暂无')}",
        "position": f"仓位策略：{bp.get('position_strategy', '暂无')}",
        "actions": actions or ["暂无具体行动项"],
        "news": intel.get("latest_news", [])[:5] or ["暂无新闻"],
        "alerts": intel.get("risk_alerts", [])[:5] or ["暂无风险警报"],
        "catalysts": intel.get("positive_catalysts", [])[:5] or ["暂无积极催化"],
        "preMarket": f"盘前：{ph.get('pre_market', '暂无')}",
        "intraday": f"盘中：{ph.get('intraday', '暂无')}",
        "postMarket": f"盘后：{ph.get('post_market', '暂无')}",
    }
    return _surface(surface_id, components, data, include_create)


def render_strategy_selector(
    strategies: list[dict[str, Any]], surface_id: str | None = None, include_create: bool = True
) -> list[dict[str, Any]]:
    surface_id = surface_id or f"strategies-{uuid4().hex}"
    items = [
        {
            "name": s.get("display_name", s.get("name", "")),
            "description": s.get("description", ""),
            "active": bool(s.get("default_active")),
        }
        for s in strategies
    ]
    components = [
        {"id": "root", "component": "Card", "children": ["title", "list"]},
        _text("title", "可用交易策略", "h3"),
        {"id": "list", "component": "List", "items": _ref("/items"), "itemTemplate": "strategyItem"},
        {
            "id": "strategyItem",
            "component": "StrategyItem",
            "name": _ref("name"),
            "description": _ref("description"),
            "active": _ref("active"),
        },
    ]
    return _surface(surface_id, components, {"items": items}, include_create)


def render_breaker_status(
    breakers: dict[str, str], surface_id: str | None = None, include_create: bool = True
) -> list[dict[str, Any]]:
    surface_id = surface_id or f"breakers-{uuid4().hex}"
    items = [{"name": name, "state": state} for name, state in breakers.items()]
    components = [
        {"id": "root", "component": "Card", "children": ["title", "list"]},
        _text("title", "数据源熔断器状态", "h3"),
        {"id": "list", "component": "List", "items": _ref("/items"), "itemTemplate": "breakerItem"},
        {"id": "breakerItem", "component": "StatusItem", "label": _ref("name"), "status": _ref("state")},
    ]
    return _surface(surface_id, components, {"items": items}, include_create)


def render_mini_chart(
    prices: list[float], surface_id: str | None = None, include_create: bool = True
) -> list[dict[str, Any]]:
    surface_id = surface_id or f"mini-chart-{uuid4().hex}"
    components = [
        {"id": "root", "component": "Card", "children": ["title", "chart"]},
        _text("title", "近期走势", "h4"),
        {"id": "chart", "component": "Sparkline", "values": _ref("/prices")},
    ]
    return _surface(surface_id, components, {"prices": [float(price) for price in prices]}, include_create)


def render_markdown(text: str, surface_id: str | None = None) -> list[dict[str, Any]]:
    """Render an agent explanation through the native Markdown component."""
    surface_id = surface_id or f"markdown-{uuid4().hex}"
    return _surface(
        surface_id,
        [{"id": "root", "component": "Markdown", "text": {"path": "/text"}}],
        {"text": text},
    )


def render_activity(
    name: str,
    status: str = "completed",
    surface_id: str | None = None,
    include_create: bool = True,
    error: str | None = None,
) -> list[dict[str, Any]]:
    """Render a tool invocation as a compact, structured activity item."""
    surface_id = surface_id or f"activity-{uuid4().hex}"
    return _surface(
        surface_id,
        [
            {
                "id": "root",
                "component": "Activity",
                "name": _ref("/name"),
                "status": _ref("/status"),
                "error": _ref("/error"),
            }
        ],
        {"name": name, "status": status, "error": error or ""},
        include_create,
    )


def _number_label(value: Any, *, suffix: str = "") -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{value}{suffix}"
    if number.is_integer():
        return f"{int(number):,}{suffix}"
    return f"{number:,.4f}{suffix}"


def _percent_label(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value) * 100:+.2f}%"
    except (TypeError, ValueError):
        return str(value)


def render_analysis_result(
    payload: dict[str, Any],
    surface_id: str | None = None,
) -> list[dict[str, Any]]:
    """Render the structured analysis result returned by the Agent tool."""
    surface_id = surface_id or f"analysis-result-{uuid4().hex}"
    ticker = payload.get("ticker", "")
    asset_type = str(payload.get("asset_type", "stock")).upper()
    decision = str(payload.get("decision", "hold"))
    decision_label = {"buy": "买入", "sell": "卖出", "hold": "持有"}.get(decision, decision)
    plan = payload.get("plan") or {}
    position_size = plan.get("position_size", payload.get("position_size"))
    take_profit = plan.get("take_profit", payload.get("take_profit", payload.get("target_price")))
    dashboard = payload.get("dashboard") or {}
    core = dashboard.get("core_conclusion") or {}
    summary = core.get("one_line_summary") or payload.get("reasoning") or "暂无结论"
    components: list[dict[str, Any]] = [
        {
            "id": "root",
            "component": "Card",
            "children": ["header", "summary", "metrics", "notice"],
        },
        {
            "id": "header",
            "component": "Row",
            "children": ["title", "decision"],
        },
        _text("title", f"Agent Analysis · {ticker} · {asset_type}", "h3"),
        {"id": "decision", "component": "Badge", "text": _ref("/decisionLabel"), "tone": _ref("/decisionTone")},
        _text("summary", _ref("/summary"), "body"),
        {
            "id": "metrics",
            "component": "Row",
            "children": ["confidence", "entry", "stop", "target", "position"],
        },
        _text("confidence", _ref("/confidence"), "metric"),
        _text("entry", _ref("/entry"), "caption"),
        _text("stop", _ref("/stop"), "caption", tone="negative"),
        _text("target", _ref("/target"), "caption", tone="positive"),
        _text("position", _ref("/position"), "caption"),
        _text("notice", "研究结果仅用于短中期研究和纸面交易，不代表真实交易或收益承诺。", "caption"),
    ]
    messages = _surface(
        surface_id,
        components,
        {
            "decisionLabel": decision_label,
            "decisionTone": decision,
            "summary": summary,
            "confidence": f"置信度 {float(payload.get('confidence', 0) or 0):.0%}",
            "entry": f"入场：{_number_label(plan.get('entry_price', payload.get('entry_price')))}",
            "stop": f"止损：{_number_label(plan.get('stop_loss', payload.get('stop_loss')))}",
            "target": f"止盈：{_number_label(take_profit)}",
            "position": f"仓位：{_percent_label(position_size)}",
        },
    )
    if dashboard:
        messages.extend(render_decision_dashboard(dashboard, f"{surface_id}-dashboard"))
    return messages


def render_backtest_result(
    payload: dict[str, Any],
    surface_id: str | None = None,
) -> list[dict[str, Any]]:
    """Render a backtest or Agent-designed experiment as an inline report."""
    surface_id = surface_id or f"backtest-result-{uuid4().hex}"
    result = payload.get("result") or {}
    is_experiment = payload.get("data_type") == "backtest_experiment"
    title = "Agent 回测实验" if is_experiment else "历史回测结果"
    ticker = result.get("ticker") or payload.get("ticker") or "股票池"
    equity_curve = result.get("equity_curve") or []
    points = [
        {"label": str(item.get("date", "")), "value": float(item.get("value", 0) or 0)}
        for item in equity_curve
        if isinstance(item, dict) and item.get("date") is not None
    ]
    trades = result.get("trades") or []
    rows = [
        {
            "date": trade.get("date", ""),
            "action": "买入" if trade.get("action") == "buy" else "卖出",
            "ticker": trade.get("ticker", ""),
            "shares": trade.get("shares", 0),
            "price": trade.get("price", 0),
            "amount": trade.get("amount", 0),
        }
        for trade in trades[-20:]
        if isinstance(trade, dict)
    ]
    children = ["title", "meta", "summary"]
    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "children": children},
        _text("title", f"{title} · {ticker}", "h3"),
        _text(
            "meta",
            (
                f"{result.get('start_date', '—')} 至 {result.get('end_date', '—')} · "
                f"初始资金 ¥{_number_label(result.get('initial_capital'))}"
            ),
            "caption",
        ),
        {
            "id": "summary",
            "component": "Row",
            "children": ["final", "return", "drawdown", "sharpe", "winrate", "trades"],
        },
        _text("final", f"最终市值 ¥{_number_label(result.get('final_value'))}", "caption"),
        _text("return", f"收益 {_percent_label(result.get('total_return'))}", "caption"),
        _text("drawdown", f"最大回撤 {_percent_label(result.get('max_drawdown'))}", "caption", tone="negative"),
        _text("sharpe", f"Sharpe {_number_label(result.get('sharpe_ratio'))}", "caption"),
        _text("winrate", f"胜率 {_percent_label(result.get('win_rate'))}", "caption"),
        _text("trades", f"交易 {result.get('total_trades', len(trades))} 次", "caption"),
    ]
    if is_experiment:
        spec = payload.get("strategy_spec") or {}
        children.insert(2, "strategy")
        components.insert(
            3,
            _text(
                "strategy",
                f"策略：{spec.get('name', 'Agent 策略')} · {spec.get('description', '')}",
                "body",
            ),
        )
    if points:
        children.append("chart")
        components.append(
            {
                "id": "chart",
                "component": "LineChart",
                "points": _ref("/points"),
                "ariaLabel": "回测资金曲线",
            }
        )
    if rows:
        children.append("trades")
        components.extend(
            [
                {
                    "id": "trades",
                    "component": "Collapsible",
                    "title": "最近交易记录",
                    "defaultExpanded": False,
                    "children": ["tradeTable"],
                },
                {
                    "id": "tradeTable",
                    "component": "DataTable",
                    "columns": [
                        {"key": "date", "label": "日期"},
                        {"key": "action", "label": "方向"},
                        {"key": "ticker", "label": "代码"},
                        {"key": "shares", "label": "数量"},
                        {"key": "price", "label": "价格"},
                        {"key": "amount", "label": "金额"},
                    ],
                    "rows": _ref("/trades"),
                },
            ]
        )
    if result.get("error"):
        children.append("error")
        components.append(_text("error", f"回测错误：{result['error']}", "body", tone="negative"))
    children.append("notice")
    components.append(_text("notice", "回测基于历史数据，仅用于策略研究，不代表未来表现。", "caption"))
    return _surface(surface_id, components, {"points": points, "trades": rows})


def render_sandbox_strategy_candidate(
    payload: dict[str, Any],
    surface_id: str | None = None,
) -> list[dict[str, Any]]:
    """Render generated code, sandbox checks, and its trusted-engine backtest."""
    surface_id = surface_id or f"sandbox-candidate-{uuid4().hex}"
    result = dict(payload.get("result") or {})
    backtest = dict(result.get("backtest") or {})
    strategy = dict(payload.get("strategy_spec") or {})
    validation = dict(payload.get("validation") or {})
    source_code = str(payload.get("source_code") or "# 未返回脚本源码")
    promotion_eligible = result.get("promotion_eligible") is True
    validation_passed = validation.get("passed") is True

    status = str(payload.get("status") or "draft")
    status_label = {
        "draft": "草稿",
        "validated": "已验证",
        "approved": "已审核",
        "rejected": "已拒绝",
        "deployed": "已部署",
    }.get(status, status)
    validation_label = "沙盒验证通过" if validation_passed else "沙盒验证未通过"
    eligibility_label = "可提交人工审核" if promotion_eligible else "仅限研究"

    check_labels = {
        "ast_parse": "Python 语法可解析",
        "source_size": "源码大小符合限制",
        "allowed_imports": "仅使用允许的依赖",
        "denied_names": "未使用危险内置函数",
        "denied_attributes": "未调用危险属性",
        "function_contract": "目标仓位函数契约正确",
        "output_length": "输出长度与行情一致",
        "binary_positions": "仓位输出仅包含 0/1",
        "deterministic_output": "重复执行结果一致",
        "causal_output": "未使用未来数据",
        "dsl_signal_equivalent": "代码信号与 StrategySpec 一致",
        "minimum_history_5y": "有效历史区间不少于 5 年",
        "strategy_spec_valid": "结构化 StrategySpec 可执行",
    }
    validation_rows = []
    for category, checks in (
        ("静态检查", validation.get("static_checks") or {}),
        ("输出检查", validation.get("output_checks") or {}),
    ):
        for name, passed in checks.items():
            validation_rows.append(
                {
                    "category": category,
                    "check": check_labels.get(str(name), str(name)),
                    "status": "通过" if passed else "未通过",
                }
            )
    validation_rows.extend(
        [
            {
                "category": "行为检查",
                "check": "重复执行结果一致",
                "status": "通过" if validation.get("deterministic") is True else "未通过",
            },
            {
                "category": "行为检查",
                "check": "未使用未来数据",
                "status": "通过" if validation.get("causal") is True else "未通过",
            },
        ]
    )
    errors = [str(item) for item in validation.get("errors") or []]

    rules = []
    for direction, conditions in (
        ("入场", strategy.get("entry_conditions") or []),
        ("退出", strategy.get("exit_conditions") or []),
    ):
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            value = condition.get("value")
            rules.append(
                {
                    "direction": direction,
                    "indicator": condition.get("indicator", ""),
                    "operator": condition.get("operator", ""),
                    "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value,
                    "window": condition.get("window") or "—",
                }
            )

    points = [
        {"label": str(item.get("date", "")), "value": float(item.get("value", 0) or 0)}
        for item in backtest.get("equity_curve") or []
        if isinstance(item, dict) and item.get("date") is not None
    ]
    trades = [
        {
            "date": trade.get("date", ""),
            "action": "买入" if str(trade.get("action", "")).lower() == "buy" else "卖出",
            "shares": trade.get("shares", 0),
            "price": trade.get("price", 0),
            "amount": trade.get("amount", 0),
        }
        for trade in (backtest.get("trades") or [])[-20:]
        if isinstance(trade, dict)
    ]

    root_children = ["header", "meta", "description", "validation-summary"]
    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "children": root_children},
        {"id": "header", "component": "Row", "children": ["title", "status", "eligibility"]},
        _text("title", f"代码策略候选 · {payload.get('name') or strategy.get('name', '未命名策略')}", "h3"),
        {
            "id": "status",
            "component": "Badge",
            "text": f"状态：{status_label}",
            "tone": "buy" if validation_passed else "sell",
        },
        {
            "id": "eligibility",
            "component": "Badge",
            "text": eligibility_label,
            "tone": "buy" if promotion_eligible else "hold",
        },
        _text(
            "meta",
            (
                f"{payload.get('candidate_id', '—')} · {payload.get('ticker', '—')} · "
                f"{str(payload.get('asset_type', '')).upper()} · v{payload.get('version', '—')}"
            ),
            "caption",
        ),
        _text("description", strategy.get("description") or "Agent 未提供策略说明。", "body"),
        _text(
            "validation-summary",
            f"{validation_label} · {eligibility_label}",
            "body",
            tone="positive" if validation_passed else "negative",
        ),
    ]

    if backtest.get("final_value") is not None:
        root_children.append("performance")
        components.extend(
            [
                {
                    "id": "performance",
                    "component": "Section",
                    "title": "可信回测引擎结果",
                    "children": ["performance-row"],
                },
                {
                    "id": "performance-row",
                    "component": "Row",
                    "children": ["final", "return", "benchmark", "drawdown", "sharpe", "trade-count"],
                },
                _text("final", f"最终市值 ¥{_number_label(backtest.get('final_value'))}", "caption"),
                _text("return", f"策略收益 {_percent_label(backtest.get('total_return'))}", "caption"),
                _text("benchmark", f"买入持有 {_percent_label(backtest.get('buy_hold_return'))}", "caption"),
                _text(
                    "drawdown",
                    f"最大回撤 {_percent_label(backtest.get('max_drawdown'))}",
                    "caption",
                    tone="negative",
                ),
                _text("sharpe", f"Sharpe {_number_label(backtest.get('sharpe_ratio'))}", "caption"),
                _text("trade-count", f"交易 {backtest.get('total_trades', len(trades))} 次", "caption"),
            ]
        )
    elif not validation_passed:
        root_children.append("backtest-unavailable")
        components.append(
            _text("backtest-unavailable", "脚本未通过验证，因此没有执行绩效回测。", "body", tone="negative")
        )

    if points:
        root_children.append("chart")
        components.append(
            {"id": "chart", "component": "LineChart", "points": _ref("/points"), "ariaLabel": "代码策略资金曲线"}
        )
    if trades:
        root_children.append("trades")
        components.extend(
            [
                {
                    "id": "trades",
                    "component": "Collapsible",
                    "title": "最近交易记录",
                    "defaultExpanded": False,
                    "children": ["trade-table"],
                },
                {
                    "id": "trade-table",
                    "component": "DataTable",
                    "columns": [
                        {"key": "date", "label": "日期"},
                        {"key": "action", "label": "方向"},
                        {"key": "shares", "label": "数量"},
                        {"key": "price", "label": "价格"},
                        {"key": "amount", "label": "金额"},
                    ],
                    "rows": _ref("/trades"),
                },
            ]
        )
    if rules:
        root_children.append("rules")
        components.extend(
            [
                {
                    "id": "rules",
                    "component": "Collapsible",
                    "title": "结构化交易规则",
                    "defaultExpanded": False,
                    "children": ["rule-table"],
                },
                {
                    "id": "rule-table",
                    "component": "DataTable",
                    "columns": [
                        {"key": "direction", "label": "阶段"},
                        {"key": "indicator", "label": "指标"},
                        {"key": "operator", "label": "关系"},
                        {"key": "value", "label": "阈值"},
                        {"key": "window", "label": "窗口"},
                    ],
                    "rows": _ref("/rules"),
                },
            ]
        )

    validation_children = ["validation-table"]
    if errors:
        validation_children.append("validation-errors")
    root_children.append("validation")
    components.extend(
        [
            {
                "id": "validation",
                "component": "Collapsible",
                "title": (
                    f"沙盒验证明细（{sum(row['status'] == '通过' for row in validation_rows)}"
                    f"/{len(validation_rows)}）"
                ),
                "defaultExpanded": not validation_passed,
                "children": validation_children,
            },
            {
                "id": "validation-table",
                "component": "DataTable",
                "columns": [
                    {"key": "category", "label": "类别"},
                    {"key": "check", "label": "检查项"},
                    {"key": "status", "label": "结果"},
                ],
                "rows": _ref("/validationRows"),
            },
        ]
    )
    if errors:
        components.append(
            {"id": "validation-errors", "component": "List", "title": "失败原因", "items": _ref("/errors")}
        )

    root_children.extend(["code", "notice"])
    components.extend(
        [
            {
                "id": "code",
                "component": "Collapsible",
                "title": "Agent 生成的 Python 信号脚本",
                "defaultExpanded": False,
                "children": ["source-hash", "source-code"],
            },
            _text("source-hash", f"SHA-256：{payload.get('source_sha256', '—')}", "caption"),
            {"id": "source-code", "component": "CodeBlock", "language": "python", "code": _ref("/sourceCode")},
            _text(
                "notice",
                "脚本仅在受限沙盒中生成目标仓位；成交、费用和绩效统一由可信回测引擎计算。仅用于研究和模拟盘。",
                "caption",
            ),
        ]
    )
    return _surface(
        surface_id,
        components,
        {
            "points": points,
            "trades": trades,
            "rules": rules,
            "validationRows": validation_rows,
            "errors": errors,
            "sourceCode": source_code,
        },
    )


def render_tool_result(tool_name: str, raw_result: str, surface_id: str | None = None) -> list[dict[str, Any]] | None:
    """Turn known tool payloads into native A2UI result surfaces."""
    try:
        payload = json.loads(raw_result)
    except (TypeError, json.JSONDecodeError):
        return None

    if tool_name == "get_realtime_quote":
        quote = dict(payload.get("quote") or {})
        quote["ticker"] = payload.get("ticker", quote.get("ticker", ""))
        return render_asset_card(quote, surface_id)

    if tool_name == "run_fund_or_stock_analysis":
        if payload.get("error"):
            return None
        return render_analysis_result(payload, surface_id)

    if tool_name in {"run_backtest", "design_and_run_backtest"}:
        return render_backtest_result(payload, surface_id)

    if tool_name == "design_and_run_sandbox_strategy":
        return render_sandbox_strategy_candidate(payload, surface_id)

    if tool_name == "compare_strategy_backtests":
        rows = [
            {
                "strategy": item.get("display_name") or item.get("strategy_name", ""),
                "return": _percent_label(item.get("total_return")),
                "annualized": _percent_label(item.get("annualized_return")),
                "excess": _percent_label(item.get("excess_return")),
                "drawdown": _percent_label(item.get("max_drawdown")),
                "sharpe": _number_label(item.get("sharpe_ratio")),
                "sortino": _number_label(item.get("sortino_ratio")),
                "winRate": _percent_label(item.get("win_rate")),
                "trades": item.get("total_trades", 0),
            }
            for item in payload.get("comparisons", [])
            if isinstance(item, dict)
        ]
        surface_id = surface_id or f"strategy-comparison-{uuid4().hex}"
        acceptance = payload.get("acceptance") or {}
        acceptance_label = (
            "验收通过"
            if acceptance.get("satisfied") is True
            else "验收未通过：" + "、".join(acceptance.get("missing") or ["结果不完整"])
        )
        by_name = {
            item.get("strategy_name"): item
            for item in payload.get("comparisons", [])
            if isinstance(item, dict)
        }
        ranked_items = [by_name[name] for name in payload.get("ranking", []) if name in by_name][:3]
        chart_components = []
        equity_children = []
        drawdown_children = []
        equity_curves = []
        drawdown_curves = []
        for index, item in enumerate(ranked_items):
            label = item.get("display_name") or item.get("strategy_name", "")
            equity_id = f"equity-{index}"
            drawdown_id = f"drawdown-{index}"
            equity_children.extend([f"equity-title-{index}", equity_id])
            drawdown_children.extend([f"drawdown-title-{index}", drawdown_id])
            equity_curves.append(
                [
                    {"label": point.get("date", ""), "value": point.get("value", 0)}
                    for point in item.get("equity_curve", [])
                    if isinstance(point, dict)
                ]
            )
            drawdown_curves.append(
                [
                    {"label": point.get("date", ""), "value": point.get("value", 0)}
                    for point in item.get("drawdown_curve", [])
                    if isinstance(point, dict)
                ]
            )
            chart_components.extend(
                [
                    _text(f"equity-title-{index}", str(label), "caption"),
                    {
                        "id": equity_id,
                        "component": "LineChart",
                        "points": _ref(f"/equityCurves/{index}"),
                        "ariaLabel": f"{label} 资金曲线",
                    },
                    _text(f"drawdown-title-{index}", str(label), "caption"),
                    {
                        "id": drawdown_id,
                        "component": "LineChart",
                        "points": _ref(f"/drawdownCurves/{index}"),
                        "ariaLabel": f"{label} 回撤曲线",
                    },
                ]
            )
        root_children = ["title", "meta", "table"]
        if equity_children:
            root_children.extend(["curves", "drawdowns"])
        root_children.append("notice")
        return _surface(
            surface_id,
            [
                {"id": "root", "component": "Card", "children": root_children},
                _text("title", f"{payload.get('ticker', '')} 多策略回测对比", "h3"),
                _text(
                    "meta",
                    (
                        f"{payload.get('actual_start_date', payload.get('start_date', ''))} 至 "
                        f"{payload.get('actual_end_date', payload.get('end_date', ''))} · "
                        f"{payload.get('strategy_count', len(rows))} 个策略 · "
                        f"按{payload.get('ranking_label', '总收益率')}排序 · {acceptance_label}"
                    ),
                    "caption",
                ),
                {
                    "id": "table",
                    "component": "DataTable",
                    "columns": [
                        {"key": "strategy", "label": "策略"},
                        {"key": "return", "label": "收益率"},
                        {"key": "annualized", "label": "年化"},
                        {"key": "excess", "label": "超额"},
                        {"key": "drawdown", "label": "最大回撤"},
                        {"key": "sharpe", "label": "夏普"},
                        {"key": "sortino", "label": "Sortino"},
                        {"key": "winRate", "label": "胜率"},
                        {"key": "trades", "label": "交易次数"},
                    ],
                    "rows": _ref("/rows"),
                },
                *(
                    [
                        {
                            "id": "curves",
                            "component": "Collapsible",
                            "title": "排名前三资金曲线",
                            "defaultExpanded": True,
                            "children": equity_children,
                        },
                        {
                            "id": "drawdowns",
                            "component": "Collapsible",
                            "title": "排名前三回撤曲线",
                            "defaultExpanded": False,
                            "children": drawdown_children,
                        },
                        *chart_components,
                    ]
                    if equity_children
                    else []
                ),
                _text("notice", "回测基于历史数据，仅用于策略研究，不代表未来表现。", "caption"),
            ],
            {"rows": rows, "equityCurves": equity_curves, "drawdownCurves": drawdown_curves},
        )

    if tool_name in {"search_web", "search_web_ddgs"}:
        items = []
        for item in payload.get("results", []) if isinstance(payload.get("results"), list) else []:
            if not isinstance(item, dict) or not str(item.get("link", "")).startswith(("http://", "https://")):
                continue
            items.append(
                {
                    "title": str(item.get("title", "")),
                    "link": str(item.get("link", "")),
                    "snippet": str(item.get("snippet", "")),
                    "source": str(item.get("source", "")),
                    "date": str(item.get("date", "")),
                }
            )
        surface_id = surface_id or f"search-{uuid4().hex}"
        components = [
            {
                "id": "root",
                "component": "Collapsible",
                "title": f"搜索结果（{len(items)} 条）",
                "defaultExpanded": len(items) <= 4,
                "children": ["list"],
            },
            {
                "id": "list",
                "component": "List",
                "items": _ref("/items"),
                "itemTemplate": "searchItem",
            },
            {
                "id": "searchItem",
                "component": "SearchResultItem",
                "title": _ref("title"),
                "link": _ref("link"),
                "snippet": _ref("snippet"),
                "source": _ref("source"),
                "date": _ref("date"),
            },
        ]
        return _surface(surface_id, components, {"items": items})

    if tool_name == "compare_quotes":
        rows = []
        for item in payload.get("quotes", []):
            quote = item.get("quote") or {}
            rows.append(
                {
                    "ticker": item.get("ticker", ""),
                    "name": quote.get("name", ""),
                    "price": quote.get("price", "-"),
                    "change": quote.get("pct_chg", 0),
                    "amount": quote.get("amount", quote.get("volume", "-")),
                    "turnover": quote.get("turnover", "-"),
                }
            )
        surface_id = surface_id or f"compare-{uuid4().hex}"
        components = [
            {"id": "root", "component": "Card", "children": ["title", "table"]},
            _text("title", "行情对比", "h3"),
            {
                "id": "table",
                "component": "DataTable",
                "columns": [
                    {"key": "ticker", "label": "代码"},
                    {"key": "name", "label": "名称"},
                    {"key": "price", "label": "最新价"},
                    {"key": "change", "label": "涨跌幅"},
                    {"key": "amount", "label": "成交额"},
                    {"key": "turnover", "label": "换手率"},
                ],
                "rows": _ref("/rows"),
            },
        ]
        return _surface(surface_id, components, {"rows": rows})

    if tool_name == "get_historical_prices":
        if payload.get("data_type") == "price_history_collection":
            items = [item for item in payload.get("items", []) if isinstance(item, dict)]
            if not items:
                return None
            if len(items) == 1:
                return render_tool_result(tool_name, json.dumps(items[0], ensure_ascii=False, default=str), surface_id)
            base_surface_id = surface_id or f"history-collection-{uuid4().hex}"
            messages: list[dict[str, Any]] = []
            for index, item in enumerate(items):
                rendered = render_tool_result(
                    tool_name,
                    json.dumps(item, ensure_ascii=False, default=str),
                    f"{base_surface_id}-{index}",
                )
                if rendered:
                    messages.extend(rendered)
            return messages or None
        records = payload.get("history", [])
        points = []
        for item in records if isinstance(records, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                close = float(item.get("close"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(close) or close <= 0:
                continue
            points.append(
                {
                    "label": str(item.get("date") or item.get("trade_date") or ""),
                    "value": close,
                }
            )
        closes = [point["value"] for point in points]
        surface_id = surface_id or f"history-{uuid4().hex}"
        components = [
            {"id": "root", "component": "Card", "children": ["title", "chart", "table"]},
            _text("title", f"{payload.get('ticker', '')} 历史走势", "h3"),
            {
                "id": "chart",
                "component": "LineChart",
                "points": _ref("/points"),
                "ariaLabel": "历史收盘价走势",
            },
            {
                "id": "table",
                "component": "DataTable",
                "columns": [
                    {"key": "date", "label": "日期"},
                    {"key": "close", "label": "收盘"},
                    {"key": "pct_chg", "label": "涨跌幅"},
                    {"key": "volume", "label": "成交量"},
                ],
                "rows": _ref("/rows"),
            },
        ]
        return _surface(surface_id, components, {"prices": closes, "points": points, "rows": records[-12:]})

    if tool_name == "get_latest_news":
        surface_id = surface_id or f"news-{uuid4().hex}"
        news = payload.get("news", [])
        items = [
            " · ".join(
                str(part) for part in (item.get("date", ""), item.get("source", ""), item.get("title", "")) if part
            )
            for item in news
        ] or ["暂无最新新闻"]
        components = [
            {"id": "root", "component": "Card", "children": ["title", "list"]},
            _text("title", f"{payload.get('ticker', '')} 最新新闻", "h3"),
            {"id": "list", "component": "List", "items": _ref("/items")},
        ]
        return _surface(surface_id, components, {"items": items})

    if tool_name == "list_trading_strategies":
        return render_strategy_selector(payload if isinstance(payload, list) else [], surface_id)

    if tool_name == "screen_assets":
        surface_id = surface_id or f"screen-{uuid4().hex}"
        rows = payload.get("results", [])
        columns = [
            {"key": "ticker", "label": "代码"},
            {"key": "name", "label": "名称"},
            {"key": "price", "label": "价格"},
            {"key": "pct_chg", "label": "涨跌幅"},
            {"key": "amount", "label": "成交额"},
            {"key": "turnover", "label": "换手率"},
        ]
        components = [
            {"id": "root", "component": "Card", "children": ["title", "table"]},
            _text("title", f"筛选结果（{payload.get('count', len(rows))} 条）", "h3"),
            {"id": "table", "component": "DataTable", "columns": columns, "rows": _ref("/rows")},
        ]
        return _surface(surface_id, components, {"rows": rows})

    if tool_name == "get_simulation_portfolio":
        portfolio = payload.get("portfolio") or {}
        surface_id = surface_id or f"simulation-portfolio-{uuid4().hex}"
        rows = [
            {
                "ticker": item.get("ticker", ""),
                "asset_type": item.get("asset_type", ""),
                "shares": item.get("shares", 0),
                "avg_cost": item.get("avg_cost", 0),
                "current_price": item.get("current_price", 0),
                "pnl_pct": item.get("pnl_pct", 0),
            }
            for item in portfolio.get("positions", [])
        ]
        components = [
            {"id": "root", "component": "Card", "children": ["title", "summary", "table"]},
            _text("title", f"模拟盘 · {payload.get('account_id', portfolio.get('account_id', 'default'))}", "h3"),
            {
                "id": "summary",
                "component": "Row",
                "children": ["cash", "value", "pnl"],
            },
            {"id": "cash", "component": "Badge", "label": "现金", "value": _ref("/summary/cash")},
            {"id": "value", "component": "Badge", "label": "总资产", "value": _ref("/summary/total_value")},
            {"id": "pnl", "component": "Badge", "label": "收益率", "value": _ref("/summary/total_return_pct")},
            {
                "id": "table",
                "component": "DataTable",
                "columns": [
                    {"key": "ticker", "label": "代码"},
                    {"key": "asset_type", "label": "类型"},
                    {"key": "shares", "label": "数量"},
                    {"key": "avg_cost", "label": "成本"},
                    {"key": "current_price", "label": "现价"},
                    {"key": "pnl_pct", "label": "收益率"},
                ],
                "rows": _ref("/rows"),
            },
        ]
        return _surface(
            surface_id,
            components,
            {"summary": {
                "cash": portfolio.get("cash", 0),
                "total_value": portfolio.get("total_value", 0),
                "total_return_pct": portfolio.get("total_return_pct", 0),
            }, "rows": rows},
        )

    if tool_name == "get_simulation_orders":
        surface_id = surface_id or f"simulation-orders-{uuid4().hex}"
        rows = [
            {
                "order_id": item.get("order_id", ""),
                "ticker": item.get("ticker", ""),
                "side": item.get("side", ""),
                "shares": item.get("shares", 0),
                "status": item.get("status", ""),
                "submitted_date": item.get("submitted_date", ""),
            }
            for item in payload.get("orders", [])
        ]
        components = [
            {"id": "root", "component": "Card", "children": ["title", "table"]},
            _text("title", f"模拟盘订单 · {payload.get('account_id', 'default')}", "h3"),
            {
                "id": "table",
                "component": "DataTable",
                "columns": [
                    {"key": "order_id", "label": "订单"},
                    {"key": "ticker", "label": "代码"},
                    {"key": "side", "label": "方向"},
                    {"key": "shares", "label": "数量"},
                    {"key": "status", "label": "状态"},
                    {"key": "submitted_date", "label": "提交日期"},
                ],
                "rows": _ref("/rows"),
            },
        ]
        return _surface(surface_id, components, {"rows": rows})

    return None


CATALOG = {
    "$id": CATALOG_ID,
    "catalogId": CATALOG_ID,
    "version": "v0.9",
    "components": {
        "Text": {},
        "Column": {},
        "Row": {},
        "Card": {},
        "Section": {},
        "Badge": {},
        "Progress": {},
        "ScoreBar": {},
        "PipelineStep": {},
        "List": {},
        "StrategyItem": {},
        "StatusItem": {},
        "Sparkline": {},
        "LineChart": {},
        "Button": {},
        "TextField": {},
        "ChoicePicker": {},
        "Markdown": {},
        "Activity": {},
        "DataTable": {},
    },
}
