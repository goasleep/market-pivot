"""A2UI v0.9 message builders for the chat surface.

The agent sends declarative component definitions and a separate data model.
The browser owns the native React widget registry; no agent-provided HTML or
JavaScript is executed on the client.
"""

from __future__ import annotations

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
    children = ["title", "meta", "skills", "objective", "outcome", "progress", "missing"]
    outcome = str(plan.get("outcome_status") or "")
    outcome_labels = {
        "satisfied": "业务验收：通过",
        "partial": "业务验收：部分完成",
        "data_unavailable": "业务验收：数据不可用",
        "invalid_result": "业务验收：结果无效",
    }
    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "children": children},
        _text("title", "股票 / 基金研究执行计划", "h3"),
        _text("meta", _ref("/meta"), "caption"),
        _text("skills", _ref("/skills"), "caption"),
        _text("objective", _ref("/objective"), "body"),
        {"id": "outcome", "component": "Badge", "text": _ref("/outcomeLabel"), "tone": _ref("/outcomeTone")},
        {"id": "progress", "component": "Progress", "value": _ref("/progress")},
        _text("missing", _ref("/missing"), "caption"),
    ]
    for index, step in enumerate(steps):
        component_id = f"step-{index}"
        children.append(component_id)
        recovery = step.get("recovery") if isinstance(step.get("recovery"), dict) else None
        detail = str(step.get("error") or "")
        evidence_status = str(step.get("evidence_status") or "")
        public_meta = " · ".join(
            item
            for item in (
                str(step.get("skill_id") or step.get("capability_id") or ""),
                f"证据 {evidence_status}" if evidence_status else "",
                f"数据 {step.get('as_of')}" if step.get("as_of") else "",
            )
            if item
        )
        evidence_issues = [str(item) for item in step.get("evidence_issues", []) if str(item)]
        if not detail and evidence_issues:
            detail = "数据说明：" + "；".join(evidence_issues)
        if recovery:
            recovery_label = {
                "retry": "自动重试",
                "adjust": "反思后调整",
                "abort": "停止重试",
            }.get(str(recovery.get("action")), "失败恢复")
            detail = f"{recovery_label}：{recovery.get('summary') or detail}"
        detail = "；".join(item for item in (public_meta, detail) if item)
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
            "skills": "能力：" + "、".join(str(item) for item in plan.get("selected_skills", [])[:12]),
            "missing": (
                "待补齐：" + "、".join(str(item) for item in plan.get("missing", [])[:8]) if plan.get("missing") else ""
            ),
            "outcomeLabel": outcome_labels.get(outcome, "业务验收：执行中"),
            "outcomeTone": (
                "positive"
                if outcome == "satisfied"
                else "negative"
                if outcome in {"data_unavailable", "invalid_result"}
                else "secondary"
            ),
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


def _percent_label(value: Any, *, signed: bool = True) -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value) * 100
        return f"{number:+.2f}%" if signed else f"{number:.2f}%"
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
    strategy_spec = payload.get("strategy_spec") or result.get("strategy_spec") or {}
    strategy_name = strategy_spec.get("description") or strategy_spec.get("name") or "回测策略"
    price_points = [
        {
            "label": str(item.get("date") or item.get("label") or ""),
            "value": float(item.get("value", item.get("close", 0)) or 0),
        }
        for item in (result.get("price_curve") or [])
        if isinstance(item, dict) and (item.get("date") is not None or item.get("label") is not None)
    ]
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
            "children": ["final", "return", "drawdown", "sharpe", "winrate", "trade-count"],
        },
        _text("final", f"最终市值 ¥{_number_label(result.get('final_value'))}", "caption"),
        _text("return", f"收益 {_percent_label(result.get('total_return'))}", "caption"),
        _text(
            "drawdown",
            f"最大回撤 {_percent_label(result.get('max_drawdown'), signed=False)}",
            "caption",
            tone="negative",
        ),
        _text("sharpe", f"Sharpe {_number_label(result.get('sharpe_ratio'))}", "caption"),
        _text("winrate", f"胜率 {_percent_label(result.get('win_rate'))}", "caption"),
        _text("trade-count", f"交易 {result.get('total_trades', len(trades))} 次", "caption"),
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
    if len(price_points) >= 2:
        children.append("tradePoints")
        components.extend(
            [
                {
                    "id": "tradePoints",
                    "component": "Collapsible",
                    "title": "标的价格与实际买卖点",
                    "defaultExpanded": True,
                    "children": ["tradePointsNote", "tradePointsChart"],
                },
                _text(
                    "tradePointsNote",
                    "买卖标记来自实际成交记录，日期和价格已反映成交时点、滑点及交易规则；并非原始信号日。",
                    "caption",
                ),
                {
                    "id": "tradePointsChart",
                    "component": "TradeChart",
                    "pricePoints": _ref("/pricePoints"),
                    "strategies": _ref("/tradeStrategies"),
                    "ariaLabel": "回测策略实际买卖点",
                },
            ]
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
    return _surface(
        surface_id,
        components,
        {
            "points": points,
            "pricePoints": price_points,
            "tradeStrategies": [{"name": strategy_name, "trades": trades}],
            "trades": rows,
        },
    )


def render_sandbox_strategy_candidate(
    payload: dict[str, Any],
    surface_id: str | None = None,
) -> list[dict[str, Any]]:
    # Strategy renderers import this facade's primitives, so loading them here avoids a cycle.
    from widgets.a2ui_strategy import render_sandbox_strategy_candidate as render

    return render(payload, surface_id)


def render_strategy_comparison(
    payload: dict[str, Any],
    surface_id: str | None = None,
) -> list[dict[str, Any]]:
    # Strategy renderers import this facade's primitives, so loading them here avoids a cycle.
    from widgets.a2ui_strategy import render_strategy_comparison as render

    return render(payload, surface_id)


def render_tool_result(
    tool_name: str,
    raw_result: str,
    surface_id: str | None = None,
) -> list[dict[str, Any]] | None:
    """Dispatch known tool payloads through the dedicated renderer module."""
    # The dispatcher imports this facade's public renderers, so loading it here avoids a cycle.
    from widgets.a2ui_tool_results import render_tool_result as dispatch_tool_result

    return dispatch_tool_result(tool_name, raw_result, surface_id)


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
        "MultiLineChart": {},
        "TradeChart": {},
        "BenchmarkChart": {},
        "Button": {},
        "TextField": {},
        "ChoicePicker": {},
        "Markdown": {},
        "Activity": {},
        "DataTable": {},
    },
}
