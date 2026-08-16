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


def render_stock_card(
    stock_data: dict[str, Any], surface_id: str | None = None, include_create: bool = True
) -> list[dict[str, Any]]:
    surface_id = surface_id or f"stock-card-{uuid4().hex}"
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
        price = float(stock_data["price"])
    except (KeyError, TypeError, ValueError):
        return []
    if not math.isfinite(price) or price <= 0:
        return []

    try:
        pct = float(stock_data["pct_chg"])
    except (KeyError, TypeError, ValueError):
        pct = None
    if pct is None or not math.isfinite(pct):
        change_label = "涨跌幅 暂无"
        change_tone = "secondary"
    else:
        change_label = f"{'▲' if pct >= 0 else '▼'} {abs(pct):.2f}%"
        change_tone = "positive" if pct >= 0 else "negative"
    data = {
        "ticker": stock_data.get("ticker", ""),
        "name": stock_data.get("name", ""),
        "priceLabel": f"¥{price:.2f}",
        "changeLabel": change_label,
        "changeTone": change_tone,
        "peLabel": f"PE {stock_data.get('pe', '-')}",
        "pbLabel": f"PB {stock_data.get('pb', '-')}",
        "volumeLabel": f"成交量 {stock_data.get('volume', '-')}",
    }
    return _surface(surface_id, components, data, include_create)


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
        if status in {"done", "complete"}:
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


def render_tool_result(tool_name: str, raw_result: str, surface_id: str | None = None) -> list[dict[str, Any]] | None:
    """Turn known tool payloads into native A2UI result surfaces."""
    try:
        payload = json.loads(raw_result)
    except (TypeError, json.JSONDecodeError):
        return None

    if tool_name == "get_realtime_quote":
        quote = dict(payload.get("quote") or {})
        quote["ticker"] = payload.get("ticker", quote.get("ticker", ""))
        return render_stock_card(quote, surface_id)

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
            {"id": "root", "component": "Card", "children": ["title", "list"]},
            _text("title", f"搜索结果（{len(items)} 条）", "h3"),
            {"id": "list", "component": "List", "items": _ref("/items"), "itemTemplate": "searchItem"},
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
        records = payload.get("history", [])
        closes = [float(item.get("close", 0) or 0) for item in records if item.get("close") is not None]
        surface_id = surface_id or f"history-{uuid4().hex}"
        components = [
            {"id": "root", "component": "Card", "children": ["title", "chart", "table"]},
            _text("title", f"{payload.get('ticker', '')} 历史走势", "h3"),
            {"id": "chart", "component": "Sparkline", "values": _ref("/prices")},
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
        return _surface(surface_id, components, {"prices": closes, "rows": records[-12:]})

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
        "Button": {},
        "TextField": {},
        "ChoicePicker": {},
        "Markdown": {},
        "Activity": {},
        "DataTable": {},
    },
}
