"""Tool-result dispatch for A2UI surfaces."""

from __future__ import annotations

import json
import math
from typing import Any
from uuid import uuid4


def render_tool_result(tool_name: str, raw_result: str, surface_id: str | None = None) -> list[dict[str, Any]] | None:
    """Turn known tool payloads into native A2UI result surfaces."""
    # Imported lazily so the public widgets.a2ui facade can re-export this
    # dispatcher without creating an import cycle during module initialization.
    from widgets.a2ui import (
        _ref,
        _surface,
        _text,
        render_analysis_result,
        render_asset_card,
        render_backtest_result,
        render_sandbox_strategy_candidate,
        render_strategy_comparison,
        render_strategy_selector,
    )

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
        return render_strategy_comparison(payload, surface_id)

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
