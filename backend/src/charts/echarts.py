"""ECharts option and HTML helpers.

The model supplies structured points; this module owns the chart option and
the safe HTML shell. ECharts uses its canvas renderer in the browser, so the
application does not need to hand-build SVG paths.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

ECHARTS_VERSION = "5.6.0"
ECHARTS_CDN = f"https://cdn.jsdelivr.net/npm/echarts@{ECHARTS_VERSION}/dist/echarts.min.js"


def _script_json(value: Any) -> str:
    """Serialize JSON for an inline script without allowing HTML breakouts."""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _normalise_points(points: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    values: list[float] = []
    for point in points[:250]:
        try:
            value = float(point["value"])
        except (KeyError, TypeError, ValueError):
            continue
        labels.append(str(point.get("label") or point.get("date") or len(labels)))
        values.append(value)
    if len(values) < 2:
        raise ValueError("至少需要两个有效的 chart points")
    return labels, values


def line_option(title: str, points: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an ECharts line option from label/value points."""
    labels, values = _normalise_points(points)
    return {
        "animation": False,
        "title": {"text": title or "走势", "left": "center", "textStyle": {"fontSize": 15}},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 52, "right": 22, "top": 52, "bottom": 46},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": labels,
            "axisLabel": {"hideOverlap": True},
        },
        "yAxis": {"type": "value", "scale": True},
        "series": [
            {
                "type": "line",
                "data": values,
                "smooth": True,
                "showSymbol": False,
                "lineStyle": {"width": 3, "color": "#2563eb"},
                "itemStyle": {"color": "#2563eb"},
                "areaStyle": {"color": "#2563eb", "opacity": 0.12},
            }
        ],
    }


def signal_attribution_option(scores: dict[str, Any]) -> dict[str, Any]:
    """Build an ECharts bar option for the decision signal dimensions."""
    fields = (
        ("技术面", "technical_score", "#2563eb"),
        ("情绪面", "sentiment_score", "#f59e0b"),
        ("基本面", "fundamental_score", "#22c55e"),
        ("市场环境", "market_regime_score", "#8b5cf6"),
    )
    data: list[dict[str, Any]] = []
    for label, key, color in fields:
        try:
            value = max(-100.0, min(100.0, float(scores.get(key, 0) or 0)))
        except (TypeError, ValueError):
            value = 0.0
        data.append({"value": value, "itemStyle": {"color": color}})
    return {
        "animation": False,
        "title": {"text": "信号归因", "left": "center", "textStyle": {"fontSize": 15}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 56, "right": 22, "top": 52, "bottom": 40},
        "xAxis": {"type": "category", "data": [field[0] for field in fields]},
        "yAxis": {"type": "value", "min": -100, "max": 100},
        "series": [{"type": "bar", "barMaxWidth": 42, "data": data}],
    }


def render_chart_container(
    chart_id: str,
    option: dict[str, Any],
    *,
    aria_label: str,
    height: int = 280,
) -> str:
    """Render one ECharts canvas container and its initialization script."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", chart_id):
        raise ValueError("chart_id 只能包含字母、数字、下划线和短横线")
    safe_id = html.escape(chart_id, quote=True)
    safe_label = html.escape(aria_label, quote=True)
    option_json = _script_json(option)
    return f"""
    <div id="{safe_id}" class="echarts-chart" style="width:100%;height:{height}px" role="img" aria-label="{safe_label}">
      <p class="chart-empty">正在加载图表…</p>
    </div>
    <script>
    (() => {{
      const element = document.getElementById({json.dumps(chart_id)});
      if (!element) return;
      if (!window.echarts) {{
        element.querySelector(".chart-empty").textContent = "图表引擎加载失败，请检查网络后重试。";
        return;
      }}
      element.replaceChildren();
      const chart = window.echarts.init(element, null, {{ renderer: "canvas" }});
      chart.setOption({option_json});
      window.addEventListener("resize", () => chart.resize(), {{ passive: true }});
    }})();
    </script>
    """


def render_chart_document(title: str, chart_html: str) -> str:
    """Wrap an ECharts chart in a standalone previewable HTML document."""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script src="{ECHARTS_CDN}"></script>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f5f7fb; color: #172033; }}
    main {{ max-width: 960px; margin: 24px auto; padding: 24px; background: white; border-radius: 16px; }}
    .chart-empty {{ color: #64748b; font-size: 13px; }}
  </style>
</head>
<body><main>{chart_html}</main></body>
</html>
"""
