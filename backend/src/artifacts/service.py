"""Generate and persist user-facing Agent artifacts in object storage."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from loguru import logger

from agents.report_agent import ReportAgent
from artifacts.storage import ArtifactStorage, LocalArtifactStorage, S3ArtifactStorage
from charts.echarts import ECHARTS_CDN, line_option, render_chart_container, signal_attribution_option
from config import get_llm_config, settings
from data.db_models import ArtifactRecord
from data.tortoise_db import init_database
from llm.service import get_llm_service
from models.schemas import TradeDecision

REPORT_COPY_SYSTEM_PROMPT = """你是 A 股研究报告编辑。请根据用户提供的结构化事实，生成一份简体中文、克制、清晰、
适合短中期研究和模拟交易用户阅读的报告叙述。

规则：
1. 只使用输入中已有的事实，不补造价格、日期、行情、新闻、数据源或结论；缺失信息请明确写“暂无足够数据”。
2. 语气统一为专业、谨慎、可执行的研究表达，先给结论，再说明依据、风险和观察条件；不要使用夸张营销语言。
3. 不承诺收益，不把研究意见写成确定性的买卖指令，不声称已经执行真实交易。
4. 当前产品面向基金短中期研究；当输入只有股票数据时，必须明确这是底层股票研究，不能冒充基金专项分析。
5. 不输出 Markdown 标题、列表、链接或原始 URL；不要重复输入中的数字字段，结构化数字由后端模板展示。
6. 必须返回 JSON 对象，键只能是：executive_summary、decision_basis、market_reading、risk_summary、
action_plan、data_limitations。每个值都是简体中文字符串。
"""

REPORT_COPY_FIELDS = (
    "executive_summary",
    "decision_basis",
    "market_reading",
    "risk_summary",
    "action_plan",
    "data_limitations",
)

# An artifact is a user-facing, independently previewable or downloadable
# file.  The chat agent may create more than one artifact for a task; these
# formats cover generated documents, structured data, and binary media.
ARTIFACT_MIME_TYPES = {
    "txt": "text/plain",
    "text": "text/plain",
    "md": "text/markdown",
    "markdown": "text/markdown",
    "html": "text/html",
    "htm": "text/html",
    "json": "application/json",
    "csv": "text/csv",
    "xml": "application/xml",
    "yaml": "application/yaml",
    "yml": "application/yaml",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "zip": "application/zip",
    "bin": "application/octet-stream",
    "png": "image/png",
    "svg": "image/svg+xml",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
}
ARTIFACT_EXTENSIONS = {
    "text/plain": "txt",
    "text/markdown": "md",
    "text/html": "html",
    "application/json": "json",
    "text/csv": "csv",
    "application/xml": "xml",
    "application/yaml": "yaml",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/zip": "zip",
    "application/octet-stream": "bin",
    "image/png": "png",
    "image/svg+xml": "svg",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/quicktime": "mov",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
}
TEXT_ARTIFACT_MIMES = {
    "text/plain",
    "text/markdown",
    "text/html",
    "application/json",
    "text/csv",
    "application/xml",
    "application/yaml",
    "image/svg+xml",
}
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _text(value: Any, fallback: str = "暂无") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def _history_points(market_context: Any | None, limit: int = 60) -> list[tuple[str, float]]:
    """Extract finite closing prices for the lightweight report trend chart."""
    records = getattr(market_context, "history", []) or []
    points: list[tuple[str, float]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            price = float(record.get("close"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price) or price <= 0:
            continue
        points.append((str(record.get("date") or ""), price))
    return points[-limit:]


def _render_price_trend_echarts(points: list[tuple[str, float]]) -> str:
    if len(points) < 2:
        return ""
    return render_chart_container(
        "price-trend-chart",
        line_option(
            "历史收盘价趋势",
            [{"label": label, "value": value} for label, value in points],
        ),
        aria_label="历史收盘价趋势图",
        height=300,
    )


def _legacy_render_price_trend_svg(points: list[tuple[str, float]]) -> str:
    """Render a compact, self-contained closing-price trend chart."""
    if len(points) < 2:
        return ""

    width, height = 760, 260
    left, top, right, bottom = 48, 20, 18, 42
    chart_width = width - left - right
    chart_height = height - top - bottom
    prices = [price for _, price in points]
    minimum, maximum = min(prices), max(prices)
    price_range = maximum - minimum or 1.0

    def x(index: int) -> float:
        return left + (index / (len(points) - 1)) * chart_width

    def y(price: float) -> float:
        return top + (1 - (price - minimum) / price_range) * chart_height

    coordinates = [(x(index), y(price)) for index, (_, price) in enumerate(points)]
    line_points = " ".join(f"{point_x:.1f},{point_y:.1f}" for point_x, point_y in coordinates)
    area_points = f"{left},{top + chart_height} {line_points} {left + chart_width},{top + chart_height}"
    first_date = html.escape(points[0][0] or "起始日")
    last_date = html.escape(points[-1][0] or "最新日")
    latest = prices[-1]
    midpoint_y = top + chart_height / 2
    bottom_y = top + chart_height

    return f"""
    <div class="price-trend" role="img" aria-label="历史收盘价趋势图">
      <div class="price-trend-header">
        <strong>历史收盘价趋势</strong>
        <span>最新 {latest:.2f}</span>
      </div>
      <svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">
        <line x1="{left}" y1="{top}" x2="{left + chart_width}" y2="{top}" class="chart-grid" />
        <line x1="{left}" y1="{midpoint_y}" x2="{left + chart_width}" y2="{midpoint_y}" class="chart-grid" />
        <line x1="{left}" y1="{bottom_y}" x2="{left + chart_width}" y2="{bottom_y}" class="chart-grid" />
        <polygon points="{area_points}" class="chart-area" />
        <polyline points="{line_points}" class="chart-line" />
        <circle cx="{coordinates[-1][0]:.1f}" cy="{coordinates[-1][1]:.1f}" r="4" class="chart-point" />
        <text x="{left - 8}" y="{top + 4}" text-anchor="end" class="chart-label">{maximum:.2f}</text>
        <text x="{left - 8}" y="{top + chart_height + 4}" text-anchor="end" class="chart-label">{minimum:.2f}</text>
        <text x="{left}" y="{height - 12}" class="chart-label">{first_date}</text>
        <text x="{left + chart_width}" y="{height - 12}" text-anchor="end" class="chart-label">{last_date}</text>
      </svg>
      <p class="price-trend-caption">区间最高 {maximum:.2f} · 区间最低 {minimum:.2f} · 共 {len(points)} 个交易日</p>
    </div>
    """


def _render_signal_attribution_echarts(attribution: dict[str, Any]) -> str:
    return render_chart_container(
        "signal-attribution-chart",
        signal_attribution_option(attribution),
        aria_label="信号归因图",
        height=260,
    )


def _legacy_render_signal_attribution_html(attribution: dict[str, Any]) -> str:
    """Render the four decision signal dimensions as a self-contained bar chart."""
    fields = (
        ("技术面", "technical_score"),
        ("情绪面", "sentiment_score"),
        ("基本面", "fundamental_score"),
        ("市场环境", "market_regime_score"),
    )
    rows: list[str] = []
    for label, key in fields:
        try:
            value = max(-100.0, min(100.0, float(attribution.get(key, 0) or 0)))
        except (TypeError, ValueError):
            value = 0.0
        tone = "signal-positive" if value >= 0 else "signal-negative"
        rows.append(
            f'<div class="signal-row"><span class="signal-name">{label}</span>'
            f'<div class="signal-track"><span class="{tone}" style="width:{abs(value):.1f}%"></span></div>'
            f'<strong>{value:+.0f}</strong></div>'
        )
    return (
        '<div class="signal-attribution" role="img" aria-label="信号归因图">'
        '<div class="signal-attribution-title">信号归因</div>'
        + "".join(rows)
        + "</div>"
    )


def _report_prompt_payload(decision: TradeDecision, market_context: Any | None) -> dict[str, Any]:
    """Build a compact fact set for the report editor without exposing URLs."""
    dashboard = decision.dashboard.model_dump(mode="json") if decision.dashboard else {}
    agent_reports = {
        str(name): _text(reasoning, "暂无观点")[:1600]
        for name, reasoning in decision.agent_reports.items()
    }
    web_results = getattr(market_context, "web_results", []) or []
    history_points = _history_points(market_context)
    return {
        "instrument": {
            "ticker": decision.ticker,
            "asset_type": decision.asset_type.value,
        },
        "decision": {
            "decision": decision.decision.value,
            "confidence": decision.confidence,
            "target_price": decision.target_price,
            "stop_loss": decision.stop_loss,
            "position_size": decision.position_size,
            "reasoning": decision.reasoning,
        },
        "dashboard": dashboard,
        "agent_reports": agent_reports,
        "market_context": {
            "as_of_date": getattr(market_context, "as_of_date", None),
            "current_price": getattr(market_context, "current_price", None),
            "market_regime": getattr(market_context, "market_regime", None),
            "data_status": getattr(market_context, "data_status", {}) or {},
            "history_summary": {
                "count": len(history_points),
                "start_date": history_points[0][0] if history_points else "",
                "end_date": history_points[-1][0] if history_points else "",
                "start_close": history_points[0][1] if history_points else None,
                "end_close": history_points[-1][1] if history_points else None,
                "high": max((price for _, price in history_points), default=None),
                "low": min((price for _, price in history_points), default=None),
            },
            "web_results": [
                {
                    "title": _text(item.get("title"), "未命名来源"),
                    "snippet": _text(item.get("snippet"), "暂无摘要")[:800],
                    "date": item.get("date"),
                }
                for item in web_results[:8]
            ],
        },
    }


async def generate_report_copy(
    decision: TradeDecision,
    market_context: Any | None = None,
) -> dict[str, str]:
    """Ask the configured LLM for consistent report prose, with safe fallback."""
    if not get_llm_config().get("api_key", "").strip():
        return {}

    prompt = _json(_report_prompt_payload(decision, market_context))
    try:
        result = await get_llm_service().chat_json(prompt, system=REPORT_COPY_SYSTEM_PROMPT)
    except Exception as exc:  # pragma: no cover - depends on external model service
        logger.warning("Report copy generation failed; using structured fallback: {}", exc)
        return {}

    if result.get("error"):
        return {}
    return {
        field: str(result[field]).strip()[:2400]
        for field in REPORT_COPY_FIELDS
        if isinstance(result.get(field), str) and result[field].strip()
    }


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z一-龥_-]+", "-", value.strip())
    return slug.strip("-") or "analysis"


def _dashboard_sections(decision: TradeDecision) -> dict[str, dict[str, Any]]:
    dashboard = decision.dashboard.model_dump(mode="json") if decision.dashboard else {}
    return {
        "core": dashboard.get("core_conclusion", {}),
        "perspective": dashboard.get("data_perspective", {}),
        "intelligence": dashboard.get("intelligence", {}),
        "battle": dashboard.get("battle_plan", {}),
        "phase": dashboard.get("phase_decision", {}),
        "attribution": dashboard.get("signal_attribution", {}),
    }


def render_analysis_markdown(
    decision: TradeDecision,
    market_context: Any | None = None,
    generated_at: str | None = None,
    report_copy: dict[str, str] | None = None,
) -> str:
    """Render a structured report with optional LLM-polished narrative copy."""
    generated_at = generated_at or _now()
    report_copy = report_copy or {}
    sections = _dashboard_sections(decision)
    context_status = getattr(market_context, "data_status", {}) or {}
    current_price = getattr(market_context, "current_price", None)
    as_of_date = getattr(market_context, "as_of_date", None)
    perspective = sections["perspective"]
    battle = sections["battle"]
    intelligence = sections["intelligence"]
    phase = sections["phase"]
    attribution = sections["attribution"]
    web_results = getattr(market_context, "web_results", []) or []
    history_points = _history_points(market_context)
    executive_summary = report_copy.get("executive_summary") or sections["core"].get("one_line_summary")

    lines = [
        f"# {_text(decision.ticker)} {_text(decision.asset_type.value)} 研究分析报告",
        "",
        f"> 生成时间：{generated_at}",
        f"> 数据截止：{_text(as_of_date, '实时数据或数据源最新日期')}",
        "> 用途声明：本报告仅用于短中期研究和模拟交易，不构成投资建议，也不承诺收益。股票分析不能替代基金专项分析。",
        "",
        "## 一、核心结论",
        "",
        f"- 决策：**{decision.decision.value}**",
        f"- 置信度：**{decision.confidence:.0%}**",
        f"- 当前价格：{_text(current_price)}",
        f"- 目标价：{_text(decision.target_price)}",
        f"- 止损价：{_text(decision.stop_loss)}",
        f"- 建议仓位：{decision.position_size:.0%}" if decision.position_size is not None else "- 建议仓位：暂无",
        f"- 一句话结论：{_text(executive_summary, decision.reasoning or '暂无')}",
        *(
            [
                "",
                "### 历史价格趋势",
                "",
                "[[PRICE_TREND_CHART]]",
            ]
            if len(history_points) >= 2
            else []
        ),
        "",
        "## 二、决策依据",
        "",
        report_copy.get("decision_basis") or decision.reasoning or "暂无综合决策依据。",
        "",
        "## 三、数据视角",
        "",
        f"- 趋势状态：{_text(perspective.get('trend_status'))}",
        f"- 价格位置：{_text(perspective.get('price_position'))}",
        f"- 量能分析：{_text(perspective.get('volume_analysis'))}",
        f"- 筹码结构：{_text(perspective.get('chip_structure'))}",
        report_copy.get("market_reading", ""),
        "",
        "## 四、交易计划",
        "",
        f"- 入场价：{_text(battle.get('entry_price'))}",
        f"- 止损价：{_text(battle.get('stop_loss'))}",
        f"- 止盈价：{_text(battle.get('take_profit'))}",
        f"- 仓位策略：{_text(battle.get('position_strategy'))}",
        report_copy.get("action_plan", ""),
    ]
    action_items = battle.get("action_items", []) or ["暂无具体行动项"]
    lines.extend(["- 行动项：", *[f"  - {_text(item)}" for item in action_items]])
    web_source_lines = []
    for item in web_results[:8]:
        title = _text(item.get("title"))
        snippet = _text(item.get("snippet"))
        link = str(item.get("link") or "").strip()
        if link.startswith(("http://", "https://")):
            web_source_lines.append(f"- {title}：{snippet} [查看来源]({link})")
        else:
            web_source_lines.append(f"- {title}：{snippet}")
    web_source_lines = web_source_lines or ["- 未启用 Serper 搜索或暂无搜索结果"]
    lines.extend(
        [
            "",
            "## 五、情报与风险",
            "",
            "### 风险警报",
            report_copy.get("risk_summary", ""),
            *[f"- {_text(item)}" for item in (intelligence.get("risk_alerts", []) or ["暂无风险警报"])[:8]],
            "",
            "### 积极催化",
            *[f"- {_text(item)}" for item in (intelligence.get("positive_catalysts", []) or ["暂无积极催化"])[:8]],
            "",
            "### 最新新闻",
            *[f"- {_text(item)}" for item in (intelligence.get("latest_news", []) or ["暂无新闻"])[:8]],
            "",
            "### 联网搜索结果",
            *web_source_lines,
            "",
            "## 六、阶段计划",
            "",
            f"- 盘前：{_text(phase.get('pre_market'))}",
            f"- 盘中：{_text(phase.get('intraday'))}",
            f"- 盘后：{_text(phase.get('post_market'))}",
            "",
            "## 七、信号归因",
            "",
            f"- 技术面：{_text(attribution.get('technical_score'), '0')}",
            f"- 舆情面：{_text(attribution.get('sentiment_score'), '0')}",
            f"- 基本面：{_text(attribution.get('fundamental_score'), '0')}",
            f"- 市场环境：{_text(attribution.get('market_regime_score'), '0')}",
            *(["", "[[SIGNAL_ATTRIBUTION_CHART]]"] if attribution else []),
            "",
            "## 八、各 Agent 观点",
            "",
        ]
    )
    for name, reasoning in decision.agent_reports.items():
        lines.extend([f"### {name}", "", reasoning or "暂无观点。", ""])
    lines.extend(
        [
            "## 数据状态",
            "",
            report_copy.get("data_limitations", ""),
            *[f"- {key}：{value}" for key, value in context_status.items()],
            "" if context_status else "- 暂无数据状态信息",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_analysis_html(
    markdown: str,
    decision: TradeDecision,
    generated_at: str,
    market_context: Any | None = None,
) -> str:
    """Render a standalone HTML report with ECharts canvas visualizations."""
    escaped = html.escape(markdown)
    link_tokens: list[str] = []
    chart_token = "__REPORT_PRICE_TREND__"
    chart_html = _render_price_trend_echarts(_history_points(market_context))
    signal_chart_token = "__REPORT_SIGNAL_ATTRIBUTION__"
    attribution = _dashboard_sections(decision)["attribution"]
    signal_chart_html = _render_signal_attribution_echarts(attribution)

    def make_link(label: str, url: str) -> str:
        token = f"__REPORT_LINK_{len(link_tokens)}__"
        link_tokens.append(
            f'<a class="report-link" href="{url}" target="_blank" '
            f'rel="noreferrer noopener">{label}</a>'
        )
        return token

    def preserve_markdown_link(match: re.Match[str]) -> str:
        return make_link(match.group(1), match.group(2))

    def preserve_bare_url(match: re.Match[str]) -> str:
        return make_link("查看链接", match.group(0))

    # Keep the visible label while moving the actual URL into href.
    body = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        preserve_markdown_link,
        escaped,
    )
    # Also hide URLs that were included in model-generated prose.
    body = re.sub(
        r"https?://[^\s<>'\")，。；：！？、】]+",
        preserve_bare_url,
        body,
    )
    body = body.replace("[[PRICE_TREND_CHART]]", chart_token)
    body = body.replace("[[SIGNAL_ATTRIBUTION_CHART]]", signal_chart_token)
    body = re.sub(r"^### (.+)$", r"<h3>\1</h3>", body, flags=re.MULTILINE)
    body = re.sub(r"^## (.+)$", r"<h2>\1</h2>", body, flags=re.MULTILINE)
    body = re.sub(r"^# (.+)$", r"<h1>\1</h1>", body, flags=re.MULTILINE)
    body = re.sub(r"^> (.+)$", r"<p class=\"meta\">\1</p>", body, flags=re.MULTILINE)
    body = re.sub(r"^(- .+)$", r"<li>\1</li>", body, flags=re.MULTILINE)
    body = re.sub(r"^(  - .+)$", r"<li>\1</li>", body, flags=re.MULTILINE)
    body = body.replace("**", "")
    paragraphs = []
    in_list = False
    for line in body.splitlines():
        if line in {chart_token, signal_chart_token}:
            if in_list:
                paragraphs.append("</ul>")
                in_list = False
            paragraphs.append(line)
            continue
        if line.startswith("<li>"):
            if not in_list:
                paragraphs.append("<ul>")
                in_list = True
            paragraphs.append(line)
        else:
            if in_list:
                paragraphs.append("</ul>")
                in_list = False
            if line.strip():
                paragraphs.append(line if line.startswith("<") else f"<p>{line}</p>")
    if in_list:
        paragraphs.append("</ul>")
    rendered = "\n".join(paragraphs)
    for index, link in enumerate(link_tokens):
        rendered = rendered.replace(f"__REPORT_LINK_{index}__", link)
    rendered = rendered.replace(chart_token, chart_html or '<p class="chart-empty">暂无足够历史数据</p>')
    rendered = rendered.replace(
        signal_chart_token,
        signal_chart_html if attribution else '<p class="chart-empty">暂无信号归因数据</p>',
    )
    title = html.escape(f"{decision.ticker} 研究分析报告")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="{ECHARTS_CDN}"></script>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f5f7fb; color: #172033; line-height: 1.7; }}
    main {{ max-width: 880px; margin: 32px auto; padding: 40px 48px; background: white;
      box-shadow: 0 8px 32px #17203312; }}
    h1 {{ margin-top: 0; font-size: 30px; color: #102a56; }}
    h2 {{ margin-top: 32px; padding-bottom: 6px; border-bottom: 1px solid #dce4f2; color: #1d4d91; }}
    h3 {{ color: #315f9f; }}
    .meta {{ color: #64748b; font-size: 13px; margin: 2px 0; }}
    .report-link {{ color: #2563eb; text-decoration: underline; text-underline-offset: 2px; }}
    .echarts-chart {{ margin: 16px 0 8px; padding: 8px; border: 1px solid #dce4f2;
      border-radius: 14px; background: #fbfcff; }}
    .chart-empty {{ margin: 0; color: #64748b; font-size: 13px; }}
    li {{ margin: 4px 0; }}
    @media (max-width: 640px) {{ main {{ margin: 0; padding: 24px; }} }}
  </style>
</head>
<body><main>{rendered}</main></body>
</html>
"""


class ArtifactService:
    """Artifact metadata repository backed by S3-compatible object storage."""

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        db_path: str | Path | None = None,
        storage: ArtifactStorage | None = None,
        report_agent: ReportAgent | None = None,
    ):
        # A local directory is only selected when explicitly supplied, which
        # keeps existing unit-test fixtures possible without making local disk
        # the production default.
        self.storage = storage or (
            LocalArtifactStorage(storage_dir)
            if storage_dir is not None
            else S3ArtifactStorage(
                endpoint_url=settings.s3_endpoint_url,
                public_endpoint_url=settings.s3_public_endpoint_url,
                bucket=settings.s3_bucket,
                region=settings.s3_region,
                access_key_id=settings.s3_access_key_id,
                secret_access_key=settings.s3_secret_access_key,
                session_token=settings.s3_session_token,
                addressing_style=settings.s3_addressing_style,
            )
        )
        self.report_agent = report_agent or ReportAgent()
        self.db_url = (
            f"sqlite://{Path(db_path).expanduser().resolve()}" if db_path is not None else settings.chat_database_url
        )

    def _record(self, row: ArtifactRecord) -> dict[str, Any]:
        item = {
            "artifact_id": row.artifact_id,
            "name": row.name,
            "artifact_type": row.artifact_type,
            "mime_type": row.mime_type,
            "size_bytes": row.size_bytes,
            "sha256": row.sha256,
            "ticker": row.ticker,
            "asset_type": row.asset_type,
            "source": row.source,
            "conversation_id": row.conversation_id,
            "task_id": row.task_id,
            "created_at": row.created_at,
            "metadata": json.loads(row.metadata_json),
            "object_key": row.relative_path,
        }
        item["preview_url"] = f"/api/artifacts/{item['artifact_id']}/preview"
        item["download_url"] = f"/api/artifacts/{item['artifact_id']}/download"
        return item

    async def _create_bytes(
        self,
        name: str,
        content: bytes,
        mime_type: str,
        artifact_type: str,
        ticker: str | None,
        asset_type: str | None,
        source: str,
        conversation_id: str | None,
        task_id: str | None,
        metadata: dict[str, Any],
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        await init_database(db_url=self.db_url)
        artifact_id = artifact_id or f"artifact-{uuid4().hex[:16]}"
        existing = await ArtifactRecord.get_or_none(artifact_id=artifact_id)
        if existing is not None:
            return self._record(existing)
        prefix = settings.s3_artifacts_prefix.strip("/")
        object_key = "/".join(part for part in (prefix, artifact_id, name) if part)
        self.storage.put(object_key, content, mime_type)
        created_at = _now()
        record = {
            "artifact_id": artifact_id,
            "name": name,
            "artifact_type": artifact_type,
            "mime_type": mime_type,
            # Keep the existing SQLite column for backward-compatible schema
            # reads; it now stores an object key rather than a local path.
            "relative_path": object_key,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "ticker": ticker,
            "asset_type": asset_type,
            "source": source,
            "conversation_id": conversation_id,
            "task_id": task_id,
            "metadata_json": _json(metadata),
            "created_at": created_at,
        }
        await ArtifactRecord.create(**record)
        response = {
            **{key: value for key, value in record.items() if key not in {"metadata_json", "relative_path"}},
            "object_key": object_key,
            "metadata": metadata,
            "preview_url": f"/api/artifacts/{artifact_id}/preview",
            "download_url": f"/api/artifacts/{artifact_id}/download",
        }
        return response

    async def create_binary_artifact(
        self,
        *,
        name: str,
        content: bytes,
        mime_type: str,
        artifact_type: str = "image",
        ticker: str | None = None,
        asset_type: str | None = None,
        source: str = "visual-evidence",
        conversation_id: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        execution_key: str | None = None,
    ) -> dict[str, Any]:
        """Persist trusted application-generated bytes without a base64 round trip."""
        if not content:
            raise ValueError("artifact content must not be empty")
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ValueError(f"artifact exceeds {MAX_ARTIFACT_BYTES // 1024 // 1024} MB limit")
        extension = ARTIFACT_EXTENSIONS.get(mime_type)
        if not extension:
            raise ValueError(f"unsupported binary artifact MIME type: {mime_type}")
        normalized_name = self._normalise_name(name, extension)
        digest = hashlib.sha256(content).hexdigest()
        identity = execution_key or f"{source}:{ticker or ''}:{asset_type or ''}:{normalized_name}:{digest}"
        artifact_id = f"artifact-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        artifact_metadata = {
            "format": extension,
            "generated_by": source,
            **(metadata or {}),
        }
        return await self._create_bytes(
            name=normalized_name,
            content=content,
            mime_type=mime_type,
            artifact_type=artifact_type,
            ticker=ticker,
            asset_type=asset_type,
            source=source,
            conversation_id=conversation_id,
            task_id=task_id,
            metadata=artifact_metadata,
            artifact_id=artifact_id,
        )

    def model_input_url(self, artifact: dict[str, Any], *, expires_in: int = 900) -> str:
        """Create an ephemeral absolute URL without storing it in artifact metadata."""
        object_key = str(artifact.get("object_key") or "")
        if not object_key:
            raise ValueError("artifact object_key is required")
        url = self.storage.presign_get_url(object_key, expires_in=expires_in)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("model artifact URL must be absolute HTTP(S)")
        return url

    async def _create_file(
        self,
        name: str,
        content: str,
        mime_type: str,
        decision: TradeDecision,
        source: str,
        conversation_id: str | None,
        task_id: str | None,
        metadata: dict[str, Any],
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """Keep the analysis-report API backed by the generic artifact writer."""
        return await self._create_bytes(
            name=name,
            content=content.encode("utf-8"),
            mime_type=mime_type,
            artifact_type="analysis_report",
            ticker=decision.ticker,
            asset_type=decision.asset_type.value,
            source=source,
            conversation_id=conversation_id,
            task_id=task_id,
            metadata=metadata,
            artifact_id=artifact_id,
        )

    @staticmethod
    def _normalise_name(name: str, extension: str) -> str:
        """Return a safe, single-file name with the requested extension."""
        candidate = Path(name.strip()).name
        candidate = re.sub(r"[^0-9A-Za-z一-龥._ -]+", "-", candidate).strip(" .-")
        if not candidate:
            candidate = "artifact"
        if not candidate.lower().endswith(f".{extension}"):
            candidate = f"{candidate}.{extension}"
        return candidate

    @staticmethod
    def _artifact_type_for_mime(mime_type: str) -> str:
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("video/"):
            return "video"
        if mime_type.startswith("audio/"):
            return "audio"
        if mime_type in {"application/pdf", "text/plain", "text/markdown", "text/html"}:
            return "document"
        if mime_type == "application/octet-stream":
            return "file"
        return "data"

    async def _find_duplicate(
        self,
        *,
        name: str,
        sha256: str,
        conversation_id: str | None,
        task_id: str | None,
    ) -> dict[str, Any] | None:
        """Make retries idempotent without limiting distinct files per task."""
        if not conversation_id and not task_id:
            return None
        query = ArtifactRecord.filter(name=name, sha256=sha256)
        if task_id:
            query = query.filter(task_id=task_id)
        else:
            query = query.filter(conversation_id=conversation_id)
        row = await query.first()
        return self._record(row) if row is not None else None

    async def create_user_artifacts(
        self,
        artifacts: list[dict[str, Any]],
        *,
        source: str = "chat",
        conversation_id: str | None = None,
        task_id: str | None = None,
        ticker: str | None = None,
        asset_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        execution_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Persist LLM-selected text, document, data, image, or video files.

        Text artifacts use ``content``. Binary artifacts use ``content_base64``;
        the server never fetches arbitrary URLs supplied by the model.
        Duplicate retries with the same task/name/content return the existing
        record, while distinct files remain allowed in the same task.
        """
        await init_database(db_url=self.db_url)
        created: list[dict[str, Any]] = []
        for index, item in enumerate(artifacts):
            if not isinstance(item, dict):
                raise ValueError("每个 artifact 必须是对象")
            raw_name = str(item.get("name") or "artifact").strip()
            raw_format = str(item.get("format") or "").strip().lower().lstrip(".")
            raw_mime = str(item.get("mime_type") or "").strip().lower()
            extension = raw_format or (ARTIFACT_EXTENSIONS.get(raw_mime) if raw_mime else None)
            extension = extension or Path(raw_name).suffix.lower().lstrip(".") or "md"
            mime_type = raw_mime or ARTIFACT_MIME_TYPES.get(extension)
            if not mime_type:
                raise ValueError(f"不支持的 artifact 格式: {raw_format or extension}")
            if extension not in ARTIFACT_MIME_TYPES:
                extension = ARTIFACT_EXTENSIONS.get(mime_type, extension)
            if not (
                mime_type in ARTIFACT_MIME_TYPES.values()
                or mime_type.startswith(("application/", "text/", "image/", "video/", "audio/"))
            ):
                raise ValueError(f"不支持的 artifact MIME 类型: {mime_type}")
            name = self._normalise_name(raw_name, extension)

            content = item.get("content")
            encoded = item.get("content_base64")
            if content is not None and encoded is not None:
                raise ValueError(f"artifact {name} 不能同时提供 content 和 content_base64")
            if encoded is not None:
                try:
                    raw = base64.b64decode(str(encoded), validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError(f"artifact {name} 的 base64 内容无效") from exc
            elif content is not None:
                if mime_type not in TEXT_ARTIFACT_MIMES and not mime_type.startswith("text/"):
                    raise ValueError(f"artifact {name} 是二进制类型，必须使用 content_base64")
                raw = str(content).encode("utf-8")
            else:
                raise ValueError(f"artifact {name} 缺少 content 或 content_base64")
            if not raw:
                raise ValueError(f"artifact {name} 不能为空")
            if len(raw) > MAX_ARTIFACT_BYTES:
                raise ValueError(f"artifact {name} 超过 {MAX_ARTIFACT_BYTES // 1024 // 1024} MB 限制")

            sha256 = hashlib.sha256(raw).hexdigest()
            duplicate = await self._find_duplicate(
                name=name,
                sha256=sha256,
                conversation_id=conversation_id,
                task_id=task_id,
            )
            if duplicate is not None:
                created.append(duplicate)
                continue
            artifact_metadata = {
                "description": str(item.get("description") or "").strip(),
                "format": extension,
                "generated_by": "chat_agent",
            }
            extra_metadata = item.get("metadata")
            if isinstance(extra_metadata, dict):
                artifact_metadata.update(extra_metadata)
            if metadata:
                artifact_metadata.update(metadata)
            created.append(
                await self._create_bytes(
                    name=name,
                    content=raw,
                    mime_type=mime_type,
                    artifact_type=str(item.get("artifact_type") or self._artifact_type_for_mime(mime_type)),
                    ticker=str(item["ticker"]) if item.get("ticker") else ticker,
                    asset_type=str(item["asset_type"]) if item.get("asset_type") else asset_type,
                    source=source,
                    conversation_id=conversation_id,
                    task_id=task_id,
                    metadata=artifact_metadata,
                    artifact_id=(
                        f"artifact-{hashlib.sha256(f'{execution_key}:{index}'.encode()).hexdigest()[:24]}"
                        if execution_key
                        else None
                    ),
                )
            )
        return created

    async def create_analysis_artifacts(
        self,
        decision: TradeDecision,
        market_context: Any | None = None,
        *,
        source: str = "analysis",
        conversation_id: str | None = None,
        task_id: str | None = None,
        execution_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Create the single user-facing HTML report through ReportAgent."""
        artifact_id = (
            f"artifact-{hashlib.sha256(f'{execution_key}:0'.encode()).hexdigest()[:24]}"
            if execution_key
            else None
        )
        if artifact_id:
            await init_database(db_url=self.db_url)
            existing = await ArtifactRecord.get_or_none(artifact_id=artifact_id)
            if existing is not None:
                return [self._record(existing)]
        generated_at = _now()
        # ReportAgent uses the provider's synchronous client and can take several
        # minutes for a full HTML document. Keep it off the FastAPI event loop so
        # task heartbeats, cancellation, status polling, and SSE remain responsive.
        report = await asyncio.to_thread(
            self.report_agent.generate,
            decision,
            market_context,
            generated_at=generated_at,
        )
        metadata = dict(report.metadata)
        metadata.setdefault("generated_at", generated_at)
        return [
            await self._create_file(
                report.name,
                report.html,
                "text/html",
                decision,
                source,
                conversation_id,
                task_id,
                metadata,
                artifact_id,
            ),
        ]

    async def get(self, artifact_id: str) -> dict[str, Any] | None:
        await init_database(db_url=self.db_url)
        row = await ArtifactRecord.get_or_none(artifact_id=artifact_id)
        if row is None:
            return None
        return self._record(row)

    async def read(self, artifact_id: str) -> tuple[dict[str, Any], bytes] | None:
        """Read artifact metadata and bytes from the configured object store."""
        artifact = await self.get(artifact_id)
        if artifact is None:
            return None
        return artifact, self.storage.get(artifact["object_key"])

    async def list(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self.list_for_scope(limit=limit)

    async def list_for_scope(
        self,
        *,
        limit: int = 50,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List artifacts, optionally restricted to one chat scope."""
        await init_database(db_url=self.db_url)
        query = ArtifactRecord.all()
        if conversation_id:
            query = query.filter(conversation_id=conversation_id)
        if task_id:
            query = query.filter(task_id=task_id)
        rows = await query.order_by("-created_at").limit(max(1, min(limit, 200)))
        return [self._record(row) for row in rows]

    async def read_text(self, artifact_id: str, max_chars: int = 20_000) -> dict[str, Any] | None:
        """Read a bounded text artifact for follow-up Agent work."""
        stored = await self.read(artifact_id)
        if stored is None:
            return None
        artifact, content = stored
        mime_type = str(artifact.get("mime_type") or "")
        readable_mimes = {"application/json", "application/xml", "application/yaml"}
        if not (mime_type.startswith("text/") or mime_type in readable_mimes):
            return {**artifact, "content": None, "content_available": False}
        return {
            **artifact,
            "content": content.decode("utf-8", errors="replace")[: max(1, min(max_chars, 100_000))],
            "content_available": True,
        }


artifact_service = ArtifactService()
