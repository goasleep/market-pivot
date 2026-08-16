"""Generate and persist user-facing Agent artifacts in object storage."""

from __future__ import annotations

import hashlib
import html
import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from artifacts.storage import ArtifactStorage, LocalArtifactStorage, S3ArtifactStorage
from config import settings
from models.schemas import TradeDecision


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _text(value: Any, fallback: str = "暂无") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


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
) -> str:
    """Render a deterministic, source-aware research report."""
    generated_at = generated_at or _now()
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
        f"- 一句话结论：{_text(sections['core'].get('one_line_summary'), decision.reasoning or '暂无')}",
        "",
        "## 二、决策依据",
        "",
        decision.reasoning or "暂无综合决策依据。",
        "",
        "## 三、数据视角",
        "",
        f"- 趋势状态：{_text(perspective.get('trend_status'))}",
        f"- 价格位置：{_text(perspective.get('price_position'))}",
        f"- 量能分析：{_text(perspective.get('volume_analysis'))}",
        f"- 筹码结构：{_text(perspective.get('chip_structure'))}",
        "",
        "## 四、交易计划",
        "",
        f"- 入场价：{_text(battle.get('entry_price'))}",
        f"- 止损价：{_text(battle.get('stop_loss'))}",
        f"- 止盈价：{_text(battle.get('take_profit'))}",
        f"- 仓位策略：{_text(battle.get('position_strategy'))}",
    ]
    action_items = battle.get("action_items", []) or ["暂无具体行动项"]
    lines.extend(["- 行动项：", *[f"  - {_text(item)}" for item in action_items]])
    web_source_lines = [
        f"- {_text(item.get('title'))}：{_text(item.get('snippet'))}（{_text(item.get('link'))}）"
        for item in web_results[:8]
    ] or ["- 未启用 Serper 搜索或暂无搜索结果"]
    lines.extend(
        [
            "",
            "## 五、情报与风险",
            "",
            "### 风险警报",
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
            *[f"- {key}：{value}" for key, value in context_status.items()],
            "" if context_status else "- 暂无数据状态信息",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_analysis_html(markdown: str, decision: TradeDecision, generated_at: str) -> str:
    """Render a standalone HTML report without adding a runtime dependency."""
    escaped = html.escape(markdown)
    body = re.sub(r"^### (.+)$", r"<h3>\1</h3>", escaped, flags=re.MULTILINE)
    body = re.sub(r"^## (.+)$", r"<h2>\1</h2>", body, flags=re.MULTILINE)
    body = re.sub(r"^# (.+)$", r"<h1>\1</h1>", body, flags=re.MULTILINE)
    body = re.sub(r"^> (.+)$", r"<p class=\"meta\">\1</p>", body, flags=re.MULTILINE)
    body = re.sub(r"^(- .+)$", r"<li>\1</li>", body, flags=re.MULTILINE)
    body = re.sub(r"^(  - .+)$", r"<li>\1</li>", body, flags=re.MULTILINE)
    body = body.replace("**", "")
    paragraphs = []
    in_list = False
    for line in body.splitlines():
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
    title = html.escape(f"{decision.ticker} 研究分析报告")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f5f7fb; color: #172033; line-height: 1.7; }}
    main {{ max-width: 880px; margin: 32px auto; padding: 40px 48px; background: white;
      box-shadow: 0 8px 32px #17203312; }}
    h1 {{ margin-top: 0; font-size: 30px; color: #102a56; }}
    h2 {{ margin-top: 32px; padding-bottom: 6px; border-bottom: 1px solid #dce4f2; color: #1d4d91; }}
    h3 {{ color: #315f9f; }}
    .meta {{ color: #64748b; font-size: 13px; margin: 2px 0; }}
    li {{ margin: 4px 0; }}
    @media (max-width: 640px) {{ main {{ margin: 0; padding: 24px; }} }}
  </style>
</head>
<body><main>{''.join(paragraphs)}</main></body>
</html>
"""


class ArtifactService:
    """Artifact metadata repository backed by S3-compatible object storage."""

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        db_path: str | Path | None = None,
        storage: ArtifactStorage | None = None,
    ):
        # A local directory is only selected when explicitly supplied, which
        # keeps existing unit-test fixtures possible without making local disk
        # the production default.
        self.storage = storage or (
            LocalArtifactStorage(storage_dir)
            if storage_dir is not None
            else S3ArtifactStorage(
                endpoint_url=settings.s3_endpoint_url,
                bucket=settings.s3_bucket,
                region=settings.s3_region,
                access_key_id=settings.s3_access_key_id,
                secret_access_key=settings.s3_secret_access_key,
                session_token=settings.s3_session_token,
                addressing_style=settings.s3_addressing_style,
            )
        )
        self.db_path = str(db_path or settings.database_file_path)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _init_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    relative_path TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    ticker TEXT,
                    asset_type TEXT,
                    source TEXT NOT NULL,
                    conversation_id TEXT,
                    task_id TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_created_at ON artifacts(created_at DESC)")
            connection.commit()

    def _record(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json"))
        item["object_key"] = item.pop("relative_path")
        item["preview_url"] = f"/api/artifacts/{item['artifact_id']}/preview"
        item["download_url"] = f"/api/artifacts/{item['artifact_id']}/download"
        return item

    def _create_file(
        self,
        name: str,
        content: str,
        mime_type: str,
        decision: TradeDecision,
        source: str,
        conversation_id: str | None,
        task_id: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        artifact_id = f"artifact-{uuid4().hex[:16]}"
        prefix = settings.s3_artifacts_prefix.strip("/")
        object_key = "/".join(part for part in (prefix, artifact_id, name) if part)
        raw = content.encode("utf-8")
        self.storage.put(object_key, raw, mime_type)
        created_at = _now()
        record = {
            "artifact_id": artifact_id,
            "name": name,
            "artifact_type": "analysis_report",
            "mime_type": mime_type,
            # Keep the existing SQLite column for backward-compatible schema
            # reads; it now stores an object key rather than a local path.
            "relative_path": object_key,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "ticker": decision.ticker,
            "asset_type": decision.asset_type.value,
            "source": source,
            "conversation_id": conversation_id,
            "task_id": task_id,
            "metadata_json": _json(metadata),
            "created_at": created_at,
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO artifacts
                   (artifact_id, name, artifact_type, mime_type, relative_path, size_bytes,
                    sha256, ticker, asset_type, source, conversation_id, task_id, metadata_json, created_at)
                   VALUES (:artifact_id, :name, :artifact_type, :mime_type, :relative_path, :size_bytes,
                    :sha256, :ticker, :asset_type, :source, :conversation_id, :task_id, :metadata_json, :created_at)""",
                record,
            )
            connection.commit()
        response = {
            **{key: value for key, value in record.items() if key not in {"metadata_json", "relative_path"}},
            "object_key": object_key,
            "metadata": metadata,
            "preview_url": f"/api/artifacts/{artifact_id}/preview",
            "download_url": f"/api/artifacts/{artifact_id}/download",
        }
        return response

    def create_analysis_artifacts(
        self,
        decision: TradeDecision,
        market_context: Any | None = None,
        *,
        source: str = "analysis",
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Create the single user-facing HTML report for an analysis.

        Markdown remains the deterministic intermediate representation used to
        render HTML, but it is not persisted as a second user-facing artifact.
        This keeps chat and analysis results from presenting duplicate report
        files while preserving the existing HTML preview/download contract.
        """
        generated_at = _now()
        markdown = render_analysis_markdown(decision, market_context, generated_at)
        html_report = render_analysis_html(markdown, decision, generated_at)
        base = f"{_safe_slug(decision.ticker)}-研究分析报告"
        web_results = getattr(market_context, "web_results", []) or []
        metadata = {
            "generated_at": generated_at,
            "report_version": "1.0",
            "web_search_count": len(web_results),
        }
        return [
            self._create_file(
                f"{base}.html",
                html_report,
                "text/html",
                decision,
                source,
                conversation_id,
                task_id,
                metadata,
            ),
        ]

    def get(self, artifact_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
        if row is None:
            return None
        return self._record(row)

    def read(self, artifact_id: str) -> tuple[dict[str, Any], bytes] | None:
        """Read artifact metadata and bytes from the configured object store."""
        artifact = self.get(artifact_id)
        if artifact is None:
            return None
        return artifact, self.storage.get(artifact["object_key"])

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 200)),)
            ).fetchall()
        return [self._record(row) for row in rows]


artifact_service = ArtifactService()
