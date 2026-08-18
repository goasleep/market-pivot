"""Artifact creation, inspection and chart tools."""

from __future__ import annotations

import asyncio
import html
import json
from typing import Any

from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field

from artifacts.service import artifact_service
from charts.echarts import line_option, render_chart_container, render_chart_document


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class ChatArtifactDraft(BaseModel):
    """One file the chat agent may choose to persist as a user artifact."""

    name: str = Field(description="文件名，例如 国家队持仓概览.html")
    format: str | None = Field(default=None, description="文件格式：md、html、pdf、png、jpg、mp4 等；可从文件名推断")
    content: str | None = Field(default=None, description="文本文件内容，与 content_base64 二选一")
    content_base64: str | None = Field(default=None, description="二进制文件的 base64 内容")
    description: str = Field(default="", description="文件用途说明")
    artifact_type: str | None = Field(default=None, description="可选分类：document、data、image、video")
    ticker: str | None = Field(default=None, description="可选的六位标的代码")
    asset_type: str | None = Field(default=None, description="可选：stock、etf 或 lof")


async def _persist_chat_artifacts(
    artifacts: list[ChatArtifactDraft],
    *,
    conversation_id: str | None = None,
    task_id: str | None = None,
) -> str:
    if not artifacts:
        raise ValueError("至少需要一个 artifact 文件")
    records = await asyncio.to_thread(
        artifact_service.create_user_artifacts,
        [item.model_dump(exclude_none=True) for item in artifacts],
        source="chat",
        conversation_id=conversation_id,
        task_id=task_id,
    )
    return _dump({"ok": True, "artifacts": records})


@tool
async def save_artifacts(artifacts: list[ChatArtifactDraft]) -> str:
    """保存一个或多个用户可预览、下载或留档的生成产物。

    长文、HTML、Markdown、PDF、JSON、CSV、图片和视频都属于 artifact；
    可以一次保存多个不同文件。文本使用 content，二进制使用 content_base64。
    """
    return await _persist_chat_artifacts(artifacts)


def build_artifact_tool(*, conversation_id: str | None = None, task_id: str | None = None) -> StructuredTool:
    """Bind chat ownership metadata to the LLM-facing artifact tool."""

    async def save_bound_artifacts(artifacts: list[ChatArtifactDraft]) -> str:
        return await _persist_chat_artifacts(artifacts, conversation_id=conversation_id, task_id=task_id)

    return StructuredTool.from_function(
        coroutine=save_bound_artifacts,
        name="save_artifacts",
        description=(
            "保存一个或多个用户可预览、下载或留档的生成产物。长文、HTML、Markdown、PDF、"
            "JSON、CSV、图片和视频都属于 artifact；文本使用 content，二进制使用 content_base64。"
            "用户要求报告、下载、保存，或内容适合独立阅读时调用；保存成功后直接给出简短总结。"
        ),
    )


def _render_line_chart_html(title: str, points: list[dict[str, Any]]) -> str:
    chart = render_chart_container(
        "standalone-chart",
        line_option(title, points),
        aria_label=title or "走势",
        height=420,
    )
    return render_chart_document(title or "走势", chart)


def _legacy_render_line_chart_svg(title: str, points: list[dict[str, Any]]) -> str:
    values: list[tuple[str, float]] = []
    for point in points[:250]:
        try:
            values.append((str(point.get("label") or point.get("date") or len(values)), float(point["value"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(values) < 2:
        raise ValueError("至少需要两个有效的 chart points")
    width, height = 900, 420
    left, top, right, bottom = 56, 44, 24, 64
    prices = [value for _, value in values]
    minimum, maximum = min(prices), max(prices)
    value_range = maximum - minimum or 1.0
    chart_width = width - left - right
    chart_height = height - top - bottom
    coordinates = [
        (
            left + index / (len(values) - 1) * chart_width,
            top + (1 - (value - minimum) / value_range) * chart_height,
        )
        for index, (_, value) in enumerate(values)
    ]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
    labels = html.escape(title or "价格走势")
    first_label = html.escape(values[0][0])
    last_label = html.escape(values[-1][0])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{labels}">
  <rect width="100%" height="100%" fill="#f8fbff"/>
  <text x="{left}" y="28" font-size="20" font-family="sans-serif" fill="#172033">{labels}</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#cbd5e1"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#cbd5e1"/>
  <polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="3"
    stroke-linecap="round" stroke-linejoin="round"/>
  <text x="{left}" y="{height - 20}" font-size="12" font-family="sans-serif" fill="#64748b">{first_label}</text>
  <text x="{width - right}" y="{height - 20}" text-anchor="end" font-size="12"
    font-family="sans-serif" fill="#64748b">{last_label}</text>
  <text x="{left - 8}" y="{top + 4}" text-anchor="end" font-size="12"
    font-family="sans-serif" fill="#64748b">{maximum:.2f}</text>
  <text x="{left - 8}" y="{height - bottom + 4}" text-anchor="end" font-size="12"
    font-family="sans-serif" fill="#64748b">{minimum:.2f}</text>
</svg>"""


def _build_list_artifacts_tool(*, conversation_id: str | None, task_id: str | None) -> StructuredTool:
    async def list_bound_artifacts(limit: int = 50) -> str:
        records = await asyncio.to_thread(
            artifact_service.list_for_scope,
            limit=limit,
            conversation_id=conversation_id,
            task_id=task_id,
        )
        return _dump({"data_type": "artifacts", "artifacts": records})

    return StructuredTool.from_function(
        coroutine=list_bound_artifacts,
        name="list_artifacts",
        description="列出当前聊天任务已生成的 artifact 文件及其预览、下载地址。",
    )


def _build_read_artifact_tool(*, conversation_id: str | None, task_id: str | None) -> StructuredTool:
    async def read_text_artifact(artifact_id: str, max_chars: int = 20_000) -> str:
        record = await asyncio.to_thread(artifact_service.read_text, artifact_id, max_chars)
        if record is None:
            raise ValueError("artifact 不存在")
        if task_id and record.get("task_id") != task_id:
            raise ValueError("artifact 不属于当前任务")
        if not task_id and conversation_id and record.get("conversation_id") != conversation_id:
            raise ValueError("artifact 不属于当前会话")
        return _dump({"data_type": "artifact_content", "artifact": record})

    return StructuredTool.from_function(
        coroutine=read_text_artifact,
        name="read_artifact",
        description="读取一个已有文本 artifact 的内容；二进制 artifact 只返回元数据，不返回 base64。",
    )


def _build_chart_artifact_tool(*, conversation_id: str | None, task_id: str | None) -> StructuredTool:
    async def create_chart_artifact(
        name: str,
        points: list[dict[str, Any]],
        title: str = "走势",
        ticker: str | None = None,
        asset_type: str | None = None,
    ) -> str:
        html_document = _render_line_chart_html(title, points)
        output_name = name if name.lower().endswith((".html", ".htm")) else f"{name.rsplit('.', 1)[0]}.html"
        records = await asyncio.to_thread(
            artifact_service.create_user_artifacts,
            [
                {
                    "name": output_name,
                    "format": "html",
                    "content": html_document,
                    "artifact_type": "document",
                    "ticker": ticker,
                    "asset_type": asset_type,
                    "description": "由结构化数据生成的 ECharts 走势图表",
                }
            ],
            source="chat-chart",
            conversation_id=conversation_id,
            task_id=task_id,
        )
        return _dump({"ok": True, "artifacts": records})

    return StructuredTool.from_function(
        coroutine=create_chart_artifact,
        name="create_chart_artifact",
        description=(
            "把一组 label/value 数据生成使用 ECharts canvas 渲染的 HTML 图表 artifact；"
            "适合需要独立预览或下载的图表。"
        ),
    )


def build_artifact_tools(*, conversation_id: str | None = None, task_id: str | None = None) -> list[StructuredTool]:
    """Return the artifact tools bound to the current chat scope."""
    return [
        build_artifact_tool(conversation_id=conversation_id, task_id=task_id),
        _build_list_artifacts_tool(conversation_id=conversation_id, task_id=task_id),
        _build_read_artifact_tool(conversation_id=conversation_id, task_id=task_id),
        _build_chart_artifact_tool(conversation_id=conversation_id, task_id=task_id),
    ]


TOOLS = [save_artifacts]
