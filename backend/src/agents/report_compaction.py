"""Chat-native compaction for generated research artifacts."""

from __future__ import annotations

import re
from typing import Any

_HTML_SOURCE_BLOCK = re.compile(
    r"```(?:html|xhtml)?\s*(?:<!doctype\s+html|<html\b).*?```",
    flags=re.IGNORECASE | re.DOTALL,
)
_RAW_HTML_SOURCE = re.compile(r"(?:<!doctype\s+html|<html\b).*", flags=re.IGNORECASE | re.DOTALL)
_GENERIC_ARTIFACT_NOTICE = re.compile(r"完整\s*HTML\s*报告已生成文件产物", flags=re.IGNORECASE)
_ARTIFACT_MARKDOWN_LINK = re.compile(
    r"\[(?P<label>[^\]\n]+)\]\(\s*/api/artifacts/[A-Za-z0-9_-]+/(?:preview|download)\s*\)",
    flags=re.IGNORECASE,
)
_ARTIFACT_LINK_ONLY_LINE = re.compile(
    r"^\s*(?:[-*+]\s+|\d+\.\s+)?"
    r"\[[^\]\n]+\]\(\s*/api/artifacts/[A-Za-z0-9_-]+/(?:preview|download)\s*\)\s*$",
    flags=re.IGNORECASE,
)
_ARTIFACT_SECTION_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:完整(?:文件|内容|成果包?)|附件(?:文件|下载|列表)?)[：:]?\s*$",
    flags=re.IGNORECASE,
)


def _artifact_label(artifact: dict[str, Any]) -> str:
    mime_type = str(artifact.get("mime_type") or "")
    return {
        "text/markdown": "Markdown",
        "text/html": "HTML",
        "application/pdf": "PDF",
        "text/csv": "CSV",
        "application/json": "JSON",
    }.get(mime_type, "文件")


def _remove_embedded_artifact_links(text: str) -> str:
    """Remove duplicated internal artifact links from model-authored prose."""
    lines = text.splitlines()
    removed_indexes: set[int] = set()
    for index, line in enumerate(lines):
        if not _ARTIFACT_LINK_ONLY_LINE.fullmatch(line):
            continue
        removed_indexes.add(index)
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous >= 0 and _ARTIFACT_SECTION_HEADING.fullmatch(lines[previous]):
            removed_indexes.add(previous)

    cleaned = "\n".join(line for index, line in enumerate(lines) if index not in removed_indexes)
    cleaned = _ARTIFACT_MARKDOWN_LINK.sub(lambda match: match.group("label"), cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def compact_generated_report(text: str, artifacts: list[dict[str, Any]] | None = None) -> str:
    """Remove embedded source while retaining a useful chat-native conclusion."""
    artifacts = artifacts or []
    artifact = artifacts[0] if artifacts else {}
    name = str(artifact.get("name") or "完整报告")
    label = _artifact_label(artifact)
    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    description = str(metadata.get("description") or "").strip()
    match = _HTML_SOURCE_BLOCK.search(text) or _RAW_HTML_SOURCE.search(text)
    if match is not None:
        lead = text[: match.start()].strip()
        tail = text[match.end() :].strip()
        text = "\n\n".join(part for part in (lead, tail) if part)
    text = _remove_embedded_artifact_links(text)
    if _GENERIC_ARTIFACT_NOTICE.search(text) and len(text.strip()) < 200:
        text = description
    if len(text) > 1800:
        prefix = text[:1600]
        boundary = max(prefix.rfind("\n\n"), prefix.rfind("。"), prefix.rfind("；"))
        text = prefix[: boundary + 1].strip() if boundary >= 400 else prefix.rstrip()
        text = f"{text}\n\n（正文已节选，完整内容保存在附件中。）"
    if not text.strip():
        text = description or "报告已经生成，核心结论和完整明细请查看附件。"
    notice = f"完整内容见下方附件：{name}（{label}），可直接预览或下载。"
    return "\n\n".join(part for part in (text.strip(), notice) if part)
