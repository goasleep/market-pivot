"""Searchable plain-text investment methodology library.

Markdown files are the canonical source of methodology knowledge.  The search
implementation deliberately has no vector-database dependency so the library
can later swap the keyword retriever for an embedding-backed implementation
without changing the Agent tool contract.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_FRONTMATTER_MARKER = "---"
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_DEFAULT_LIMIT = 5
_MAX_LIMIT = 20


class MethodologyRetriever(Protocol):
    """Stable retrieval contract for future keyword/vector implementations."""

    def search(
        self,
        query: str,
        *,
        methodology_type: str | None = None,
        asset_type: str | None = None,
        horizon: str | None = None,
        limit: int = _DEFAULT_LIMIT,
    ) -> list[dict[str, Any]]:
        """Return methodology excerpts with their source metadata."""


@dataclass(frozen=True)
class MethodologyDocument:
    """Normalized Markdown document and searchable metadata."""

    document_id: str
    title: str
    methodology_type: str
    author: str
    source: str
    source_url: str
    published_at: str
    market_scope: tuple[str, ...]
    asset_types: tuple[str, ...]
    horizon: str
    tags: tuple[str, ...]
    status: str
    authority: str
    path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        """Build one normalized search field without losing the raw content."""
        metadata_text = " ".join(
            [
                self.title,
                self.methodology_type,
                self.author,
                self.source,
                self.horizon,
                *self.market_scope,
                *self.asset_types,
                *self.tags,
            ]
        )
        return f"{metadata_text}\n{self.content}".lower()

    def as_result(self, score: float) -> dict[str, Any]:
        """Return a compact, citeable result for the Agent tool."""
        return {
            "id": self.document_id,
            "title": self.title,
            "type": self.methodology_type,
            "author": self.author,
            "source": self.source,
            "source_url": self.source_url,
            "published_at": self.published_at,
            "market_scope": list(self.market_scope),
            "asset_types": list(self.asset_types),
            "horizon": self.horizon,
            "tags": list(self.tags),
            "authority": self.authority,
            "score": round(score, 4),
            "excerpt": _build_excerpt(self.content),
            "path": self.path,
        }


class MethodologyLibrary:
    """Load and search Markdown methodology documents from one or more roots."""

    def __init__(self, roots: list[Path] | None = None) -> None:
        backend_root = Path(__file__).resolve().parents[2]
        packaged_root = backend_root / "resources" / "methodology"
        user_root = backend_root / "data" / "methodology_user"
        self.roots = tuple(roots or [packaged_root, user_root])

    def load_documents(self) -> list[MethodologyDocument]:
        """Load all Markdown files in deterministic path order."""
        documents: list[MethodologyDocument] = []
        seen_ids: set[str] = set()
        for root in self.roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.md")):
                if path.name.lower() == "readme.md":
                    continue
                document = _parse_document(path)
                if document.document_id in seen_ids:
                    raise ValueError(f"重复的方法论文档 id: {document.document_id}")
                seen_ids.add(document.document_id)
                documents.append(document)
        return documents

    def search(
        self,
        query: str,
        *,
        methodology_type: str | None = None,
        asset_type: str | None = None,
        horizon: str | None = None,
        limit: int = _DEFAULT_LIMIT,
    ) -> list[dict[str, Any]]:
        """Search methodology by metadata filters and lightweight relevance."""
        normalized_query = query.strip().lower()
        if not normalized_query:
            raise ValueError("query 不能为空")
        bounded_limit = max(1, min(int(limit), _MAX_LIMIT))
        query_terms = _tokenize(normalized_query)
        normalized_type = _normalize_filter(methodology_type)
        normalized_asset = _normalize_filter(asset_type)
        normalized_horizon = _normalize_filter(horizon)
        ranked: list[tuple[float, MethodologyDocument]] = []

        for document in self.load_documents():
            if document.status.lower() not in {"active", "draft"}:
                continue
            if normalized_type and document.methodology_type.lower() != normalized_type:
                continue
            if normalized_asset and normalized_asset not in {item.lower() for item in document.asset_types}:
                continue
            if normalized_horizon and document.horizon.lower() != normalized_horizon:
                continue
            score = _relevance_score(document, normalized_query, query_terms)
            if score > 0:
                ranked.append((score, document))

        ranked.sort(key=lambda item: (-item[0], item[1].title, item[1].document_id))
        return [document.as_result(score) for score, document in ranked[:bounded_limit]]


def _parse_document(path: Path) -> MethodologyDocument:
    raw = path.read_text(encoding="utf-8")
    metadata, content = _split_frontmatter(raw, path)
    document_id = str(metadata.get("id") or path.stem).strip()
    title = str(metadata.get("title") or path.stem).strip()
    return MethodologyDocument(
        document_id=document_id,
        title=title,
        methodology_type=str(metadata.get("type") or "opinion").strip(),
        author=str(metadata.get("author") or "").strip(),
        source=str(metadata.get("source") or "").strip(),
        source_url=str(metadata.get("source_url") or "").strip(),
        published_at=str(metadata.get("published_at") or "").strip(),
        market_scope=tuple(_as_list(metadata.get("market_scope"))),
        asset_types=tuple(_as_list(metadata.get("asset_types"))),
        horizon=str(metadata.get("horizon") or "").strip(),
        tags=tuple(_as_list(metadata.get("tags"))),
        status=str(metadata.get("status") or "active").strip(),
        authority=str(metadata.get("authority") or "methodology").strip(),
        path=str(path),
        content=content.strip(),
        metadata=metadata,
    )


def _split_frontmatter(raw: str, path: Path) -> tuple[dict[str, Any], str]:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_MARKER:
        raise ValueError(f"方法论文档缺少 YAML frontmatter: {path}")
    try:
        closing_index = next(index for index in range(1, len(lines)) if lines[index].strip() == _FRONTMATTER_MARKER)
    except StopIteration as exc:
        raise ValueError(f"方法论文档 frontmatter 未闭合: {path}") from exc

    metadata: dict[str, Any] = {}
    for line_number, line in enumerate(lines[1:closing_index], start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line or line.startswith((" ", "\t")):
            raise ValueError(f"方法论文档 frontmatter 格式错误: {path}:{line_number}")
        key, value = line.split(":", 1)
        metadata[key.strip()] = _parse_scalar(value.strip())
    return metadata, "\n".join(lines[closing_index + 1 :])


def _parse_scalar(value: str) -> Any:
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value).strip()] if str(value).strip() else []


def _normalize_filter(value: str | None) -> str:
    return str(value or "").strip().lower()


def _tokenize(value: str) -> list[str]:
    return _TOKEN_PATTERN.findall(value.lower())


def _relevance_score(document: MethodologyDocument, query: str, query_terms: list[str]) -> float:
    title = document.title.lower()
    tags = " ".join(document.tags).lower()
    searchable_text = document.searchable_text
    score = 0.0
    if query in title:
        score += 10.0
    if query in tags:
        score += 7.0
    if query in searchable_text:
        score += 2.0
    for term in set(query_terms):
        if term in title:
            score += 3.0
        elif term in tags:
            score += 2.0
        elif term in searchable_text:
            score += 1.0
    return score


def _build_excerpt(content: str, max_length: int = 900) -> str:
    compact = re.sub(r"\n{3,}", "\n\n", content).strip()
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 1].rstrip()}…"
