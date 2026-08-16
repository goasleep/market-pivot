"""Fetch and clean web pages without using an LLM.

Search providers only return snippets.  This module turns a bounded subset of
those URLs into deterministic, untrusted text evidence for downstream agents.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger

from config import settings
from data.akshare_provider import DataCache

_cache = DataCache(settings.database_file_path)
_CACHE_TTL_SECONDS = 30 * 60
_MAX_RESULTS_TO_FETCH = 5
_MAX_RESPONSE_BYTES = 1_500_000
_MAX_CONTENT_CHARS = 6_000
_MIN_CONTENT_CHARS = 80
_MAX_REDIRECTS = 2
_TIMEOUT = httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0)


def _cache_key(url: str) -> str:
    digest = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:24]
    return f"web:content:{digest}"


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class _HtmlTextExtractor(HTMLParser):
    """Small deterministic HTML-to-text extractor for article-like pages."""

    _BLOCK_TAGS = {
        "article",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "section",
        "tr",
    }
    _SKIP_TAGS = {"aside", "canvas", "footer", "form", "iframe", "nav", "noscript", "script", "style", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.title = ""
        self.description = ""
        self.published_at = ""
        self._buffer: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def _flush(self) -> None:
        text = _clean_text("".join(self._buffer))
        if text:
            self.blocks.append(text)
        self._buffer = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag in self._SKIP_TAGS:
                self._skip_depth += 1
            return
        if tag in self._SKIP_TAGS:
            self._flush()
            self._skip_depth = 1
            return
        if tag in self._BLOCK_TAGS:
            self._flush()
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            attributes = {key.lower(): value or "" for key, value in attrs}
            name = (attributes.get("name") or attributes.get("property") or "").lower()
            content = _clean_text(attributes.get("content", ""))
            if name in {"description", "og:description"} and content and not self.description:
                self.description = content
            if name in {"article:published_time", "date", "pubdate"} and content and not self.published_at:
                self.published_at = content

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag in self._SKIP_TAGS:
                self._skip_depth -= 1
            return
        if tag == "title":
            self.title = _clean_text("".join(self._buffer))
            self._buffer = []
            self._in_title = False
            return
        if tag in self._BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._buffer.append(data)


def extract_article_content(html: str) -> dict[str, str]:
    """Extract article text and metadata using local deterministic rules."""
    parser = _HtmlTextExtractor()
    parser.feed(html)
    parser.close()

    unique_blocks: list[str] = []
    seen: set[str] = set()
    for block in parser.blocks:
        if len(block) < 8 or block in seen:
            continue
        seen.add(block)
        unique_blocks.append(block)

    content = "\n".join(unique_blocks)[:_MAX_CONTENT_CHARS].strip()
    return {
        "content": content,
        "content_status": "full_text" if len(content) >= _MIN_CONTENT_CHARS else "empty",
        "page_title": parser.title,
        "description": parser.description,
        "published_at": parser.published_at,
    }


def _is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return False
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return parsed.hostname.lower() not in {"localhost", "localhost.localdomain"}
    return not (address.is_private or address.is_loopback or address.is_link_local or address.is_reserved)


async def _fetch_result(client: httpx.AsyncClient, item: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(item)
    url = str(item.get("link", "")).strip()
    if not _is_safe_url(url):
        enriched.update({"content": "", "content_status": "unavailable", "fetch_error": "不支持或不安全的网页地址"})
        return enriched

    cached = _cache.get(_cache_key(url), ttl=_CACHE_TTL_SECONDS)
    if cached is not None:
        enriched.update(cached)
        return enriched

    try:
        current_url = url
        for redirect_count in range(_MAX_REDIRECTS + 1):
            async with client.stream("GET", current_url) as response:
                if 300 <= response.status_code < 400:
                    if redirect_count >= _MAX_REDIRECTS:
                        raise ValueError("网页重定向次数超过限制")
                    redirected_url = urljoin(current_url, response.headers.get("location", ""))
                    if not _is_safe_url(redirected_url):
                        raise ValueError("网页重定向到不安全地址")
                    current_url = redirected_url
                    continue
                if response.status_code >= 400:
                    raise httpx.HTTPStatusError("网页返回错误状态", request=response.request, response=response)
                content_type = response.headers.get("content-type", "").lower()
                if content_type and "html" not in content_type and "text/plain" not in content_type:
                    raise ValueError("网页不是 HTML 或纯文本")
                if int(response.headers.get("content-length", "0") or 0) > _MAX_RESPONSE_BYTES:
                    raise ValueError("网页内容超过大小限制")

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > _MAX_RESPONSE_BYTES:
                        raise ValueError("网页内容超过大小限制")
                    chunks.append(chunk)
                raw_html = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
                break
        else:
            raise ValueError("无法获取网页内容")

        extracted = extract_article_content(raw_html)
        if extracted["content_status"] == "full_text":
            _cache.set(_cache_key(url), extracted)
        enriched.update(extracted)
        return enriched
    except (httpx.HTTPError, UnicodeError, ValueError, TypeError) as exc:
        logger.debug("Web content fetch failed for {!r}: {}", url, exc)
        enriched.update({"content": "", "content_status": "snippet_only", "fetch_error": str(exc)})
        return enriched


async def async_enrich_web_results(
    results: list[dict[str, Any]], *, limit: int = _MAX_RESULTS_TO_FETCH
) -> list[dict[str, Any]]:
    """Fetch a bounded number of result pages concurrently, preserving order."""
    if not results:
        return []

    fetch_limit = max(0, min(int(limit), _MAX_RESULTS_TO_FETCH, len(results)))
    enriched = [dict(item) for item in results]
    for item in enriched[fetch_limit:]:
        item.setdefault("content", "")
        item.setdefault("content_status", "not_fetched")

    if fetch_limit == 0:
        return enriched

    limits = httpx.Limits(max_connections=fetch_limit, max_keepalive_connections=fetch_limit)
    headers = {"User-Agent": "A-Share-Agent/1.0 (research-only; +https://example.invalid)"}
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=False, limits=limits, headers=headers) as client:
        fetched = await asyncio.gather(*(_fetch_result(client, item) for item in enriched[:fetch_limit]))
    return fetched + enriched[fetch_limit:]
