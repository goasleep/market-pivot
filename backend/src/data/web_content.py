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
_CONTENT_FILTER_VERSION = "v3"
_MAX_RESULTS_TO_FETCH = 5
_MAX_RESPONSE_BYTES = 1_500_000
_MAX_CONTENT_CHARS = 3_000
_MIN_CONTENT_CHARS = 80
_MAX_REDIRECTS = 2
_TIMEOUT = httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0)


def _cache_key(url: str) -> str:
    digest = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:24]
    # Include the extraction version so cached boilerplate from an older
    # cleaner cannot leak back into prompts after the filtering rules change.
    return f"web:content:{_CONTENT_FILTER_VERSION}:{digest}"


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
    _SKIP_TAGS = {
        "aside",
        "canvas",
        "footer",
        "form",
        "header",
        "iframe",
        "menu",
        "nav",
        "noscript",
        "script",
        "style",
        "svg",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, float, bool]] = []
        self.title = ""
        self.description = ""
        self.published_at = ""
        self._buffer: list[str] = []
        self._anchor_chars = 0
        self._anchor_depth = 0
        self._article_depth = 0
        self._in_title = False
        self._skip_depth = 0

    def _flush(self) -> None:
        text = _clean_text("".join(self._buffer))
        if text:
            visible_chars = len(re.sub(r"\s+", "", text))
            link_ratio = self._anchor_chars / max(visible_chars, 1)
            self.blocks.append((text, link_ratio, self._article_depth > 0))
        self._buffer = []
        self._anchor_chars = 0

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
        if tag in {"article", "main"}:
            self._article_depth += 1
        if tag == "a":
            self._anchor_depth += 1
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
        if tag in {"article", "main"}:
            self._article_depth = max(0, self._article_depth - 1)
        if tag == "a":
            self._anchor_depth = max(0, self._anchor_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._buffer.append(data)
        if self._anchor_depth:
            self._anchor_chars += len(re.sub(r"\s+", "", data))


_BOILERPLATE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"欢迎扫码(?:安装|下载)",
        r"扫[一\-]?扫(?:下载|安装)?\s*(?:APP)?",
        r"(?:下载|安装|打开)\s*(?:手机)?APP",
        r"股吧首页|热门个股吧|热门主题吧",
        r"上海证券交易所服务热线|上交所APP.*通办.*栏目",
        r"行情和统计.*相关公告.*基本信息",
        r"APP下载|微博微信|English繁",
        r"郑重声明|免责声明",
        r"本(?:网|网站|页面)(?:站)?所(?:刊载|提供|发布)",
        r"投资者依据本网站|盈亏与本网站无关|不负任何责任",
        r"信息网络传播视听节目许可证|经营证券期货业务许可证",
        r"违法和不良信息举报|举报邮箱|举报电话",
        r"ICP备|ICP证|公安网安备|网站备案号",
        r"版权所有|©\s*\d{4}",
        r"关于我们.*(?:广告服务|联系我们|法律声明|隐私保护)",
        r"可持续发展.*供应商平台|诚聘英才.*法律声明",
        r"友情链接|意见与建议",
        r"东方财富(?:Level-2|证券开户|在线交易|证券交易)",
        r"数据加载中",
        r"名称最新价涨跌幅|股票名称持仓占比涨跌幅",
    )
)


def _is_boilerplate_block(text: str, link_ratio: float) -> bool:
    """Reject navigation, legal chrome, and promotional page furniture."""
    if any(pattern.search(text) for pattern in _BOILERPLATE_PATTERNS):
        return True
    if len(re.findall(r"[:：]\s*-", text)) >= 3:
        return True
    # Link-heavy blocks are normally menus, related-link grids, or footer
    # navigation. Long prose can contain citations, so only apply the rule to
    # relatively short blocks.
    return link_ratio >= 0.55 and len(text) <= 500


def _looks_like_prose(text: str) -> bool:
    """Require sentence-like evidence when a page has no semantic article container."""
    if len(text) < 30:
        return False
    if re.search(r"[。！？；.!?]", text):
        return True
    # Some publishers omit punctuation in extracted blocks. Keep sufficiently
    # long text, while rejecting short quote widgets and category labels.
    return len(text) >= 100


def extract_article_content(html: str) -> dict[str, Any]:
    """Extract article text and metadata using local deterministic rules."""
    parser = _HtmlTextExtractor()
    parser.feed(html)
    parser.close()

    cleaned_blocks = [
        (text, in_article)
        for text, link_ratio, in_article in parser.blocks
        if len(text) >= 8 and not _is_boilerplate_block(text, link_ratio)
    ]
    article_blocks = [text for text, in_article in cleaned_blocks if in_article]
    candidate_blocks = (
        article_blocks
        if sum(map(len, article_blocks)) >= _MIN_CONTENT_CHARS
        else [text for text, _ in cleaned_blocks if _looks_like_prose(text)]
    )

    unique_blocks: list[str] = []
    seen: set[str] = set()
    for block in candidate_blocks:
        normalized = block.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_blocks.append(block)

    content = "\n".join(unique_blocks)[:_MAX_CONTENT_CHARS].strip()
    return {
        "content": content,
        "content_status": "full_text" if len(content) >= _MIN_CONTENT_CHARS else "empty",
        "page_title": parser.title,
        "description": parser.description,
        "published_at": parser.published_at,
        "content_filter_version": _CONTENT_FILTER_VERSION,
        "content_blocks": len(unique_blocks),
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
