"""DDGS-backed metasearch with cache and failure isolation."""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from config import settings
from data.akshare_provider import DataCache

_cache = DataCache(settings.database_file_path)
_CACHE_TTL_SECONDS = 15 * 60
_FAILURE_TTL_SECONDS = 30


def _cache_key(query: str, num_results: int, timelimit: str | None) -> str:
    raw = f"{query.strip()}|{num_results}|{timelimit or ''}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"ddgs:search:{digest}"


def _normalise_results(raw_results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for position, item in enumerate(raw_results[:limit], start=1):
        link = item.get("href") or item.get("link")
        if not link:
            continue
        results.append(
            {
                "position": position,
                "title": str(item.get("title", "")),
                "link": str(link),
                "snippet": str(item.get("body") or item.get("snippet") or ""),
                "date": str(item.get("date", "")),
                "source": "DDGS",
            }
        )
    return results


def search_web_ddgs(query: str, *, num_results: int = 8, timelimit: str | None = None) -> dict[str, Any]:
    """Search the web through the open-source DDGS metasearch library."""
    query = query.strip()
    limit = max(1, min(int(num_results), 10))
    if not query:
        return {"available": False, "query": "", "results": [], "error": "搜索关键词不能为空"}

    key = _cache_key(query, limit, timelimit)
    cached = _cache.get(key, ttl=_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    failure_key = f"{key}:failure"
    if _cache.get(failure_key, ttl=_FAILURE_TTL_SECONDS) is not None:
        return {"available": False, "query": query, "results": [], "error": "DDGS 暂时不可用，请稍后重试"}

    try:
        # Keep the optional search backend inside its failure-isolation boundary.
        from ddgs import DDGS

        raw_results = DDGS(timeout=12).text(
            query,
            region=settings.ddgs_region,
            safesearch=settings.ddgs_safesearch,
            timelimit=timelimit,
            max_results=limit,
        )
        result = {
            "available": True,
            "query": query,
            "results": _normalise_results(raw_results or [], limit),
            "searched_at": datetime.now(timezone.utc).isoformat(),
            "source": "DDGS metasearch",
        }
        _cache.set(key, result)
        return result
    except Exception as exc:  # DDGS aggregates external engines with varying failures.
        logger.warning("DDGS search failed for {!r}: {}", query, exc)
        _cache.set(failure_key, {"error": str(exc), "timestamp": time.time()})
        return {
            "available": False,
            "query": query,
            "results": [],
            "error": "DDGS 搜索失败，可能是上游限流或网络不可用",
        }


async def async_search_web_ddgs(
    query: str,
    *,
    num_results: int = 8,
    timelimit: str | None = None,
) -> dict[str, Any]:
    """Async wrapper for the blocking DDGS client."""
    return await asyncio.to_thread(search_web_ddgs, query, num_results=num_results, timelimit=timelimit)
