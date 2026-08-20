"""AnySearch-backed web search with cache and failure isolation."""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger

from config import settings
from data.akshare_provider import DataCache

_cache = DataCache(settings.database_file_path)
_CACHE_TTL_SECONDS = 15 * 60
_FAILURE_TTL_SECONDS = 30
_TIMEOUT_SECONDS = 12.0


def _cache_key(query: str, num_results: int) -> str:
    raw = f"{query.strip()}|{num_results}|{settings.anysearch_zone}|{settings.anysearch_language}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"anysearch:search:{digest}"


def _normalise_results(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    raw_results = data.get("results", []) if isinstance(data, dict) else []
    results: list[dict[str, Any]] = []
    for position, item in enumerate(raw_results[:limit], start=1):
        if not isinstance(item, dict):
            continue
        link = str(item.get("url") or item.get("link") or "").strip()
        if not link:
            continue
        results.append(
            {
                "position": position,
                "title": str(item.get("title", "")),
                "link": link,
                "snippet": str(item.get("snippet") or item.get("content") or ""),
                "date": str(item.get("date", "")),
                "source": "AnySearch",
            }
        )
    return results


async def async_search_web_anysearch(query: str, *, num_results: int = 8) -> dict[str, Any]:
    """Search the web through AnySearch without blocking the event loop."""
    query = query.strip()
    limit = max(1, min(int(num_results), 20))
    if not query:
        return {"available": False, "query": "", "results": [], "error": "搜索关键词不能为空"}

    key = _cache_key(query, limit)
    cached = _cache.get(key, ttl=_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    failure_key = f"{key}:failure"
    if _cache.get(failure_key, ttl=_FAILURE_TTL_SECONDS) is not None:
        return {"available": False, "query": query, "results": [], "error": "AnySearch 暂时不可用，请稍后重试"}

    payload: dict[str, Any] = {"query": query, "max_results": limit, "format": "json"}
    if settings.anysearch_zone.strip():
        payload["zone"] = settings.anysearch_zone.strip()
    if settings.anysearch_language.strip():
        payload["language"] = settings.anysearch_language.strip()

    headers = {"Content-Type": "application/json"}
    api_key = settings.anysearch_api_key.strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    endpoint = f"{settings.anysearch_base_url.rstrip('/')}/v1/search"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            raw = response.json()
        if not isinstance(raw, dict):
            raise ValueError("AnySearch returned a non-object JSON response")
        if raw.get("code") not in (None, 0):
            raise ValueError(str(raw.get("message") or "AnySearch returned an error"))
        result = {
            "available": True,
            "query": query,
            "results": _normalise_results(raw, limit),
            "searched_at": datetime.now(timezone.utc).isoformat(),
            "source": "AnySearch",
        }
        _cache.set(key, result)
        return result
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        logger.warning("AnySearch search failed for {!r}: {}", query, exc)
        _cache.set(failure_key, {"error": str(exc), "timestamp": time.time()})
        return {
            "available": False,
            "query": query,
            "results": [],
            "error": "AnySearch 搜索失败，请检查 API Key、网络或额度",
        }


def search_web_anysearch(query: str, *, num_results: int = 8) -> dict[str, Any]:
    """Synchronous compatibility entry point for non-FastAPI callers.

    FastAPI and agent code should use :func:`async_search_web_anysearch`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_search_web_anysearch(query, num_results=num_results))
    raise RuntimeError("请在异步调用方中使用 async_search_web_anysearch")
