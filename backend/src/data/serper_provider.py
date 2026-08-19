"""Serper-backed web search with a small ORM cache and failure isolation."""

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
from data.anysearch_provider import async_search_web_anysearch
from data.ddgs_provider import async_search_web_ddgs
from data.source_registry import data_sources, utc_now

_cache = DataCache(settings.database_file_path)
_CACHE_TTL_SECONDS = 15 * 60
_FAILURE_TTL_SECONDS = 30
_TIMEOUT_SECONDS = 12.0


def _cache_key(query: str, num_results: int, tbs: str | None) -> str:
    raw = f"{query.strip()}|{num_results}|{tbs or ''}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"serper:search:{digest}"


def _normalise_results(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in payload.get("organic", [])[:limit]:
        if not isinstance(item, dict) or not item.get("link"):
            continue
        results.append(
            {
                "position": item.get("position"),
                "title": str(item.get("title", "")),
                "link": str(item.get("link", "")),
                "snippet": str(item.get("snippet", "")),
                "date": str(item.get("date", "")),
                "source": str(item.get("source", "")),
            }
        )
    return results


def search_web(query: str, *, num_results: int = 8, tbs: str | None = None) -> dict[str, Any]:
    """Search Google through Serper and return source-aware organic results."""
    query = query.strip()
    limit = max(1, min(int(num_results), 10))
    if not query:
        return {"available": False, "query": "", "results": [], "error": "搜索关键词不能为空"}
    if not settings.serper_api_key.strip():
        return {
            "available": False,
            "query": query,
            "results": [],
            "error": "SERPER_API_KEY 未配置，联网搜索未启用",
        }

    key = _cache_key(query, limit, tbs)
    cached = _cache.get(key, ttl=_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    failure_key = f"{key}:failure"
    if _cache.get(failure_key, ttl=_FAILURE_TTL_SECONDS) is not None:
        return {"available": False, "query": query, "results": [], "error": "Serper 暂时不可用，请稍后重试"}

    payload: dict[str, Any] = {"q": query, "gl": settings.serper_gl, "hl": settings.serper_hl, "num": limit}
    if tbs:
        payload["tbs"] = tbs

    endpoint = f"{settings.serper_base_url.rstrip('/')}/search"
    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            response = client.post(
                endpoint,
                headers={"X-API-KEY": settings.serper_api_key.strip(), "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()
        result = {
            "available": True,
            "query": query,
            "results": _normalise_results(raw, limit),
            "searched_at": datetime.now(timezone.utc).isoformat(),
            "source": "Serper / Google Search",
        }
        _cache.set(key, result)
        return result
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        logger.warning("Serper search failed for {!r}: {}", query, exc)
        _cache.set(failure_key, {"error": str(exc), "timestamp": time.time()})
        return {
            "available": False,
            "query": query,
            "results": [],
            "error": "Serper 搜索失败，请检查 API Key、网络或额度",
        }


async def async_search_web(query: str, *, num_results: int = 8, tbs: str | None = None) -> dict[str, Any]:
    """Async wrapper for callers that must not block the event loop."""
    return await asyncio.to_thread(search_web, query, num_results=num_results, tbs=tbs)


async def async_search_web_parallel(
    query: str,
    *,
    num_results: int = 8,
    tbs: str | None = None,
) -> dict[str, Any]:
    """Run configured premium providers and DDGS concurrently, then merge results."""
    timelimit = {"qdr:h": "h", "qdr:d": "d", "qdr:w": "w", "qdr:m": "m", "qdr:y": "y"}.get(tbs or "")
    tasks = []
    if settings.anysearch_api_key.strip():
        tasks.append(async_search_web_anysearch(query, num_results=num_results))
    if settings.serper_api_key.strip():
        tasks.append(async_search_web(query, num_results=num_results, tbs=tbs))
    tasks.append(async_search_web_ddgs(query, num_results=num_results, timelimit=timelimit))
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    merged: list[dict[str, Any]] = []
    providers: list[str] = []
    source_ids: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    for response in responses:
        if isinstance(response, Exception):
            errors.append(str(response))
            continue
        if response.get("available"):
            source = str(response.get("source", "web search"))
            providers.append(source)
            try:
                source_ids.append(data_sources.get(source).source_id)
            except ValueError:
                pass
        else:
            error = response.get("error")
            if error:
                errors.append(str(error))
        for item in response.get("results", []):
            identity = str(item.get("link") or item.get("title") or "").strip().lower()
            if identity and identity not in seen:
                seen.add(identity)
                merged.append(item)

    return {
        "available": bool(merged),
        "query": query,
        "results": merged[: max(1, min(int(num_results), 10))],
        "providers": providers,
        "source_ids": list(dict.fromkeys(source_ids)),
        "source": " + ".join(providers) if providers else "",
        "searched_at": utc_now(),
        "errors": errors,
    }
