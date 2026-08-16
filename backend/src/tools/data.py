"""External data acquisition and enrichment tools."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from data.anysearch_provider import async_search_web_anysearch
from data.ddgs_provider import async_search_web_ddgs
from data.serper_provider import async_search_web_parallel
from data.stock_provider import async_get_stock_news
from data.web_content import async_enrich_web_results
from models.schemas import AssetType


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


@tool
async def search_web(query: str, num_results: int = 8, freshness: str | None = None) -> str:
    """获取资讯数据：新闻、公告、政策、行业事件和市场催化；不要用于精确行情数据。"""
    allowed_freshness = {None, "qdr:h", "qdr:d", "qdr:w", "qdr:m", "qdr:y"}
    if freshness not in allowed_freshness:
        freshness = None
    result = await async_search_web_parallel(query, num_results=num_results, tbs=freshness)
    return _dump({"data_type": "news", **result})


@tool
async def fetch_web_content(url: str) -> str:
    """抓取一个安全的网页正文并返回清洗后的可引用文本。"""
    results = await async_enrich_web_results([{"link": url}], limit=1)
    result = results[0] if results else {"link": url, "content_status": "unavailable"}
    return _dump({"data_type": "web_content", **result})


@tool
async def get_latest_news(ticker: str, asset_type: str = "stock") -> str:
    """聚合股票或基金标的的最新新闻与公告。"""
    kind = AssetType(asset_type)
    label = "股票" if kind == AssetType.STOCK else f"{kind.value.upper()} 基金"
    akshare_news = await async_get_stock_news(ticker, limit=10) if kind == AssetType.STOCK else []
    web_result = await async_search_web_parallel(
        f"{ticker} {label} 最新新闻 公告 走势",
        num_results=10,
        tbs="qdr:m",
    )
    web_news = web_result.get("results", [])
    return _dump(
        {
            "data_type": "news",
            "ticker": ticker,
            "asset_type": kind.value,
            "news": [*akshare_news, *web_news],
            "akshare_news": akshare_news,
            "web_news": web_news,
            "web_providers": web_result.get("providers", []),
            "web_errors": web_result.get("errors", []),
        }
    )


@tool
async def search_web_anysearch(query: str, num_results: int = 8) -> str:
    """明确使用 AnySearch 统一搜索，返回标题、摘要、来源和链接。API Key 可选。"""
    return _dump(await async_search_web_anysearch(query, num_results=num_results))


@tool
async def search_web_ddgs(query: str, num_results: int = 8, freshness: str | None = None) -> str:
    """明确使用 DDGS 免费元搜索，返回标题、摘要、来源和链接。"""
    allowed_freshness = {None, "qdr:h", "qdr:d", "qdr:w", "qdr:m", "qdr:y"}
    if freshness not in allowed_freshness:
        freshness = None
    timelimit = {"qdr:h": "h", "qdr:d": "d", "qdr:w": "w", "qdr:m": "m", "qdr:y": "y"}.get(freshness or "")
    return _dump(await async_search_web_ddgs(query, num_results=num_results, timelimit=timelimit))


TOOLS = [search_web, fetch_web_content]
