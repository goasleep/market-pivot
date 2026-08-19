"""Agent tools for the plain-text investment methodology library."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import tool

from data.source_registry import provenance
from methodology.library import MethodologyLibrary


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


methodology_library = MethodologyLibrary()


@tool
async def search_methodology(
    query: str,
    methodology_type: str | None = None,
    asset_type: str | None = None,
    horizon: str | None = None,
    limit: int = 5,
) -> str:
    """检索投资理念、投资经验、市场观点和论文方法摘要。

    该工具只提供方法论和研究依据，不提供实时行情，不替代回测，也不能
    单独作为买卖结论。涉及当前价格、指标、风险或历史收益时必须继续调用
    对应的结构化数据或确定性计算工具。
    """
    results = await asyncio.to_thread(
        methodology_library.search,
        query,
        methodology_type=methodology_type,
        asset_type=asset_type,
        horizon=horizon,
        limit=limit,
    )
    return _dump(
        {
            "data_type": "methodology_search",
            "query": query,
            "filters": {
                "type": methodology_type,
                "asset_type": asset_type,
                "horizon": horizon,
                "limit": limit,
            },
            "results": results,
            "provenance": provenance("methodology_library", freshness="versioned_local_text"),
            "usage_note": "方法论用于形成和解释可验证假设；当前行情、技术指标、风险和回测必须使用对应工具。",
        }
    )


TOOLS = [search_methodology]
