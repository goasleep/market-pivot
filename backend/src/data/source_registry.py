"""Data-source registry and provenance helpers.

Business tools answer *what* the agent needs.  This module describes *where*
the evidence came from without forcing callers to know provider-specific
names or metadata conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


@dataclass(frozen=True)
class DataSourceSpec:
    source_id: str
    name: str
    kind: str
    capabilities: tuple[str, ...]
    priority: int
    citation_required: bool = False


class DataSourceRegistry:
    def __init__(self, sources: Iterable[DataSourceSpec]) -> None:
        self._sources = {source.source_id: source for source in sources}
        self._aliases = {
            "AnySearch": "anysearch",
            "DDGS": "ddgs",
            "DDGS metasearch": "ddgs",
            "Serper / Google Search": "serper",
            "AkShare / 东方财富": "akshare",
            "AkShare": "akshare",
        }

    def get(self, source_id: str) -> DataSourceSpec:
        normalized = self._aliases.get(source_id, source_id)
        try:
            return self._sources[normalized]
        except KeyError as exc:
            raise ValueError(f"未知数据源: {source_id}") from exc

    def resolve(self, capability: str, preferred: str | None = None) -> DataSourceSpec:
        """Resolve the preferred source or the highest-priority source."""
        if preferred:
            source = self.get(preferred)
            if capability not in source.capabilities:
                raise ValueError(f"数据源 {preferred} 不支持能力 {capability}")
            return source
        candidates = [source for source in self._sources.values() if capability in source.capabilities]
        if not candidates:
            raise ValueError(f"没有支持能力 {capability} 的数据源")
        return min(candidates, key=lambda source: source.priority)

    def metadata(
        self,
        source_id: str,
        *,
        fetched_at: str | None = None,
        as_of: str | None = None,
        freshness: str | None = None,
        status: str = "available",
        url: str | None = None,
    ) -> dict[str, Any]:
        source = self.get(source_id)
        return {
            "source_id": source.source_id,
            "name": source.name,
            "kind": source.kind,
            "capabilities": list(source.capabilities),
            "citation_required": source.citation_required,
            "status": status,
            "fetched_at": fetched_at or utc_now(),
            "as_of": as_of,
            "freshness": freshness,
            "url": url,
        }

    def metadata_for_labels(
        self,
        labels: Iterable[str],
        *,
        fetched_at: str | None = None,
        as_of: str | None = None,
        freshness: str | None = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for label in labels:
            try:
                metadata = self.metadata(
                    label,
                    fetched_at=fetched_at,
                    as_of=as_of,
                    freshness=freshness,
                )
            except ValueError:
                continue
            if metadata["source_id"] not in seen:
                seen.add(metadata["source_id"])
                result.append(metadata)
        return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


data_sources = DataSourceRegistry(
    [
        DataSourceSpec(
            "akshare",
            "AkShare / 东方财富",
            "market_data",
            (
                "market.quote",
                "market.history",
                "market.fundamentals",
                "market.news",
                "market.screen",
                "market.dividends",
            ),
            priority=10,
        ),
        DataSourceSpec(
            "eastmoney",
            "东方财富",
            "market_data",
            ("market.quote", "market.history", "market.fundamentals"),
            priority=10,
        ),
        DataSourceSpec(
            "tencent",
            "腾讯证券",
            "market_data",
            ("market.history",),
            priority=20,
        ),
        DataSourceSpec(
            "sina",
            "新浪财经",
            "market_data",
            ("market.quote", "market.history"),
            priority=30,
        ),
        DataSourceSpec(
            "serper",
            "Serper / Google Search",
            "web_search",
            ("web.search",),
            priority=10,
            citation_required=True,
        ),
        DataSourceSpec(
            "anysearch",
            "AnySearch",
            "web_search",
            ("web.search",),
            priority=20,
            citation_required=True,
        ),
        DataSourceSpec(
            "ddgs",
            "DDGS metasearch",
            "web_search",
            ("web.search",),
            priority=30,
            citation_required=True,
        ),
        DataSourceSpec(
            "web_page",
            "网页正文",
            "web_content",
            ("web.content",),
            priority=10,
            citation_required=True,
        ),
        DataSourceSpec(
            "paper_trading_db",
            "本地纸面交易数据库",
            "simulation",
            ("simulation.portfolio", "simulation.orders"),
            priority=10,
        ),
        DataSourceSpec(
            "derived",
            "系统确定性计算",
            "derived",
            ("research.indicator", "research.risk", "research.plan"),
            priority=10,
        ),
        DataSourceSpec(
            "sandbox",
            "受限策略研究沙盒",
            "sandbox",
            ("research.strategy_code", "research.backtest", "research.data_transform"),
            priority=10,
        ),
        DataSourceSpec(
            "methodology_library",
            "本地投资方法论库",
            "knowledge",
            ("knowledge.methodology",),
            priority=10,
            citation_required=True,
        ),
    ]
)


def provenance(
    source_id: str,
    *,
    fetched_at: str | None = None,
    as_of: str | None = None,
    freshness: str | None = None,
    status: str = "available",
    url: str | None = None,
) -> list[dict[str, Any]]:
    return [
        data_sources.metadata(
            source_id,
            fetched_at=fetched_at,
            as_of=as_of,
            freshness=freshness,
            status=status,
            url=url,
        )
    ]


def provenance_for_labels(
    labels: Iterable[str],
    *,
    fetched_at: str | None = None,
    as_of: str | None = None,
    freshness: str | None = None,
) -> list[dict[str, Any]]:
    return data_sources.metadata_for_labels(
        labels,
        fetched_at=fetched_at,
        as_of=as_of,
        freshness=freshness,
    )
