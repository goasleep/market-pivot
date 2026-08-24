"""Searchable, provider-neutral catalog of structured financial datasets."""

from __future__ import annotations

import re
from collections.abc import Iterable

from models.market_data import DatasetDefinition, DatasetField

DIVIDEND_DATASET = DatasetDefinition(
    dataset_id="cn_a_share_cash_dividends",
    title="A股上市公司现金分红实施明细",
    description=(
        "沪深北交易所A股上市公司的年度现金分红方案。包含报告年度、实施状态及税前每10股现金分红，"
        "适用于连续分红筛选、年度聚合和累计每股分红排名。"
    ),
    asset_types=["stock"],
    aliases=[
        "A股分红",
        "现金分红",
        "每股分红",
        "派息",
        "股息",
        "dividend",
        "cash dividend",
    ],
    fields=[
        DatasetField(name="ticker", dtype="string", description="六位A股代码", aliases=["股票代码"]),
        DatasetField(name="name", dtype="string", description="股票简称", aliases=["公司简称"]),
        DatasetField(name="fiscal_year", dtype="int", description="分红对应报告年度", aliases=["报告年度"]),
        DatasetField(
            name="cash_dividend_per_10_shares",
            dtype="float",
            description="税前每10股现金分红",
            unit="CNY/10 shares",
            aliases=["现金分红比例", "派息比例"],
        ),
        DatasetField(
            name="cash_dividend_per_share",
            dtype="float",
            description="税前每股现金分红",
            unit="CNY/share",
            aliases=["每股分红"],
        ),
        DatasetField(name="status", dtype="string", description="方案进度", aliases=["实施状态"]),
        DatasetField(name="record_date", dtype="date", description="股权登记日"),
        DatasetField(name="ex_dividend_date", dtype="date", description="除权除息日"),
    ],
    provider_ids=["akshare"],
    capabilities=["screen", "rank", "aggregate", "time_series"],
    temporal_field="fiscal_year",
    required_query_terms=["a股", "股票", "上市公司", "公司分红", "每股分红"],
)


class MarketDataCatalog:
    def __init__(self, datasets: Iterable[DatasetDefinition]) -> None:
        self._datasets = {item.dataset_id: item for item in datasets}

    def get(self, dataset_id: str) -> DatasetDefinition:
        try:
            return self._datasets[dataset_id]
        except KeyError as exc:
            raise ValueError(f"未知市场数据集: {dataset_id}") from exc

    def supports_asset_type(self, asset_type: str) -> bool:
        """Return whether the catalog contains a dataset for this research asset type."""
        return any(self.supports_dataset_asset_type(dataset, asset_type) for dataset in self._datasets.values())

    @staticmethod
    def supports_dataset_asset_type(dataset: DatasetDefinition, asset_type: str) -> bool:
        return asset_type in dataset.asset_types or (
            asset_type == "fund" and any(item in dataset.asset_types for item in ("etf", "lof"))
        )

    def search(self, query: str, *, asset_type: str | None = None, limit: int = 5) -> list[DatasetDefinition]:
        """Rank datasets by catalog vocabulary; this selects data, not conversation routes."""
        normalized = re.sub(r"\s+", " ", query.lower()).strip()
        query_tokens = {
            token
            for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", normalized)
            if len(token) >= 2 and not token.isdigit()
        }
        scored: list[tuple[int, DatasetDefinition]] = []
        for dataset in self._datasets.values():
            if asset_type and not self.supports_dataset_asset_type(dataset, asset_type):
                continue
            if dataset.required_query_terms and not any(
                term.lower() in normalized for term in dataset.required_query_terms
            ):
                continue
            phrases = [dataset.title, dataset.description, dataset.dataset_id, *dataset.aliases]
            score = 0
            for phrase in phrases:
                phrase_lower = phrase.lower()
                if phrase_lower in normalized:
                    score += 8
                score += sum(1 for token in query_tokens if token and token in phrase_lower)
            # Chinese financial concepts are often embedded in a full sentence rather than tokenized.
            score += sum(5 for alias in dataset.aliases if alias.lower() in normalized)
            if score:
                scored.append((score, dataset))
        return [item for _, item in sorted(scored, key=lambda row: (-row[0], row[1].dataset_id))[:limit]]

    def list(self) -> list[DatasetDefinition]:
        return list(self._datasets.values())


market_data_catalog = MarketDataCatalog([DIVIDEND_DATASET])
