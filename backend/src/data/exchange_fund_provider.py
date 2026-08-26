"""Provider-neutral data boundary for exchange-traded ETF and LOF products."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from data.akshare_provider import (
    async_get_exchange_fund_history,
    async_get_exchange_fund_nav_history,
    async_get_exchange_fund_quote,
    get_exchange_fund_history,
    get_exchange_fund_nav_history,
    get_exchange_fund_quote,
)
from data.source_registry import provenance
from models.fund_data import ExchangeFundProfile, ProviderResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exchange(ticker: str) -> str:
    return "SH" if ticker.startswith(("50", "51", "52", "56", "58")) else "SZ" if ticker.isdigit() else "unknown"


class ExchangeFundDataProvider(Protocol):
    async def resolve_instrument(
        self, ticker: str, asset_type: str = "etf"
    ) -> ProviderResult[ExchangeFundProfile]: ...

    async def profile(self, ticker: str, asset_type: str = "etf") -> ProviderResult[ExchangeFundProfile]: ...

    async def history(
        self, ticker: str, asset_type: str = "etf", **kwargs: Any
    ) -> ProviderResult[list[dict[str, Any]]]: ...

    async def nav_history(self, ticker: str, asset_type: str = "etf") -> ProviderResult[list[dict[str, Any]]]: ...


class AkShareExchangeFundDataProvider:
    """Normalize free-source responses; unsupported fields remain explicit gaps."""

    async def resolve_instrument(
        self,
        ticker: str,
        asset_type: str = "etf",
    ) -> ProviderResult[ExchangeFundProfile]:
        normalized = str(ticker).strip().lower().removeprefix("sh").removeprefix("sz")
        fetched_at = _now()
        if len(normalized) != 6 or not normalized.isdigit() or asset_type not in {"etf", "lof"}:
            return ProviderResult(
                status="data_unavailable",
                fetched_at=fetched_at,
                freshness="24h",
                errors=[{"code": "invalid_instrument", "message": "代码或资产类型无效"}],
            )
        try:
            quote = await async_get_exchange_fund_quote(normalized, asset_type=asset_type)
        except Exception as exc:
            quote = {}
            error = str(exc)[:300]
        else:
            error = ""
        name = str(quote.get("name") or quote.get("名称") or "").strip()
        # Prefixes may narrow candidates, but only a provider response verifies one.
        verified = bool(name or quote.get("price") or quote.get("ticker"))
        if not verified:
            return ProviderResult(
                status="data_unavailable",
                fetched_at=fetched_at,
                freshness="24h",
                sources=provenance("akshare", freshness="24h", status="unavailable"),
                errors=[{"code": "instrument_not_verified", "message": error or "Provider 未核验到该场内基金"}],
            )
        qdii = "qdii" in name.lower() or "跨境" in name
        profile = ExchangeFundProfile(
            ticker=normalized,
            name=name or normalized,
            exchange=_exchange(normalized),
            asset_type=asset_type,
            is_qdii=qdii,
            verified=True,
            size_cny=quote.get("total_mv") or quote.get("market_cap"),
        )
        return ProviderResult(
            status="available",
            data=profile,
            fetched_at=fetched_at,
            as_of=str(quote.get("data_date") or quote.get("date") or fetched_at),
            freshness="24h",
            sources=provenance("akshare", freshness="24h"),
        )

    async def profile(self, ticker: str, asset_type: str = "etf") -> ProviderResult[ExchangeFundProfile]:
        return await self.resolve_instrument(ticker, asset_type)

    async def history(
        self,
        ticker: str,
        asset_type: str = "etf",
        **kwargs: Any,
    ) -> ProviderResult[list[dict[str, Any]]]:
        fetched_at = _now()
        try:
            frame = await async_get_exchange_fund_history(ticker, asset_type=asset_type, **kwargs)
            records = frame.to_dict("records")
        except Exception as exc:
            records = []
            error = str(exc)[:300]
        else:
            error = ""
        as_of = str(records[-1].get("date") or "") if records else None
        return ProviderResult(
            status="available" if records else "data_unavailable",
            data=records or None,
            as_of=as_of,
            fetched_at=fetched_at,
            freshness="immutable_incremental",
            sources=provenance("akshare", as_of=as_of, freshness="historical"),
            errors=[] if records else [{"code": "history_unavailable", "message": error or "无历史行情"}],
        )

    async def nav_history(self, ticker: str, asset_type: str = "etf") -> ProviderResult[list[dict[str, Any]]]:
        fetched_at = _now()
        try:
            frame = await async_get_exchange_fund_nav_history(ticker, asset_type=asset_type)
            records = frame.to_dict("records")
        except Exception as exc:
            records = []
            error = str(exc)[:300]
        else:
            error = ""
        as_of = str(records[-1].get("date") or records[-1].get("净值日期") or "") if records else None
        return ProviderResult(
            status="available" if records else "data_unavailable",
            data=records or None,
            as_of=as_of,
            fetched_at=fetched_at,
            freshness="immutable_incremental",
            sources=provenance("akshare", as_of=as_of, freshness="historical"),
            errors=[] if records else [{"code": "nav_unavailable", "message": error or "无净值数据"}],
        )


exchange_fund_data_provider: ExchangeFundDataProvider = AkShareExchangeFundDataProvider()

__all__ = [
    "async_get_exchange_fund_history",
    "async_get_exchange_fund_nav_history",
    "async_get_exchange_fund_quote",
    "get_exchange_fund_history",
    "get_exchange_fund_nav_history",
    "get_exchange_fund_quote",
    "ExchangeFundDataProvider",
    "AkShareExchangeFundDataProvider",
    "exchange_fund_data_provider",
]
