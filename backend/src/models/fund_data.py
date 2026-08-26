"""Provider-neutral contracts for exchange-traded and open-ended funds."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")
ProviderStatus = Literal["available", "limited", "data_unavailable", "conflicting"]


class ProviderResult(BaseModel, Generic[T]):
    status: ProviderStatus
    data: T | None = None
    as_of: str | None = None
    fetched_at: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    freshness: str = "unknown"
    errors: list[dict[str, str]] = Field(default_factory=list)


class ExchangeFundProfile(BaseModel):
    ticker: str
    name: str
    exchange: Literal["SH", "SZ", "unknown"] = "unknown"
    asset_type: Literal["etf", "lof"]
    is_qdii: bool = False
    verified: bool = False
    tracking_index: str | None = None
    manager: str | None = None
    listing_date: str | None = None
    management_fee_rate: float | None = None
    custody_fee_rate: float | None = None
    size_cny: float | None = None
    shares: float | None = None


class ExchangeFundExposure(BaseModel):
    ticker: str
    holdings: list[dict[str, Any]] = Field(default_factory=list)
    sector_exposure: list[dict[str, Any]] = Field(default_factory=list)
    asset_exposure: list[dict[str, Any]] = Field(default_factory=list)
    top10_concentration: float | None = None
    report_date: str | None = None


class TrackingMetrics(BaseModel):
    ticker: str
    tracking_difference: float | None = None
    tracking_error: float | None = None
    sample_start: str | None = None
    sample_end: str | None = None
    observations: int = 0
    benchmark_source: str | None = None


class LiquidityMetrics(BaseModel):
    ticker: str
    average_amount: float | None = None
    turnover_rate: float | None = None
    spread_bps: float | None = None
    order_participation_rate: float | None = None
    impact_risk: Literal["low", "medium", "high", "unknown"] = "unknown"


class PremiumDiscountMetrics(BaseModel):
    ticker: str
    price: float | None = None
    nav_or_iopv: float | None = None
    premium_discount_rate: float | None = None
    price_at: str | None = None
    nav_at: str | None = None
    comparable: bool = False
    reason: str = ""


class OpenFundProfile(BaseModel):
    fund_code: str
    name: str
    asset_type: Literal["open_fund"] = "open_fund"
    product_category: Literal[
        "equity",
        "hybrid",
        "bond",
        "money_market",
        "index",
        "enhanced_index",
        "qdii",
        "fof",
        "unknown",
    ] = "unknown"
    share_class: str | None = None
    parent_fund_code: str | None = None
    manager: str | None = None
    inception_date: str | None = None
    management_fee_rate: float | None = None
    custody_fee_rate: float | None = None
    sales_service_fee_rate: float | None = None
    size_cny: float | None = None
    subscription_status: str | None = None
    redemption_status: str | None = None
    verified: bool = False


class OpenFundNavPoint(BaseModel):
    date: str
    unit_nav: float | None = None
    cumulative_nav: float | None = None
    pct_chg: float | None = None


class MoneyFundYieldPoint(BaseModel):
    date: str
    yield_per_10k: float | None = None
    seven_day_annualized: float | None = None


class OpenFundExposure(BaseModel):
    fund_code: str
    stock_holdings: list[dict[str, Any]] = Field(default_factory=list)
    bond_holdings: list[dict[str, Any]] = Field(default_factory=list)
    industry_allocation: list[dict[str, Any]] = Field(default_factory=list)
    report_date: str | None = None
