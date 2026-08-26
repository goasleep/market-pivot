"""Structured provider boundary for off-exchange open-ended public funds."""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timezone
from typing import Any, Protocol

import akshare as ak
import pandas as pd

from data.source_registry import provenance
from models.fund_data import MoneyFundYieldPoint, OpenFundExposure, OpenFundNavPoint, OpenFundProfile, ProviderResult

SUPPORTED_OPEN_FUND_CATEGORIES = {
    "equity",
    "hybrid",
    "bond",
    "money_market",
    "index",
    "enhanced_index",
}
RECOGNIZED_LIMITED_CATEGORIES = {"qdii", "fof"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    return frame.to_dict("records") if isinstance(frame, pd.DataFrame) and not frame.empty else []


def _code(value: Any) -> str:
    text = str(value or "").strip()
    return text.zfill(6) if text.isdigit() else text


def _number(value: Any) -> float | None:
    if value is None or value == "" or pd.isna(value):
        return None
    text = str(value).replace(",", "").replace("%", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _amount_cny(value: Any) -> float | None:
    text = str(value or "").replace(",", "").strip()
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(亿|万)?", text)
    if not match:
        return None
    amount = float(match.group(1))
    return round(amount * {"亿": 100_000_000, "万": 10_000}.get(match.group(2) or "", 1), 2)


def _date_text(value: Any) -> str | None:
    match = re.search(r"(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})", str(value or ""))
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}" if match else None


def _overview_map(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    if {"item", "value"} <= set(records[0]):
        return {str(row.get("item") or ""): row.get("value") for row in records}
    return records[0]


def normalize_open_fund_category(value: Any) -> str:
    text = str(value or "").lower()
    if "qdii" in text:
        return "qdii"
    if "fof" in text:
        return "fof"
    if "指数增强" in text or "增强指数" in text:
        return "enhanced_index"
    if "股票" in text:
        return "equity"
    if "混合" in text:
        return "hybrid"
    if "债券" in text or "债券型" in text:
        return "bond"
    if "货币" in text:
        return "money_market"
    if "指数" in text:
        return "index"
    return "unknown"


def _share_class(name: str) -> str | None:
    match = re.search(r"(?:[-－ ]|基金)?([ABCEDI])(?:类)?$", name, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _find_row(records: list[dict[str, Any]], fund_code: str) -> dict[str, Any]:
    for row in records:
        if _code(row.get("基金代码") or row.get("代码") or row.get("fund_code")) == fund_code:
            return row
    return {}


def _dynamic_value(row: dict[str, Any], *needles: str) -> Any:
    for key, value in row.items():
        normalized = str(key).lower()
        if all(needle.lower() in normalized for needle in needles):
            return value
    return None


def _report_date(value: str | None) -> date | None:
    text = str(value or "").strip()
    quarter = re.search(r"(20\d{2}).*?([1-4])\s*季", text)
    if quarter:
        year, number = int(quarter.group(1)), int(quarter.group(2))
        month = number * 3
        day = 31 if month in {3, 12} else 30
        return date(year, month, day)
    parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


async def fetch_open_fund_catalog() -> list[dict[str, Any]]:
    return _records(await asyncio.to_thread(ak.fund_name_em))


async def fetch_open_fund_daily() -> list[dict[str, Any]]:
    return _records(await asyncio.to_thread(ak.fund_open_fund_daily_em))


async def fetch_money_fund_daily() -> list[dict[str, Any]]:
    return _records(await asyncio.to_thread(ak.fund_money_fund_daily_em))


async def fetch_open_fund_overview(fund_code: str) -> list[dict[str, Any]]:
    return _records(await asyncio.to_thread(ak.fund_overview_em, symbol=fund_code))


class OpenFundDataProvider(Protocol):
    async def resolve_instrument(self, fund_code: str) -> ProviderResult[OpenFundProfile]: ...

    async def profile(self, fund_code: str) -> ProviderResult[OpenFundProfile]: ...

    async def nav_history(
        self,
        fund_code: str,
        product_category: str = "unknown",
    ) -> ProviderResult[list[dict[str, Any]]]: ...

    async def money_yield_history(self, fund_code: str) -> ProviderResult[list[dict[str, Any]]]: ...

    async def fees(self, fund_code: str) -> ProviderResult[dict[str, Any]]: ...

    async def exposure(self, fund_code: str, year: str = "") -> ProviderResult[OpenFundExposure]: ...

    async def universe(self, product_category: str) -> ProviderResult[list[dict[str, Any]]]: ...


class AkShareOpenFundDataProvider:
    async def resolve_instrument(self, fund_code: str) -> ProviderResult[OpenFundProfile]:
        normalized = _code(fund_code)
        fetched_at = _now()
        if len(normalized) != 6 or not normalized.isdigit():
            return ProviderResult(
                status="data_unavailable",
                fetched_at=fetched_at,
                freshness="24h",
                errors=[{"code": "invalid_open_fund_code", "message": "场外基金代码必须是六位数字"}],
            )
        try:
            catalog = await fetch_open_fund_catalog()
        except Exception as exc:
            catalog = []
            error = str(exc)[:300]
        else:
            error = ""
        row = _find_row(catalog, normalized)
        if not row:
            return ProviderResult(
                status="data_unavailable",
                fetched_at=fetched_at,
                freshness="24h",
                sources=provenance("akshare", freshness="24h", status="unavailable"),
                errors=[{"code": "open_fund_not_verified", "message": error or "Provider 未核验到该场外基金"}],
            )
        name = str(row.get("基金简称") or row.get("基金名称") or row.get("name") or normalized).strip()
        category = normalize_open_fund_category(row.get("基金类型") or row.get("类型"))
        status = "limited" if category in RECOGNIZED_LIMITED_CATEGORIES else "available"
        errors = (
            [{"code": "category_recognized_but_limited", "message": f"{category} 首期只提供身份识别和数据缺口"}]
            if status == "limited"
            else []
        )
        return ProviderResult(
            status=status,
            data=OpenFundProfile(
                fund_code=normalized,
                name=name,
                product_category=category,
                share_class=_share_class(name),
                verified=True,
            ),
            as_of=fetched_at,
            fetched_at=fetched_at,
            freshness="24h",
            sources=provenance("akshare", as_of=fetched_at, freshness="24h", status=status),
            errors=errors,
        )

    async def profile(self, fund_code: str) -> ProviderResult[OpenFundProfile]:
        result = await self.resolve_instrument(fund_code)
        if result.data is None or result.status == "data_unavailable":
            return result
        try:
            daily = (
                await fetch_money_fund_daily()
                if result.data.product_category == "money_market"
                else await fetch_open_fund_daily()
            )
        except Exception:
            daily = []
        row = _find_row(daily, result.data.fund_code)
        if row:
            daily_name = str(row.get("基金简称") or row.get("基金名称") or "").strip()
            if daily_name and daily_name != result.data.name:
                result.status = "conflicting"
                result.errors.append(
                    {
                        "code": "profile_source_conflict",
                        "message": f"产品目录名称 {result.data.name} 与净值源名称 {daily_name} 不一致",
                    }
                )
            result.data.subscription_status = str(row.get("申购状态") or "") or None
            result.data.redemption_status = str(row.get("赎回状态") or "") or None
        try:
            overview = _overview_map(await fetch_open_fund_overview(result.data.fund_code))
        except Exception as exc:
            overview = {}
            result.errors.append({"code": "profile_metadata_unavailable", "message": str(exc)[:200]})
        if overview:
            result.data.manager = str(overview.get("基金管理人") or overview.get("基金公司") or "") or None
            result.data.inception_date = _date_text(overview.get("成立日期/规模") or overview.get("成立时间"))
            result.data.management_fee_rate = _number(overview.get("管理费率"))
            result.data.custody_fee_rate = _number(overview.get("托管费率"))
            result.data.sales_service_fee_rate = _number(overview.get("销售服务费率"))
            result.data.size_cny = _amount_cny(overview.get("净资产规模") or overview.get("最新规模"))
            overview_codes = re.findall(r"(?<!\d)\d{6}(?!\d)", str(overview.get("基金代码") or ""))
            if overview_codes and overview_codes[0] != result.data.fund_code:
                result.data.parent_fund_code = overview_codes[0]
            overview_as_of = _date_text(overview.get("净资产规模"))
            if overview_as_of:
                result.as_of = overview_as_of
        else:
            result.status = "limited" if result.status == "available" else result.status
            if not any(error.get("code") == "profile_metadata_unavailable" for error in result.errors):
                result.errors.append(
                    {"code": "profile_metadata_unavailable", "message": "管理人、规模、成立日期和费率元数据不可用"}
                )
        return result

    async def nav_history(
        self,
        fund_code: str,
        product_category: str = "unknown",
    ) -> ProviderResult[list[dict[str, Any]]]:
        normalized = _code(fund_code)
        fetched_at = _now()
        pricing_basis = "money_yield" if product_category == "money_market" else "nav"
        try:
            if product_category == "money_market":
                frame = await asyncio.to_thread(ak.fund_money_fund_info_em, symbol=normalized)
                rows = []
                for row in _records(frame):
                    point = MoneyFundYieldPoint(
                        date=str(row.get("净值日期") or row.get("日期") or ""),
                        yield_per_10k=_number(row.get("每万份收益") or row.get("万份收益")),
                        seven_day_annualized=_number(row.get("7日年化收益率") or row.get("七日年化收益率")),
                    )
                    rows.append(point.model_dump(mode="json"))
            else:
                frame = await asyncio.to_thread(
                    ak.fund_open_fund_info_em,
                    symbol=normalized,
                    indicator="单位净值走势",
                    period="成立来",
                )
                rows = []
                for row in _records(frame):
                    point = OpenFundNavPoint(
                        date=str(row.get("净值日期") or row.get("日期") or ""),
                        unit_nav=_number(row.get("单位净值")),
                        cumulative_nav=_number(row.get("累计净值")),
                        pct_chg=_number(row.get("日增长率")),
                    )
                    rows.append(point.model_dump(mode="json"))
        except Exception as exc:
            rows = []
            error = str(exc)[:300]
        else:
            error = ""
        rows = [row for row in rows if row.get("date")]
        as_of = str(rows[-1]["date"]) if rows else None
        return ProviderResult(
            status="available" if rows else "data_unavailable",
            data=rows or None,
            as_of=as_of,
            fetched_at=fetched_at,
            freshness="immutable_incremental",
            sources=provenance("akshare", as_of=as_of, freshness="historical"),
            errors=[] if rows else [{"code": f"{pricing_basis}_unavailable", "message": error or "历史数据不可用"}],
        )

    async def money_yield_history(self, fund_code: str) -> ProviderResult[list[dict[str, Any]]]:
        return await self.nav_history(fund_code, "money_market")

    async def fees(self, fund_code: str) -> ProviderResult[dict[str, Any]]:
        normalized = _code(fund_code)
        fetched_at = _now()
        indicators = ("运作费用", "申购费率（前端）", "赎回费率", "交易状态", "交易确认日")
        data: dict[str, Any] = {}
        errors: list[dict[str, str]] = []
        for indicator in indicators:
            try:
                frame = await asyncio.to_thread(ak.fund_fee_em, symbol=normalized, indicator=indicator)
                data[indicator] = _records(frame)
            except Exception as exc:
                errors.append({"code": "fee_field_unavailable", "message": f"{indicator}: {str(exc)[:160]}"})
        available = any(data.values())
        return ProviderResult(
            status="available" if available and not errors else "limited" if available else "data_unavailable",
            data=data or None,
            as_of=fetched_at,
            fetched_at=fetched_at,
            freshness="24h",
            sources=provenance("akshare", as_of=fetched_at, freshness="24h"),
            errors=errors,
        )

    async def exposure(self, fund_code: str, year: str = "") -> ProviderResult[OpenFundExposure]:
        normalized = _code(fund_code)
        fetched_at = _now()
        report_year = year or str(datetime.now().year)
        calls = (
            ("stock_holdings", ak.fund_portfolio_hold_em),
            ("bond_holdings", ak.fund_portfolio_bond_hold_em),
            ("industry_allocation", ak.fund_portfolio_industry_allocation_em),
        )
        values: dict[str, list[dict[str, Any]]] = {}
        errors: list[dict[str, str]] = []
        for key, function in calls:
            try:
                values[key] = _records(await asyncio.to_thread(function, symbol=normalized, date=report_year))
            except Exception as exc:
                values[key] = []
                errors.append({"code": f"{key}_unavailable", "message": str(exc)[:200]})
        report_dates = [
            str(row.get("季度") or row.get("报告期") or row.get("截止时间") or "")
            for rows in values.values()
            for row in rows
        ]
        report_date = max((item for item in report_dates if item), default=None)
        exposure = OpenFundExposure(fund_code=normalized, report_date=report_date, **values)
        available = any(values.values())
        parsed_report_date = _report_date(report_date)
        stale = parsed_report_date is not None and (datetime.now(timezone.utc).date() - parsed_report_date).days > 120
        if stale:
            errors.append({"code": "exposure_stale", "message": "持仓报告期距今超过 120 天"})
        status = (
            "available"
            if available and not errors and not stale
            else "limited"
            if available
            else "data_unavailable"
        )
        return ProviderResult(
            status=status,
            data=exposure if available else None,
            as_of=report_date,
            fetched_at=fetched_at,
            freshness="report_period",
            sources=provenance("akshare", as_of=report_date, freshness="report_period"),
            errors=errors,
        )

    async def universe(self, product_category: str) -> ProviderResult[list[dict[str, Any]]]:
        fetched_at = _now()
        category_names = {
            "equity": "股票型",
            "hybrid": "混合型",
            "bond": "债券型",
            "index": "指数型",
            "enhanced_index": "指数型",
        }
        try:
            if product_category == "money_market":
                frame = await asyncio.to_thread(ak.fund_money_rank_em)
            elif product_category in category_names:
                frame = await asyncio.to_thread(ak.fund_open_fund_rank_em, symbol=category_names[product_category])
            elif product_category in RECOGNIZED_LIMITED_CATEGORIES:
                return ProviderResult(
                    status="limited",
                    fetched_at=fetched_at,
                    freshness="24h",
                    errors=[
                        {
                            "code": "category_recognized_but_limited",
                            "message": f"{product_category} 暂不提供正式筛选",
                        }
                    ],
                )
            else:
                return ProviderResult(
                    status="data_unavailable",
                    fetched_at=fetched_at,
                    freshness="24h",
                    errors=[{"code": "category_required", "message": "场外基金筛选必须指定同一产品类别"}],
                )
            rows = _records(frame)
        except Exception as exc:
            rows = []
            error = str(exc)[:300]
        else:
            error = ""
        normalized_rows = []
        for row in rows:
            normalized_rows.append(
                {
                    **row,
                    "fund_code": _code(row.get("基金代码") or row.get("代码")),
                    "name": str(row.get("基金简称") or row.get("基金名称") or ""),
                    "product_category": product_category,
                    "as_of": str(row.get("日期") or row.get("净值日期") or fetched_at),
                    "unit_nav": _number(row.get("单位净值")),
                    "cumulative_nav": _number(row.get("累计净值")),
                    "seven_day_annualized": _number(_dynamic_value(row, "7日年化")),
                    "yield_per_10k": _number(_dynamic_value(row, "万份收益")),
                    "provider_verified": True,
                }
            )
        return ProviderResult(
            status="available" if normalized_rows else "data_unavailable",
            data=normalized_rows or None,
            as_of=max((row["as_of"] for row in normalized_rows), default=None),
            fetched_at=fetched_at,
            freshness="24h",
            sources=provenance("akshare", as_of=fetched_at, freshness="24h"),
            errors=(
                []
                if normalized_rows
                else [{"code": "open_fund_universe_unavailable", "message": error or "无候选数据"}]
            ),
        )


open_fund_data_provider: OpenFundDataProvider = AkShareOpenFundDataProvider()
