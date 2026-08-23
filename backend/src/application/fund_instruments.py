"""Context-aware fund instrument resolution without treating arbitrary numbers as tickers."""

from __future__ import annotations

import re
from collections.abc import Iterable

from models.fund_task import FundInstrumentRef, InstrumentResolutionStatus

_EXPLICIT_CODE = re.compile(
    r"(?:基金代码|代码|ticker|ETF|LOF|场内基金)\s*[:：为是]?\s*([0-9]{6})(?!\d)",
    re.IGNORECASE,
)
_ETF_PREFIXES = ("15", "51", "56", "58")
_LOF_PREFIXES = ("16", "50")


def resolve_fund_instruments(
    message: str,
    tickers: Iterable[str] = (),
    *,
    asset_type: str = "stock",
) -> list[FundInstrumentRef]:
    explicit = [match.group(1) for match in _EXPLICIT_CODE.finditer(message)]
    candidates = list(dict.fromkeys([*explicit, *(str(item) for item in tickers)]))
    results: list[FundInstrumentRef] = []
    for code in candidates:
        is_etf = code.startswith(_ETF_PREFIXES)
        is_lof = code.startswith(_LOF_PREFIXES)
        strong_context = code in explicit or asset_type in {"etf", "lof"}
        if not strong_context or not (is_etf or is_lof):
            results.append(
                FundInstrumentRef(
                    status=InstrumentResolutionStatus.AMBIGUOUS,
                    fund_code=code,
                    resolution_reason="六位数字缺少可验证的基金代码上下文",
                )
            )
            continue
        product_type = "etf" if is_etf else "lof"
        results.append(
            FundInstrumentRef(
                status=InstrumentResolutionStatus.VERIFIED,
                fund_code=code,
                exchange_ticker=code,
                product_type=product_type,
                trading_venue="exchange",
                provider_id="exchange-code-rules",
                resolution_reason="代码前缀与明确的ETF/LOF上下文一致；数据工具仍需核验产品存在性",
            )
        )
    return results
