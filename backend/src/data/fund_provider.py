"""ETF/LOF market-data boundary.

The legacy AkShare module remains the low-level compatibility facade. Research
and execution code should import fund data through this module so asset-specific
normalization stays isolated from stock data.
"""

from data.akshare_provider import (
    async_get_fund_history,
    async_get_fund_nav_history,
    async_get_fund_realtime,
    get_fund_history,
    get_fund_nav_history,
    get_fund_realtime,
)

__all__ = [
    "async_get_fund_history",
    "async_get_fund_nav_history",
    "async_get_fund_realtime",
    "get_fund_history",
    "get_fund_nav_history",
    "get_fund_realtime",
]
