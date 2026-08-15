"""A-share trading-day calendar with a safe weekday fallback."""

from __future__ import annotations

import threading
import time
from datetime import date

from loguru import logger


class TradingCalendar:
    """Load official trading dates lazily and cache them in memory.

    AkShare is an external data source, so an unavailable calendar must not
    make the simulator crash.  In that case weekends remain blocked and
    weekdays are allowed; the state is logged so operators can see the
    degraded-calendar condition.
    """

    def __init__(self, ttl_seconds: int = 7 * 86400):
        self.ttl_seconds = ttl_seconds
        self._trade_dates: set[date] | None = None
        self._loaded_at = 0.0
        self._lock = threading.RLock()

    def is_trading_day(self, target: date) -> bool:
        if target.weekday() >= 5:
            return False
        self._ensure_loaded()
        with self._lock:
            if self._trade_dates is None:
                return True
            return target in self._trade_dates

    def refresh(self) -> bool:
        try:
            import akshare as ak

            frame = ak.tool_trade_date_hist_sina()
            column = "trade_date" if "trade_date" in frame.columns else frame.columns[0]
            dates = {
                value.date() if hasattr(value, "date") else date.fromisoformat(str(value)[:10])
                for value in frame[column].dropna()
            }
            if not dates:
                raise ValueError("official trading calendar returned no dates")
            with self._lock:
                self._trade_dates = dates
                self._loaded_at = time.monotonic()
            return True
        except Exception as exc:
            with self._lock:
                self._trade_dates = None
                self._loaded_at = time.monotonic()
            logger.warning("Official A-share trading calendar unavailable; using weekday fallback: {}", exc)
            return False

    def _ensure_loaded(self) -> None:
        with self._lock:
            fresh = self._trade_dates is not None and time.monotonic() - self._loaded_at < self.ttl_seconds
            recently_failed = self._trade_dates is None and time.monotonic() - self._loaded_at < 300
        if fresh or recently_failed:
            return
        self.refresh()


trading_calendar = TradingCalendar()


def is_trading_day(target: date) -> bool:
    return trading_calendar.is_trading_day(target)
