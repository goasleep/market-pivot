"""AkShare data source wrapper with circuit breaker and multi-layer cache.

Provides:
- get_stock_history() - historical daily OHLCV
- get_stock_realtime() - realtime quote
- get_financial_data() - fundamental indicators
- get_stock_news() - news articles

Enhancements over basic version:
- CircuitBreaker: auto-trips after N consecutive failures, half-open probe
- Multi-layer TTL: realtime 60s, daily 24h, financial 7d, news 1h, failures 30s
- Random sleep to avoid rate-limiting
- Exponential backoff retry
"""

import json
import random
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from loguru import logger

from config import settings
from data.database import SQLiteDatabase

# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Circuit breaker for protecting data sources.

    States:
    - CLOSED: normal operation, requests pass through
    - OPEN: tripped, requests fail fast without hitting the source
    - HALF_OPEN: probing - one request allowed to test recovery

    Trip after `failure_threshold` consecutive failures.
    Auto-transition to HALF_OPEN after `recovery_timeout` seconds.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                # Check if we should transition to half-open
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"[CircuitBreaker:{self.name}] OPEN -> HALF_OPEN")
            return self._state

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        state = self.state  # triggers open->half_open check
        if state == self.CLOSED:
            return True
        if state == self.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        return False  # OPEN

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                self._state = self.CLOSED
                self._failure_count = 0
                logger.info(f"[CircuitBreaker:{self.name}] HALF_OPEN -> CLOSED (recovered)")
            else:
                self._failure_count = 0

    def record_failure(self):
        """Record a failed call."""
        with self._lock:
            self._last_failure_time = time.time()
            if self._state == self.HALF_OPEN:
                # Half-open failure -> back to open
                self._state = self.OPEN
                logger.warning(f"[CircuitBreaker:{self.name}] HALF_OPEN -> OPEN (probe failed)")
            else:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    logger.warning(f"[CircuitBreaker:{self.name}] CLOSED -> OPEN (failures={self._failure_count})")

    def reset(self):
        """Manually reset the breaker."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0


# ---------------------------------------------------------------------------
# Enhanced Cache with failure caching
# ---------------------------------------------------------------------------


class DataCache:
    """SQLite-based cache with TTL support and failure caching."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._database = SQLiteDatabase(db_path)

    def get(self, key: str, ttl: int = 3600) -> Any | None:
        row = self._database.get_cache(key)
        if row is None:
            return None
        value_str, timestamp = row
        if time.time() - timestamp > ttl:
            return None
        return json.loads(value_str)

    def set(self, key: str, value: Any):
        self._database.set_cache(key, value)

    def clear(self):
        self._database.clear_cache()


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

_cache = DataCache(settings.database_file_path)

# One breaker per data category
_breakers: dict[str, CircuitBreaker] = {
    "history": CircuitBreaker("akshare_history", failure_threshold=5, recovery_timeout=120),
    "realtime": CircuitBreaker("akshare_realtime", failure_threshold=8, recovery_timeout=30),
    "financial": CircuitBreaker("akshare_financial", failure_threshold=5, recovery_timeout=180),
    "news": CircuitBreaker("akshare_news", failure_threshold=5, recovery_timeout=60),
    "stock_list": CircuitBreaker("akshare_stock_list", failure_threshold=5, recovery_timeout=300),
}

# TTL constants (seconds)
TTL_REALTIME = 60  # 1 min
TTL_DAILY = 86400  # 1 day (historical data doesn't change)
TTL_FINANCIAL = 86400 * 7  # 7 days
TTL_NEWS = 3600  # 1 hour
TTL_FAILURE = 30  # cache failures for 30s to avoid hammering a broken source

# Retry config
MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # base sleep seconds for exponential backoff
BACKOFF_MAX = 10.0
RANDOM_SLEEP_MIN = 0.3
RANDOM_SLEEP_MAX = 1.5


def _random_sleep():
    """Random sleep to avoid rate-limiting."""
    time.sleep(random.uniform(RANDOM_SLEEP_MIN, RANDOM_SLEEP_MAX))


def _retry_with_backoff(
    func: Callable,
    breaker: CircuitBreaker,
    operation_name: str,
    max_retries: int = MAX_RETRIES,
) -> Any:
    """Execute a function with circuit breaker + exponential backoff retry.

    Args:
        func: Callable that returns the result (or raises)
        breaker: CircuitBreaker for this data source
        operation_name: Human-readable name for logging
        max_retries: Max retry attempts

    Returns:
        Result of func, or raises the last exception
    """
    if not breaker.can_execute():
        logger.warning(f"[{operation_name}] Circuit breaker OPEN, skipping request")
        raise CircuitBreakerOpenError(f"Circuit breaker open for {operation_name}")

    last_exc = None
    for attempt in range(max_retries):
        try:
            _random_sleep()
            result = func()
            breaker.record_success()
            return result
        except CircuitBreakerOpenError:
            raise
        except Exception as e:
            last_exc = e
            logger.warning(f"[{operation_name}] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                sleep_time = min(
                    BACKOFF_BASE * (2**attempt) + random.uniform(0, 0.5),
                    BACKOFF_MAX,
                )
                time.sleep(sleep_time)

    breaker.record_failure()
    raise last_exc  # type: ignore[misc]


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


# ---------------------------------------------------------------------------
# Public API (unchanged signatures)
# ---------------------------------------------------------------------------


def _format_ticker(ticker: str) -> str:
    """Ensure ticker is 6-digit code."""
    ticker = ticker.strip()
    if ticker.lower().startswith(("sh", "sz")):
        ticker = ticker[2:]
    return ticker.zfill(6)


def get_stock_history(
    ticker: str,
    start_date: str = "",
    end_date: str = "",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Get historical daily OHLCV data via AkShare.

    Uses circuit breaker + multi-layer cache + retry with backoff.
    """
    ticker = _format_ticker(ticker)
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    else:
        start_date = start_date.replace("-", "")
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")
    else:
        end_date = end_date.replace("-", "")

    cache_key = f"hist:{ticker}:{start_date}:{end_date}:{adjust}"
    cached = _cache.get(cache_key, ttl=TTL_DAILY)
    if cached is not None:
        logger.debug(f"Cache hit: {cache_key}")
        return pd.DataFrame(cached)

    # Check failure cache to avoid hammering a broken source
    fail_key = f"fail:{cache_key}"
    if _cache.get(fail_key, ttl=TTL_FAILURE) is not None:
        logger.debug(f"Failure cache hit: {cache_key}")
        return pd.DataFrame()

    breaker = _breakers["history"]
    logger.info(f"Fetching history for {ticker} ({start_date} - {end_date})")

    def _fetch():
        import akshare as ak

        return ak.stock_zh_a_hist(
            symbol=ticker,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

    try:
        df = _retry_with_backoff(_fetch, breaker, f"history:{ticker}")
        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "换手率": "turnover",
        }
        df = df.rename(columns=col_map)
        df["ticker"] = ticker
        _cache.set(cache_key, df.to_dict(orient="records"))
        return df
    except Exception as e:
        logger.error(f"Failed to fetch history for {ticker}: {e}")
        _cache.set(fail_key, {"error": str(e)})
        return pd.DataFrame()


def get_stock_realtime(ticker: str) -> dict:
    """Get realtime stock quote via AkShare (East Money).

    Uses circuit breaker + cache + retry with backoff.
    """
    ticker = _format_ticker(ticker)
    cache_key = f"rt:{ticker}"
    cached = _cache.get(cache_key, ttl=TTL_REALTIME)
    if cached is not None:
        return cached

    fail_key = f"fail:{cache_key}"
    if _cache.get(fail_key, ttl=TTL_FAILURE) is not None:
        logger.debug(f"Failure cache hit: {cache_key}")
        return {}

    breaker = _breakers["realtime"]
    logger.info(f"Fetching realtime quote for {ticker}")

    def _fetch():
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == ticker]
        if row.empty:
            raise ValueError(f"Stock {ticker} not found in spot data")
        return row.iloc[0]

    try:
        row = _retry_with_backoff(_fetch, breaker, f"realtime:{ticker}")
        data = {
            "ticker": ticker,
            "name": row.get("名称", ""),
            "price": float(row.get("最新价", 0)),
            "pct_chg": float(row.get("涨跌幅", 0)),
            "change": float(row.get("涨跌额", 0)),
            "volume": float(row.get("成交量", 0)),
            "amount": float(row.get("成交额", 0)),
            "high": float(row.get("最高", 0)),
            "low": float(row.get("最低", 0)),
            "open": float(row.get("今开", 0)),
            "prev_close": float(row.get("昨收", 0)),
            "turnover": float(row.get("换手率", 0)),
            "pe": float(row.get("市盈率-动态", 0)),
            "pb": float(row.get("市净率", 0)),
            "total_mv": float(row.get("总市值", 0)),
            "circ_mv": float(row.get("流通市值", 0)),
        }
        _cache.set(cache_key, data)
        return data
    except Exception as e:
        logger.error(f"Failed to fetch realtime for {ticker}: {e}")
        _cache.set(fail_key, {"error": str(e)})
        return {}


def get_financial_data(ticker: str) -> dict:
    """Get fundamental financial indicators via AkShare.

    Uses circuit breaker + cache + retry with backoff.
    """
    ticker = _format_ticker(ticker)
    cache_key = f"fin:{ticker}"
    cached = _cache.get(cache_key, ttl=TTL_FINANCIAL)
    if cached is not None:
        return cached

    fail_key = f"fail:{cache_key}"
    if _cache.get(fail_key, ttl=TTL_FAILURE) is not None:
        logger.debug(f"Failure cache hit: {cache_key}")
        return {"ticker": ticker}

    breaker = _breakers["financial"]
    logger.info(f"Fetching financials for {ticker}")

    def _fetch():
        import akshare as ak

        result = {"ticker": ticker}

        # Financial indicators
        try:
            df = ak.stock_financial_analysis_indicator(symbol=ticker)
            if not df.empty:
                latest = df.iloc[0]
                result.update(
                    {
                        "roe": float(latest.get("加权净资产收益率(%)", 0)),
                        "gross_profit_margin": float(latest.get("销售毛利率(%)", 0)),
                        "net_profit_margin": float(latest.get("销售净利率(%)", 0)),
                        "debt_ratio": float(latest.get("资产负债率(%)", 0)),
                    }
                )
        except Exception:
            pass

        # Revenue and profit growth
        try:
            df = ak.stock_financial_report_sina(stock=f"sh{ticker}" if ticker.startswith("6") else f"sz{ticker}")
            if not df.empty:
                result["report_data"] = df.head(4).to_dict(orient="records")
        except Exception:
            pass

        return result

    try:
        result = _retry_with_backoff(_fetch, breaker, f"financial:{ticker}")
        _cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Failed to fetch financials for {ticker}: {e}")
        _cache.set(fail_key, {"error": str(e)})
        return {"ticker": ticker}


def get_stock_news(ticker: str, limit: int = 10) -> list[dict]:
    """Get recent news for a stock via AkShare.

    Uses circuit breaker + cache + retry with backoff.
    """
    ticker = _format_ticker(ticker)
    cache_key = f"news:{ticker}:{limit}"
    cached = _cache.get(cache_key, ttl=TTL_NEWS)
    if cached is not None:
        return cached

    fail_key = f"fail:{cache_key}"
    if _cache.get(fail_key, ttl=TTL_FAILURE) is not None:
        logger.debug(f"Failure cache hit: {cache_key}")
        return []

    breaker = _breakers["news"]
    logger.info(f"Fetching news for {ticker}")

    def _fetch():
        import akshare as ak

        df = ak.stock_news_em(symbol=ticker)
        if df.empty:
            return []
        news_list = []
        for _, row in df.head(limit).iterrows():
            news_list.append(
                {
                    "title": str(row.get("新闻标题", "")),
                    "content": str(row.get("新闻内容", ""))[:500],
                    "source": str(row.get("文章来源", "")),
                    "date": str(row.get("发布时间", "")),
                }
            )
        return news_list

    try:
        news_list = _retry_with_backoff(_fetch, breaker, f"news:{ticker}")
        _cache.set(cache_key, news_list)
        return news_list
    except Exception as e:
        logger.error(f"Failed to fetch news for {ticker}: {e}")
        _cache.set(fail_key, {"error": str(e)})
        return []


def get_stock_list() -> list[dict]:
    """Get all A-share stock list.

    Uses circuit breaker + cache + retry with backoff.
    """
    cache_key = "stock_list"
    cached = _cache.get(cache_key, ttl=TTL_DAILY)
    if cached is not None:
        return cached

    fail_key = f"fail:{cache_key}"
    if _cache.get(fail_key, ttl=TTL_FAILURE) is not None:
        logger.debug(f"Failure cache hit: {cache_key}")
        return []

    breaker = _breakers["stock_list"]

    def _fetch():
        import akshare as ak

        df = ak.stock_info_a_code_name()
        return df.to_dict(orient="records")

    try:
        stocks = _retry_with_backoff(_fetch, breaker, "stock_list")
        _cache.set(cache_key, stocks)
        return stocks
    except Exception as e:
        logger.error(f"Failed to fetch stock list: {e}")
        _cache.set(fail_key, {"error": str(e)})
        return []


def get_breaker_status() -> dict[str, str]:
    """Get status of all circuit breakers (for monitoring API)."""
    return {name: breaker.state for name, breaker in _breakers.items()}
