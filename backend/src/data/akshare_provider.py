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

import asyncio
import json
import math
import random
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import akshare as ak
import pandas as pd
import requests
from loguru import logger

from config import settings
from data.history_cache import HistorySeries, history_cache

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
    """Process-local cache with TTL support and failure caching.

    Market data is an optimization rather than business state. Keeping it
    in-memory removes database coupling from the synchronous AkShare adapter;
    durable application state is handled exclusively by Tortoise repositories.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._entries: dict[str, tuple[str, float]] = {}

    def get(self, key: str, ttl: int = 3600) -> Any | None:
        row = self._entries.get(key)
        if row is None:
            return None
        value_str, timestamp = row
        if time.time() - timestamp > ttl:
            return None
        return json.loads(value_str)

    def get_stale(self, key: str) -> Any | None:
        """Read an expired cache entry for an immutable, fully historical query."""
        row = self._entries.get(key)
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, value: Any):
        self._entries[key] = (json.dumps(value, ensure_ascii=False, default=str), time.time())

    def clear(self):
        self._entries.clear()


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
    "fund_nav": CircuitBreaker("akshare_fund_nav", failure_threshold=5, recovery_timeout=180),
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
UPSTREAM_TIMEOUT_SECONDS = 12.0
ETF_NAV_PAGE_SIZE = 20
ETF_NAV_URL = "https://api.fund.eastmoney.com/f10/lsjz"
ETF_NAV_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; A-Share-Agent/1.0)",
}


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


def _fetch_etf_history_sina(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch ETF OHLCV from Sina with an explicit request timeout.

    AkShare's Sina adapter does not expose a timeout and therefore cannot be
    used directly inside the provider's bounded retry loop.  Reuse its public
    response decoder, but keep the network request under our own timeout.
    """
    # py_mini_racer is platform-specific; keep this optional fallback dependency isolated.
    import py_mini_racer
    from akshare.stock.cons import hk_js_decode

    market = "sh" if ticker.startswith(("5", "6", "9")) else "sz"
    url = f"https://finance.sina.com.cn/realstock/company/{market}{ticker}/hisdata_klc2/klc_kl.js"
    response = requests.get(url, timeout=UPSTREAM_TIMEOUT_SECONDS)
    response.raise_for_status()
    encoded = response.text.split("=", 1)[1].split(";", 1)[0].replace('"', "")
    js_code = py_mini_racer.MiniRacer()
    js_code.eval(hk_js_decode)
    rows = js_code.call("d", encoded)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["date"].notna()].copy()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame = frame[(frame["date"] >= _date_string(start_date)) & (frame["date"] <= _date_string(end_date))]
    return frame.sort_values("date").reset_index(drop=True)


def _date_string(value: str) -> str:
    """Normalize a YYYYMMDD/ISO date into the provider's canonical form."""
    normalized = str(value).replace("-", "")
    return f"{normalized[:4]}-{normalized[4:6]}-{normalized[6:8]}"


def _normalize_fund_price_history(frame: pd.DataFrame, ticker: str, asset_type: str) -> pd.DataFrame:
    """Normalize primary, fallback, and cached fund OHLCV rows to one schema."""
    source_metadata = dict(frame.attrs.get("source_metadata") or {})
    frame = frame.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "开盘价": "open",
            "收盘": "close",
            "最高": "high",
            "最高价": "high",
            "最低": "low",
            "最低价": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "换手率": "turnover",
        }
    ).copy()
    if "close" in frame.columns:
        close = pd.to_numeric(frame["close"], errors="coerce")
        calculated_change = close.pct_change(fill_method=None).mul(100)
        if "pct_chg" in frame.columns:
            reported_change = pd.to_numeric(frame["pct_chg"], errors="coerce")
            frame["pct_chg"] = reported_change.fillna(calculated_change).fillna(0.0)
        else:
            frame["pct_chg"] = calculated_change.fillna(0.0)
    elif "pct_chg" not in frame.columns:
        frame["pct_chg"] = 0.0
    frame["ticker"] = ticker
    frame["asset_type"] = asset_type
    if source_metadata:
        frame.attrs["source_metadata"] = source_metadata
    return frame


def _fund_history_source_metadata(
    *,
    source_id: str,
    source_name: str,
    endpoint: str,
    fallback: bool,
    fallback_reason: str = "",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_id": source_id,
        "source_name": source_name,
        "endpoint": endpoint,
        "fallback": fallback,
        "source_chain": ["eastmoney", source_id] if fallback else [source_id],
        "cache": "miss",
    }
    if fallback_reason:
        metadata["fallback_reason"] = fallback_reason[:500]
    return metadata


def _fetch_etf_nav_history(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch ETF NAV pages and preserve the upstream JSON field names.

    AkShare 1.18.x assumes each NAV row has 13 fields, while Eastmoney now
    returns 14 fields.  Reading the JSON keys directly avoids that brittle
    positional schema and also gives every page a bounded HTTP timeout.
    """
    params = {
        "fundCode": ticker,
        "pageIndex": 1,
        "pageSize": ETF_NAV_PAGE_SIZE,
        "startDate": _date_string(start_date),
        "endDate": _date_string(end_date),
        "_": round(time.time() * 1000),
    }
    response = requests.get(
        ETF_NAV_URL,
        params=params,
        headers={**ETF_NAV_HEADERS, "Referer": f"https://fundf10.eastmoney.com/jjjz_{ticker}.html"},
        timeout=UPSTREAM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("Data") or {}
    total_count = int(payload.get("TotalCount") or data.get("TotalCount") or 0)
    if total_count <= 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = list(data.get("LSJZList") or [])
    total_pages = math.ceil(total_count / ETF_NAV_PAGE_SIZE)
    for page in range(2, total_pages + 1):
        page_params = {**params, "pageIndex": page}
        page_response = requests.get(
            ETF_NAV_URL,
            params=page_params,
            headers={**ETF_NAV_HEADERS, "Referer": f"https://fundf10.eastmoney.com/jjjz_{ticker}.html"},
            timeout=UPSTREAM_TIMEOUT_SECONDS,
        )
        page_response.raise_for_status()
        page_data = page_response.json().get("Data") or {}
        rows.extend(page_data.get("LSJZList") or [])

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.rename(
        columns={
            "FSRQ": "净值日期",
            "DWJZ": "单位净值",
            "LJJZ": "累计净值",
            "JZZZL": "日增长率",
        }
    )


def _fetch_stock_history_upstream(ticker: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    breaker = _breakers["history"]
    logger.info(f"Fetching history for {ticker} ({start_date} - {end_date})")

    def _fetch():
        try:
            frame = ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                timeout=UPSTREAM_TIMEOUT_SECONDS,
            )
            frame.attrs["source_metadata"] = _fund_history_source_metadata(
                source_id="eastmoney",
                source_name="东方财富（AkShare）",
                endpoint="stock_zh_a_hist",
                fallback=False,
            )
            return frame
        except Exception as primary_exc:
            # Eastmoney occasionally closes the connection before returning a
            # response. AkShare also exposes Tencent's historical endpoint;
            # use it as a bounded fallback before marking the data missing.
            market = "sh" if ticker.startswith(("5", "6", "9")) else "sz"
            logger.warning(
                f"History primary source failed for {ticker}, trying Tencent fallback: {primary_exc}"
            )
            try:
                frame = ak.stock_zh_a_hist_tx(
                    symbol=f"{market}{ticker}",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                    timeout=UPSTREAM_TIMEOUT_SECONDS,
                )
                frame.attrs["source_metadata"] = _fund_history_source_metadata(
                    source_id="tencent",
                    source_name="腾讯证券（AkShare）",
                    endpoint="stock_zh_a_hist_tx",
                    fallback=True,
                    fallback_reason=str(primary_exc),
                )
                return frame
            except Exception:
                raise primary_exc

    df = _retry_with_backoff(_fetch, breaker, f"history:{ticker}")
    source_metadata = dict(df.attrs.get("source_metadata") or {})
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
    df.attrs["source_metadata"] = source_metadata
    return df


def get_stock_history(
    ticker: str,
    start_date: str = "",
    end_date: str = "",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Get historical daily OHLCV data via AkShare."""
    ticker = _format_ticker(ticker)
    start_date = (start_date or (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")).replace("-", "")
    end_date = (end_date or datetime.now().strftime("%Y%m%d")).replace("-", "")
    cache_key = f"hist:v2:{ticker}:{start_date}:{end_date}:{adjust}"
    fail_key = f"fail:{cache_key}"
    if not history_cache.enabled and _cache.get(fail_key, ttl=TTL_FAILURE) is not None:
        logger.debug(f"Failure cache hit: {cache_key}")
        return pd.DataFrame()

    if history_cache.enabled:
        try:
            return history_cache.get_or_fetch(
                HistorySeries(dataset="price", asset_type="stock", ticker=ticker, adjustment=adjust or "none"),
                start_date,
                end_date,
                lambda fetch_start, fetch_end: _fetch_stock_history_upstream(
                    ticker, fetch_start, fetch_end, adjust
                ),
            )
        except Exception as exc:
            logger.error(f"Failed to fetch history for {ticker}: {exc}")
            _cache.set(fail_key, {"error": str(exc)})
            return pd.DataFrame()

    cached = _cache.get(cache_key, ttl=TTL_DAILY)
    if cached is None and end_date < datetime.now().strftime("%Y%m%d"):
        cached = _cache.get_stale(cache_key)
    if cached is not None:
        if isinstance(cached, dict) and isinstance(cached.get("records"), list):
            frame = pd.DataFrame(cached["records"])
            frame.attrs["source_metadata"] = dict(cached.get("source_metadata") or {}) | {"cache": "hit"}
            return frame
        return pd.DataFrame(cached)

    try:
        df = _fetch_stock_history_upstream(ticker, start_date, end_date, adjust)
        _cache.set(
            cache_key,
            {
                "records": df.to_dict(orient="records"),
                "source_metadata": dict(df.attrs.get("source_metadata") or {}),
            },
        )
        return df
    except Exception as exc:
        logger.error(f"Failed to fetch history for {ticker}: {exc}")
        _cache.set(fail_key, {"error": str(exc)})
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


def get_fund_realtime(ticker: str, asset_type: str = "etf") -> dict:
    """Get an exchange-traded ETF/LOF quote via AkShare."""
    ticker = _format_ticker(ticker)
    if asset_type not in {"etf", "lof"}:
        raise ValueError("asset_type must be etf or lof")

    cache_key = f"fund_rt:{asset_type}:{ticker}"
    cached = _cache.get(cache_key, ttl=TTL_REALTIME)
    if cached is not None:
        return cached
    fail_key = f"fail:{cache_key}"
    if _cache.get(fail_key, ttl=TTL_FAILURE) is not None:
        return {}

    breaker = _breakers["realtime"]
    endpoint_name = "fund_etf_spot_em" if asset_type == "etf" else "fund_lof_spot_em"

    def _fetch():
        endpoint = getattr(ak, endpoint_name)
        df = endpoint()
        code_column = "代码"
        row = df[df[code_column].astype(str).str.zfill(6) == ticker]
        if row.empty:
            raise ValueError(f"Fund {ticker} not found in {endpoint_name}")
        return row.iloc[0]

    try:
        row = _retry_with_backoff(_fetch, breaker, f"{asset_type}_realtime:{ticker}")
        data = {
            "ticker": ticker,
            "asset_type": asset_type,
            "name": row.get("名称", ""),
            "price": float(row.get("最新价", 0) or 0),
            "pct_chg": float(row.get("涨跌幅", 0) or 0),
            "change": float(row.get("涨跌额", 0) or 0),
            "volume": float(row.get("成交量", 0) or 0),
            "amount": float(row.get("成交额", 0) or 0),
            "high": float(row.get("最高", row.get("最高价", 0)) or 0),
            "low": float(row.get("最低", row.get("最低价", 0)) or 0),
            "open": float(row.get("今开", row.get("开盘价", 0)) or 0),
            "prev_close": float(row.get("昨收", 0) or 0),
            "turnover": float(row.get("换手率", 0) or 0),
            "discount_rate": float(row.get("基金折价率", 0) or 0),
            "iopv": float(row.get("IOPV实时估值", 0) or 0),
            "data_date": str(row.get("数据日期", "")),
            "updated_at": str(row.get("更新时间", "")),
        }
        _cache.set(cache_key, data)
        return data
    except Exception as e:
        logger.error(f"Failed to fetch {asset_type} realtime for {ticker}: {e}")
        _cache.set(fail_key, {"error": str(e)})
        return {}


def _fetch_fund_history_upstream(
    ticker: str,
    asset_type: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> pd.DataFrame:
    breaker = _breakers["history"]
    endpoint_name = "fund_etf_hist_em" if asset_type == "etf" else "fund_lof_hist_em"

    def _fetch():
        endpoint = getattr(ak, endpoint_name)
        try:
            frame = endpoint(
                symbol=ticker,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            if not frame.empty:
                frame.attrs["source_metadata"] = _fund_history_source_metadata(
                    source_id="eastmoney",
                    source_name="东方财富（AkShare）",
                    endpoint=endpoint_name,
                    fallback=False,
                )
                return frame
            if asset_type != "etf":
                return frame
            raise ValueError(f"{asset_type} history returned no rows")
        except Exception as primary_exc:
            if asset_type != "etf":
                raise
            logger.warning(
                f"ETF history primary source failed for {ticker}, trying Sina fallback: {primary_exc}"
            )
            fallback = _fetch_etf_history_sina(ticker, start_date, end_date)
            if fallback.empty:
                raise primary_exc
            fallback.attrs["source_metadata"] = _fund_history_source_metadata(
                source_id="sina",
                source_name="新浪财经",
                endpoint="hisdata_klc2",
                fallback=True,
                fallback_reason=str(primary_exc),
            )
            return fallback

    df = _retry_with_backoff(_fetch, breaker, f"{asset_type}_history:{ticker}")
    return _normalize_fund_price_history(df, ticker, asset_type)


def get_fund_history(
    ticker: str,
    asset_type: str = "etf",
    start_date: str = "",
    end_date: str = "",
    adjust: str = "",
) -> pd.DataFrame:
    """Get daily historical data for an exchange-traded ETF/LOF."""
    ticker = _format_ticker(ticker)
    if asset_type not in {"etf", "lof"}:
        raise ValueError("asset_type must be etf or lof")
    start_date = (start_date or (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")).replace("-", "")
    end_date = (end_date or datetime.now().strftime("%Y%m%d")).replace("-", "")
    cache_key = f"fund_hist:v2:{asset_type}:{ticker}:{start_date}:{end_date}:{adjust}"
    fail_key = f"fail:{cache_key}"
    if not history_cache.enabled and _cache.get(fail_key, ttl=TTL_FAILURE) is not None:
        return pd.DataFrame()

    if history_cache.enabled:
        try:
            return history_cache.get_or_fetch(
                HistorySeries(
                    dataset="price",
                    asset_type=asset_type,
                    ticker=ticker,
                    adjustment=adjust or "none",
                ),
                start_date,
                end_date,
                lambda fetch_start, fetch_end: _fetch_fund_history_upstream(
                    ticker, asset_type, fetch_start, fetch_end, adjust
                ),
            )
        except Exception as exc:
            logger.error(f"Failed to fetch {asset_type} history for {ticker}: {exc}")
            _cache.set(fail_key, {"error": str(exc)})
            return pd.DataFrame()

    cached = _cache.get(cache_key, ttl=TTL_DAILY)
    if cached is None and end_date < datetime.now().strftime("%Y%m%d"):
        cached = _cache.get_stale(cache_key)
    if cached is not None:
        if isinstance(cached, dict) and isinstance(cached.get("records"), list):
            frame = pd.DataFrame(cached["records"])
            frame.attrs["source_metadata"] = dict(cached.get("source_metadata") or {}) | {"cache": "hit"}
        else:
            frame = pd.DataFrame(cached)
        return _normalize_fund_price_history(frame, ticker, asset_type)

    try:
        df = _fetch_fund_history_upstream(ticker, asset_type, start_date, end_date, adjust)
        _cache.set(
            cache_key,
            {
                "records": df.to_dict(orient="records"),
                "source_metadata": dict(df.attrs.get("source_metadata") or {}),
            },
        )
        return df
    except Exception as exc:
        logger.error(f"Failed to fetch {asset_type} history for {ticker}: {exc}")
        _cache.set(fail_key, {"error": str(exc)})
        return pd.DataFrame()


def _fetch_fund_nav_history_upstream(
    ticker: str,
    asset_type: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    breaker = _breakers["fund_nav"]

    def _fetch():
        if asset_type == "etf":
            return _fetch_etf_nav_history(ticker, start_date, end_date)
        return ak.fund_open_fund_info_em(
            symbol=ticker,
            indicator="单位净值走势",
        )

    df = _retry_with_backoff(_fetch, breaker, f"{asset_type}_nav:{ticker}")
    if df.empty:
        return pd.DataFrame()
    col_map = {
        "净值日期": "date",
        "日期": "date",
        "单位净值": "unit_nav",
        "累计净值": "cumulative_nav",
        "日增长率": "pct_chg",
    }
    df = df.rename(columns=col_map)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df[df["date"].notna()]
    df = df[
        (df["date"] >= f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}")
        & (df["date"] <= f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}")
    ]
    keep = [name for name in ("date", "unit_nav", "cumulative_nav", "pct_chg") if name in df.columns]
    result = df[keep].copy()
    for name in keep:
        if name != "date":
            result[name] = pd.to_numeric(result[name], errors="coerce")
    result = result.sort_values("date").reset_index(drop=True)
    result.attrs["source_metadata"] = _fund_history_source_metadata(
        source_id="eastmoney",
        source_name="东方财富",
        endpoint="fund_etf_fund_info_em" if asset_type == "etf" else "fund_open_fund_info_em",
        fallback=False,
    )
    return result


def get_fund_nav_history(
    ticker: str,
    asset_type: str = "etf",
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """Get historical NAV data for an exchange-traded fund via AkShare."""
    ticker = _format_ticker(ticker)
    if asset_type not in {"etf", "lof"}:
        raise ValueError("asset_type must be etf or lof")
    start_date = (start_date or "20000101").replace("-", "")
    end_date = (end_date or datetime.now().strftime("%Y%m%d")).replace("-", "")
    cache_key = f"fund_nav:{asset_type}:{ticker}:{start_date}:{end_date}"
    fail_key = f"fail:{cache_key}"
    if not history_cache.enabled and _cache.get(fail_key, ttl=TTL_FAILURE) is not None:
        return pd.DataFrame()

    if history_cache.enabled:
        try:
            return history_cache.get_or_fetch(
                HistorySeries(dataset="nav", asset_type=asset_type, ticker=ticker),
                start_date,
                end_date,
                lambda fetch_start, fetch_end: _fetch_fund_nav_history_upstream(
                    ticker, asset_type, fetch_start, fetch_end
                ),
            )
        except Exception as exc:
            logger.error(f"Failed to fetch {asset_type} NAV for {ticker}: {exc}")
            _cache.set(fail_key, {"error": str(exc)})
            return pd.DataFrame()

    cached = _cache.get(cache_key, ttl=TTL_DAILY)
    if cached is not None:
        return pd.DataFrame(cached)
    try:
        result = _fetch_fund_nav_history_upstream(ticker, asset_type, start_date, end_date)
        _cache.set(cache_key, result.to_dict(orient="records"))
        return result
    except Exception as exc:
        logger.error(f"Failed to fetch {asset_type} NAV for {ticker}: {exc}")
        _cache.set(fail_key, {"error": str(exc)})
        return pd.DataFrame()


def get_asset_spot(asset_type: str = "stock", limit: int = 1000) -> list[dict]:
    """Get a cached realtime universe snapshot for screening.

    This deliberately fetches one market-wide snapshot instead of requesting
    one quote per symbol, which keeps screening within AkShare rate limits.
    """
    if asset_type not in {"stock", "etf", "lof"}:
        raise ValueError("asset_type must be stock, etf, or lof")
    cache_key = f"spot_universe:{asset_type}"
    cached = _cache.get(cache_key, ttl=TTL_REALTIME)
    if cached is not None:
        return cached[:limit]
    fail_key = f"fail:{cache_key}"
    if _cache.get(fail_key, ttl=TTL_FAILURE) is not None:
        return []

    endpoint_name = {
        "stock": "stock_zh_a_spot_em",
        "etf": "fund_etf_spot_em",
        "lof": "fund_lof_spot_em",
    }[asset_type]
    breaker = _breakers["realtime"]

    def _fetch():
        return getattr(ak, endpoint_name)()

    def _number(row, *names):
        for name in names:
            value = row.get(name)
            if value is not None and not pd.isna(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0

    try:
        df = _retry_with_backoff(_fetch, breaker, f"spot_universe:{asset_type}")
        records = []
        for _, row in df.iterrows():
            ticker = str(row.get("代码", "")).zfill(6)
            if len(ticker) != 6:
                continue
            records.append(
                {
                    "ticker": ticker,
                    "asset_type": asset_type,
                    "name": str(row.get("名称", "")),
                    "price": _number(row, "最新价", "收盘价"),
                    "pct_chg": _number(row, "涨跌幅"),
                    "amount": _number(row, "成交额"),
                    "volume": _number(row, "成交量"),
                    "turnover": _number(row, "换手率"),
                    "total_mv": _number(row, "总市值"),
                }
            )
        _cache.set(cache_key, records)
        return records[:limit]
    except Exception as exc:
        logger.error(f"Failed to fetch {asset_type} spot universe: {exc}")
        _cache.set(fail_key, {"error": str(exc)})
        return []


def get_breaker_status() -> dict[str, str]:
    """Get status of all circuit breakers (for monitoring API)."""
    return {name: breaker.state for name, breaker in _breakers.items()}


# Async adapters keep the legacy synchronous provider usable while ensuring
# FastAPI and LangGraph async paths never block the event loop on AkShare,
# SQLite, retry backoff, or rate-limit sleeps.
async def async_get_stock_history(*args, **kwargs) -> pd.DataFrame:
    return await asyncio.to_thread(get_stock_history, *args, **kwargs)


async def async_get_stock_realtime(*args, **kwargs) -> dict:
    return await asyncio.to_thread(get_stock_realtime, *args, **kwargs)


async def async_get_financial_data(*args, **kwargs) -> dict:
    return await asyncio.to_thread(get_financial_data, *args, **kwargs)


async def async_get_stock_news(*args, **kwargs) -> list[dict]:
    return await asyncio.to_thread(get_stock_news, *args, **kwargs)


async def async_get_stock_list(*args, **kwargs) -> list[dict]:
    return await asyncio.to_thread(get_stock_list, *args, **kwargs)


async def async_get_fund_realtime(*args, **kwargs) -> dict:
    return await asyncio.to_thread(get_fund_realtime, *args, **kwargs)


async def async_get_fund_history(*args, **kwargs) -> pd.DataFrame:
    return await asyncio.to_thread(get_fund_history, *args, **kwargs)


async def async_get_fund_nav_history(*args, **kwargs) -> pd.DataFrame:
    return await asyncio.to_thread(get_fund_nav_history, *args, **kwargs)


async def async_get_asset_spot(*args, **kwargs) -> list[dict]:
    return await asyncio.to_thread(get_asset_spot, *args, **kwargs)
