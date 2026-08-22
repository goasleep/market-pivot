import pandas as pd
import pytest
import requests

import data.akshare_provider as provider


def test_historical_fund_query_can_reuse_expired_immutable_cache(monkeypatch):
    class StaleCache:
        def get(self, key, ttl=3600):
            return None

        def get_stale(self, key):
            assert key == "fund_hist:etf:510300:20250101:20250105:"
            return [
                {
                    "date": "2025-01-02",
                    "open": 4.0,
                    "high": 4.1,
                    "low": 3.9,
                    "close": 4.05,
                    "volume": 100,
                }
            ]

    monkeypatch.setattr(provider, "_cache", StaleCache())
    monkeypatch.setattr(
        provider,
        "_retry_with_backoff",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network must not be used")),
    )
    result = provider.get_fund_history(
        "510300",
        asset_type="etf",
        start_date="2025-01-01",
        end_date="2025-01-05",
    )
    assert isinstance(result, pd.DataFrame)
    assert result.iloc[0]["close"] == 4.05
    assert result.iloc[0]["pct_chg"] == 0.0


def test_etf_history_uses_sina_when_eastmoney_returns_empty(monkeypatch):
    class EmptyCache:
        def get(self, key, ttl=3600):
            return None

        def get_stale(self, key):
            return None

        def set(self, key, value):
            pass

    fallback = pd.DataFrame(
        [
            {
                "date": "2025-08-18",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 100,
            },
            {
                "date": "2025-08-19",
                "open": 1.05,
                "high": 1.2,
                "low": 1.0,
                "close": 1.155,
                "volume": 120,
            },
        ]
    )
    monkeypatch.setattr(provider, "_cache", EmptyCache())
    monkeypatch.setattr(provider, "_fetch_etf_history_sina", lambda *args: fallback)
    monkeypatch.setattr(
        provider,
        "_retry_with_backoff",
        lambda func, *args, **kwargs: func(),
    )

    import akshare as ak

    monkeypatch.setattr(ak, "fund_etf_hist_em", lambda **kwargs: pd.DataFrame())
    result = provider.get_fund_history(
        "159667",
        asset_type="etf",
        start_date="2025-08-15",
        end_date="2025-08-20",
    )

    assert len(result) == 2
    assert result.iloc[0]["close"] == 1.05
    assert result.iloc[0]["pct_chg"] == 0.0
    assert result.iloc[1]["pct_chg"] == pytest.approx(10.0)
    assert result.iloc[0]["ticker"] == "159667"


def test_etf_nav_accepts_eastmoney_rows_with_fourteen_fields(monkeypatch):
    class EmptyCache:
        def get(self, key, ttl=3600):
            return None

        def set(self, key, value):
            pass

    row = {
        "FSRQ": "2025-08-18",
        "DWJZ": "1.0500",
        "LJJZ": "2.1000",
        "SDATE": None,
        "ACTUALSYI": "",
        "NAVTYPE": "1",
        "JZZZL": "1.25",
        "SGZT": "场内买入",
        "SHZT": "场内卖出",
        "FHFCZ": "",
        "FHFCZ10": "",
        "FHFCBZ": "",
        "DTYPE": None,
        "FHSP": "",
    }

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def fake_get(url, **kwargs):
        assert kwargs["timeout"] == provider.UPSTREAM_TIMEOUT_SECONDS
        return Response({"Data": {"TotalCount": 1, "LSJZList": [row]}})

    monkeypatch.setattr(provider, "_cache", EmptyCache())
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(
        provider,
        "_retry_with_backoff",
        lambda func, *args, **kwargs: func(),
    )

    result = provider.get_fund_nav_history(
        "159667",
        asset_type="etf",
        start_date="2025-08-15",
        end_date="2025-08-20",
    )

    assert len(result) == 1
    assert result.iloc[0]["date"] == "2025-08-18"
    assert result.iloc[0]["unit_nav"] == 1.05
    assert result.iloc[0]["cumulative_nav"] == 2.1
    assert result.iloc[0]["pct_chg"] == 1.25
