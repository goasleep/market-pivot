import pandas as pd

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
