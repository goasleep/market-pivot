import pandas as pd

from data.market_index import _normalize_index_history


def test_market_index_history_normalizes_chinese_columns_for_backtesting():
    frame = pd.DataFrame(
        {
            "日期": ["2026-01-02", "2026-01-05"],
            "开盘": [3900, 3910],
            "收盘": [3905, 3920],
            "最高": [3910, 3930],
            "最低": [3890, 3900],
            "成交量": [100, 120],
        }
    )

    normalized = _normalize_index_history(frame, "000300", "eastmoney")

    assert list(normalized["date"]) == ["2026-01-02", "2026-01-05"]
    assert list(normalized["close"]) == [3905, 3920]
    assert normalized.iloc[0]["ticker"] == "000300"
    assert normalized.attrs["source_metadata"]["endpoint"] == "index_zh_a_hist"
