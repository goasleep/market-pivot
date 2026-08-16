from screening.fund_screener import FundScreener


def test_fund_screener_filters_and_ranks_with_polars():
    records = [
        {
            "ticker": "510300",
            "asset_type": "etf",
            "name": "沪深300ETF",
            "price": 3.9,
            "pct_chg": 1.2,
            "amount": 100_000_000,
            "turnover": 2.0,
            "iopv": 3.89,
            "discount_rate": 0.2,
        },
        {
            "ticker": "159915",
            "asset_type": "etf",
            "name": "创业板ETF",
            "price": 2.1,
            "pct_chg": -2.0,
            "amount": 50_000_000,
            "turnover": 1.0,
        },
    ]

    results = FundScreener().screen_snapshot(
        records,
        asset_type="etf",
        min_pct_chg=0,
        sort_by="screen_score",
        limit=10,
    )

    assert [item["ticker"] for item in results] == ["510300"]
    assert results[0]["screen_score"] > 0
    assert "成交额" in results[0]["screen_basis"][0]


def test_fund_screener_rejects_stock_asset_type():
    try:
        FundScreener().screen_snapshot([], asset_type="stock")
    except ValueError as exc:
        assert "etf and lof" in str(exc)
    else:  # pragma: no cover - assertion makes the failure explicit
        raise AssertionError("stock should not be accepted by FundScreener")
