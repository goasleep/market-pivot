# ADR 005: Financial product classification and naming boundary

## Status

Accepted — direct cutover in Harness runtime 2.0.

## Decision

The only routable `asset_type` values are `stock`, `etf`, `lof`, and `open_fund`. `fund` remains a human-facing family term and is not a routable asset type, Skill ID, Capability ID, Validator ID, or Provider name.

ETF and LOF share the `exchange_fund` domain because both require exchange price, volume, liquidity, NAV/IOPV comparability, and exchange trading rules. Off-exchange public funds use the `open_fund` domain and NAV, fee, holding, subscription, and redemption evidence. Stock remains an independent research product; the stock comprehensive graph is exposed only through `stock.comprehensive_analysis`.

The Provider boundaries are `ExchangeFundDataProvider` and `OpenFundDataProvider`. A six-digit code prefix may form a candidate but never verifies product identity. The selected domain and its Provider must verify the instrument before a formal conclusion or ranking.

Supported open-fund categories are equity, hybrid, bond, money market, index, and enhanced index. QDII and FOF are recognized but return structured `limited` or `data_unavailable` results when the required evidence is unavailable.

Open funds have no real-time exchange price, order book, bid-ask spread, turnover, IOPV, or premium/discount. Those fields are `not_applicable`, never zero. Money-market funds use yield per 10,000 units and seven-day annualized yield rather than changes in a nominally stable NAV. Cross-category open-fund rankings are forbidden unless the user explicitly requests an asset-allocation comparison.

Open-fund backtests use published NAV and execute signals at the next available NAV. Cumulative NAV is preferred; unit NAV fallback and all fee assumptions are disclosed. Intraday, volume, spread, slippage, and order-book strategies are not applicable. Open-fund paper subscription/redemption and settlement simulation are outside the first release.

## Consequences

Legacy `fund.*` and shared `etf.*` identifiers fail Registry startup and are not aliased. Existing completed conversations remain historical text; active incompatible checkpoints are interrupted by the existing dry-run-first runtime cleanup script.
