"""Trading engine for A-share simulation.

Features:
- Virtual cash and position management
- A-share trading rules: 100-share lots, T+1, commissions, stamp tax
- Time-aware: prevents future data leakage
- Buy/sell execution with fee calculation
"""

from loguru import logger

from models.schemas import Decision, PortfolioState, Position, SimulationAccountConfig, TradeDecision, TradeRecord

# A-share trading cost constants
BUY_COMMISSION_RATE = 0.0003  # 0.03%
SELL_COMMISSION_RATE = 0.0003  # 0.03%
MIN_COMMISSION = 5.0  # minimum 5 CNY per trade
STAMP_TAX_RATE = 0.001  # 0.1% (sell only)
TRANSFER_FEE_RATE = 0.00002  # 0.002% (very small, often waived)
MIN_LOT = 100  # minimum 100 shares


def decision_shares(
    portfolio: PortfolioState,
    rules: SimulationAccountConfig,
    decision: TradeDecision,
    price: float,
) -> int:
    """Calculate an Agent order quantity using the shared A-share rules.

    Both historical backtests and the persistent simulation account call this
    helper.  The actual engine still performs the final cash, lot, T+1 and
    position-limit checks at fill time.
    """

    if price <= 0 or decision.decision == Decision.HOLD:
        return 0
    if decision.decision == Decision.BUY:
        position_pct = min(max(decision.position_size or 0.2, 0.0), rules.max_single_position_pct)
        existing_invested = sum(
            position.shares * (position.current_price or position.avg_cost)
            for position in portfolio.positions
        )
        total_value = max(portfolio.total_value, portfolio.cash)
        max_invest = min(
            portfolio.cash * position_pct,
            max(total_value * rules.max_total_position_pct - existing_invested, 0),
        )
        return int(max_invest / price)
    position = next((item for item in portfolio.positions if item.ticker == decision.ticker), None)
    if not position:
        return 0
    if decision.stop_loss and price <= decision.stop_loss:
        return position.available_shares
    return position.available_shares // 2


class TradingEngine:
    """Core trading engine for A-share simulation."""

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        rules: SimulationAccountConfig | None = None,
        portfolio: PortfolioState | None = None,
        current_date: str = "",
    ):
        self.rules = rules or SimulationAccountConfig(initial_cash=initial_capital)
        self.portfolio = portfolio or PortfolioState(
            cash=initial_capital,
            initial_capital=initial_capital,
        )
        self._current_date: str = current_date

    def set_date(self, d: str):
        """Set current simulation date and release frozen shares (T+1 settlement).

        When the date changes, all frozen shares (bought on the previous day)
        become available for selling.
        """
        if d != self._current_date:
            for pos in self.portfolio.positions:
                if pos.frozen_shares > 0:
                    pos.available_shares += pos.frozen_shares
                    pos.frozen_shares = 0
        self._current_date = d

    @property
    def current_date(self) -> str:
        return self._current_date

    def buy(self, ticker: str, shares: int, price: float, trade_date: str = "") -> TradeRecord | None:
        """Execute a buy order.

        A-share rules:
        - Shares must be multiple of 100
        - Commission: 0.03% (min 5 CNY)
        - No stamp tax on buy
        """
        trade_date = trade_date or self._current_date
        if price <= 0:
            return None
        min_lot = self.rules.min_lot
        shares = (shares // min_lot) * min_lot  # round down to the configured lot size
        if shares < min_lot:
            logger.warning(f"Buy {ticker}: shares < {min_lot}, skipping")
            return None

        execution_price = price * (1 + self.rules.slippage_bps / 10_000)
        amount = shares * execution_price
        commission = max(amount * self.rules.buy_commission_rate, self.rules.minimum_commission)
        transfer_fee = amount * self.rules.transfer_fee_rate
        total_cost = amount + commission + transfer_fee

        if total_cost > self.portfolio.cash:
            # Adjust shares to fit budget
            max_affordable = int(
                self.portfolio.cash
                / (execution_price * (1 + self.rules.buy_commission_rate + self.rules.transfer_fee_rate))
            )
            shares = (max_affordable // min_lot) * min_lot
            if shares < min_lot:
                logger.warning(
                    f"Buy {ticker}: insufficient funds (need {total_cost:.2f}, have {self.portfolio.cash:.2f})"
                )
                return None
            amount = shares * execution_price
            commission = max(amount * self.rules.buy_commission_rate, self.rules.minimum_commission)
            transfer_fee = amount * self.rules.transfer_fee_rate
            total_cost = amount + commission + transfer_fee

        # Update position — new shares are frozen (T+1: cannot sell today)
        existing = self._find_position(ticker)
        if existing:
            new_avg_cost = (existing.avg_cost * existing.shares + amount) / (existing.shares + shares)
            existing.avg_cost = new_avg_cost
            existing.shares += shares
            existing.frozen_shares += shares
        else:
            self.portfolio.positions.append(
                Position(
                    ticker=ticker,
                    asset_type=self.rules.asset_type,
                    shares=shares,
                    avg_cost=execution_price,
                    available_shares=0,
                    frozen_shares=shares,
                )
            )

        # Deduct cash
        self.portfolio.cash -= total_cost

        # Record trade
        trade = TradeRecord(
            date=trade_date,
            action=Decision.BUY,
            ticker=ticker,
            asset_type=self.rules.asset_type,
            shares=shares,
            price=execution_price,
            amount=amount,
            commission=commission,
            tax=0.0,
        )
        self.portfolio.trades.append(trade)
        logger.info(f"BUY {ticker} {shares}@{execution_price:.2f} cost={total_cost:.2f} cash={self.portfolio.cash:.2f}")
        return trade

    def sell(self, ticker: str, shares: int, price: float, trade_date: str = "") -> TradeRecord | None:
        """Execute a sell order.

        A-share rules:
        - Shares must be multiple of 100
        - Commission: 0.03% (min 5 CNY)
        - Stamp tax: 0.1% (sell only)
        - T+1: can only sell shares bought before today (enforced via available_shares/frozen_shares)
        """
        trade_date = trade_date or self._current_date
        if price <= 0:
            return None
        pos = self._find_position(ticker)
        min_lot = self.rules.min_lot
        if not pos or pos.available_shares <= 0:
            logger.warning(
                f"Sell {ticker}: no position or insufficient available shares "
                f"(available={pos.available_shares if pos else 0}, frozen={pos.frozen_shares if pos else 0})"
            )
            return None

        shares = min(shares, pos.available_shares)
        shares = (shares // min_lot) * min_lot
        if shares < min_lot:
            # Odd-lot cleanup is allowed only when closing the entire holding.
            if shares == pos.available_shares:
                shares = pos.available_shares
            else:
                logger.warning(f"Sell {ticker}: shares < {min_lot} after rounding")
                return None

        execution_price = price * (1 - self.rules.slippage_bps / 10_000)
        amount = shares * execution_price
        commission = max(amount * self.rules.sell_commission_rate, self.rules.minimum_commission)
        stamp_tax = amount * self.rules.stamp_tax_rate
        transfer_fee = amount * self.rules.transfer_fee_rate
        net_proceeds = amount - commission - stamp_tax - transfer_fee

        # Update position — deduct from available shares
        pos.available_shares -= shares
        pos.shares -= shares
        if pos.shares == 0:
            self.portfolio.positions = [p for p in self.portfolio.positions if p.ticker != ticker]

        # Add cash
        self.portfolio.cash += net_proceeds

        # Record trade
        trade = TradeRecord(
            date=trade_date,
            action=Decision.SELL,
            ticker=ticker,
            asset_type=self.rules.asset_type,
            shares=shares,
            price=execution_price,
            amount=amount,
            commission=commission,
            tax=stamp_tax,
        )
        self.portfolio.trades.append(trade)
        logger.info(
            f"SELL {ticker} {shares}@{execution_price:.2f} "
            f"net={net_proceeds:.2f} cash={self.portfolio.cash:.2f}"
        )
        return trade

    def update_prices(self, price_map: dict[str, float]):
        """Update current prices for all positions."""
        for pos in self.portfolio.positions:
            if pos.ticker in price_map:
                pos.current_price = price_map[pos.ticker]

    def get_portfolio_summary(self) -> dict:
        """Get portfolio summary as dict."""
        return {
            "cash": self.portfolio.cash,
            "total_value": self.portfolio.total_value,
            "positions": [p.model_dump() for p in self.portfolio.positions],
            "trades": [t.model_dump() for t in self.portfolio.trades],
            "initial_capital": self.portfolio.initial_capital,
            "total_pnl": self.portfolio.total_pnl,
            "total_return_pct": self.portfolio.total_return_pct,
        }

    def reset(self, initial_capital: float = 1_000_000.0):
        """Reset portfolio to initial state."""
        self.rules = self.rules.model_copy(update={"initial_cash": initial_capital})
        self.portfolio = PortfolioState(
            cash=initial_capital,
            initial_capital=initial_capital,
        )
        self._current_date = ""

    def seed_position(self, ticker: str, shares: int, price: float) -> None:
        """Add a pre-existing holding while preserving total initial assets."""
        shares = int(shares)
        if shares <= 0 or price <= 0:
            return
        amount = shares * price
        if amount > self.portfolio.cash:
            raise ValueError("Initial position value exceeds initial capital")
        self.portfolio.cash -= amount
        self.portfolio.positions.append(
            Position(
                ticker=ticker,
                shares=shares,
                avg_cost=price,
                current_price=price,
                available_shares=shares,
                frozen_shares=0,
            )
        )

    def _find_position(self, ticker: str) -> Position | None:
        for p in self.portfolio.positions:
            if p.ticker == ticker:
                return p
        return None


class TimeAwareTradingEngine(TradingEngine):
    """Trading engine with time-awareness to prevent data leakage.

    Ensures that the agent can only see data up to the current simulation date.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        rules: SimulationAccountConfig | None = None,
        portfolio: PortfolioState | None = None,
    ):
        super().__init__(initial_capital, rules=rules, portfolio=portfolio)
        self._available_dates: list[str] = []
        self._date_index: int = 0

    def set_available_dates(self, dates: list[str]):
        """Set the list of trading dates available in the simulation."""
        self._available_dates = sorted(dates)
        self._date_index = 0

    def advance_to_date(self, target_date: str) -> bool:
        """Advance simulation to the target date.

        Returns False if target date is beyond available dates.
        """
        # Find the index of target_date
        for i, d in enumerate(self._available_dates):
            if d >= target_date:
                self._date_index = i
                self.set_date(self._available_dates[i])
                return True
        logger.warning(f"Date {target_date} beyond available range")
        return False

    def advance_next(self) -> str | None:
        """Advance to next trading day."""
        self._date_index += 1
        if self._date_index >= len(self._available_dates):
            return None
        d = self._available_dates[self._date_index]
        self.set_date(d)
        return d

    @property
    def is_finished(self) -> bool:
        return self._date_index >= len(self._available_dates) - 1

    @property
    def progress_pct(self) -> float:
        if not self._available_dates:
            return 0.0
        return (self._date_index + 1) / len(self._available_dates) * 100
