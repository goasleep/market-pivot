"""Data models for A-Share Agent."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Decision(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class AssetType(str, Enum):
    """Supported exchange-traded assets."""

    STOCK = "stock"
    ETF = "etf"
    LOF = "lof"


class StockData(BaseModel):
    """Historical/realtime stock data."""

    ticker: str
    name: str = ""
    date: str = ""
    open: float = 0.0
    close: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    amount: float = 0.0
    pct_chg: float = 0.0  # daily change %
    turnover: float = 0.0
    pe: float = 0.0
    pb: float = 0.0
    total_mv: float = 0.0  # total market value
    circ_mv: float = 0.0  # circulating market value


class FinancialData(BaseModel):
    """Fundamental financial indicators."""

    ticker: str
    roe: float = 0.0  # return on equity %
    net_profit_margin: float = 0.0
    gross_profit_margin: float = 0.0
    revenue_growth: float = 0.0  # YoY %
    profit_growth: float = 0.0  # YoY %
    debt_ratio: float = 0.0  # asset-liability ratio %
    current_ratio: float = 0.0
    eps: float = 0.0
    bps: float = 0.0  # book value per share


class NewsItem(BaseModel):
    """Single news article."""

    title: str
    content: str = ""
    source: str = ""
    date: str = ""
    sentiment: float = 0.0  # -1 to 1


class AgentReport(BaseModel):
    """Individual agent analysis report."""

    agent_name: str
    signal: Decision = Decision.HOLD
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    key_data: dict = Field(default_factory=dict)


class SignalType(str, Enum):
    """Detailed signal types beyond simple buy/sell/hold."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WATCH = "watch"  # 观望，等待信号
    REDUCE = "reduce"  # 减仓
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class CoreConclusion(BaseModel):
    """Core conclusion block of the decision dashboard."""

    signal: SignalType = SignalType.WATCH
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    one_line_summary: str = ""  # 一句话结论
    position_advice: str = ""  # 仓位建议文字描述


class DataPerspective(BaseModel):
    """Data-driven perspective block."""

    trend_status: str = ""  # e.g. "多头排列", "空头排列", "震荡"
    price_position: str = ""  # e.g. "接近MA20支撑", "远离均线超买"
    volume_analysis: str = ""  # e.g. "放量上涨", "缩量回调"
    chip_structure: str = ""  # e.g. "筹码集中", "上方套牢盘多"


class Intelligence(BaseModel):
    """Intelligence / news block."""

    latest_news: list[str] = Field(default_factory=list)  # 最新新闻摘要
    risk_alerts: list[str] = Field(default_factory=list)  # 风险警报
    positive_catalysts: list[str] = Field(default_factory=list)  # 利好催化
    earnings_outlook: str = ""  # 盈利展望


class BattlePlan(BaseModel):
    """Actionable battle plan block."""

    entry_price: float | None = None  # 狙击点 / 买入价
    stop_loss: float | None = None
    take_profit: float | None = None  # 止盈目标
    position_strategy: str = ""  # e.g. "分批建仓", "一次到位", "空仓观望"
    action_items: list[str] = Field(default_factory=list)  # 行动清单


class PhaseDecision(BaseModel):
    """Phase-based decision block (pre-market / intraday / post-market)."""

    pre_market: str = ""  # 盘前观察条件
    intraday: str = ""  # 盘中执行计划
    post_market: str = ""  # 盘后复盘要点


class SignalAttribution(BaseModel):
    """Signal attribution by dimension, each -100 to +100."""

    technical_score: float = Field(default=0.0, ge=-100.0, le=100.0)
    sentiment_score: float = Field(default=0.0, ge=-100.0, le=100.0)
    fundamental_score: float = Field(default=0.0, ge=-100.0, le=100.0)
    market_regime_score: float = Field(default=0.0, ge=-100.0, le=100.0)


class DecisionDashboard(BaseModel):
    """Structured decision dashboard - rich analysis output."""

    core_conclusion: CoreConclusion = Field(default_factory=CoreConclusion)
    data_perspective: DataPerspective = Field(default_factory=DataPerspective)
    intelligence: Intelligence = Field(default_factory=Intelligence)
    battle_plan: BattlePlan = Field(default_factory=BattlePlan)
    phase_decision: PhaseDecision = Field(default_factory=PhaseDecision)
    signal_attribution: SignalAttribution = Field(default_factory=SignalAttribution)


class TradeDecision(BaseModel):
    """Final portfolio manager decision with dashboard."""

    ticker: str
    asset_type: AssetType = AssetType.STOCK
    decision: Decision = Decision.HOLD
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    target_price: float | None = None
    stop_loss: float | None = None
    position_size: float | None = Field(default=None, ge=0.0, le=1.0)  # 0-1 ratio of portfolio
    reasoning: str = ""
    agent_reports: dict[str, str] = Field(default_factory=dict)
    dashboard: DecisionDashboard | None = None


class MarketContext(BaseModel):
    """Immutable market snapshot used by every analysis agent.

    ``as_of_date`` is set for backtests. In that mode the context is built only
    from data available on or before the date and does not include live news or
    current financial data.
    """

    ticker: str
    asset_type: AssetType = AssetType.STOCK
    as_of_date: str | None = None
    current_price: float = 0.0
    realtime: dict = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list)
    financial: dict = Field(default_factory=dict)
    news: list[dict] = Field(default_factory=list)
    market_regime: str = "unknown"
    is_backtest: bool = False
    data_status: dict[str, Any] = Field(default_factory=dict)


class TradeRecord(BaseModel):
    """Executed trade record."""

    date: str
    action: Decision
    ticker: str
    asset_type: AssetType = AssetType.STOCK
    shares: int
    price: float
    amount: float
    commission: float = 0.0
    tax: float = 0.0


class Position(BaseModel):
    """Current position in portfolio.

    T+1 rule: shares bought today are 'frozen' and cannot be sold
    until the next trading day. When the date advances, frozen shares
    are released to available.
    """

    ticker: str
    asset_type: AssetType = AssetType.STOCK
    shares: int  # total = available + frozen
    avg_cost: float
    current_price: float = 0.0
    available_shares: int = 0  # sellable now (bought before today)
    frozen_shares: int = 0  # bought today, cannot sell until next day

    def model_post_init(self, __context) -> None:
        """Ensure available_shares defaults to shares for backward compat."""
        if self.available_shares == 0 and self.frozen_shares == 0:
            self.available_shares = self.shares

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.shares

    @property
    def pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


class PortfolioState(BaseModel):
    """Full portfolio state."""

    cash: float = 1_000_000.0
    positions: list[Position] = Field(default_factory=list)
    trades: list[TradeRecord] = Field(default_factory=list)
    initial_capital: float = 1_000_000.0

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions)

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.initial_capital

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return self.total_pnl / self.initial_capital


class ExternalSimulationConfig(BaseModel):
    """Configuration reserved for an external paper-trading provider.

    The current implementation stores this configuration but does not connect
    to a remote provider. A broker adapter can be enabled later without
    changing the account model or Agent workflow.
    """

    provider: Literal[
        "internal",
        "eastmoney_emt",
        "juejin",
        "joinquant",
        "ricequant",
        "custom",
    ] = "internal"
    enabled: bool = False
    simulation_only: bool = True
    endpoint: str = ""
    account_id: str = ""
    token: str = ""
    options: dict[str, Any] = Field(default_factory=dict)


class LiveTradingConfig(BaseModel):
    """Explicit configuration for a live broker adapter.

    Live execution is deliberately opt-in at both the service and account
    levels.  ``provider=none`` is the safe default and no credentials are
    returned by API payloads.
    """

    provider: str = "none"
    enabled: bool = False
    account_id: str = ""
    endpoint: str = ""
    token: str = ""
    require_manual_approval: bool = True
    max_order_value: float = Field(default=0.0, ge=0)


class LiveOrderIntent(BaseModel):
    """Provider-neutral order request sent after Agent/risk approval."""

    client_order_id: str
    account_id: str
    ticker: str
    asset_type: AssetType = AssetType.STOCK
    side: Decision
    shares: int = Field(gt=0)
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = Field(default=None, gt=0)
    submitted_date: str
    fill_policy: Literal["next_open", "same_close", "manual"] = "next_open"


class LiveOrderResult(BaseModel):
    """Normalized result returned by a live broker adapter."""

    client_order_id: str
    broker_order_id: str | None = None
    status: Literal["submitted", "filled", "rejected", "cancelled", "unknown"]
    message: str = ""
    filled_shares: int = 0
    fill_price: float | None = None


class SimulationAccountConfig(BaseModel):
    """User-configurable rules for a daily A-share simulation account."""

    name: str = "默认日级 Agent 模拟账户"
    initial_cash: float = Field(default=1_000_000.0, gt=0)
    execution_frequency: Literal["1d"] = "1d"
    signal_time: Literal["after_close"] = "after_close"
    fill_time: Literal["next_open", "same_close", "manual"] = "next_open"
    slippage_bps: float = Field(default=5.0, ge=0)
    buy_commission_rate: float = Field(default=0.0003, ge=0)
    sell_commission_rate: float = Field(default=0.0003, ge=0)
    minimum_commission: float = Field(default=5.0, ge=0)
    stamp_tax_rate: float = Field(default=0.001, ge=0)
    transfer_fee_rate: float = Field(default=0.00002, ge=0)
    min_lot: int = Field(default=100, ge=1)
    max_single_position_pct: float = Field(default=0.2, ge=0, le=1)
    max_total_position_pct: float = Field(default=0.95, ge=0, le=1)
    default_stop_loss_pct: float = Field(default=0.08, ge=0, le=1)
    benchmark: str = "000300"
    asset_type: AssetType = AssetType.STOCK
    universe: list[str] = Field(default_factory=list)
    external: ExternalSimulationConfig = Field(default_factory=ExternalSimulationConfig)
    live: LiveTradingConfig = Field(default_factory=LiveTradingConfig)


class SimulationAccount(BaseModel):
    """Persisted simulation account metadata and current portfolio state."""

    account_id: str
    status: Literal["active", "paused"] = "active"
    current_date: str = ""
    config: SimulationAccountConfig = Field(default_factory=SimulationAccountConfig)
    portfolio: PortfolioState = Field(default_factory=PortfolioState)


class SimulationOrder(BaseModel):
    """A simulation order, optionally waiting for a future daily fill."""

    order_id: str
    account_id: str
    ticker: str
    asset_type: AssetType = AssetType.STOCK
    side: Decision
    shares: int
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = None
    status: Literal["pending", "filled", "rejected", "cancelled"] = "pending"
    submitted_date: str = ""
    fill_date: str | None = None
    fill_price: float | None = None
    reject_reason: str | None = None
    source: Literal["manual", "agent", "backtest", "system"] = "manual"
    run_id: str | None = None
    fill_policy: Literal["next_open", "same_close", "manual"] = "next_open"


class SimulationSnapshot(BaseModel):
    """Daily mark-to-market snapshot for the simulation account."""

    account_id: str
    date: str
    cash: float
    total_value: float
    total_pnl: float
    total_return_pct: float
    positions: list[Position] = Field(default_factory=list)


class AutomationTaskConfig(BaseModel):
    """Persistent configuration for an unattended daily Agent task.

    Paper mode remains the default.  Live mode requires an independent service
    feature flag, account-level broker configuration, and ``live_armed``.
    """

    enabled: bool = False
    mode: Literal["observe", "confirm", "auto"] = "observe"
    execution_mode: Literal["paper", "live"] = "paper"
    live_armed: bool = False
    schedule_time: str = "15:10"
    weekdays: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    universe: list[str] = Field(default_factory=list)
    asset_type: AssetType = AssetType.STOCK
    strategy_name: str | None = None
    max_symbols_per_run: int = Field(default=50, ge=1, le=500)
    max_orders_per_run: int = Field(default=20, ge=1, le=200)
    daily_loss_limit_pct: float = Field(default=0.03, ge=0, le=1)
    data_max_age_seconds: int = Field(default=86400, ge=0)
    fill_time: Literal["next_open", "same_close", "manual"] = "next_open"
    simulation_only: bool = True

    @field_validator("schedule_time")
    @classmethod
    def validate_schedule_time(cls, value: str) -> str:
        from datetime import time

        try:
            time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("schedule_time 必须是 HH:MM 格式") from exc
        return value[:5]

    @field_validator("weekdays")
    @classmethod
    def validate_weekdays(cls, value: list[int]) -> list[int]:
        if any(day < 0 or day > 4 for day in value):
            raise ValueError("weekdays 只能包含 0 到 4")
        return sorted(set(value))

    @model_validator(mode="after")
    def validate_live_mode(self):
        if self.execution_mode == "live" and self.simulation_only:
            raise ValueError("实盘模式必须明确设置 simulation_only=false")
        return self


class AgentRunSummary(BaseModel):
    """Durable execution/audit summary for one Agent run."""

    run_id: str
    account_id: str
    run_date: str
    trigger: Literal["schedule", "manual", "retry", "settlement"] = "manual"
    status: Literal["queued", "running", "completed", "failed", "cancelled", "skipped"] = "queued"
    mode: Literal["observe", "confirm", "auto"] = "observe"
    strategy_name: str | None = None
    symbols_total: int = 0
    symbols_processed: int = 0
    decisions_count: int = 0
    orders_count: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    idempotency_key: str


class AgentDecisionAudit(BaseModel):
    """Decision-level audit record, including risk and execution outcome."""

    decision_id: str
    run_id: str
    account_id: str
    ticker: str
    decision: TradeDecision
    current_price: float = 0.0
    risk_status: Literal["pending", "approved", "rejected"] = "pending"
    risk_reason: str | None = None
    order_id: str | None = None
    created_at: str
