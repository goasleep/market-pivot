"""Data models for A-Share Agent."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class Decision(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class AssetType(str, Enum):
    """Supported exchange-traded assets."""

    STOCK = "stock"
    ETF = "etf"
    LOF = "lof"


class Instrument(BaseModel):
    """Common identity shared by stocks, ETFs, and LOFs."""

    ticker: str
    asset_type: AssetType
    name: str = ""
    exchange: str = ""
    currency: str = "CNY"


class PriceBar(BaseModel):
    """Normalized daily bar shared by all supported instruments."""

    date: str
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    amount: float = 0.0
    pct_chg: float = 0.0
    turnover: float = 0.0


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


class PriceEvidence(BaseModel):
    """One traceable fact used to justify a price level or strategy rule."""

    metric: str
    value: Any = None
    source: str = ""
    as_of: str = ""
    calculation: str = ""


class TradePlan(BaseModel):
    """Canonical, executable plan shared by analysis, simulation, and UI."""

    entry_price: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)
    position_size: float | None = Field(default=None, ge=0.0, le=1.0)
    position_strategy: str = ""  # e.g. "分批建仓", "一次到位", "空仓观望"
    action_items: list[str] = Field(default_factory=list)  # 行动清单
    entry_explanation: str = ""
    stop_loss_explanation: str = ""
    take_profit_explanation: str = ""
    price_evidence: list[PriceEvidence] = Field(default_factory=list)


class StrategyCondition(BaseModel):
    """Small deterministic condition vocabulary executable by the backtester."""

    indicator: str
    operator: Literal["gt", "gte", "lt", "lte", "eq", "between"]
    value: float | list[float]
    window: int | None = Field(default=None, ge=1)
    description: str = ""


class IndicatorSpec(BaseModel):
    """A bounded, auditable indicator requested by an Agent strategy."""

    name: str
    alias: str | None = None
    source: Literal["close", "ohlcv"] = "close"
    window: int | None = Field(default=None, ge=1)
    role: Literal["entry", "exit", "filter", "confirmation", "risk"] = "filter"
    params: dict[str, float | int | str] = Field(default_factory=dict)


class PositionModel(BaseModel):
    """Bounded target-exposure model evaluated by the trusted runtime."""

    type: Literal["fixed", "volatility_target", "trend_volatility_target"] = "fixed"
    volatility_window: int = Field(default=20, ge=2, le=252)
    target_volatility: float = Field(default=0.15, gt=0, le=1)
    trend_window: int = Field(default=60, ge=2, le=252)
    min_exposure: float = Field(default=0.0, ge=0, le=1)
    max_exposure: float = Field(default=0.95, ge=0, le=1)
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = "weekly"

    @model_validator(mode="after")
    def validate_exposure_bounds(self):
        if self.min_exposure > self.max_exposure:
            raise ValueError("min_exposure 不能大于 max_exposure")
        return self


class StrategySpec(BaseModel):
    """Versioned strategy definition produced by YAML or an LLM."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    asset_types: list[AssetType] = Field(default_factory=lambda: [AssetType.ETF, AssetType.LOF])
    indicators: list[str] = Field(default_factory=list)
    indicator_specs: list[IndicatorSpec] = Field(default_factory=list)
    entry_conditions: list[StrategyCondition] = Field(default_factory=list)
    exit_conditions: list[StrategyCondition] = Field(default_factory=list)
    entry_condition_logic: Literal["all", "any"] = "all"
    exit_condition_logic: Literal["all", "any"] = "any"
    stop_loss_pct: float | None = Field(default=None, ge=0, le=1)
    take_profit_pct: float | None = Field(default=None, ge=0)
    position_size_pct: float = Field(default=0.2, ge=0, le=1)
    rebalance_frequency: Literal["daily", "weekly", "manual"] = "daily"
    position_model: PositionModel | None = None
    source: Literal["yaml", "llm", "user", "sandbox"] = "yaml"


class PortfolioSpec(BaseModel):
    """Deterministic portfolio construction rules for multi-asset backtests."""

    allocation_method: Literal["equal_weight"] = "equal_weight"
    rebalance_frequency: Literal["daily", "weekly", "monthly", "manual"] = "weekly"
    max_position_weight: float = Field(default=0.4, gt=0.0, le=1.0)
    max_positions: int = Field(default=3, ge=1, le=100)
    cash_reserve: float = Field(default=0.1, ge=0.0, lt=1.0)


class BattlePlan(TradePlan):
    """Backward-compatible dashboard view of the canonical trade plan."""


class StrategyPlan(BaseModel):
    """The strategy selected or synthesized for this decision."""

    name: str = ""
    thesis: str = ""
    entry_conditions: list[str] = Field(default_factory=list)
    exit_conditions: list[str] = Field(default_factory=list)
    indicators_used: list[str] = Field(default_factory=list)
    data_basis: list[PriceEvidence] = Field(default_factory=list)
    spec: StrategySpec | None = None


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
    strategy_plan: StrategyPlan = Field(default_factory=StrategyPlan)
    phase_decision: PhaseDecision = Field(default_factory=PhaseDecision)
    signal_attribution: SignalAttribution = Field(default_factory=SignalAttribution)


class TradeDecision(BaseModel):
    """Final portfolio manager decision with dashboard."""

    ticker: str
    asset_type: AssetType = AssetType.STOCK
    decision: Decision = Decision.HOLD
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    plan: TradePlan = Field(default_factory=TradePlan)
    reasoning: str = ""
    agent_reports: dict[str, str] = Field(default_factory=dict)
    dashboard: DecisionDashboard | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_plan(cls, value):
        """Accept the old flat decision shape while storing one canonical plan."""
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        plan = payload.get("plan")
        dashboard = payload.get("dashboard")
        dashboard_plan = dashboard.get("battle_plan") if isinstance(dashboard, dict) else None
        if not plan:
            plan = dashboard_plan or {}
        plan = dict(plan)
        for field_name in (
            "entry_price",
            "stop_loss",
            "take_profit",
            "position_strategy",
            "action_items",
            "entry_explanation",
            "stop_loss_explanation",
            "take_profit_explanation",
            "price_evidence",
        ):
            if field_name in payload and payload[field_name] is not None and field_name not in plan:
                plan[field_name] = payload[field_name]
        if "position_size" in payload and payload["position_size"] is not None and "position_size" not in plan:
            plan["position_size"] = payload["position_size"]
        if payload.get("target_price") is not None and "take_profit" not in plan:
            plan["take_profit"] = payload["target_price"]
        payload["plan"] = plan
        return payload

    @model_validator(mode="after")
    def _sync_dashboard_plan(self):
        """Keep the rich dashboard representation derived from ``plan``."""
        if self.dashboard is not None:
            existing = self.dashboard.battle_plan.model_dump()
            existing.update(self.plan.model_dump(exclude_none=False))
            self.dashboard.battle_plan = BattlePlan.model_validate(existing)
        return self

    @computed_field
    @property
    def entry_price(self) -> float | None:
        return self.plan.entry_price

    @computed_field
    @property
    def target_price(self) -> float | None:
        return self.plan.take_profit

    @computed_field
    @property
    def stop_loss(self) -> float | None:
        return self.plan.stop_loss

    @computed_field
    @property
    def take_profit(self) -> float | None:
        return self.plan.take_profit

    @computed_field
    @property
    def position_size(self) -> float | None:
        return self.plan.position_size


class MarketContext(BaseModel):
    """Immutable market snapshot used by every analysis agent.

    ``as_of_date`` is set for backtests. In that mode the context is built only
    from data available on or before the date and does not include live news or
    current financial data.
    """

    ticker: str
    asset_type: AssetType = AssetType.STOCK
    instrument: Instrument | None = None
    as_of_date: str | None = None
    current_price: float = 0.0
    realtime: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    financial: dict[str, Any] = Field(default_factory=dict)
    news: list[dict[str, Any]] = Field(default_factory=list)
    web_results: list[dict[str, Any]] = Field(default_factory=list)
    fund_data: "FundSnapshot | None" = None
    market_regime: str = "unknown"
    is_backtest: bool = False
    data_status: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_instrument(self):
        if self.instrument is None:
            self.instrument = Instrument(
                ticker=self.ticker,
                asset_type=self.asset_type,
                name=str(self.realtime.get("name", "")),
            )
        return self


class FundSnapshot(BaseModel):
    """Typed ETF/LOF snapshot normalized from AkShare."""

    source: str = ""
    as_of: str = ""
    realtime_fields: dict[str, Any] = Field(default_factory=dict)
    derived_metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    nav_history: list[dict[str, Any]] = Field(default_factory=list)


MarketContext.model_rebuild()
AssetResearchContext = MarketContext


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
    transfer_fee: float = 0.0
    external_id: str | None = None


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
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)

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
    """Configuration for an external paper-trading provider."""

    provider: Literal[
        "internal",
        "eastmoney_emt",
        "eastmoney_file",
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
    input_dir: str = ""
    output_dir: str = ""
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


class AssetTradingRules(BaseModel):
    """Trading constraints that vary by asset type."""

    asset_type: AssetType
    min_lot: int = Field(default=100, ge=1)
    t_plus_one: bool = True
    slippage_bps: float = Field(default=5.0, ge=0)
    buy_commission_rate: float = Field(default=0.0003, ge=0)
    sell_commission_rate: float = Field(default=0.0003, ge=0)
    minimum_commission: float = Field(default=5.0, ge=0)
    stamp_tax_rate: float = Field(default=0.001, ge=0)
    transfer_fee_rate: float = Field(default=0.00002, ge=0)
    auto_exit_levels: bool = True
    max_single_position_pct: float = Field(default=0.2, ge=0, le=1)
    max_total_position_pct: float = Field(default=0.95, ge=0, le=1)

    @classmethod
    def defaults_for(cls, asset_type: AssetType) -> "AssetTradingRules":
        """Return conservative defaults for the supported A-share asset types."""
        if asset_type in {AssetType.ETF, AssetType.LOF}:
            return cls(
                asset_type=asset_type,
                stamp_tax_rate=0.0,
                transfer_fee_rate=0.0,
                t_plus_one=True,
            )
        return cls(asset_type=asset_type)


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
    trading_rules: AssetTradingRules | None = None

    @model_validator(mode="after")
    def _set_trading_rules(self):
        if self.trading_rules is None or self.trading_rules.asset_type != self.asset_type:
            self.trading_rules = AssetTradingRules(
                asset_type=self.asset_type,
                min_lot=self.min_lot,
                slippage_bps=self.slippage_bps,
                buy_commission_rate=self.buy_commission_rate,
                sell_commission_rate=self.sell_commission_rate,
                minimum_commission=self.minimum_commission,
                stamp_tax_rate=self.stamp_tax_rate,
                transfer_fee_rate=self.transfer_fee_rate,
                max_single_position_pct=self.max_single_position_pct,
                max_total_position_pct=self.max_total_position_pct,
            )
            if self.asset_type in {AssetType.ETF, AssetType.LOF}:
                self.trading_rules.stamp_tax_rate = 0.0
                self.trading_rules.transfer_fee_rate = 0.0
        return self

    def effective_trading_rules(self, asset_type: AssetType | str | None = None) -> AssetTradingRules:
        """Return rules for the requested asset while preserving legacy fields."""
        requested = AssetType(asset_type or self.asset_type)
        if requested == self.asset_type and self.trading_rules is not None:
            return self.trading_rules
        return AssetTradingRules.defaults_for(requested).model_copy(
            update={
                "slippage_bps": self.slippage_bps,
                "buy_commission_rate": self.buy_commission_rate,
                "sell_commission_rate": self.sell_commission_rate,
                "minimum_commission": self.minimum_commission,
                "min_lot": self.min_lot,
                "max_single_position_pct": self.max_single_position_pct,
                "max_total_position_pct": self.max_total_position_pct,
            }
        )


class SimulationAccount(BaseModel):
    """Persisted simulation account metadata and current portfolio state."""

    account_id: str
    status: Literal["active", "paused", "archived"] = "active"
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
    deployment_id: str | None = None
    decision_id: str | None = None
    fill_policy: Literal["next_open", "same_close", "manual"] = "next_open"
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)


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
    deployment_id: str | None = None
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
    deployment_id: str | None = None
    strategy_sha256: str | None = None
    llm_runtime: dict[str, Any] = Field(default_factory=dict)
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
    signal_source: Literal["agent", "deployed_strategy"] = "agent"
    strategy_evaluation: dict[str, Any] = Field(default_factory=dict)
    agent_gate: dict[str, Any] = Field(default_factory=dict)
    proposed_order: dict[str, Any] | None = None
    confirmation_status: Literal["none", "pending", "confirmed", "rejected", "expired"] = "none"
    created_at: str


class StrategyDeployment(BaseModel):
    """Immutable strategy snapshot bound to one paper account."""

    deployment_id: str
    experiment_id: str
    account_id: str
    status: Literal["active", "paused", "archived"] = "active"
    strategy_name: str
    strategy_version: str
    strategy_sha256: str
    strategy_spec: StrategySpec
    portfolio_spec: PortfolioSpec | None = None
    universe: list[str]
    asset_type: AssetType
    execution: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    activated_at: str | None = None
    archived_at: str | None = None
