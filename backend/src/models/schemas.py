"""Data models for A-Share Agent."""

from pydantic import BaseModel, Field
from enum import Enum
from datetime import date


class Decision(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


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
    confidence: float = 0.5
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
    confidence: float = 0.5
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

    technical_score: float = 0.0
    sentiment_score: float = 0.0
    fundamental_score: float = 0.0
    market_regime_score: float = 0.0


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
    decision: Decision = Decision.HOLD
    confidence: float = 0.5
    target_price: float | None = None
    stop_loss: float | None = None
    position_size: float | None = None  # 0-1 ratio of portfolio
    reasoning: str = ""
    agent_reports: dict[str, str] = Field(default_factory=dict)
    dashboard: DecisionDashboard | None = None


class TradeRecord(BaseModel):
    """Executed trade record."""

    date: str
    action: Decision
    ticker: str
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
