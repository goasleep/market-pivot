"""Contracts for reproducible multi-strategy research and sandbox candidates."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.schemas import AssetType, StrategySpec


class TaskContract(BaseModel):
    """Machine-checkable acceptance criteria derived before task execution."""

    model_config = ConfigDict(frozen=True)

    operation: Literal["research", "backtest", "strategy_comparison", "sandbox_research"]
    comparison_axis: Literal["none", "asset", "strategy", "parameter"] = "none"
    minimum_strategy_count: int = Field(default=1, ge=1, le=20)
    required_benchmark: str | None = None
    minimum_history_years: float = Field(default=1.0, gt=0, le=30)
    required_metrics: tuple[str, ...] = ()
    required_outputs: tuple[str, ...] = ()


class TaskAcceptance(BaseModel):
    satisfied: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)


class CostScenario(BaseModel):
    name: Literal["low", "base", "stress"]
    slippage_bps: float = Field(ge=0, le=100)
    buy_commission_rate: float = Field(ge=0, le=0.02)
    sell_commission_rate: float = Field(ge=0, le=0.02)
    minimum_commission: float = Field(ge=0, le=100)
    stamp_tax_rate: float = Field(ge=0, le=0.02)
    transfer_fee_rate: float = Field(ge=0, le=0.02)


class StrategyComparisonSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str = Field(min_length=6, max_length=16)
    asset_type: AssetType
    start_date: str
    end_date: str
    initial_capital: float = Field(default=1_000_000, gt=0)
    fill_time: Literal["next_open", "same_close"] = "next_open"
    benchmark: str = "buy_hold"
    ranking_metric: Literal["total_return", "sharpe_ratio", "calmar_ratio", "out_of_sample_return"] = (
        "total_return"
    )
    strategies: tuple[StrategySpec, ...]
    task_contract: TaskContract
    cost_scenarios: tuple[CostScenario, ...] = ()
    out_of_sample_ratio: float = Field(default=0.3, gt=0, lt=0.5)

    @model_validator(mode="after")
    def validate_strategy_count(self) -> "StrategyComparisonSpec":
        if len(self.strategies) < self.task_contract.minimum_strategy_count:
            raise ValueError("策略数量未达到任务契约要求")
        names = [item.name for item in self.strategies]
        if len(names) != len(set(names)):
            raise ValueError("策略名称不能重复")
        return self


class StrategyPerformance(BaseModel):
    strategy_name: str
    display_name: str
    description: str = ""
    metrics: dict[str, float | int | None]
    total_return: float
    annualized_return: float | None = None
    annualized_volatility: float | None = None
    max_drawdown: float
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    win_rate: float
    profit_factor: float | None = None
    exposure: float | None = None
    turnover: float | None = None
    final_value: float
    total_trades: int
    total_fees: float
    excess_return: float | None = None
    equity_curve: list[dict[str, Any]] = Field(default_factory=list)
    drawdown_curve: list[dict[str, Any]] = Field(default_factory=list)
    trades: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class SandboxPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    timeout_seconds: int = Field(default=10, ge=1, le=60)
    memory_mb: int = Field(default=512, ge=64, le=2048)
    max_source_bytes: int = Field(default=32_000, ge=100, le=256_000)
    allowed_imports: tuple[str, ...] = ("numpy", "pandas")
    function_name: str = "generate_target_positions"


class SandboxValidation(BaseModel):
    passed: bool
    static_checks: dict[str, bool] = Field(default_factory=dict)
    output_checks: dict[str, bool] = Field(default_factory=dict)
    deterministic: bool = False
    causal: bool = False
    errors: list[str] = Field(default_factory=list)


class ResearchStrategyCandidate(BaseModel):
    candidate_id: str
    status: Literal["draft", "validated", "approved", "rejected", "deployed"] = "draft"
    name: str
    version: str = "1.0.0"
    asset_type: AssetType
    ticker: str
    source_code: str
    source_sha256: str
    data_sha256: str
    strategy_spec: StrategySpec
    validation: SandboxValidation
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    reviewed_at: str | None = None
    reviewed_by: str | None = None
    review_note: str = ""


def strategy_comparison_contract(*, minimum_history_years: float = 5.0) -> TaskContract:
    """Default acceptance contract for requests that compare several strategies."""
    return TaskContract(
        operation="strategy_comparison",
        comparison_axis="strategy",
        minimum_strategy_count=7,
        required_benchmark="buy_hold",
        minimum_history_years=minimum_history_years,
        required_metrics=(
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
            "exposure",
            "turnover",
            "total_fees",
            "excess_return",
        ),
        required_outputs=(
            "comparison_table",
            "equity_curves",
            "drawdown_curves",
            "cost_scenarios",
            "out_of_sample",
            "stability",
        ),
    )
