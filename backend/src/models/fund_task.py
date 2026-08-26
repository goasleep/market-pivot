"""Machine-checkable contracts for the fund-focused conversation entry point."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class FundTaskKind(str, Enum):
    EDUCATION = "education"
    CALCULATION = "calculation"
    RULE_DESIGN = "rule_design"
    SCENARIO_PLAN = "scenario_plan"
    INSTRUMENT_RESEARCH = "instrument_research"
    UNIVERSE_RESEARCH = "universe_research"
    EVENT_RESEARCH = "event_research"
    SIMULATION_QUERY = "simulation_query"
    SIMULATION_MUTATION = "simulation_mutation"
    CLARIFICATION = "clarification"


class EvidenceMode(str, Enum):
    NONE = "none"
    USER_PROVIDED = "user_provided"
    FUND_PROFILE = "fund_profile"
    NAV_HISTORY = "nav_history"
    REALTIME_MARKET = "realtime_market"
    ANNOUNCEMENTS = "announcements"
    UNIVERSE_DATA = "universe_data"
    SIMULATION_STATE = "simulation_state"


class InstrumentResolutionStatus(str, Enum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"
    INVALID = "invalid"


class TaskOutcome(str, Enum):
    SATISFIED = "satisfied"
    PARTIAL = "partial"
    NEEDS_INPUT = "needs_input"
    DATA_UNAVAILABLE = "data_unavailable"
    FAILED = "failed"


class FundDomain(str, Enum):
    EXCHANGE_FUND = "exchange_fund"
    OPEN_FUND = "open_fund"


class FundProductCategory(str, Enum):
    EQUITY = "equity"
    HYBRID = "hybrid"
    BOND = "bond"
    MONEY_MARKET = "money_market"
    INDEX = "index"
    ENHANCED_INDEX = "enhanced_index"
    QDII = "qdii"
    FOF = "fof"
    UNKNOWN = "unknown"


class FundSubject(BaseModel):
    scope: Literal[
        "fund_concept",
        "fund_instrument",
        "fund_universe",
        "portfolio",
        "account",
        "underlying_stock",
    ] = "fund_concept"
    fund_domain: FundDomain
    asset_type: Literal["etf", "lof", "open_fund"]
    product_category: FundProductCategory = FundProductCategory.UNKNOWN
    pricing_basis: Literal["market_price", "nav", "money_yield"] = "nav"


class FundInstrumentRef(BaseModel):
    status: InstrumentResolutionStatus
    fund_code: str | None = None
    exchange_ticker: str | None = None
    name: str | None = None
    share_class: str | None = None
    asset_type: Literal["etf", "lof", "open_fund"]
    product_category: FundProductCategory = FundProductCategory.UNKNOWN
    trading_venue: Literal["exchange", "otc", "unknown"] = "unknown"
    provider_id: str | None = None
    verified_at: str | None = None
    resolution_reason: str = ""


class FundSelectionRequirements(BaseModel):
    """Deterministic delivery requirements for fund universe selection tasks."""

    selection_mode: Literal["single", "rank"] = "rank"
    holding_horizon: Literal["short_term", "medium_term", "unspecified"] = "unspecified"
    minimum_candidates: int = Field(default=3, ge=1, le=20)
    require_comparison: bool = True
    require_primary: bool = True
    require_alternative: bool = True
    require_exclusions: bool = True
    require_data_as_of: bool = True
    comparison_dimensions: list[str] = Field(default_factory=list)


class FundIntentSpec(BaseModel):
    task_kind: FundTaskKind
    operation: str
    subject: FundSubject
    evidence_mode: EvidenceMode = EvidenceMode.NONE
    user_inputs: dict[str, Any] = Field(default_factory=dict)
    instruments: list[FundInstrumentRef] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    requires_live_data: bool = False
    requires_verified_instrument: bool = False
    required_outputs: list[str] = Field(default_factory=list)
    selection_requirements: FundSelectionRequirements | None = None
    allowed_capabilities: list[str] = Field(default_factory=list)
    forbidden_capabilities: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)


class FundAnswerAcceptance(BaseModel):
    outcome: TaskOutcome
    satisfied: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
