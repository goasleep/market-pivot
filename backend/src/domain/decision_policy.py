"""Deterministic validation and sizing rules for Agent decisions."""

from dataclasses import dataclass

from models.schemas import AssetTradingRules, Decision, TradeDecision


@dataclass(frozen=True)
class DecisionIssue:
    code: str
    message: str


class DecisionValidator:
    """Validate an LLM decision before it reaches an execution boundary."""

    @staticmethod
    def validate(decision: TradeDecision, *, current_price: float = 0.0) -> list[DecisionIssue]:
        issues: list[DecisionIssue] = []
        if decision.decision == Decision.BUY:
            plan = decision.plan
            if plan.entry_price is None:
                issues.append(DecisionIssue("missing_entry", "买入决策缺少入场价"))
            if plan.stop_loss is None:
                issues.append(DecisionIssue("missing_stop", "买入决策缺少止损价"))
            if plan.take_profit is None:
                issues.append(DecisionIssue("missing_take_profit", "买入决策缺少止盈价"))
            if plan.stop_loss and plan.entry_price and plan.stop_loss >= plan.entry_price:
                issues.append(DecisionIssue("invalid_stop_order", "止损价必须低于入场价"))
            if plan.take_profit and plan.entry_price and plan.take_profit <= plan.entry_price:
                issues.append(DecisionIssue("invalid_take_profit_order", "止盈价必须高于入场价"))
            if not plan.entry_explanation.strip():
                issues.append(DecisionIssue("missing_entry_reason", "入场价缺少解释"))
            if not plan.stop_loss_explanation.strip():
                issues.append(DecisionIssue("missing_stop_reason", "止损价缺少解释"))
            if not plan.take_profit_explanation.strip():
                issues.append(DecisionIssue("missing_take_profit_reason", "止盈价缺少解释"))
            if not plan.price_evidence:
                issues.append(DecisionIssue("missing_price_evidence", "价格方案缺少数据依据"))
        if current_price < 0:
            issues.append(DecisionIssue("invalid_current_price", "当前价格不能为负数"))
        return issues


class OrderSizer:
    """Calculate a deterministic order quantity from portfolio and risk limits."""

    @staticmethod
    def shares(portfolio, rules: AssetTradingRules, decision: TradeDecision, price: float) -> int:
        if price <= 0 or decision.decision == Decision.HOLD:
            return 0
        min_lot = rules.min_lot
        if decision.decision == Decision.BUY:
            position_pct = min(max(decision.position_size or 0.2, 0.0), rules.max_single_position_pct)
            existing_invested = sum(
                position.shares * (position.current_price or position.avg_cost) for position in portfolio.positions
            )
            total_value = max(portfolio.total_value, portfolio.cash)
            max_invest = min(
                portfolio.cash * position_pct,
                max(total_value * rules.max_total_position_pct - existing_invested, 0),
            )
            return (int(max_invest / price) // min_lot) * min_lot

        position = next((item for item in portfolio.positions if item.ticker == decision.ticker), None)
        if not position:
            return 0
        if decision.stop_loss and price <= decision.stop_loss:
            return position.available_shares
        return position.available_shares // min_lot * min_lot
