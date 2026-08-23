"""Deterministic calculations for common retail fund questions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FundCalculationResult:
    title: str
    inputs: dict[str, float | str]
    formula: str
    result: str
    assumptions: list[str] = field(default_factory=list)

    def render(self) -> str:
        inputs = "；".join(f"{key}={value}" for key, value in self.inputs.items())
        assumptions = "\n".join(f"- {item}" for item in self.assumptions)
        suffix = f"\n\n假设：\n{assumptions}" if assumptions else ""
        return f"### {self.title}\n\n输入：{inputs}\n\n公式：`{self.formula}`\n\n结果：**{self.result}**{suffix}"


_PERCENT = r"(\d+(?:\.\d+)?)\s*%"
_AMOUNT = r"(\d+(?:\.\d+)?)\s*(万元|万|元)"


def _money(number: str, unit: str) -> float:
    return float(number) * (10_000 if unit in {"万元", "万"} else 1)


def calculate_from_question(message: str) -> FundCalculationResult | None:
    """Return a deterministic result for recognized formulas; otherwise let the direct executor explain."""
    if "赎回费" in message:
        rate = re.search(rf"赎回费率(?:为|是)?\s*{_PERCENT}", message)
        amount = re.search(rf"(?:买入金额|赎回金额|本金)(?:为|是)?\s*{_AMOUNT}", message)
        if rate and amount:
            rate_pct = float(rate.group(1))
            principal = _money(amount.group(1), amount.group(2))
            cost = principal * rate_pct / 100
            return FundCalculationResult(
                title="赎回费用计算",
                inputs={"赎回金额": f"{principal:.2f}元", "赎回费率": f"{rate_pct}%"},
                formula="赎回费用 = 赎回金额 × 赎回费率",
                result=f"{cost:.2f}元；赎回后金额约为{principal - cost:.2f}元",
                assumptions=["基金净值没有变化", "不考虑其他可能费用及税费"],
            )

    if "佣金" in message and "万分之" in message:
        rate = re.search(r"万分之\s*(\d+(?:\.\d+)?)", message)
        minimum = re.search(r"最低\s*(\d+(?:\.\d+)?)\s*元", message)
        amount = re.search(rf"(?:买入金额|成交金额)(?:为|是)?\s*{_AMOUNT}", message)
        if rate and minimum and amount:
            rate_value = float(rate.group(1)) / 10_000
            minimum_value = float(minimum.group(1))
            principal = _money(amount.group(1), amount.group(2))
            one_way = max(principal * rate_value, minimum_value)
            round_trip = one_way * 2
            return FundCalculationResult(
                title="ETF佣金计算",
                inputs={
                    "单边成交金额": f"{principal:.2f}元",
                    "佣金率": f"万分之{float(rate.group(1)):g}",
                    "单笔最低佣金": f"{minimum_value:.2f}元",
                },
                formula="单边佣金 = max(成交金额 × 佣金率, 最低佣金)；往返佣金 = 买入佣金 + 卖出佣金",
                result=f"单边{one_way:.2f}元；按相同卖出金额估算，买卖合计{round_trip:.2f}元",
                assumptions=["卖出成交金额与买入金额相同", "未计入价差、滑点及券商可能收取的其他费用"],
            )

    if "A类" in message and "C类" in message and ("销售服务费" in message or "申购费" in message):
        a_rate = re.search(r"A类[^%]{0,40}?(\d+(?:\.\d+)?)\s*%", message, re.IGNORECASE)
        c_rate = re.search(r"C类[^%]{0,40}?(\d+(?:\.\d+)?)\s*%", message, re.IGNORECASE)
        months_match = re.search(r"预计持有\s*(\d+)\s*个?月", message)
        chinese_months = {"三个月": 3, "九个月": 9, "六个月": 6}
        months = int(months_match.group(1)) if months_match else next(
            (value for key, value in chinese_months.items() if key in message),
            None,
        )
        if a_rate and c_rate and months:
            a_pct = float(a_rate.group(1))
            c_annual_pct = float(c_rate.group(1))
            c_holding_pct = c_annual_pct * months / 12
            difference = c_holding_pct - a_pct
            winner = "A类费用较低" if difference > 0 else "C类费用较低" if difference < 0 else "两者费用近似"
            return FundCalculationResult(
                title="A类与C类持有成本比较",
                inputs={"A类申购费率": f"{a_pct}%", "C类年销售服务费": f"{c_annual_pct}%", "持有期": f"{months}个月"},
                formula="C类持有期销售服务费率 ≈ 年费率 × 持有月数 ÷ 12",
                result=(
                    f"A类约{a_pct:.3f}%，C类约{c_holding_pct:.3f}%；仅比较这两项时，{winner}，"
                    f"差约{abs(difference):.3f}个百分点"
                ),
                assumptions=["未计入赎回费、管理费、托管费和净值差异", "销售服务费按时间线性近似"],
            )
    return None
