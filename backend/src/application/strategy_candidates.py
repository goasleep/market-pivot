"""Lifecycle for LLM-authored research signals and paper-only promotion."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from application.backtest_experiment import BacktestExperimentStore, backtest_experiments
from application.research_sandbox import (
    replay_target_positions,
    source_sha256,
    validate_and_run_signals,
)
from data.db_models import ResearchStrategyCandidateRecord
from data.tortoise_db import init_database
from engine.backtester import prepare_single_backtest_data
from engine.strategy_runtime import decision_from_strategy
from llm.service import get_llm_service
from models.schemas import AssetType, Decision, Position, StrategySpec
from models.strategy_research import ResearchStrategyCandidate
from strategies.compiler import available_indicators, strategy_from_mapping

SANDBOX_DESIGN_SYSTEM = """你是量化研究代码 Agent。只返回 JSON，不要 Markdown。
JSON 必须包含 source_code 和 strategy_spec。source_code 只能定义同步函数
generate_target_positions(frame)，输入列仅有 date/open/high/low/close/volume，返回与 frame 等长且只含 0/1 的序列。
函数必须只使用截至当前行的 rolling/expanding/shift 数据，禁止 iloc[-1] 影响历史行，禁止负数 shift，禁止网络、文件、
进程、线程、反射、动态执行和随机数。pandas 以 pd、numpy 以 np 预置。
strategy_spec 必须使用给定受控指标描述与代码完全相同的信号，source=sandbox；只有两者逐日信号完全一致才可进入模拟盘审批。
不要计算成交价、订单、资金、费用或绩效，这些由可信交易引擎负责。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrategyCandidateService:
    def __init__(self, db_path: str | Path | None = None, *, experiments: BacktestExperimentStore | None = None):
        self.db_path = db_path
        self.experiments = experiments or backtest_experiments

    async def _ready(self) -> None:
        await init_database(db_path=self.db_path)

    @staticmethod
    def _from_row(row: ResearchStrategyCandidateRecord) -> ResearchStrategyCandidate:
        return ResearchStrategyCandidate.model_validate(json.loads(row.payload_json))

    async def save(self, candidate: ResearchStrategyCandidate) -> ResearchStrategyCandidate:
        await self._ready()
        timestamp = _now()
        await ResearchStrategyCandidateRecord.update_or_create(
            candidate_id=candidate.candidate_id,
            defaults={
                "status": candidate.status,
                "payload_json": candidate.model_dump_json(),
                "created_at": candidate.created_at,
                "updated_at": timestamp,
            },
        )
        return candidate

    async def get(self, candidate_id: str) -> ResearchStrategyCandidate:
        await self._ready()
        row = await ResearchStrategyCandidateRecord.get_or_none(candidate_id=candidate_id)
        if row is None:
            raise KeyError(f"研究策略候选不存在: {candidate_id}")
        return self._from_row(row)

    async def list(self, *, status: str | None = None) -> list[ResearchStrategyCandidate]:
        await self._ready()
        query = ResearchStrategyCandidateRecord.all()
        if status:
            query = query.filter(status=status)
        rows = await query.order_by("-created_at").limit(100)
        return [self._from_row(row) for row in rows]

    async def generate(
        self,
        *,
        objective: str,
        ticker: str,
        asset_type: AssetType | str,
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000,
    ) -> ResearchStrategyCandidate:
        kind = AssetType(asset_type)
        prompt = json.dumps(
            {
                "objective": objective,
                "ticker": ticker,
                "asset_type": kind.value,
                "available_indicators": available_indicators(),
            },
            ensure_ascii=False,
        )
        raw = await get_llm_service().chat_json(prompt, system=SANDBOX_DESIGN_SYSTEM)
        if not isinstance(raw, dict) or not isinstance(raw.get("source_code"), str):
            raise ValueError("代码 Agent 未返回 source_code 和 strategy_spec")
        source = _strip_code_fence(raw["source_code"])
        strategy_payload = dict(raw.get("strategy_spec") or {})
        strategy_payload.setdefault("name", f"sandbox_{ticker}_{uuid4().hex[:6]}")
        strategy_payload.setdefault("asset_types", [kind.value])
        strategy = strategy_from_mapping(strategy_payload, source="sandbox")
        prepared = await prepare_single_backtest_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            asset_type=kind,
        )
        frame, snapshot = prepared
        positions, validation = await validate_and_run_signals(source, frame)
        dsl_positions = _strategy_target_positions(strategy, frame) if validation.passed else []
        equivalent = bool(positions) and positions == dsl_positions
        history_years = (
            date.fromisoformat(str(snapshot["actual_end_date"]))
            - date.fromisoformat(str(snapshot["actual_start_date"]))
        ).days / 365.25
        history_sufficient = history_years >= 5 and len(frame) >= 750
        validation.output_checks["dsl_signal_equivalent"] = equivalent
        validation.output_checks["minimum_history_5y"] = history_sufficient
        if not equivalent:
            validation.errors.append("代码信号与可部署 StrategySpec 不完全一致，仅可保留为研究候选")
        if not history_sufficient:
            validation.errors.append("有效历史区间不足 5 年，仅可保留为研究候选")
        backtest = (
            replay_target_positions(
                ticker=ticker,
                asset_type=kind,
                frame=frame,
                positions=positions,
                initial_capital=initial_capital,
            )
            if validation.passed
            else {}
        )
        backtest.update(
            {
                "start_date": start_date,
                "end_date": end_date,
                "data_snapshot": snapshot,
                "strategy_spec": strategy.model_dump(mode="json"),
                "buy_hold_return": round(float(frame.iloc[-1]["close"] / frame.iloc[0]["close"] - 1), 6),
                "history_years": round(history_years, 2),
                "out_of_sample": _out_of_sample(backtest.get("equity_curve") or []),
            }
        )
        promotion_eligible = validation.passed and equivalent and history_sufficient
        candidate = ResearchStrategyCandidate(
            candidate_id=f"candidate-{uuid4().hex[:16]}",
            status="validated" if promotion_eligible else "draft",
            name=strategy.name,
            version=strategy.version,
            asset_type=kind,
            ticker=ticker,
            source_code=source,
            source_sha256=source_sha256(source),
            data_sha256=str(snapshot["sha256"]),
            strategy_spec=strategy,
            validation=validation,
            result={
                "objective": objective,
                "promotion_eligible": promotion_eligible,
                "backtest": backtest,
            },
            created_at=_now(),
        )
        return await self.save(candidate)

    async def review(
        self,
        candidate_id: str,
        *,
        approved: bool,
        reviewed_by: str,
        note: str = "",
    ) -> ResearchStrategyCandidate:
        candidate = await self.get(candidate_id)
        if not reviewed_by.strip():
            raise ValueError("审批人不能为空")
        if approved and (
            candidate.status != "validated"
            or not candidate.validation.passed
            or not candidate.result.get("promotion_eligible")
        ):
            raise ValueError("只有通过安全、因果、确定性和 DSL 等价验证的候选才能审批")
        candidate.status = "approved" if approved else "rejected"
        candidate.reviewed_at = _now()
        candidate.reviewed_by = reviewed_by.strip()
        candidate.review_note = note
        return await self.save(candidate)

    async def deploy_to_paper(
        self,
        candidate_id: str,
        *,
        account_id: str,
        execution_mode: str = "paper",
    ) -> Any:
        if execution_mode != "paper":
            raise ValueError("LLM/沙盒研究策略禁止部署到实盘，只能进入模拟盘")
        candidate = await self.get(candidate_id)
        if candidate.status != "approved":
            raise ValueError("候选策略必须经过人工审批后才能部署到模拟盘")
        experiment_id = f"sandbox-{candidate.candidate_id}"
        result = dict(candidate.result.get("backtest") or {})
        payload = {
            "objective": candidate.result.get("objective", "沙盒研究策略"),
            "mode": "single",
            "strategy_spec": candidate.strategy_spec.model_dump(mode="json"),
            "portfolio_spec": None,
            "result": result,
            "artifacts": [],
            "candidate_id": candidate.candidate_id,
            "source_sha256": candidate.source_sha256,
            "data_sha256": candidate.data_sha256,
        }
        await self.experiments.save(experiment_id, "completed", payload)
        from application.deployments import deployment_service

        deployment = await deployment_service.create_from_experiment(
            experiment_id,
            account_id=account_id,
            mode="confirm",
            enabled=True,
            execution_key=f"{candidate.candidate_id}:{account_id}",
        )
        candidate.status = "deployed"
        await self.save(candidate)
        return deployment


def _strip_code_fence(source: str) -> str:
    text = source.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        return "\n".join(lines).strip()
    return text


def _strategy_target_positions(spec: StrategySpec, frame: pd.DataFrame) -> list[int]:
    holding = False
    entry_price = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    output: list[int] = []
    for index, row in frame.reset_index(drop=True).iterrows():
        current_price = float(row["close"])
        position = (
            Position(
                ticker="candidate",
                asset_type=spec.asset_types[0],
                shares=100,
                available_shares=100,
                avg_cost=entry_price,
                current_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            if holding
            else None
        )
        decision, _ = decision_from_strategy(
            spec,
            frame.iloc[: index + 1],
            asset_type=spec.asset_types[0],
            ticker="candidate",
            current_price=current_price,
            position=position,
        )
        if decision.decision == Decision.BUY:
            holding = True
            entry_price = current_price
            stop_loss = decision.stop_loss
            take_profit = decision.take_profit
        elif decision.decision == Decision.SELL:
            holding = False
            stop_loss = None
            take_profit = None
        output.append(int(holding))
    return output


def _out_of_sample(curve: list[dict[str, Any]]) -> dict[str, Any]:
    if len(curve) < 3:
        return {}
    split = max(2, min(len(curve) - 1, int(len(curve) * 0.7)))
    first = float(curve[0]["value"])
    middle = float(curve[split - 1]["value"])
    final = float(curve[-1]["value"])
    return {
        "split_date": curve[split]["date"],
        "in_sample_return": round(middle / first - 1, 6) if first else 0.0,
        "out_of_sample_return": round(final / middle - 1, 6) if middle else 0.0,
    }


strategy_candidates = StrategyCandidateService()
