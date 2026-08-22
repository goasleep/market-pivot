from types import SimpleNamespace

import pandas as pd
import pytest

import application.strategy_candidates as candidate_module
from application.backtest_experiment import BacktestExperimentStore
from application.strategy_candidates import StrategyCandidateService


def _history():
    close = [10 + index * 0.002 + (0.2 if index % 30 < 15 else -0.2) for index in range(800)]
    return pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", "2026-01-01", periods=len(close)).strftime("%Y-%m-%d"),
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": [1_000_000] * len(close),
        }
    )


@pytest.mark.asyncio
async def test_candidate_requires_equivalence_and_human_review_then_remains_paper_only(monkeypatch, tmp_path):
    frame = _history()
    snapshot = {
        "sha256": "a" * 64,
        "actual_start_date": frame.iloc[0]["date"],
        "actual_end_date": frame.iloc[-1]["date"],
    }

    class FakeLLM:
        async def chat_json(self, *_args, **_kwargs):
            return {
                "source_code": """
def generate_target_positions(frame):
    fast = frame["close"].rolling(2).mean()
    slow = frame["close"].rolling(4).mean()
    return (fast > slow).fillna(False).astype(int)
""",
                "strategy_spec": {
                    "name": "sandbox_ma_2_4",
                    "asset_types": ["etf"],
                    "indicators": ["spread"],
                    "indicator_specs": [
                        {
                            "name": "ma_spread_pct",
                            "alias": "spread",
                            "params": {"fast_window": 2, "slow_window": 4},
                        }
                    ],
                    "entry_conditions": [{"indicator": "spread", "operator": "gt", "value": 0}],
                    "exit_conditions": [{"indicator": "spread", "operator": "lte", "value": 0}],
                    "position_size_pct": 0.95,
                    "source": "sandbox",
                },
            }

    async def fake_prepared(**_kwargs):
        return frame, snapshot

    monkeypatch.setattr(candidate_module, "get_llm_service", lambda: FakeLLM())
    monkeypatch.setattr(candidate_module, "prepare_single_backtest_data", fake_prepared)
    experiments = BacktestExperimentStore(tmp_path / "candidates.sqlite3")
    service = StrategyCandidateService(tmp_path / "candidates.sqlite3", experiments=experiments)

    candidate = await service.generate(
        objective="测试 MA2/4",
        ticker="510300",
        asset_type="etf",
        start_date="2016-01-01",
        end_date="2026-01-01",
    )

    assert candidate.status == "validated"
    assert candidate.validation.output_checks["dsl_signal_equivalent"] is True
    assert candidate.result["promotion_eligible"] is True
    with pytest.raises(ValueError, match="实盘"):
        await service.deploy_to_paper(candidate.candidate_id, account_id="live", execution_mode="live")

    approved = await service.review(candidate.candidate_id, approved=True, reviewed_by="risk-owner")
    assert approved.status == "approved"

    class FakeDeployments:
        async def create_from_experiment(self, experiment_id, **kwargs):
            assert experiment_id == f"sandbox-{candidate.candidate_id}"
            assert kwargs["mode"] == "confirm"
            return SimpleNamespace(deployment_id="dep-1")

    import application.deployments as deployments_module

    monkeypatch.setattr(deployments_module, "deployment_service", FakeDeployments())
    deployment = await service.deploy_to_paper(candidate.candidate_id, account_id="paper-1")

    assert deployment.deployment_id == "dep-1"
    assert (await service.get(candidate.candidate_id)).status == "deployed"
    experiment = await experiments.get(f"sandbox-{candidate.candidate_id}")
    assert experiment["source_sha256"] == candidate.source_sha256


@pytest.mark.asyncio
async def test_invalid_strategy_spec_is_preserved_as_research_only_candidate(monkeypatch, tmp_path):
    frame = _history()
    snapshot = {
        "sha256": "b" * 64,
        "actual_start_date": frame.iloc[0]["date"],
        "actual_end_date": frame.iloc[-1]["date"],
    }

    class FakeLLM:
        async def chat_json(self, *_args, **_kwargs):
            return {
                "source_code": """
def generate_target_positions(frame):
    return (frame["close"].rolling(5).mean() > frame["close"].rolling(20).mean()).fillna(False).astype(int)
""",
                "strategy_spec": {
                    "name": "invalid_aliases",
                    "asset_types": ["etf"],
                    "indicators": ["fast_ma", "slow_ma"],
                    "entry_conditions": [{"indicator": "fast_ma", "operator": "gt", "value": 0}],
                },
            }

    async def fake_prepared(**_kwargs):
        return frame, snapshot

    monkeypatch.setattr(candidate_module, "get_llm_service", lambda: FakeLLM())
    monkeypatch.setattr(candidate_module, "prepare_single_backtest_data", fake_prepared)
    service = StrategyCandidateService(tmp_path / "invalid-candidate.sqlite3")

    candidate = await service.generate(
        objective="测试非法别名",
        ticker="510300",
        asset_type="etf",
        start_date="2016-01-01",
        end_date="2026-01-01",
    )

    assert candidate.status == "draft"
    assert candidate.validation.passed is True
    assert candidate.validation.output_checks["strategy_spec_valid"] is False
    assert candidate.result["promotion_eligible"] is False
    assert candidate.result["backtest"]["final_value"] > 0
    assert any("StrategySpec" in error for error in candidate.validation.errors)
