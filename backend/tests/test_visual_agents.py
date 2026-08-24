from types import SimpleNamespace

import pandas as pd
import pytest

from agents import fundamentals_analyst, risk_manager
from models.schemas import AgentReport, AssetType, FundSnapshot, MarketContext


def _history(periods: int = 25) -> list[dict]:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-07-01", periods=periods).strftime("%Y-%m-%d"),
            "open": [4.0 + index * 0.01 for index in range(periods)],
            "high": [4.1 + index * 0.01 for index in range(periods)],
            "low": [3.9 + index * 0.01 for index in range(periods)],
            "close": [4.05 + index * 0.01 for index in range(periods)],
            "volume": [100_000 + index for index in range(periods)],
        }
    ).to_dict(orient="records")


class FakeVisionLLM:
    def __init__(self):
        self.calls = []

    def supports_vision(self):
        return True

    async def chat_json_with_images(self, prompt, urls, **_kwargs):
        self.calls.append((prompt, urls))
        if "fund" in prompt.lower():
            return {
                "signal": "hold",
                "confidence": 0.65,
                "reasoning": "基金折溢价结构稳定",
                "key_metrics": {"premium_discount": "stable"},
            }
        return {
            "signal": "hold",
            "confidence": 0.6,
            "reasoning": "回撤和波动可控",
            "risk_level": "medium",
            "max_position_pct": 0.3,
            "stop_loss_pct": 0.06,
            "risk_factors": ["波动"],
        }

    async def chat_json(self, *_args, **_kwargs):
        raise AssertionError("visual evidence should be sent to the model")


@pytest.mark.asyncio
async def test_fund_structure_stage_returns_visual_artifact(monkeypatch):
    history = _history()
    context = MarketContext(
        ticker="510300",
        asset_type=AssetType.ETF,
        realtime={"price": 4.29},
        history=history,
        fund_data=FundSnapshot(
            derived_metrics={"latest_premium_pct": 0.2},
            nav_history=[{"date": row["date"], "unit_nav": row["close"] * 0.998} for row in history],
        ),
    )

    class FakeVisuals:
        async def prepare_fund_structure(self, **_kwargs):
            return SimpleNamespace(
                artifact={"artifact_id": "fund-chart", "mime_type": "image/png"},
                model_url="https://objects.example.test/fund.png",
            )

    llm = FakeVisionLLM()
    monkeypatch.setattr(fundamentals_analyst, "visual_evidence_service", FakeVisuals())
    stage = await fundamentals_analyst.analyze_stage("510300", context=context, llm=llm)

    assert stage.report.agent_name == "fund_structure"
    assert stage.artifacts[0]["artifact_id"] == "fund-chart"
    assert llm.calls[0][1] == ["https://objects.example.test/fund.png"]
    assert '"nav_history"' not in llm.calls[0][0]


@pytest.mark.asyncio
async def test_risk_stage_includes_metrics_and_visual_artifact(monkeypatch):
    context = MarketContext(ticker="510300", asset_type=AssetType.ETF, history=_history())

    class FakeVisuals:
        async def prepare_risk(self, **_kwargs):
            return SimpleNamespace(
                artifact={"artifact_id": "risk-chart", "mime_type": "image/png"},
                model_url="https://objects.example.test/risk.png",
            )

    llm = FakeVisionLLM()
    monkeypatch.setattr(risk_manager, "visual_evidence_service", FakeVisuals())
    stage = await risk_manager.assess_stage(
        "510300",
        {"technical": AgentReport(agent_name="technical", reasoning="震荡")},
        context=context,
        asset_type=AssetType.ETF,
        llm=llm,
    )

    assert stage.artifacts[0]["artifact_id"] == "risk-chart"
    assert stage.report.key_data["market_risk_metrics"]["return_20d_pct"] is not None
    assert llm.calls[0][1] == ["https://objects.example.test/risk.png"]
