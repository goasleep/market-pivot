from types import SimpleNamespace

import pandas as pd
import pytest

from agents import technical_analyst
from models.schemas import AssetType, Decision


@pytest.mark.asyncio
async def test_technical_analysis_derives_missing_pct_change():
    history = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=20).strftime("%Y-%m-%d"),
            "open": [4.0 + index * 0.01 for index in range(20)],
            "high": [4.1 + index * 0.01 for index in range(20)],
            "low": [3.9 + index * 0.01 for index in range(20)],
            "close": [4.05 + index * 0.01 for index in range(20)],
            "volume": [1000 + index for index in range(20)],
        }
    )
    context = SimpleNamespace(
        history=history.to_dict(orient="records"),
        asset_type=AssetType.ETF,
        market_regime="range",
    )

    class FakeLLM:
        async def chat_json(self, prompt, **_kwargs):
            assert "pct_chg" in prompt
            return {
                "signal": "hold",
                "confidence": 0.6,
                "reasoning": "测试分析",
                "key_indicators": {},
            }

    result = await technical_analyst.analyze("510300", context=context, llm=FakeLLM())

    assert result.signal == Decision.HOLD
    assert result.reasoning == "测试分析"


@pytest.mark.asyncio
async def test_technical_stage_uses_chart_url_for_allowlisted_model(monkeypatch):
    history = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=25).strftime("%Y-%m-%d"),
            "open": [4.0 + index * 0.01 for index in range(25)],
            "high": [4.1 + index * 0.01 for index in range(25)],
            "low": [3.9 + index * 0.01 for index in range(25)],
            "close": [4.05 + index * 0.01 for index in range(25)],
            "volume": [1000 + index for index in range(25)],
        }
    )
    context = SimpleNamespace(
        history=history.to_dict(orient="records"),
        asset_type=AssetType.ETF,
        market_regime="range",
    )

    class FakeVisuals:
        async def prepare_technical(self, **kwargs):
            assert kwargs["conversation_id"] == "conversation-test"
            return SimpleNamespace(
                artifact={"artifact_id": "artifact-chart", "mime_type": "image/png"},
                model_url="https://objects.example.test/chart.png",
            )

    class FakeLLM:
        def supports_vision(self):
            return True

        async def chat_json_with_images(self, prompt, urls, **_kwargs):
            assert "Exact latest indicators" in prompt
            assert urls == ["https://objects.example.test/chart.png"]
            return {"signal": "hold", "confidence": 0.7, "reasoning": "视觉分析", "key_indicators": {}}

        async def chat_json(self, *_args, **_kwargs):
            raise AssertionError("vision path should not use text-only chat")

    monkeypatch.setattr(technical_analyst, "visual_evidence_service", FakeVisuals())
    stage = await technical_analyst.analyze_stage(
        "510300",
        context=context,
        llm=FakeLLM(),
        conversation_id="conversation-test",
        task_id="task-test",
    )

    assert stage.report.reasoning == "视觉分析"
    assert stage.artifacts == [{"artifact_id": "artifact-chart", "mime_type": "image/png"}]
