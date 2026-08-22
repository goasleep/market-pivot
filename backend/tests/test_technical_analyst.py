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
