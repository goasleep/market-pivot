import pytest

from agents.debate_room import debate
from models.schemas import AgentReport, Decision


class RejectedLLM:
    async def chat(self, prompt: str, system: str | None = None) -> str:
        raise RuntimeError("sensitive_words_detected")

    async def chat_json(self, prompt: str, system: str | None = None) -> dict:
        raise AssertionError("judge should not run after a debate generation failure")


@pytest.mark.asyncio
async def test_debate_degrades_when_argument_generation_fails():
    report = await debate(
        "510300",
        {"technical": AgentReport(agent_name="technical", reasoning="trend data")},
        llm=RejectedLLM(),
        asset_type="etf",
    )

    assert report.signal == Decision.HOLD
    assert report.reasoning == "多空辩论模型暂时不可用，已降级为中性结论。"
    assert report.key_data == {"degraded": True, "reason": "llm_unavailable"}
