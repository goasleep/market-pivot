"""Regression tests for Supervisor LLM limits and completion fallback."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from graph import agent_loop as agent_loop_module
from graph.agent_loop import AgentLoopContext, decide_next_action, judge_completion


class FakeLLMService:
    async def chat_with_tools(self, messages, tools, **kwargs):
        del messages, tools, kwargs
        return AIMessage(content="直接回答")

    async def chat_json(self, prompt, **kwargs):
        del prompt, kwargs
        return {
            "outcome": "satisfied",
            "satisfied": True,
            "terminal": True,
            "missing": [],
            "next_action": "",
            "reason": "回答完整",
        }


@pytest.mark.asyncio
async def test_supervisor_llm_calls_use_thirty_minute_limit(monkeypatch):
    fake_service = FakeLLMService()
    captured: list[int] = []
    original_wait_for = asyncio.wait_for

    async def capture_wait_for(awaitable, timeout):
        captured.append(timeout)
        return await original_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(agent_loop_module, "get_llm_service", lambda: fake_service)
    monkeypatch.setattr(agent_loop_module.asyncio, "wait_for", capture_wait_for)

    decision = await decide_next_action(
        {"messages": [], "step": 0, "max_steps": 4},
        SimpleNamespace(context=AgentLoopContext(tools=[], tool_map={})),
    )
    assert decision["candidate_response"] == "直接回答"

    completion = await judge_completion(
        {
            "candidate_response": "直接回答",
            "task_contract": {"requires_tools": False},
            "tool_events": [],
            "step": 1,
            "max_steps": 4,
        }
    )
    assert completion["completion_result"]["outcome"] == "satisfied"
    assert completion["final_response"] == "直接回答"
    assert captured == [1800, 1800]


@pytest.mark.asyncio
async def test_completion_judge_failure_is_terminal_partial(monkeypatch):
    class BrokenJudgeService(FakeLLMService):
        async def chat_json(self, prompt, **kwargs):
            del prompt, kwargs
            raise RuntimeError("judge unavailable")

    monkeypatch.setattr(agent_loop_module, "get_llm_service", lambda: BrokenJudgeService())

    completion = await judge_completion(
        {
            "candidate_response": "已形成可供用户参考的阶段性答案。",
            "task_contract": {"requires_tools": False},
            "tool_events": [],
            "step": 1,
            "max_steps": 4,
        }
    )

    assert completion["completion_result"]["outcome"] == "partial"
    assert completion["completion_result"]["satisfied"] is False
    assert completion["completion_result"]["terminal"] is True
    assert completion["final_response"] == "已形成可供用户参考的阶段性答案。"
