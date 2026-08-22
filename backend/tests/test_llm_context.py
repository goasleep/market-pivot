import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from llm.context import (
    ContextBudget,
    ContextWindowExceededError,
    TokenCounter,
    select_conversation_history,
    select_messages_for_model,
)

MODEL = "gpt-4o-mini"


def _budget(input_limit: int) -> ContextBudget:
    return ContextBudget(
        model=MODEL,
        context_window=input_limit + 2048,
        output_reserve=1024,
        safety_margin=1024,
        input_limit=input_limit,
    )


def test_token_counter_counts_chinese_chat_content():
    counter = TokenCounter(MODEL)

    assert counter.count_messages([{"role": "user", "content": "分析沪深300短期趋势"}]) > 5


def test_conversation_history_uses_token_budget_and_keeps_complete_recent_turn():
    counter = TokenCounter(MODEL)
    p0 = [
        {"role": "system", "content": "系统安全规则"},
        {"role": "user", "content": "当前问题"},
    ]
    history = [
        {"role": "user", "content": "旧问题" * 300},
        {"role": "assistant", "content": "旧回答" * 300},
        {"role": "user", "content": "最近问题"},
        {"role": "assistant", "content": "最近回答"},
    ]
    input_limit = counter.count_messages([*p0, *history[-2:]]) + 4

    selection = select_conversation_history(history, p0_messages=p0, budget=_budget(input_limit))

    assert selection.messages == history[-2:]
    assert selection.dropped_messages == 2
    assert selection.selected_tokens <= selection.input_limit


def test_pending_interaction_is_p0_even_when_older_turns_are_dropped():
    counter = TokenCounter(MODEL)
    p0 = [
        {"role": "system", "content": "系统安全规则"},
        {"role": "user", "content": "当前问题"},
    ]
    pending = {
        "role": "assistant",
        "content": "需要用户确认",
        "parts": [
            {
                "type": "interaction",
                "content": {"status": "pending", "question": "是否继续？"},
            }
        ],
    }
    history = [
        {"role": "user", "content": "旧问题" * 500},
        {"role": "assistant", "content": "旧回答" * 500},
        pending,
    ]
    input_limit = counter.count_messages([*p0, pending]) + 4

    selection = select_conversation_history(history, p0_messages=p0, budget=_budget(input_limit))

    assert selection.messages == [pending]


def test_agent_context_keeps_system_current_user_and_latest_tool_exchange():
    counter = TokenCounter(MODEL)
    latest_call = AIMessage(
        content="",
        tool_calls=[{"id": "latest", "name": "get_quote", "args": {"ticker": "510300"}}],
    )
    latest_result = ToolMessage(content='{"price": 4.68}', tool_call_id="latest")
    messages = [
        SystemMessage(content="产品边界与安全规则"),
        HumanMessage(content="很早的问题"),
        AIMessage(content="很早的回答"),
        HumanMessage(content="当前问题"),
        AIMessage(
            content="",
            tool_calls=[{"id": "old", "name": "get_history", "args": {"ticker": "510300"}}],
        ),
        ToolMessage(content="大量历史数据" * 1000, tool_call_id="old"),
        latest_call,
        latest_result,
    ]
    p0 = [messages[0], messages[3], latest_call, latest_result]
    input_limit = counter.count_messages(p0) + 4

    selection = select_messages_for_model(messages, budget=_budget(input_limit))

    assert selection.messages == p0
    assert selection.selected_tokens <= selection.input_limit


def test_non_compressible_context_never_gets_silently_truncated():
    messages = [
        SystemMessage(content="安全规则" * 100),
        HumanMessage(content="当前用户消息" * 100),
    ]

    with pytest.raises(ContextWindowExceededError, match="不可压缩上下文"):
        select_messages_for_model(messages, budget=_budget(20))
