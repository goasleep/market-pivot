"""Request-scoped LLM profile selection."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator

_llm_profile_context: ContextVar[dict[str, Any] | None] = ContextVar("llm_profile_context", default=None)


@contextmanager
def use_llm_profile(profile: dict[str, Any]) -> Iterator[None]:
    """Pin one immutable profile snapshot for all nested LLM calls."""
    token: Token[dict[str, Any] | None] = _llm_profile_context.set(dict(profile))
    try:
        yield
    finally:
        _llm_profile_context.reset(token)


def current_llm_profile() -> dict[str, Any] | None:
    profile = _llm_profile_context.get()
    return dict(profile) if profile else None
