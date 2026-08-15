"""Shared tracing configuration for LangChain and LangGraph executions."""

from __future__ import annotations

from typing import Any

from loguru import logger

from config import settings


def build_trace_config(
    run_name: str,
    *,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Build a LangGraph config with an optional per-run Langfuse handler.

    A fresh callback handler is created for every root execution so concurrent
    requests do not share trace state. Langfuse's documented metadata keys are
    used for dynamic session/user/tag attributes.
    """
    trace_tags = list(tags or [])
    trace_metadata = dict(metadata or {})
    if session_id:
        trace_metadata["langfuse_session_id"] = session_id
    if user_id:
        trace_metadata["langfuse_user_id"] = user_id
    if trace_tags:
        trace_metadata["langfuse_tags"] = trace_tags

    config: dict[str, Any] = {
        "run_name": run_name,
        "tags": trace_tags,
        "metadata": trace_metadata,
    }
    handler = _new_langfuse_handler()
    if handler is not None:
        config["callbacks"] = [handler]
    return config


def _new_langfuse_handler() -> Any | None:
    """Create a Langfuse callback only when credentials are configured."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as exc:  # pragma: no cover - protects app startup/runtime
        logger.warning("Langfuse callback unavailable; continuing without tracing: {}", exc)
        return None
