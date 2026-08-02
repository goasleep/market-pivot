"""Provider-independent application interface for LLM calls."""

from llm.deepseek import MODEL_CONFIGS, get_chat_model
from llm.service import LLMService, chat, chat_json, chat_langchain, get_llm_service

__all__ = [
    "LLMService",
    "MODEL_CONFIGS",
    "chat",
    "chat_json",
    "chat_langchain",
    "get_chat_model",
    "get_llm_service",
]
