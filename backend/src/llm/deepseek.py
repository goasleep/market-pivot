"""DeepSeek LLM adapter using OpenAI-compatible API.

Supports:
- deepseek-chat: general purpose chat model
- deepseek-reasoner: reasoning model (R1)

Configuration is hot-reloadable: reads from config.get_llm_config() on each
call, so settings changed via the UI API take effect immediately without restart.
"""

from openai import AsyncOpenAI
from loguru import logger
from config import get_llm_config


# Default model presets (temperature & max_tokens defaults per model)
MODEL_CONFIGS = {
    "deepseek-chat": {
        "max_tokens": 8192,
        "temperature": 0.3,
        "description": "General purpose chat model (V3)",
    },
    "deepseek-reasoner": {
        "max_tokens": 16384,
        "temperature": 0.0,
        "description": "Reasoning model (R1) for complex analysis",
    },
}


def get_client() -> AsyncOpenAI:
    """Create a fresh AsyncOpenAI client using current config.

    No singleton caching — config may change at runtime via the UI.
    """
    cfg = get_llm_config()
    return AsyncOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )


async def chat(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Simple chat completion.

    Args:
        prompt: User message
        system: System prompt
        model: Model name (default from config)
        temperature: Sampling temperature
        max_tokens: Max output tokens

    Returns:
        Response text
    """
    cfg = get_llm_config()
    model = model or cfg["model"]

    # Use model preset defaults, then override with config values, then explicit args
    preset = MODEL_CONFIGS.get(model, {})
    temperature = temperature if temperature is not None else cfg.get("temperature", preset.get("temperature", 0.3))
    max_tokens = max_tokens or cfg.get("max_tokens", preset.get("max_tokens", 8192))

    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.debug(f"LLM response ({model}): {len(content)} chars")
        return content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


async def chat_json(
    prompt: str,
    system: str = "",
    model: str | None = None,
) -> dict:
    """Chat completion expecting JSON response.

    The prompt should instruct the model to return valid JSON.
    Uses json enforcement via system message suffix.
    """
    if system:
        system = system + "\n\nYou must respond with valid JSON only, no markdown, no explanation."
    else:
        system = "You must respond with valid JSON only, no markdown, no explanation."

    raw = await chat(prompt, system=system, model=model, temperature=0.0)

    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first and last line (fences)
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines)

    import json

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}\nRaw: {raw[:500]}")
        return {"error": "json_parse_failed", "raw": raw[:500]}


async def chat_langchain(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
) -> str:
    """LangChain-compatible chat using langchain-openai ChatOpenAI.

    Args:
        messages: List of {"role": "...", "content": "..."} dicts
        model: Model name
        temperature: Sampling temperature

    Returns:
        Response text
    """
    from langchain_openai import ChatOpenAI

    cfg = get_llm_config()
    model = model or cfg["model"]
    llm = ChatOpenAI(
        model=model,
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        temperature=temperature,
    )

    lc_messages = []
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    response = await llm.ainvoke(lc_messages)
    return response.content
