"""Tortoise-backed application settings."""

from __future__ import annotations

import json
import time
from typing import Any

from config import get_llm_config, save_llm_config
from data.db_models import AppSetting
from data.tortoise_db import init_database

LLM_SETTINGS_KEY = "llm_config"


async def load_llm_config() -> dict[str, Any]:
    await init_database()
    row = await AppSetting.get_or_none(key=LLM_SETTINGS_KEY)
    if row is None:
        return get_llm_config()
    try:
        persisted = json.loads(row.value)
    except json.JSONDecodeError:
        persisted = {}
    if not isinstance(persisted, dict):
        return get_llm_config()
    return save_llm_config(persisted)


async def update_llm_config(updates: dict[str, Any]) -> dict[str, Any]:
    await init_database()
    config = save_llm_config(updates)
    payload = json.dumps(config, ensure_ascii=False)
    await AppSetting.update_or_create(
        defaults={"value": payload, "updated_at": time.time()},
        key=LLM_SETTINGS_KEY,
    )
    return config
