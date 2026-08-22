"""Application-wide Tortoise ORM lifecycle."""

from __future__ import annotations

from pathlib import Path

from tortoise import Tortoise
from tortoise.context import TortoiseContext

from config import settings

_active_db_url: str | None = None
_contexts: set[TortoiseContext] = set()


def database_url(db_path: str | Path | None = None, db_url: str | None = None) -> str:
    if db_url:
        return db_url
    if db_path is not None:
        return f"sqlite://{Path(db_path).expanduser().resolve()}"
    return settings.chat_database_url


async def init_database(
    db_path: str | Path | None = None,
    db_url: str | None = None,
    *,
    generate_schemas: bool = True,
) -> None:
    """Initialize the single ORM connection used by the application."""
    global _active_db_url
    url = database_url(db_path, db_url)
    if Tortoise.is_inited():
        if _active_db_url == url:
            return
        # Isolated temporary databases are useful during the demo/test phase.
        # The running application initializes one URL for its whole lifespan.
        await close_database()
    context = await Tortoise.init(
        db_url=url,
        modules={"models": ["data.db_models", "data.chat_models"]},
        _enable_global_fallback=True,
    )
    _contexts.add(context)
    if generate_schemas:
        await Tortoise.generate_schemas(safe=True)
    _active_db_url = url


async def close_database() -> None:
    global _active_db_url
    contexts = list(_contexts)
    _contexts.clear()
    for context in contexts:
        await context.close_connections()
    if Tortoise.is_inited() and not contexts:
        await Tortoise.close_connections()
    _active_db_url = None
