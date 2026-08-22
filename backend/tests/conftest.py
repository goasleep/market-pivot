import aiosqlite
import pytest_asyncio

from data.tortoise_db import close_database


@pytest_asyncio.fixture(scope="session", autouse=True)
async def close_all_test_sqlite_connections():
    """Own every aiosqlite connection created by the test process."""
    original_connect = aiosqlite.connect
    connections: list[aiosqlite.Connection] = []

    def tracked_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    aiosqlite.connect = tracked_connect
    try:
        yield
    finally:
        for connection in reversed(connections):
            try:
                await connection.close()
            except RuntimeError:
                connection.stop()
        aiosqlite.connect = original_connect


@pytest_asyncio.fixture(autouse=True)
async def close_tortoise_after_test():
    """Release aiosqlite worker threads between isolated async test loops."""
    yield
    await close_database()
