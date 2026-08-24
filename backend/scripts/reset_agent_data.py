"""Destructively reset Agent-owned runtime data while preserving trading data."""

from __future__ import annotations

import argparse
import asyncio
import os

from artifacts.service import ArtifactService
from data.chat_models import (
    ChatConversation,
    ChatMessage,
    ChatMessageReference,
    ChatMessageSearch,
    ChatTask,
    ChatTaskEvent,
    ChatTaskInteraction,
    ChatTaskState,
)
from data.db_models import AgentDecisionRecord, AgentRunRecord, ArtifactRecord
from data.tortoise_db import close_database, init_database
from graph.checkpointing import checkpoint_manager

RESET_MODELS = (
    ChatMessageReference,
    ChatMessageSearch,
    ChatTaskEvent,
    ChatTaskInteraction,
    ChatTaskState,
    ChatTask,
    ChatMessage,
    ChatConversation,
    AgentDecisionRecord,
    AgentRunRecord,
)


async def reset(*, execute: bool) -> None:
    await init_database()
    artifact_rows = await ArtifactRecord.all()
    counts = {model._meta.db_table: await model.all().count() for model in RESET_MODELS}
    counts[ArtifactRecord._meta.db_table] = len(artifact_rows)
    print("Agent data reset scope:")
    for table, count in counts.items():
        print(f"  {table}: {count}")
    print("Preserved: app settings, simulation accounts/orders/snapshots, backtests, strategies, market cache")
    if not execute:
        print("Dry run only. Use --execute with both confirmation environment variables.")
        await close_database()
        return

    if os.getenv("CONFIRM_AGENT_DATA_RESET") != "yes":
        raise RuntimeError("Set CONFIRM_AGENT_DATA_RESET=yes to authorize the destructive reset")
    if os.getenv("AGENT_WORKERS_STOPPED") != "yes":
        raise RuntimeError("Stop API/Agent workers, then set AGENT_WORKERS_STOPPED=yes")

    artifact_service = ArtifactService()
    for row in artifact_rows:
        artifact_service.storage.delete(row.relative_path)
    await ArtifactRecord.all().delete()
    for model in RESET_MODELS:
        await model.all().delete()

    saver = await checkpoint_manager.start()
    thread_ids: set[str] = set()
    async for item in saver.alist(None):
        thread_id = str(item.config.get("configurable", {}).get("thread_id", ""))
        if thread_id:
            thread_ids.add(thread_id)
    for thread_id in thread_ids:
        await saver.adelete_thread(thread_id)
    await checkpoint_manager.stop()
    await close_database()
    print(f"Reset complete; deleted {len(thread_ids)} checkpoint threads.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="perform the reset instead of showing counts")
    args = parser.parse_args()
    asyncio.run(reset(execute=args.execute))


if __name__ == "__main__":
    main()
