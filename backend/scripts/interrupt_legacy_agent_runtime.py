"""Interrupt only active legacy Agent runtime while preserving completed research and trading data."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from data.chat_models import ChatMessage, ChatTask, ChatTaskEvent, ChatTaskInteraction, ChatTaskState  # noqa: E402
from data.tortoise_db import close_database, init_database  # noqa: E402
from graph.checkpointing import checkpoint_manager  # noqa: E402

ACTIVE_STATUSES = ("pending", "running", "waiting_user", "cancel_requested")


async def interrupt_runtime(*, confirm: bool) -> dict[str, int]:
    await init_database()
    tasks = await ChatTask.filter(status__in=ACTIVE_STATUSES).all()
    task_ids = {row.task_id for row in tasks}
    counts = {
        "tasks": len(task_ids),
        "interactions": await ChatTaskInteraction.filter(task_id__in=task_ids, status="pending").count()
        if task_ids
        else 0,
        "states": await ChatTaskState.filter(task_id__in=task_ids).count() if task_ids else 0,
        "events": await ChatTaskEvent.filter(task_id__in=task_ids).count() if task_ids else 0,
        "checkpoints": 0,
    }
    saver = await checkpoint_manager.start()
    checkpoint_threads: set[str] = set()
    async for item in saver.alist(None):
        thread_id = str(item.config.get("configurable", {}).get("thread_id", ""))
        if any(thread_id == task_id or thread_id.startswith(f"{task_id}:") for task_id in task_ids):
            checkpoint_threads.add(thread_id)
    counts["checkpoints"] = len(checkpoint_threads)
    print("Legacy Agent runtime cleanup scope:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    print("Preserved: completed conversations/messages/references/artifacts/backtests/strategies/simulation accounts")
    if not confirm:
        print("Dry run only. Re-run with --confirm to apply this idempotent cleanup.")
        await checkpoint_manager.stop()
        await close_database()
        return counts

    timestamp = datetime.now(timezone.utc).isoformat()
    if task_ids:
        await ChatTask.filter(task_id__in=task_ids).update(
            status="interrupted",
            error="Interrupted during Financial Harness runtime migration",
            updated_at=timestamp,
        )
        await ChatMessage.filter(task_id__in=task_ids).update(status="interrupted", updated_at=timestamp)
        await ChatTaskInteraction.filter(task_id__in=task_ids, status="pending").delete()
        await ChatTaskState.filter(task_id__in=task_ids).delete()
        await ChatTaskEvent.filter(task_id__in=task_ids).delete()
    for thread_id in checkpoint_threads:
        await saver.adelete_thread(thread_id)
    await checkpoint_manager.stop()
    await close_database()
    print("Cleanup complete.")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="apply cleanup; default is dry-run")
    args = parser.parse_args()
    asyncio.run(interrupt_runtime(confirm=args.confirm))


if __name__ == "__main__":
    main()
