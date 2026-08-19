from datetime import datetime, timezone

from data.sqlite_checkpoint import SQLiteCheckpointSaver
from data.sqlite_coordination import SQLiteCoordination


def test_sqlite_lease_allows_one_owner_and_failover(tmp_path):
    db_path = tmp_path / "coordination.db"
    first = SQLiteCoordination(db_path, owner_id="node-a")
    second = SQLiteCoordination(db_path, owner_id="node-b")

    assert first.acquire_lease("scheduler", ttl_seconds=60)
    assert not second.acquire_lease("scheduler", ttl_seconds=60)
    first.release_lease("scheduler")
    assert second.acquire_lease("scheduler", ttl_seconds=60)


def test_sqlite_job_claim_and_event_cursor_support_failover(tmp_path):
    db_path = tmp_path / "coordination.db"
    first = SQLiteCoordination(db_path, owner_id="node-a")
    second = SQLiteCoordination(db_path, owner_id="node-b")
    first.create_job("job-1", "backtest", {"ticker": "000001"})

    assert first.claim_job("job-1") is not None
    assert second.claim_job("job-1") is None
    first.update_job("job-1", lease_expires_at=0)
    assert second.claim_job("job-1") is not None

    first_event = first.append_event("backtest", "job-1", "progress", {"stage": "data"})
    second_event = second.append_event("backtest", "job-1", "complete", {"ok": True})
    events = first.list_events("backtest", "job-1", after_id=first_event["event_id"])
    assert [event["event_id"] for event in events] == [second_event["event_id"]]


def test_sqlite_checkpoint_round_trip(tmp_path):
    saver = SQLiteCheckpointSaver(tmp_path / "checkpoints.db")
    config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "id": "00000000000000000000000000000001",
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {"messages": ["hello"]},
        "channel_versions": {"messages": 1},
        "versions_seen": {},
    }

    saved = saver.put(config, checkpoint, {"source": "input", "step": 0}, {"messages": 1})
    saver.put_writes(saved | {"configurable": {**saved["configurable"]}}, [("messages", "pending")], "task-1")

    restored = saver.get_tuple(saved)
    assert restored is not None
    assert restored.checkpoint["channel_values"] == {"messages": ["hello"]}
    assert restored.metadata["source"] == "input"
    assert restored.pending_writes == [("task-1", "messages", "pending")]
