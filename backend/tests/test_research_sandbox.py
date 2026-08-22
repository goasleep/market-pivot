import pandas as pd
import pytest

from application.research_sandbox import (
    SandboxError,
    replay_target_positions,
    validate_and_run_signals,
    validate_source,
)


def _frame(rows=12):
    close = [10 + index * 0.1 + (-0.25 if index % 4 == 0 else 0) for index in range(rows)]
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="D").strftime("%Y-%m-%d"),
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": [1_000_000] * rows,
        }
    )


@pytest.mark.asyncio
async def test_sandbox_accepts_deterministic_causal_binary_signal_and_core_replays_it():
    source = """
def generate_target_positions(frame):
    fast = frame["close"].rolling(2).mean()
    slow = frame["close"].rolling(4).mean()
    return (fast > slow).fillna(False).astype(int)
"""
    frame = _frame()

    positions, validation = await validate_and_run_signals(source, frame)
    result = replay_target_positions(ticker="510300", asset_type="etf", frame=frame, positions=positions)

    assert validation.passed is True
    assert validation.deterministic is True
    assert validation.causal is True
    assert len(positions) == len(frame)
    assert result["execution"]["fill_time"] == "next_open"
    assert result["total_trades"] >= 1


@pytest.mark.asyncio
async def test_sandbox_rejects_future_dependent_signal_by_prefix_invariance():
    source = """
def generate_target_positions(frame):
    target = int(frame["close"].iloc[-1] > 10.7)
    return [target] * len(frame)
"""

    _, validation = await validate_and_run_signals(source, _frame())

    assert validation.passed is False
    assert validation.causal is False
    assert "未来数据" in validation.errors[0]


@pytest.mark.parametrize(
    "source",
    [
        'import os\ndef generate_target_positions(frame):\n    return [0] * len(frame)',
        'def generate_target_positions(frame):\n    open("/tmp/leak", "w")\n    return [0] * len(frame)',
        'def generate_target_positions(frame):\n    frame.to_csv("/tmp/leak")\n    return [0] * len(frame)',
    ],
)
def test_sandbox_static_gate_rejects_process_and_file_access(source):
    with pytest.raises(SandboxError):
        validate_source(source)
