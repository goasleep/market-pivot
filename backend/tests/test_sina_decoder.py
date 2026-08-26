import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from data.sina_decoder import (
    SinaDecoderExecutionError,
    SinaDecoderSupervisor,
    SinaDecoderUnavailableError,
)


def _test_decoder_worker(connection):
    """Spawn-safe worker used to exercise the parent supervision protocol."""
    connection.send(("ready", os.getpid()))
    try:
        while True:
            message = connection.recv()
            if message == ("shutdown",):
                return
            command, request_id, encoded = message
            assert command == "decode"
            if encoded == "crash":
                os._exit(23)
            if encoded == "error":
                connection.send(("error", request_id, "decoder rejected payload"))
                continue
            if encoded.startswith("sleep:"):
                time.sleep(float(encoded.split(":", 1)[1]))
            connection.send(
                (
                    "result",
                    request_id,
                    [{"encoded": encoded, "pid": os.getpid()}],
                )
            )
    except EOFError:
        return
    finally:
        connection.close()


def _supervisor(*, timeout_seconds: float = 1.0) -> SinaDecoderSupervisor:
    return SinaDecoderSupervisor(
        timeout_seconds=timeout_seconds,
        startup_timeout_seconds=3.0,
        worker_target=_test_decoder_worker,
    )


def test_supervisor_reuses_one_spawned_worker_for_concurrent_callers():
    supervisor = _supervisor()
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            rows = list(executor.map(supervisor.decode, ("one", "two", "three", "four")))
    finally:
        supervisor.close()

    assert [item[0]["encoded"] for item in rows] == ["one", "two", "three", "four"]
    assert len({item[0]["pid"] for item in rows}) == 1


def test_supervisor_contains_native_style_exit_and_rebuilds_worker():
    supervisor = _supervisor()
    try:
        original_pid = supervisor.decode("healthy")[0]["pid"]

        with pytest.raises(SinaDecoderUnavailableError, match="decoder worker"):
            supervisor.decode("crash")

        replacement = supervisor.decode("recovered")
    finally:
        supervisor.close()

    assert replacement[0]["encoded"] == "recovered"
    assert replacement[0]["pid"] != original_pid


def test_supervisor_replaces_worker_after_timeout():
    supervisor = _supervisor(timeout_seconds=0.05)
    try:
        original_pid = supervisor.decode("healthy")[0]["pid"]

        with pytest.raises(SinaDecoderUnavailableError, match="timed out"):
            supervisor.decode("sleep:0.2")

        replacement_pid = supervisor.decode("recovered")[0]["pid"]
    finally:
        supervisor.close()

    assert replacement_pid != original_pid


def test_supervisor_returns_decoder_errors_without_restarting_worker():
    supervisor = _supervisor()
    try:
        original_pid = supervisor.decode("healthy")[0]["pid"]

        with pytest.raises(SinaDecoderExecutionError, match="rejected payload"):
            supervisor.decode("error")

        current_pid = supervisor.decode("still-healthy")[0]["pid"]
    finally:
        supervisor.close()

    assert current_pid == original_pid
