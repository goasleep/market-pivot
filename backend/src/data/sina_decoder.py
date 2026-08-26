"""Crash-isolated JavaScript decoder for Sina ETF history payloads."""

from __future__ import annotations

import atexit
import multiprocessing
import os
import signal
import threading
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from loguru import logger

SINA_DECODER_TIMEOUT_SECONDS = 30.0
SINA_DECODER_STARTUP_TIMEOUT_SECONDS = 30.0

# V8 owns process-global native state. The worker currently handles one request
# at a time, and this lock keeps that invariant explicit if its loop evolves.
_SINA_V8_LOCK = threading.Lock()


class SinaDecoderError(RuntimeError):
    """Base error raised by the isolated Sina decoder."""


class SinaDecoderExecutionError(SinaDecoderError):
    """The worker stayed healthy but rejected one payload."""


class SinaDecoderUnavailableError(SinaDecoderError):
    """The worker crashed, timed out, or broke its IPC channel."""


def _mini_racer_worker(connection: Connection) -> None:
    """Own one V8 context and service decode requests inside a child process."""
    # Uvicorn's interactive SIGINT reaches the whole terminal process group.
    # The parent owns this worker's lifecycle and will send a shutdown command,
    # so suppress a child traceback during otherwise graceful server shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        # Keep both native imports in the spawned process. A V8 assertion can
        # then terminate only this worker rather than the FastAPI process.
        import py_mini_racer
        from akshare.stock.cons import hk_js_decode

        with _SINA_V8_LOCK:
            decoder = py_mini_racer.MiniRacer()
            decoder.eval(hk_js_decode)
        connection.send(("ready", os.getpid()))
    except BaseException as exc:
        try:
            connection.send(("startup_error", f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()
        return

    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                return
            if message == ("shutdown",):
                return

            command, request_id, encoded = message
            if command != "decode":
                connection.send(("error", request_id, f"unsupported command: {command}"))
                continue
            try:
                with _SINA_V8_LOCK:
                    rows = decoder.call("d", encoded)
                connection.send(("result", request_id, rows))
            except Exception as exc:
                connection.send(("error", request_id, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


WorkerTarget = Callable[[Connection], None]


class SinaDecoderSupervisor:
    """Serialize calls to one spawned decoder and replace unhealthy workers."""

    def __init__(
        self,
        *,
        timeout_seconds: float = SINA_DECODER_TIMEOUT_SECONDS,
        startup_timeout_seconds: float = SINA_DECODER_STARTUP_TIMEOUT_SECONDS,
        worker_target: WorkerTarget = _mini_racer_worker,
    ):
        self.timeout_seconds = timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self._worker_target = worker_target
        self._context = multiprocessing.get_context("spawn")
        self._request_lock = threading.Lock()
        self._process: Any | None = None
        self._connection: Connection | None = None
        self._request_id = 0

    def decode(self, encoded: str) -> list[dict[str, Any]]:
        """Decode one payload, rebuilding the worker once after infrastructure failure."""
        failures: list[str] = []
        with self._request_lock:
            for attempt in range(2):
                try:
                    return self._decode_once(encoded)
                except SinaDecoderExecutionError:
                    raise
                except SinaDecoderUnavailableError as exc:
                    failures.append(str(exc))
                    self._discard_worker()
                    if attempt == 0:
                        logger.warning(f"Sina decoder worker unavailable; rebuilding once: {exc}")

        detail = failures[-1] if failures else "unknown decoder worker failure"
        raise SinaDecoderUnavailableError(f"Sina decoder worker unavailable after restart: {detail}")

    def close(self) -> None:
        """Stop the child process without leaving a native worker behind."""
        with self._request_lock:
            self._discard_worker(graceful=True)

    def __enter__(self) -> SinaDecoderSupervisor:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _decode_once(self, encoded: str) -> list[dict[str, Any]]:
        self._ensure_worker()
        process = self._process
        connection = self._connection
        if process is None or connection is None:
            raise SinaDecoderUnavailableError("Sina decoder worker did not start")

        self._request_id += 1
        request_id = self._request_id
        try:
            connection.send(("decode", request_id, encoded))
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise SinaDecoderUnavailableError(f"Sina decoder worker pipe failed: {exc}") from exc

        if not connection.poll(self.timeout_seconds):
            if process.is_alive():
                raise SinaDecoderUnavailableError(
                    f"Sina decoder worker timed out after {self.timeout_seconds:g} seconds"
                )
            raise SinaDecoderUnavailableError(
                f"Sina decoder worker exited with code {process.exitcode}"
            )

        try:
            message = connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise SinaDecoderUnavailableError(
                f"Sina decoder worker exited with code {process.exitcode}: {exc}"
            ) from exc

        if not isinstance(message, tuple) or len(message) != 3:
            raise SinaDecoderUnavailableError("Sina decoder worker returned a malformed response")
        status, response_id, payload = message
        if response_id != request_id:
            raise SinaDecoderUnavailableError("Sina decoder worker returned a mismatched response")
        if status == "error":
            raise SinaDecoderExecutionError(f"Sina decoder rejected payload: {payload}")
        if status != "result" or not isinstance(payload, list):
            raise SinaDecoderUnavailableError("Sina decoder worker returned an invalid result")
        return payload

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive() and self._connection is not None:
            return
        self._discard_worker()

        parent_connection, child_connection = self._context.Pipe()
        process = self._context.Process(
            target=self._worker_target,
            args=(child_connection,),
            name="sina-etf-decoder",
            daemon=True,
        )
        try:
            process.start()
        except Exception as exc:
            parent_connection.close()
            child_connection.close()
            process.close()
            raise SinaDecoderUnavailableError(f"Sina decoder worker could not start: {exc}") from exc
        child_connection.close()
        self._process = process
        self._connection = parent_connection

        if not parent_connection.poll(self.startup_timeout_seconds):
            if process.is_alive():
                detail = f"timed out after {self.startup_timeout_seconds:g} seconds"
            else:
                detail = f"exited with code {process.exitcode}"
            raise SinaDecoderUnavailableError(f"Sina decoder worker startup {detail}")
        try:
            message = parent_connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise SinaDecoderUnavailableError(
                f"Sina decoder worker startup failed with exit code {process.exitcode}: {exc}"
            ) from exc
        if not isinstance(message, tuple) or len(message) != 2:
            raise SinaDecoderUnavailableError("Sina decoder worker returned a malformed startup response")
        status, detail = message
        if status != "ready":
            raise SinaDecoderUnavailableError(f"Sina decoder worker startup failed: {detail}")

    def _discard_worker(self, *, graceful: bool = False) -> None:
        process, connection = self._process, self._connection
        self._process = None
        self._connection = None
        if process is None:
            if connection is not None:
                connection.close()
            return

        if graceful and process.is_alive() and connection is not None:
            try:
                connection.send(("shutdown",))
            except (BrokenPipeError, EOFError, OSError):
                pass
            process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
        if connection is not None:
            connection.close()
        process.close()


sina_decoder = SinaDecoderSupervisor()
atexit.register(sina_decoder.close)


def decode_sina_payload(encoded: str) -> list[dict[str, Any]]:
    """Decode through the process-isolated singleton used by data providers."""
    return sina_decoder.decode(encoded)
