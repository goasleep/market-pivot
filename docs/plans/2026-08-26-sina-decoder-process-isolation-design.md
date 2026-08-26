# Sina ETF Decoder Process Isolation Design

## Context

The Sina ETF history fallback decodes an encoded JavaScript payload with
`py_mini_racer`. Concurrent construction of embedded V8 runtimes has caused a
native `address_pool_manager.cc` fatal check on macOS. Because that failure
aborts the native process, Python exception handling cannot protect the FastAPI
service.

## Decision

Run all Sina payload decoding in one lazily started child process created with
the multiprocessing `spawn` context. The worker owns one reusable `MiniRacer`
context and processes requests serially over a pipe. A process-local lock still
guards V8 initialization and calls, preserving the existing first layer of
concurrency protection.

The parent owns a thread-safe supervisor. It starts the worker on demand,
validates request/response identifiers, applies a bounded timeout, and tears
down unhealthy workers. If the worker exits, breaks its pipe, or times out, the
supervisor rebuilds it once and retries the request. A second infrastructure
failure is returned as a normal Python exception, allowing the existing data
source retry and graceful-unavailability behavior to continue without taking
down the API process.

## Alternatives Considered

- Keep only the process-wide thread lock: lowest overhead, but another V8
  assertion or segmentation fault can still terminate FastAPI.
- Start one child process per payload: strong isolation, but repeatedly pays
  process and V8 startup costs during multi-fund research.
- Use a single supervised worker: isolates native failures while amortizing
  startup cost and naturally serializing access. This is the selected design.

## Lifecycle and Failure Handling

- No child process is created during module import; startup is lazy so reload
  and test imports remain safe.
- The worker imports and initializes `py_mini_racer` only inside the spawned
  process and signals readiness before accepting requests.
- Ordinary JavaScript/Python decoding errors are returned without killing the
  worker. Native exits, broken pipes, malformed responses, and timeouts cause
  worker replacement.
- Application shutdown sends a graceful stop request, then terminates the
  worker if it does not exit promptly. The worker ignores terminal SIGINT so
  the parent can perform that shutdown without a child traceback.
- Network access remains in the calling process. Only the encoded string and
  decoded rows cross the process boundary.

## Verification

Focused tests cover worker reuse, concurrent callers, native-style child exit,
restart behavior, and timeout cleanup. Provider tests verify that decoded rows
still flow through the existing normalization and source metadata path. The
full backend Ruff and pytest suites remain the release gate.
