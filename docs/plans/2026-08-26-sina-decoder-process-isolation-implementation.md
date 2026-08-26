# Sina ETF Decoder Process Isolation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent native MiniRacer failures in the Sina ETF fallback from terminating the FastAPI process.

**Architecture:** Add a lazy, thread-safe supervisor around one `spawn`-created decoder process. The child reuses one V8 context, while the parent restarts an unhealthy child once and converts repeated failures into ordinary provider errors.

**Tech Stack:** Python 3.12, `multiprocessing`, `threading`, pytest, pandas, AkShare, PyMiniRacer

---

### Task 1: Add the supervised decoder worker

**Files:**
- Create: `backend/src/data/sina_decoder.py`
- Test: `backend/tests/test_sina_decoder.py`

**Step 1: Write failing lifecycle tests**

Add spawn-safe test workers that echo their PID, exit abruptly on a sentinel
payload, and sleep past a short timeout. Assert that concurrent calls reuse one
worker, an abrupt exit raises a Python exception rather than killing pytest,
and the next call starts a different healthy worker.

**Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_sina_decoder.py -q`

Expected: failure because `data.sina_decoder` does not exist.

**Step 3: Implement the minimal supervisor and worker protocol**

Implement a top-level spawn target, lazy process startup, request identifiers,
readiness and result messages, bounded waits, one restart attempt, and explicit
shutdown. Keep V8 imports and its reusable context inside the child process.

**Step 4: Run focused lifecycle tests**

Run: `cd backend && uv run pytest tests/test_sina_decoder.py -q`

Expected: all decoder lifecycle tests pass.

### Task 2: Route Sina ETF decoding through the worker

**Files:**
- Modify: `backend/src/data/akshare_provider.py:16-37,282-312`
- Modify: `backend/tests/test_akshare_provider.py:1-183`

**Step 1: Update the provider test**

Replace direct MiniRacer monkeypatching with a decoder-service stub and assert
that concurrent Sina fetches call the service and preserve normalized rows.

**Step 2: Run the provider test to verify it fails**

Run: `cd backend && uv run pytest tests/test_akshare_provider.py -q`

Expected: failure until the provider delegates to the decoder service.

**Step 3: Replace in-process V8 calls**

Keep the bounded Sina HTTP request in the provider, then pass only the encoded
payload to the decoder supervisor. Preserve date filtering, numeric conversion,
fallback metadata, retry behavior, and public function signatures.

**Step 4: Run focused provider tests**

Run: `cd backend && uv run pytest tests/test_akshare_provider.py -q`

Expected: all provider tests pass.

### Task 3: Validate and commit

**Files:**
- Verify all files changed by Tasks 1 and 2.

**Step 1: Run backend quality gates**

Run: `cd backend && uv run ruff check src tests`

Run: `cd backend && uv run pytest`

Expected: both commands pass.

**Step 2: Inspect the final diff**

Run: `git diff --check && git status --short && git diff --stat`

Expected: no whitespace errors; unrelated `outputs/` content remains unstaged.

**Step 3: Commit the logical fix**

```bash
git add backend/src/data/sina_decoder.py backend/src/data/akshare_provider.py \
  backend/tests/test_sina_decoder.py backend/tests/test_akshare_provider.py \
  docs/plans/2026-08-26-sina-decoder-process-isolation-design.md \
  docs/plans/2026-08-26-sina-decoder-process-isolation-implementation.md
git commit -m "fix(data): isolate Sina decoder process"
```
