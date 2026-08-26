# Sandbox Strategy Capability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make explicit Python/code/sandbox backtest requests deterministically select a dedicated sandbox-research capability while keeping the sandbox tool unavailable to ordinary backtests.

**Architecture:** Split `design_and_run_sandbox_strategy` out of the shared `backtest.execute` skill and descriptor into a dedicated `strategy.sandbox_research` capability. Compile explicit sandbox backtest language into that capability, retain `market.history` as its dependency, and add a Harness execution instruction that requires the single sandbox tool when the capability is present.

**Tech Stack:** Python 3.12, FastAPI backend, Pydantic contracts, LangGraph Harness, YAML skill manifests, pytest.

---

### Task 1: Specify sandbox routing behavior with failing tests

**Files:**
- Modify: `backend/tests/test_harness_compiler.py`

**Step 1: Add a test for an explicit code-strategy backtest**

Create a BACKTEST request such as `用 Python 生成代码策略并回测 510300` and assert that its required capabilities are `market.history` and `strategy.sandbox_research`.

**Step 2: Assert the selected tool surface**

Resolve the contract through the default skill registry and assert that `design_and_run_sandbox_strategy` is present while `run_backtest`, `design_and_run_backtest`, and `compare_strategy_backtests` are absent.

**Step 3: Add a plain-backtest regression test**

Compile `回测 510300` and assert that it retains `backtest.execute`; its selected tools must exclude `design_and_run_sandbox_strategy`.

**Step 4: Add a direct-explanation regression test**

Compile a direct-response request that discusses Python backtest code and assert that it loads no capabilities or tools.

**Step 5: Run the focused tests and verify they fail**

Run: `cd backend && uv run pytest tests/test_harness_compiler.py -q`

Expected: the new sandbox capability/tool-isolation assertions fail before implementation.

### Task 2: Add the dedicated sandbox capability and skill

**Files:**
- Modify: `backend/src/harness/bootstrap.py`
- Modify: `backend/resources/agent_skills/research/backtest/skill.yaml`
- Create: `backend/resources/agent_skills/research/sandbox/skill.yaml`
- Create: `backend/resources/agent_skills/research/sandbox/instructions.md`

**Step 1: Reassign the sandbox tool descriptor**

Change the `design_and_run_sandbox_strategy` descriptor capability from `backtest.execute` to `strategy.sandbox_research` without changing its asset-type, evidence, cost, or read-only boundary.

**Step 2: Remove the sandbox tool from ordinary backtests**

Delete `design_and_run_sandbox_strategy` from the `backtest.execute` skill tool list.

**Step 3: Create a single-purpose sandbox skill**

Define `strategy.sandbox_research` for stock, ETF, and LOF assets. Require `market.history`, expose only `design_and_run_sandbox_strategy`, require backtest/sandbox evidence, and keep the skill read-only and high-cost.

**Step 4: Document execution boundaries**

State that the skill generates only a bounded target-position function, must report sandbox validation and trusted-engine replay, and never implies live execution.

### Task 3: Compile explicit sandbox requests into the new capability

**Files:**
- Modify: `backend/src/harness/compiler.py`
- Modify: `backend/src/agents/financial_harness_agent.py`

**Step 1: Add an explicit sandbox-request predicate**

Recognize a sandbox request when the message combines an explicit code action with strategy, factor, position, or backtest context. Do not depend on the legacy intent, and apply the capability only to stock, ETF, and LOF.

**Step 2: Replace the ordinary backtest capability**

For matching requests, require `market.history` and `strategy.sandbox_research` instead of `backtest.execute`.

If the routing model incorrectly chooses a tool-free response for an explicit sandbox execution request, upgrade it to controlled backtest execution. Preserve direct responses for code review, explanation-only requests, and unrelated Python generation.

**Step 3: Strengthen the scoped Harness instruction**

When the compiled contract requires `strategy.sandbox_research`, instruct the executor that it must call `design_and_run_sandbox_strategy`; ordinary backtest requests will not have this tool available.

**Step 4: Run the focused tests**

Run: `cd backend && uv run pytest tests/test_harness_compiler.py -q`

Expected: all compiler and tool-isolation tests pass.

### Task 4: Verify registry, behavior, and backend quality gates

**Files:**
- Test: `backend/tests/test_harness_compiler.py`
- Test: `backend/tests/test_chat_tools.py`
- Test: `backend/tests/test_strategy_candidates.py`

**Step 1: Run focused sandbox tests**

Run: `cd backend && uv run pytest tests/test_harness_compiler.py tests/test_chat_tools.py -k 'sandbox or harness or backtest' -q`

Expected: PASS.

**Step 2: Run candidate lifecycle tests**

Run: `cd backend && uv run pytest tests/test_strategy_candidates.py -q`

Expected: PASS.

**Step 3: Run backend lint**

Run: `cd backend && uv run ruff check src tests`

Expected: PASS.

**Step 4: Run the complete backend suite**

Run: `cd backend && uv run pytest`

Expected: PASS.

**Step 5: Check the final diff**

Run: `git diff --check`

Expected: no whitespace errors. Do not commit unless the user explicitly requests it.
