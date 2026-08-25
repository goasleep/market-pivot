# Delivery-First Supervisor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use the Code workflow to implement this plan task-by-task.

**Goal:** Make conversational tasks model-routed, delivery-first, progressively visible, and reliable for long multi-step work.

**Architecture:** Add a structured LLM routing decision before task-contract compilation, carry it through the existing single-Supervisor graph, and enforce one 30-minute timeout for asynchronous model/tool calls without adding an outer task timeout. Add durable progress snapshots and preserve a useful正文 whenever artifacts are produced.

**Tech Stack:** Python 3.12, FastAPI, LangChain/LangGraph, Pydantic, pytest/pytest-asyncio, React SSE/A2UI client.

---

### Task 1: Structured model routing

**Files:**
- Modify: `backend/src/models/supervisor.py`
- Modify: `backend/src/application/task_contract.py`
- Modify: `backend/src/agents/stock_agent.py`
- Create: `backend/tests/test_task_contract_routing.py`

**Steps:**
1. Add failing tests for code review, plan design, data research, classifier failure, and serialized routing metadata.
2. Add `TaskRoutingDecision` and execution-mode enums.
3. Implement asynchronous model classification with strict JSON validation and `supervisor_decides` fallback.
4. Compile the task contract from the routing decision and expose the decision to the Supervisor prompt/UI.
5. Run the focused routing tests.

### Task 2: Uniform 30-minute node limits

**Files:**
- Modify: `backend/src/llm/service.py`
- Modify: `backend/src/graph/agent_loop.py`
- Modify: `backend/src/graph/research_planning.py`
- Modify: `backend/tests/test_llm_service.py`
- Modify: `backend/tests/test_chat_tools.py`
- Modify: `backend/tests/test_supervisor_end_to_end.py`
- Modify: `backend/tests/test_research_plan.py`
- Modify: `backend/tests/test_agent_loop_llm_timeout.py`

**Steps:**
1. Update tests to require 1,800 seconds for asynchronous model/tool calls and all research depths.
2. Add a shared asynchronous LLM call timeout in the LLM service.
3. Set default, Research Plan, and long-running tool timeouts to 1,800 seconds.
4. Set all Research Plan time deadlines to 1,800 seconds while preserving logical step/tool-count guards.
5. Run focused timeout tests.

### Task 3: Delivery-first completion and safe judge fallback

**Files:**
- Modify: `backend/src/agents/stock_agent.py`
- Modify: `backend/src/graph/agent_loop.py`
- Modify: `backend/tests/test_chat_tools.py`
- Modify: `backend/tests/test_agent_loop_llm_timeout.py`

**Steps:**
1. Add failing tests that a judge exception yields terminal `partial`, never `satisfied`.
2. Rewrite the Supervisor prompt around minimum useful deliverables and optional-evidence disclosure.
3. Change completion fallback to terminal partial while retaining the candidate response.
4. Emit a concise stage result after each tool completion.
5. Run focused Supervisor tests.

### Task 4: Durable periodic progress

**Files:**
- Modify: `backend/src/application/chat_service.py`
- Modify: `backend/tests/test_chat_service.py`

**Steps:**
1. Add tests for progress-state tracking and periodic progress emission.
2. Track routing mode, completed tools, last public stage summary, and deliverables per task.
3. Start a periodic progress reporter alongside the existing heartbeat and cancel it on every terminal path.
4. Ensure progress is persisted as ordinary A2UI Markdown so reconnecting clients can replay it.
5. Run chat-service tests.

### Task 5: Artifact plus chat正文

**Files:**
- Modify: `backend/src/agents/stock_agent.py`
- Modify: `backend/tests/test_chat_tools.py`

**Steps:**
1. Add failing tests for long Markdown responses and embedded HTML source removal.
2. Replace generic HTML-only compaction with summary-preserving, format-neutral compaction.
3. Track artifact metadata and append a truthful file summary when needed.
4. Run artifact-focused tests.

### Task 6: Verification and live regression

**Files:**
- Test: `backend/tests/`
- Output: local JSON/Excel evaluation artifacts only; do not commit them.

**Steps:**
1. Run Ruff and focused backend tests.
2. Run the complete backend test suite.
3. Restart/reload the services and confirm the active model is `gpt-5.6-sol`.
4. Rerun B8, B12, C1, and C2 in independent conversations with only the original prompt.
5. Compare completion, routing/tool usage, stage-result visibility,正文/附件 behavior, score, and elapsed time.

### Task 7: Compact backtest delivery

**Files:**
- Modify: `backend/src/tools/research.py`
- Modify: `backend/src/agents/stock_agent.py`
- Modify: `backend/tests/test_chat_tools.py`

**Steps:**
1. Return core metrics, cost scenarios, periods, provenance, acceptance, and limitations directly from strategy comparison tools.
2. Keep complete curves, daily values, signals, and trades only in audit artifacts.
3. Hide `read_artifact` from backtest and mixed-workflow Supervisor tool surfaces.
4. Add payload-size and tool-surface regression tests.

Create a commit only after explicit user authorization, per repository instructions.
