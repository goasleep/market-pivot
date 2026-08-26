# P0 Tool Skills Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use Code workflow guidance to implement this plan task-by-task.

**Goal:** Expose structured market-dataset research and the executable strategy catalog through the Financial Harness Skill registry.

**Architecture:** Add two declarative Skill packages over existing read-only tools. Extend the deterministic task compiler with narrow semantic routing so whole-market structured stock queries select `market.dataset`, while explicit requests for the system's available strategies select `strategy.list`; general strategy methodology remains on `methodology.search`.

**Tech Stack:** Python 3.12, Pydantic, LangChain structured tools, YAML Skill manifests, pytest.

---

### Task 1: Add failing compiler and registry coverage

**Files:**
- Modify: `backend/tests/test_harness_compiler.py`

**Step 1:** Add a test proving an A-share multi-year dividend screen compiles to only `market.dataset`, resolves the new Skill, and exposes `search_market_data_catalog` followed by `query_market_data`.

**Step 2:** Add a test proving an explicit system strategy-catalog request compiles to only `strategy.list` and exposes `list_trading_strategies`.

**Step 3:** Add a regression test proving a general strategy-methodology request still compiles to `methodology.search`.

**Step 4:** Run `cd backend && uv run pytest tests/test_harness_compiler.py -q`; expect the new assertions to fail before implementation.

### Task 2: Add the declarative Skill packages

**Files:**
- Create: `backend/resources/agent_skills/market/dataset/skill.yaml`
- Create: `backend/resources/agent_skills/market/dataset/instructions.md`
- Create: `backend/resources/agent_skills/strategy/list/skill.yaml`
- Create: `backend/resources/agent_skills/strategy/list/instructions.md`

**Step 1:** Declare `market.dataset` for stock research with both catalog and query tools, structured-data evidence fields, and high cost.

**Step 2:** Require catalog discovery before query execution, prohibit invented dataset IDs, stop when the catalog reports unavailable, and report coverage/acceptance/provenance.

**Step 3:** Declare `strategy.list` over `list_trading_strategies`, distinguishing configured executable strategies from methodology or performance claims.

### Task 3: Route requests to the new capabilities

**Files:**
- Modify: `backend/src/harness/compiler.py`

**Step 1:** Add narrow predicates for whole-market screening/ranking/aggregation or multi-period financial queries and explicit system strategy-catalog requests.

**Step 2:** Select `market.dataset` before generic stock analysis when the structured-query predicate matches.

**Step 3:** Select `strategy.list` only for catalog requests; preserve `methodology.search` for ordinary strategy questions.

### Task 4: Verify

**Files:**
- Verify all files above.

**Step 1:** Run the focused compiler tests.

**Step 2:** Run `cd backend && uv run ruff check src tests`.

**Step 3:** Run `cd backend && uv run pytest`.

**Step 4:** Run `git diff --check` and inspect the final diff. Do not commit without explicit user authorization.
