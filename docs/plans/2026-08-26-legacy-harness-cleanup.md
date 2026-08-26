# Legacy Harness Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use Code workflow guidance to implement this plan task-by-task.

**Goal:** Remove the unreachable legacy Supervisor and ResearchPlan implementation while preserving the production Financial Harness and controlled stock comprehensive-analysis capability.

**Architecture:** First extract the two production dependencies from `stock_agent.py`. Then delete the orphaned ResearchPlan subsystem and migrate or remove legacy-only tests. Finally simplify the tool registry and shared Agent Loop by removing APIs and special cases that existed only for the deleted runtime.

**Tech Stack:** Python 3.12, FastAPI, LangGraph, LangChain structured tools, Pydantic, pytest, Ruff, Git hooks.

---

### Task 1: Extract production stock-analysis dependencies

**Files:**
- Create: `backend/src/agents/stock_analysis.py`
- Create: `backend/src/agents/report_compaction.py`
- Modify: `backend/src/agents/stock_executor.py`
- Modify: `backend/src/agents/financial_harness_agent.py`
- Modify: `backend/src/agents/stock_agent.py`
- Modify: `backend/tests/test_chat_tools.py`

**Step 1:** Move report compaction and its private helpers into `report_compaction.py`; update the Financial Harness and tests to import the new public function.

**Step 2:** Move stock analysis tool construction, synchronous result collection, streaming progress, and trace/checkpoint configuration into `StockAnalysisRuntime`.

**Step 3:** Update `StockComprehensiveExecutor` to own `StockAnalysisRuntime` directly. Temporarily let the legacy class inherit the runtime so the extraction commit remains behavior-preserving.

**Step 4:** Run `cd backend && uv run pytest tests/test_chat_tools.py tests/test_harness_compiler.py -q` and `uv run ruff check src tests`.

**Step 5:** Commit as `refactor(agents): extract stock analysis runtime`.

### Task 2: Remove the legacy Supervisor and ResearchPlan subsystem

**Files:**
- Delete: `backend/src/agents/stock_agent.py`
- Delete: `backend/src/application/research_plan.py`
- Delete: `backend/src/graph/research_plan.py`
- Delete: `backend/src/graph/research_planning.py`
- Delete: `backend/src/graph/research_evidence.py`
- Delete: `backend/src/models/research_plan.py`
- Modify: `backend/src/api/main.py`
- Delete: `backend/tests/test_research_plan.py`
- Modify: legacy-importing backend tests as identified by the static reference scan.

**Step 1:** Replace remaining tests that use the legacy agent only for request resolution or stock analysis with `AssetRequestResolver`, `FinancialHarnessAgent`, or `StockAnalysisRuntime`.

**Step 2:** Remove tests whose subject is specifically the deleted nested ResearchPlan or old Supervisor tool surface.

**Step 3:** Delete the legacy modules and remove ResearchPlan graph configuration from FastAPI lifespan.

**Step 4:** Run focused chat, Harness, financial-data, methodology, and open-fund tests, followed by Ruff.

**Step 5:** Commit as `refactor(harness): remove legacy research supervisor`.

### Task 3: Remove legacy tool and Agent Loop plumbing

**Files:**
- Modify: `backend/src/tools/registry.py`
- Modify: `backend/src/harness/bootstrap.py`
- Modify: `backend/src/harness/registry.py`
- Modify: `backend/src/graph/agent_loop.py`
- Modify: related backend tests.

**Step 1:** Remove `build_chat_tools`, `build_task_tools`, their fund allowlists, and the unused `build_artifact_tool` wrapper. Preserve `build_artifact_tools` and `build_named_tools`.

**Step 2:** Remove the unreachable `run_research_plan` descriptor and Agent Loop timeout/retry special cases.

**Step 3:** Remove legacy `screen_assets` completion-count logic while retaining the atomic tool and renderer.

**Step 4:** Simplify `ToolCatalog` by retaining descriptor validation and runtime name binding only; delete executable lookup APIs used solely by old tests.

**Step 5:** Run focused registry and Agent Loop tests, Ruff, and the complete backend suite.

**Step 6:** Commit as `refactor(harness): remove legacy tool plumbing`.

### Task 4: Final audit

**Step 1:** Search for references to `agents.stock_agent`, `application.research_plan`, `graph.research_plan`, `models.research_plan`, `run_research_plan`, `build_chat_tools`, and `build_task_tools`; expect none in production or tests.

**Step 2:** Run `cd backend && uv run ruff check src tests` and `cd backend && uv run pytest`.

**Step 3:** Run `git diff --check`, inspect the commit sequence, and confirm the unrelated `outputs/` directory remains untracked and uncommitted.
