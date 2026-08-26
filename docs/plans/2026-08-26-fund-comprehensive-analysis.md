# Fund Comprehensive Analysis Implementation Plan

> **For Codex:** Use the executing-plans workflow to implement this plan task-by-task.

**Goal:** Add a first-class `fund.comprehensive_analysis` Harness capability for ETF/LOF product research without entering the stock multi-agent graph.

**Architecture:** Implement the capability as a declarative composite Skill. Its fixed dependency closure covers identity, market/NAV, trend, liquidity and risk; the compiler adds exposure, tracking, premium/discount, event and portfolio branches only when the request requires them. Composite completion is determined from child evidence and a fund-specific deterministic validator.

**Tech Stack:** Python, Pydantic, LangGraph, LangChain StructuredTool, pytest.

---

### Task 1: Composite Skill contract

**Files:**
- Modify: `backend/src/harness/models.py`
- Modify: `backend/src/harness/registry.py`
- Create: `backend/resources/agent_skills/fund/comprehensive_analysis/skill.yaml`
- Create: `backend/resources/agent_skills/fund/comprehensive_analysis/instructions.md`
- Test: `backend/tests/test_fund_comprehensive_analysis.py`

1. Add declarative composite capability metadata without executable code.
2. Validate that composite capabilities have dependencies and no direct tools.
3. Verify Registry dependency closure for ETF and LOF.

### Task 2: Contract and conditional branches

**Files:**
- Modify: `backend/src/harness/compiler.py`
- Test: `backend/tests/test_fund_comprehensive_analysis.py`

1. Route ETF/LOF instrument analysis to `fund.comprehensive_analysis`.
2. Preserve the hard prohibition on `stock.comprehensive_analysis`.
3. Add exposure, tracking, premium/discount, event and portfolio capabilities only from matching task requirements.
4. Verify direct educational requests still expose no tools.

### Task 3: Composite planning and acceptance

**Files:**
- Modify: `backend/src/harness/planner.py`
- Modify: `backend/src/harness/bootstrap.py`
- Modify: `backend/src/agents/financial_harness_agent.py`
- Test: `backend/tests/test_fund_comprehensive_analysis.py`

1. Place the composite step after its dependency steps.
2. Add a deterministic fund-comprehensive validator over child evidence.
3. Mark the composite capability covered only when its required child evidence is available.
4. Expose composite status through existing plan and acceptance payloads.

### Task 4: Verification

1. Run focused Harness and ETF tests.
2. Run Ruff and the complete backend test suite.
3. Run frontend build and lint because public capability/plan payloads are consumed by the UI.
4. Do not commit unless explicitly requested.
