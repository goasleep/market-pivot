# Legacy Harness Cleanup Design

## Context

The public chat path now enters `AssetAgent`, which subclasses `FinancialHarnessAgent`. The declarative Harness compiler, Skill registry, constrained planner, and `build_named_tools` own capability selection and execution.

The previous `agents.stock_agent.AssetAgent` still contains a second Supervisor loop, a dynamically created `run_research_plan` tool, checkpoint recovery, and a separate ResearchPlan graph. Production imports that legacy module only to reuse the stock comprehensive-analysis tool and generated-report compaction. Tests and application startup keep the rest of the old subsystem importable even though public requests do not enter it.

## Decision

Retain one public Financial Harness runtime and one controlled stock-analysis executor. Move the two production dependencies out of `stock_agent.py`:

- A dedicated stock-analysis runtime builds `run_stock_comprehensive_analysis` and invokes `research_service`.
- A report-compaction utility removes embedded artifact source and bounds chat-native summaries.

After production and focused tests use those modules, remove the legacy Supervisor and its ResearchPlan graph as one coherent subsystem. Remove its startup configuration, models, evidence/planning helpers, dynamic tool descriptor, timeout/retry special cases, broad tool builders, and legacy-only tests.

Keep `get_fundamentals`, `compare_quotes`, and `screen_assets`. They remain deterministic atomic tools with focused tests and may receive Skills later; this cleanup does not make that product decision. Keep their A2UI renderers for the same reason.

## Runtime Flow After Cleanup

1. `application.chat_service` uses `agents.asset_agent.asset_agent`.
2. `FinancialHarnessAgent` compiles a contract and selects declarative Skills.
3. `build_named_tools` builds only the selected runtime tools.
4. `StockComprehensiveExecutor` delegates only the stock comprehensive-analysis capability to the stock-analysis runtime.
5. The shared Agent Loop executes tools and validates the Harness contract; it has no nested ResearchPlan tool.

## Safety and Compatibility

- No public API path or frontend contract changes.
- Existing checkpointed `financial-harness` and stock comprehensive-analysis graph names remain stable.
- Legacy `supervisor-agent` and `market-research-plan` checkpoints are intentionally no longer resumable because no production route creates them.
- Simulation mutations retain their existing confirmation boundary.
- The three unmanifested atomic tools remain implemented but unavailable to the Harness until explicitly assigned a Skill.

## Verification

Each commit must pass focused tests for the moved surface. The final cleanup must pass Ruff, the complete backend pytest suite, `git diff --check`, and a static import/reference scan proving that deleted legacy module names and tool builders no longer have callers.
