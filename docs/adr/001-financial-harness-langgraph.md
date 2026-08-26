# ADR 001: Retain LangGraph and build a Financial Harness

Status: Accepted

The public chat runtime uses a single `FinancialHarnessAgent`. LangGraph remains the checkpointed execution engine, while a new kernel graph compiles contracts, selects Skills and validates plans before the existing LLM/tool loop runs. The ETF/LOF path never defaults to the fixed stock multi-agent graph.
