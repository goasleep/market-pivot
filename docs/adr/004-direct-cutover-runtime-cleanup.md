# ADR 004: Direct cutover and runtime-only cleanup

Status: Accepted

New tasks persist `graph_name=financial-harness` and explicit runtime, contract and Skill versions. Old checkpoints are not migrated. `scripts/interrupt_legacy_agent_runtime.py` provides an idempotent dry-run-first cleanup that interrupts only active Agent tasks and removes their interactions, state, runtime events and checkpoint threads while preserving completed conversations and trading/research records.
