# Delivery-First Supervisor Design

## Requirements

- Use an LLM-generated structured routing decision; do not classify user tasks with new keyword rules.
- Apply a uniform 30-minute timeout to asynchronous LLM and tool calls while leaving the outer chat task without a wall-clock timeout.
- Make the Supervisor delivery-first: satisfy the minimum useful deliverables before optional evidence expansion.
- A completion-judge failure must never be reported as fully satisfied.
- Stream an immediate routing result, tool-completion summaries, and periodic durable progress for long tasks.
- When an artifact is generated, retain a useful chat summary and describe the actual artifact type.
- Preserve the existing confirmation boundary for mutating paper-trading tools.

## Chosen Architecture

The chat entry point asks the configured model for a small `TaskRoutingDecision` before building the task contract. The decision distinguishes direct response, artifact-only work, evidence research, backtest execution, and mixed workflows. Its fields are stored in the task contract and injected into the Supervisor prompt. If classification fails, routing is marked `supervisor_decides`; no keyword classifier is used as a fallback.

The Supervisor remains the only orchestrator. Its prompt and completion contract become delivery-first: it should answer once the required deliverables are supported, disclose missing optional evidence, and avoid reopening completed work. Tool and asynchronous model calls use the same 1,800-second timeout. The durable chat task itself has no wall-clock timeout.

The LLM timeout is a hard caller-side deadline: the Agent stops waiting after 1,800 seconds even when an OpenAI-compatible transport delays cancellation. Backtest comparison tools return a compact Supervisor payload containing core metrics, cost scenarios, periods, provenance, acceptance, and limitations. Full curves, daily values, signals, and trades remain in user-facing audit artifacts. Backtest and mixed workflows do not expose `read_artifact` to the Supervisor, so attachments cannot be reread to construct the chat answer.

Progress is emitted at three levels: the model routing decision before execution, a concise stage result after each completed tool, and a periodic snapshot from the durable task manager while work is still running. Progress snapshots describe completed tools and outstanding deliverables without exposing chain-of-thought.

Artifact compaction removes embedded source payloads but never replaces the response with a generic notice. The final chat response keeps a concise conclusion, references the generated file using its real MIME type/name, and leaves the full content in the artifact card.

## Alternatives Considered

1. Let the existing Supervisor classify implicitly. This is the smallest change but is difficult to test and did not prevent B8 from calling live ETF tools.
2. Generate a complete DAG before every task. This offers stronger control but adds latency and duplicates the existing Research Plan graph.
3. Use deterministic keyword routing. This is fast but conflicts with the requirement that the model classify tasks.

## Failure Handling

- Routing model failure: use `supervisor_decides`; do not infer a task type in code.
- Completion judge failure: terminate with `partial`, retain the candidate answer, and expose a verification warning.
- Tool timeout: return a structured `tool_timeout` observation after 30 minutes; the outer task may continue.
- LLM transport ignores cancellation: stop awaiting at the hard deadline and observe the late provider task separately.
- Long-running task: durable progress continues every configured interval and includes completed stage names.
- Artifact generation: chat summary is always emitted even if the artifact card is the primary long-form delivery.
- Backtest artifacts: return core metrics directly and keep full files audit-only; never require artifact content to form the answer.

## Verification

- Unit tests for model routing decisions and classifier failure fallback.
- Agent-loop tests for 1,800-second limits and non-satisfied judge fallback.
- Chat-service tests for immediate and periodic stage results.
- Artifact tests ensuring Markdown is not called HTML and long answers are not replaced by a generic notice.
- Backtest payload tests ensuring full curves/trades are excluded and `read_artifact` is hidden from backtest workflows.
- Regression tests and live reruns for B8, B12, C1, and C2.
