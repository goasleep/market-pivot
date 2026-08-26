# ADR 003: Constrained planning and deterministic acceptance

Status: Accepted

Asset type, task contract, permissions and data requirements hard-filter Skills. Plans use capability IDs and must fit dependency, DAG, tool-permission and budget constraints. Acceptance order is safety, required inputs, deterministic domain validation, evidence coverage/freshness, then semantic judgment. A semantic judge cannot override a failed hard check.
