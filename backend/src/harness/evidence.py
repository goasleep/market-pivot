"""Run-scoped evidence store that keeps full tool payloads out of model context."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from harness.models import EvidenceRecord


class EvidenceStore:
    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}
        self._raw: dict[str, str] = {}

    def add_tool_result(self, capability_id: str, tool_name: str, raw_result: str) -> EvidenceRecord:
        try:
            payload = json.loads(raw_result)
        except (TypeError, json.JSONDecodeError):
            payload = {"result": str(raw_result)}
        status = str(payload.get("status") or payload.get("data_status") or "available")
        status = {
            "degraded": "limited",
            "unverified": "limited",
            "conflict": "conflicting",
            "unavailable": "unavailable",
            "data_unavailable": "unavailable",
            "not_applicable": "unavailable",
        }.get(status, status)
        if status not in {"available", "limited", "unavailable", "conflicting"}:
            status = "available" if payload.get("available", True) else "unavailable"
        raw_provenance = payload.get("provenance") or {}
        provenance_items = (
            [raw_provenance]
            if isinstance(raw_provenance, dict) and raw_provenance
            else [item for item in raw_provenance if isinstance(item, dict)]
            if isinstance(raw_provenance, list)
            else []
        )
        explicit_sources = payload.get("sources")
        sources = (
            [item for item in explicit_sources if isinstance(item, dict)]
            if isinstance(explicit_sources, list)
            else provenance_items
        )
        primary_source = next((item for item in sources if item), {})
        data = payload.get("data", payload.get("quote", payload.get("history", payload)))
        summary = json.dumps(data, ensure_ascii=False, default=str)
        record = EvidenceRecord(
            capability_id=capability_id,
            tool_name=tool_name,
            source_type=str(payload.get("data_type") or capability_id),
            status=status,
            as_of=payload.get("as_of") or primary_source.get("as_of"),
            fetched_at=payload.get("fetched_at")
            or primary_source.get("fetched_at")
            or datetime.now(timezone.utc).isoformat(),
            freshness=payload.get("freshness") or primary_source.get("freshness"),
            sources=tuple(item for item in sources if isinstance(item, dict)),
            summary=summary[:2400],
            artifact_ids=tuple(
                str(item.get("artifact_id") or item.get("id"))
                for item in payload.get("artifacts", [])
                if isinstance(item, dict) and (item.get("artifact_id") or item.get("id"))
            ),
            content_hash=hashlib.sha256(str(raw_result).encode()).hexdigest(),
        )
        self._records[record.evidence_id] = record
        self._raw[record.evidence_id] = str(raw_result)
        return record

    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records.values())

    def raw(self, evidence_id: str) -> str:
        return self._raw[evidence_id]

    def compressed_context(self) -> list[dict[str, Any]]:
        return [record.model_dump(mode="json", exclude={"raw_result"}) for record in self._records.values()]
