"""Create model-facing financial chart artifacts through the default object store."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from artifacts.service import ArtifactService, artifact_service
from charts.financial import (
    RenderedFinancialChart,
    render_fund_structure_chart,
    render_risk_chart,
    render_technical_chart,
)

_RENDER_LOCK = threading.Lock()


def _render_chart(
    renderer: Callable[..., RenderedFinancialChart],
    renderer_args: tuple[Any, ...],
) -> RenderedFinancialChart:
    """Keep Matplotlib rendering deterministic across parallel graph nodes."""
    with _RENDER_LOCK:
        return renderer(*renderer_args)


@dataclass(frozen=True)
class VisualEvidence:
    """Persistent artifact metadata plus an ephemeral model-only URL."""

    artifact: dict[str, Any]
    model_url: str | None


class VisualEvidenceService:
    def __init__(self, artifacts: ArtifactService | None = None):
        self.artifacts = artifacts or artifact_service

    async def _prepare(
        self,
        renderer: Callable[..., RenderedFinancialChart],
        *,
        chart_type: str,
        ticker: str,
        asset_type: str,
        conversation_id: str | None,
        task_id: str | None,
        renderer_args: tuple[Any, ...],
    ) -> VisualEvidence | None:
        try:
            rendered = await asyncio.to_thread(_render_chart, renderer, renderer_args)
            end_date = str(rendered.metadata.get("end_date") or "latest")
            scope = task_id or conversation_id
            execution_key = f"{scope}:visual:{chart_type}:{ticker}:{end_date}" if scope else None
            artifact = await self.artifacts.create_binary_artifact(
                name=rendered.name,
                content=rendered.content,
                mime_type="image/png",
                artifact_type="image",
                ticker=ticker,
                asset_type=asset_type,
                source="analysis-visual",
                conversation_id=conversation_id,
                task_id=task_id,
                metadata=rendered.metadata,
                execution_key=execution_key,
            )
        except Exception as exc:
            logger.warning("Unable to create {} visual evidence for {}: {}", chart_type, ticker, exc)
            return None

        try:
            model_url = self.artifacts.model_input_url(artifact)
        except Exception as exc:
            logger.warning("Unable to create model URL for {} visual evidence {}: {}", chart_type, ticker, exc)
            model_url = None
        return VisualEvidence(artifact=artifact, model_url=model_url)

    async def prepare_technical(
        self,
        *,
        ticker: str,
        asset_type: str,
        history: list[dict[str, Any]],
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> VisualEvidence | None:
        return await self._prepare(
            render_technical_chart,
            chart_type="technical",
            ticker=ticker,
            asset_type=asset_type,
            conversation_id=conversation_id,
            task_id=task_id,
            renderer_args=(ticker, asset_type, history),
        )

    async def prepare_risk(
        self,
        *,
        ticker: str,
        asset_type: str,
        history: list[dict[str, Any]],
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> VisualEvidence | None:
        return await self._prepare(
            render_risk_chart,
            chart_type="risk",
            ticker=ticker,
            asset_type=asset_type,
            conversation_id=conversation_id,
            task_id=task_id,
            renderer_args=(ticker, asset_type, history),
        )

    async def prepare_fund_structure(
        self,
        *,
        ticker: str,
        asset_type: str,
        history: list[dict[str, Any]],
        nav_history: list[dict[str, Any]],
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> VisualEvidence | None:
        return await self._prepare(
            render_fund_structure_chart,
            chart_type="fund_structure",
            ticker=ticker,
            asset_type=asset_type,
            conversation_id=conversation_id,
            task_id=task_id,
            renderer_args=(ticker, asset_type, history, nav_history),
        )


visual_evidence_service = VisualEvidenceService()
