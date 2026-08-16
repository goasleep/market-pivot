"""Artifact metadata, preview, and download endpoints."""

import asyncio
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from artifacts.service import artifact_service
from artifacts.storage import ArtifactNotFoundError

router = APIRouter()


@router.get("")
async def list_artifacts(limit: int = 50):
    return {"artifacts": artifact_service.list(limit)}


@router.get("/{artifact_id}")
async def get_artifact(artifact_id: str):
    artifact = artifact_service.get(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="产物不存在")
    return artifact


async def _file_response(artifact_id: str, disposition: str) -> Response:
    try:
        stored = await asyncio.to_thread(artifact_service.read, artifact_id)
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail="产物文件不存在") from exc
    if stored is None:
        raise HTTPException(status_code=404, detail="产物不存在")
    artifact, content = stored
    filename = quote(str(artifact["name"]))
    return Response(
        content=content,
        media_type=artifact["mime_type"],
        headers={"Content-Disposition": f"{disposition}; filename*=UTF-8''{filename}"},
    )


@router.get("/{artifact_id}/preview")
async def preview_artifact(artifact_id: str):
    return await _file_response(artifact_id, "inline")


@router.get("/{artifact_id}/download")
async def download_artifact(artifact_id: str):
    return await _file_response(artifact_id, "attachment")
