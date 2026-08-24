"""Storage adapters for generated artifacts.

The production adapter is S3-compatible and keeps report bytes outside the
application container.  The local adapter is intentionally explicit so unit
tests can remain deterministic without requiring an object-store emulator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when an artifact metadata record has no corresponding object."""


class ArtifactStorage(Protocol):
    """Minimal byte-oriented object storage contract."""

    def put(self, object_key: str, content: bytes, content_type: str) -> None:
        """Write one object."""

    def get(self, object_key: str) -> bytes:
        """Read one object or raise a storage-specific error."""

    def delete(self, object_key: str) -> None:
        """Delete exactly one object; missing objects are treated as already deleted."""

    def presign_get_url(self, object_key: str, expires_in: int = 900) -> str:
        """Return a temporary HTTP(S) URL suitable for external model input."""


class LocalArtifactStorage:
    """Explicit local adapter for tests and development-only fixtures."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, object_key: str) -> Path:
        root = self.root.resolve()
        path = (root / object_key).resolve()
        if root != path and root not in path.parents:
            raise ValueError("非法产物路径")
        return path

    def put(self, object_key: str, content: bytes, content_type: str) -> None:
        del content_type
        path = self._path(object_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def get(self, object_key: str) -> bytes:
        try:
            return self._path(object_key).read_bytes()
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError(object_key) from exc

    def delete(self, object_key: str) -> None:
        try:
            self._path(object_key).unlink()
        except FileNotFoundError:
            return

    def presign_get_url(self, object_key: str, expires_in: int = 900) -> str:
        del object_key, expires_in
        raise RuntimeError("本地产物存储无法生成供外部模型访问的 HTTP(S) URL")


class S3ArtifactStorage:
    """S3 and S3-compatible object storage adapter.

    ``endpoint_url`` may point to AWS S3, MinIO, Ceph RGW, Cloudflare R2, or
    another service implementing the S3 API.  The boto3 client is created
    lazily so importing the application does not make a network request.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        public_endpoint_url: str = "",
        bucket: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        session_token: str = "",
        addressing_style: str = "path",
    ):
        self.endpoint_url = endpoint_url.strip()
        self.public_endpoint_url = public_endpoint_url.strip() or self.endpoint_url
        self.bucket = bucket.strip()
        self.region = region.strip() or "us-east-1"
        self.access_key_id = access_key_id.strip()
        self.secret_access_key = secret_access_key.strip()
        self.session_token = session_token.strip()
        self.addressing_style = addressing_style.strip() or "path"
        self._client = None
        self._presign_client = None

    def _client_options(self, endpoint_url: str) -> dict:
        options = {
            "endpoint_url": endpoint_url or None,
            "region_name": self.region,
            "config": Config(s3={"addressing_style": self.addressing_style}),
        }
        if self.access_key_id:
            options.update(
                {
                    "aws_access_key_id": self.access_key_id,
                    "aws_secret_access_key": self.secret_access_key,
                    "aws_session_token": self.session_token or None,
                }
            )
        return options

    def _get_client(self):
        if not self.bucket:
            raise RuntimeError("S3 产物存储未配置完整，请设置 S3_BUCKET")
        if bool(self.access_key_id) != bool(self.secret_access_key):
            raise RuntimeError("S3_ACCESS_KEY_ID 和 S3_SECRET_ACCESS_KEY 必须同时设置")
        if self._client is None:
            self._client = boto3.client("s3", **self._client_options(self.endpoint_url))
        return self._client

    def _get_presign_client(self):
        self._get_client()
        if self.public_endpoint_url == self.endpoint_url:
            return self._client
        if self._presign_client is None:
            self._presign_client = boto3.client("s3", **self._client_options(self.public_endpoint_url))
        return self._presign_client

    def put(self, object_key: str, content: bytes, content_type: str) -> None:
        self._get_client().put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=content,
            ContentType=content_type,
        )

    def get(self, object_key: str) -> bytes:
        try:
            response = self._get_client().get_object(Bucket=self.bucket, Key=object_key)
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code in {"404", "NoSuchKey", "NoSuchObject"}:
                raise ArtifactNotFoundError(object_key) from exc
            raise
        return response["Body"].read()

    def delete(self, object_key: str) -> None:
        self._get_client().delete_object(Bucket=self.bucket, Key=object_key)

    def presign_get_url(self, object_key: str, expires_in: int = 900) -> str:
        return self._get_presign_client().generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": object_key},
            ExpiresIn=expires_in,
        )
