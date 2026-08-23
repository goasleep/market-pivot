"""Permanent object-storage cache for confirmed historical market data."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Callable, Protocol
from zoneinfo import ZoneInfo

import boto3
import pandas as pd
import polars as pl
from botocore.config import Config
from botocore.exceptions import ClientError
from loguru import logger

from config import settings

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_CACHE_SECONDS = 30.0
FRAME_CACHE_LIMIT = 128
PERMANENT_HISTORY_LAG_DAYS = 3


@dataclass(frozen=True)
class HistorySeries:
    dataset: str
    asset_type: str
    ticker: str
    adjustment: str = "none"

    @property
    def series_id(self) -> str:
        return ":".join((self.dataset, self.asset_type, self.ticker, self.adjustment))

    def data_key(self, year: int, digest: str) -> str:
        return (
            f"data/{self.dataset}/{self.asset_type}/{self.ticker}/"
            f"{self.adjustment}/{year}/snapshot-{digest}.parquet"
        )


@dataclass(frozen=True)
class ManifestSnapshot:
    manifest: dict[str, Any]
    raw: bytes | None
    etag: str | None
    sha256: str | None


@dataclass(frozen=True)
class FetchedRange:
    start_date: date
    end_date: date
    frame: pd.DataFrame
    source_metadata: dict[str, Any]


class HistoryObjectStore(Protocol):
    enabled: bool

    def read_manifest(self, *, force: bool = False) -> ManifestSnapshot: ...

    def write_manifest(self, previous: ManifestSnapshot, manifest: dict[str, Any]) -> ManifestSnapshot: ...

    def get_bytes(self, object_key: str) -> bytes: ...

    def put_bytes(self, object_key: str, content: bytes, content_type: str) -> str | None: ...

    def status(self) -> dict[str, Any]: ...


def _empty_manifest() -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entries": {},
    }


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


class QiniuSingleManifestStore:
    """Single-writer manifest store for Qiniu Kodo.

    This implementation deliberately does not use S3 conditional writes or
    Bucket Versioning because Qiniu Kodo does not provide the required S3
    compatibility guarantees. It requires one backend process and protects
    concurrent threads with an in-process lock in ``HistoryCache``.
    """

    def __init__(self) -> None:
        self.enabled = bool(
            settings.market_history_cache_enabled
            and settings.market_history_s3_endpoint_url
            and settings.market_history_s3_bucket
        )
        self.endpoint_url = settings.market_history_s3_endpoint_url.strip()
        self.bucket = settings.market_history_s3_bucket.strip()
        self.region = settings.market_history_s3_region.strip() or "us-east-1"
        self.access_key_id = settings.market_history_s3_access_key_id.strip()
        self.secret_access_key = settings.market_history_s3_secret_access_key.strip()
        # Qiniu does not support x-amz-security-token. The field remains part
        # of the generic adapter configuration for other S3-compatible stores.
        self.session_token = settings.market_history_s3_session_token.strip()
        self.addressing_style = settings.market_history_s3_addressing_style.strip() or "path"
        self.prefix = settings.market_history_s3_prefix.strip("/")
        self._client = None
        self._manifest_cache: tuple[ManifestSnapshot, float] | None = None
        self._state_lock = threading.RLock()
        self._state = "disabled" if not self.enabled else "not_checked"
        self._last_success_at: str | None = None
        self._last_error: str | None = None

    def _key(self, relative_key: str) -> str:
        return "/".join(part for part in (self.prefix, relative_key.strip("/")) if part)

    def _get_client(self):
        if not self.enabled:
            raise RuntimeError("历史行情永久缓存未启用")
        if bool(self.access_key_id) != bool(self.secret_access_key):
            raise RuntimeError("MARKET_HISTORY_S3_ACCESS_KEY_ID 和 SECRET_ACCESS_KEY 必须同时设置")
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url or None,
                region_name=self.region,
                aws_access_key_id=self.access_key_id or None,
                aws_secret_access_key=self.secret_access_key or None,
                aws_session_token=self.session_token or None,
                config=Config(signature_version="s3v4", s3={"addressing_style": self.addressing_style}),
            )
        return self._client

    def _record_success(self) -> None:
        with self._state_lock:
            self._state = "ready"
            self._last_success_at = datetime.now(timezone.utc).isoformat()
            self._last_error = None

    def record_error(self, exc: Exception) -> None:
        with self._state_lock:
            self._state = "degraded"
            self._last_error = str(exc)[:500]

    def read_manifest(self, *, force: bool = False) -> ManifestSnapshot:
        with self._state_lock:
            cached = self._manifest_cache
            if not force and cached is not None and time.monotonic() - cached[1] < MANIFEST_CACHE_SECONDS:
                return cached[0]
        try:
            response = self._get_client().get_object(Bucket=self.bucket, Key=self._key("manifest.json"))
            raw = response["Body"].read()
            manifest = json.loads(raw)
            if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or not isinstance(
                manifest.get("entries"), dict
            ):
                raise ValueError("历史缓存 Manifest 结构无效")
            snapshot = ManifestSnapshot(
                manifest=manifest,
                raw=raw,
                etag=str(response.get("ETag") or "").strip('"') or None,
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code not in {"NoSuchKey", "NoSuchObject", "404"} and status != 404:
                self.record_error(exc)
                raise
            snapshot = ManifestSnapshot(manifest=_empty_manifest(), raw=None, etag=None, sha256=None)
        except Exception as exc:
            self.record_error(exc)
            raise
        with self._state_lock:
            self._manifest_cache = (snapshot, time.monotonic())
        self._record_success()
        return snapshot

    def write_manifest(self, previous: ManifestSnapshot, manifest: dict[str, Any]) -> ManifestSnapshot:
        raw = _canonical_json(manifest)
        try:
            if previous.raw is not None and previous.sha256:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archive_key = f"manifest-archive/manifest-{timestamp}-{previous.sha256}.json"
                self.put_bytes(archive_key, previous.raw, "application/json")
            response = self._get_client().put_object(
                Bucket=self.bucket,
                Key=self._key("manifest.json"),
                Body=raw,
                ContentType="application/json",
            )
            with self._state_lock:
                self._manifest_cache = None
            verified = self.read_manifest(force=True)
            expected_sha = hashlib.sha256(raw).hexdigest()
            if verified.sha256 != expected_sha:
                raise RuntimeError("历史缓存 Manifest 覆盖后的 SHA-256 校验失败")
            if response.get("ETag") and not verified.etag:
                raise RuntimeError("历史缓存 Manifest 覆盖后缺少 ETag")
            return verified
        except Exception as exc:
            self.record_error(exc)
            raise

    def get_bytes(self, object_key: str) -> bytes:
        try:
            response = self._get_client().get_object(Bucket=self.bucket, Key=self._key(object_key))
            content = response["Body"].read()
            self._record_success()
            return content
        except Exception as exc:
            self.record_error(exc)
            raise

    def put_bytes(self, object_key: str, content: bytes, content_type: str) -> str | None:
        try:
            response = self._get_client().put_object(
                Bucket=self.bucket,
                Key=self._key(object_key),
                Body=content,
                ContentType=content_type,
            )
            self._record_success()
            return str(response.get("ETag") or "").strip('"') or None
        except Exception as exc:
            self.record_error(exc)
            raise

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            cached = self._manifest_cache[0] if self._manifest_cache else None
            entry_count = len(cached.manifest.get("entries", {})) if cached else 0
            return {
                "enabled": self.enabled,
                "state": self._state,
                "mode": "qiniu_single_writer_manifest",
                "bucket_configured": bool(self.bucket and self.endpoint_url),
                "manifest_etag": cached.etag if cached else None,
                "manifest_sha256": cached.sha256 if cached else None,
                "entry_count": entry_count,
                "last_success_at": self._last_success_at,
                "last_error": self._last_error,
            }


# Qiniu Kodo does not expose AWS S3 Bucket Versioning semantics and its
# compatibility documentation does not guarantee conditional PutObject
# support for If-Match or If-None-Match. Therefore this manifest does not
# rely on S3 version IDs or ETag-based compare-and-swap.
#
# Writes are serialized with an in-process lock and the previous committed
# manifest is archived before overwrite. This is safe only with one backend
# process. Multiple workers or replicas require an append-only index or an
# external distributed lock.
_MANIFEST_WRITE_LOCK = threading.RLock()


class HistoryCache:
    def __init__(self, store: HistoryObjectStore) -> None:
        self.store = store
        self._frame_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._frame_lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.store.enabled

    @staticmethod
    def _parse_date(value: str) -> date:
        normalized = value.replace("-", "")
        return datetime.strptime(normalized, "%Y%m%d").date()  # noqa: DTZ007 -- date-only input

    @staticmethod
    def _date_arg(value: date) -> str:
        return value.strftime("%Y%m%d")

    @staticmethod
    def _coverage_ranges(entry: dict[str, Any]) -> list[tuple[date, date]]:
        result = []
        for item in entry.get("coverage", []):
            try:
                result.append((date.fromisoformat(item["start_date"]), date.fromisoformat(item["end_date"])))
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(result)

    @staticmethod
    def _merge_ranges(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
        merged: list[tuple[date, date]] = []
        for start, end in sorted(ranges):
            if start > end:
                continue
            if not merged or start > merged[-1][1] + timedelta(days=1):
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    @classmethod
    def _missing_ranges(
        cls,
        start: date,
        end: date,
        coverage: list[tuple[date, date]],
    ) -> list[tuple[date, date]]:
        if start > end:
            return []
        missing: list[tuple[date, date]] = []
        cursor = start
        for covered_start, covered_end in cls._merge_ranges(coverage):
            if covered_end < cursor or covered_start > end:
                continue
            if covered_start > cursor:
                missing.append((cursor, min(end, covered_start - timedelta(days=1))))
            cursor = max(cursor, covered_end + timedelta(days=1))
            if cursor > end:
                break
        if cursor <= end:
            missing.append((cursor, end))
        return missing

    @staticmethod
    def _filter_frame(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        if frame is None or frame.empty or "date" not in frame.columns:
            return pd.DataFrame(columns=list(frame.columns) if frame is not None else [])
        filtered = frame.copy()
        filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        filtered = filtered[filtered["date"].notna()]
        return filtered[(filtered["date"] >= start.isoformat()) & (filtered["date"] <= end.isoformat())].copy()

    @staticmethod
    def _merge_frames(frames: list[pd.DataFrame], start: date, end: date) -> pd.DataFrame:
        usable = [frame for frame in frames if frame is not None and not frame.empty]
        if not usable:
            return pd.DataFrame()
        merged = pd.concat(usable, ignore_index=True, sort=False)
        merged = HistoryCache._filter_frame(merged, start, end)
        if merged.empty:
            return merged
        return merged.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

    @staticmethod
    def _serialize_frame(frame: pd.DataFrame) -> bytes:
        normalized = frame.copy()
        normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.strftime("%Y-%m-%d")
        records = normalized.where(pd.notna(normalized), None).to_dict(orient="records")
        polars_frame = pl.DataFrame(records, infer_schema_length=None)
        output = BytesIO()
        polars_frame.write_parquet(output, compression="zstd")
        return output.getvalue()

    @staticmethod
    def _deserialize_frame(content: bytes) -> pd.DataFrame:
        return pd.DataFrame(pl.read_parquet(BytesIO(content)).to_dicts())

    def _read_partition(self, partition: dict[str, Any]) -> pd.DataFrame:
        object_key = str(partition["object_key"])
        with self._frame_lock:
            cached = self._frame_cache.get(object_key)
            if cached is not None:
                self._frame_cache.move_to_end(object_key)
                return cached.copy()
        content = self.store.get_bytes(object_key)
        digest = hashlib.sha256(content).hexdigest()
        if digest != partition.get("sha256"):
            raise ValueError(f"历史缓存 Parquet SHA-256 不匹配: {object_key}")
        frame = self._deserialize_frame(content)
        with self._frame_lock:
            self._frame_cache[object_key] = frame.copy()
            self._frame_cache.move_to_end(object_key)
            while len(self._frame_cache) > FRAME_CACHE_LIMIT:
                self._frame_cache.popitem(last=False)
        return frame

    def _read_cached(
        self,
        snapshot: ManifestSnapshot,
        series: HistorySeries,
        start: date,
        end: date,
    ) -> tuple[pd.DataFrame, list[str]]:
        entry = snapshot.manifest.get("entries", {}).get(series.series_id, {})
        frames: list[pd.DataFrame] = []
        sources: set[str] = set()
        for year in range(start.year, end.year + 1):
            partition = entry.get("partitions", {}).get(str(year))
            if not partition:
                continue
            frames.append(self._read_partition(partition))
            sources.update(str(source) for source in partition.get("sources", []) if source)
        return self._merge_frames(frames, start, end), sorted(sources)

    @staticmethod
    def _source_metadata(frame: pd.DataFrame) -> dict[str, Any]:
        return dict(frame.attrs.get("source_metadata") or {})

    def _commit(
        self,
        series: HistorySeries,
        start: date,
        end: date,
        fetched: list[FetchedRange],
    ) -> ManifestSnapshot:
        # Upstream requests run outside the global lock. Another request may
        # commit while they are in flight, so coverage must be recalculated
        # from a forced manifest read after acquiring the lock.
        with _MANIFEST_WRITE_LOCK:
            previous = self.store.read_manifest(force=True)
            entry = previous.manifest.get("entries", {}).get(series.series_id, {})
            missing = self._missing_ranges(start, end, self._coverage_ranges(entry))
            if not missing:
                return previous

            selected_frames = []
            selected_sources: set[str] = set()
            for item in fetched:
                for missing_start, missing_end in missing:
                    overlap_start = max(item.start_date, missing_start)
                    overlap_end = min(item.end_date, missing_end)
                    if overlap_start <= overlap_end:
                        selected_frames.append(self._filter_frame(item.frame, overlap_start, overlap_end))
                        source_id = item.source_metadata.get("source_id")
                        if source_id:
                            selected_sources.add(str(source_id))
            manifest = copy.deepcopy(previous.manifest)
            entries = manifest.setdefault("entries", {})
            mutable_entry = entries.setdefault(
                series.series_id,
                {
                    "dataset": series.dataset,
                    "asset_type": series.asset_type,
                    "ticker": series.ticker,
                    "adjustment": series.adjustment,
                    "coverage": [],
                    "partitions": {},
                },
            )
            all_coverage = self._coverage_ranges(mutable_entry) + missing
            mutable_entry["coverage"] = [
                {"start_date": covered_start.isoformat(), "end_date": covered_end.isoformat()}
                for covered_start, covered_end in self._merge_ranges(all_coverage)
            ]

            new_rows = self._merge_frames(selected_frames, start, end)
            for year in sorted(set(new_rows["date"].str[:4].astype(int)) if not new_rows.empty else set()):
                year_start = date(year, 1, 1)
                year_end = date(year, 12, 31)
                year_new = self._filter_frame(new_rows, year_start, year_end)
                old_partition = mutable_entry.get("partitions", {}).get(str(year))
                old_frame = self._read_partition(old_partition) if old_partition else pd.DataFrame()
                # Existing rows come first so confirmed permanent history is
                # never silently replaced by a later upstream response.
                year_frame = self._merge_frames([old_frame, year_new], year_start, year_end)
                content = self._serialize_frame(year_frame)
                digest = hashlib.sha256(content).hexdigest()
                object_key = series.data_key(year, digest)
                etag = self.store.put_bytes(object_key, content, "application/vnd.apache.parquet")
                old_sources = set(old_partition.get("sources", [])) if old_partition else set()
                sources = sorted(old_sources | selected_sources)
                mutable_entry.setdefault("partitions", {})[str(year)] = {
                    "object_key": object_key,
                    "sha256": digest,
                    "etag": etag,
                    "row_count": len(year_frame),
                    "min_date": str(year_frame.iloc[0]["date"]),
                    "max_date": str(year_frame.iloc[-1]["date"]),
                    "sources": sources,
                }
                with self._frame_lock:
                    self._frame_cache[object_key] = year_frame.copy()

            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            return self.store.write_manifest(previous, manifest)

    def get_or_fetch(
        self,
        series: HistorySeries,
        start_date: str,
        end_date: str,
        fetcher: Callable[[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        if not self.enabled:
            return fetcher(self._date_arg(start), self._date_arg(end))

        cutoff = datetime.now(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=PERMANENT_HISTORY_LAG_DAYS)
        permanent_end = min(end, cutoff)
        try:
            snapshot = self.store.read_manifest()
            cached = pd.DataFrame()
            cached_sources: list[str] = []
            fetched_permanent: list[FetchedRange] = []
            if start <= permanent_end:
                cached, cached_sources = self._read_cached(snapshot, series, start, permanent_end)
                entry = snapshot.manifest.get("entries", {}).get(series.series_id, {})
                for missing_start, missing_end in self._missing_ranges(
                    start,
                    permanent_end,
                    self._coverage_ranges(entry),
                ):
                    frame = fetcher(self._date_arg(missing_start), self._date_arg(missing_end))
                    fetched_permanent.append(
                        FetchedRange(
                            start_date=missing_start,
                            end_date=missing_end,
                            frame=frame,
                            source_metadata=self._source_metadata(frame),
                        )
                    )
                if fetched_permanent:
                    snapshot = self._commit(series, start, permanent_end, fetched_permanent)
                    cached, cached_sources = self._read_cached(snapshot, series, start, permanent_end)

            dynamic = pd.DataFrame()
            dynamic_start = max(start, cutoff + timedelta(days=1))
            if dynamic_start <= end:
                dynamic = fetcher(self._date_arg(dynamic_start), self._date_arg(end))

            result = self._merge_frames([cached, dynamic], start, end)
            fetched_sources = {
                str(item.source_metadata.get("source_id"))
                for item in fetched_permanent
                if item.source_metadata.get("source_id")
            }
            dynamic_metadata = self._source_metadata(dynamic)
            if dynamic_metadata.get("source_id"):
                fetched_sources.add(str(dynamic_metadata["source_id"]))
            sources = sorted(set(cached_sources) | fetched_sources)
            if sources:
                result.attrs["source_metadata"] = {
                    "source_id": sources[0] if len(sources) == 1 else "mixed",
                    "source_ids": sources,
                    "source_name": "对象存储历史缓存" if not fetched_sources else "历史缓存与上游数据",
                    "source_chain": sources,
                    "cache": (
                        "hit"
                        if not fetched_permanent and dynamic.empty
                        else "partial_hit"
                        if not cached.empty
                        else "miss"
                    ),
                }
            return result
        except Exception as exc:
            logger.warning("Historical object cache unavailable for {}: {}; bypassing", series.series_id, exc)
            if hasattr(self.store, "record_error"):
                self.store.record_error(exc)  # type: ignore[attr-defined]
            frame = fetcher(self._date_arg(start), self._date_arg(end))
            metadata = self._source_metadata(frame)
            metadata["cache"] = "bypass"
            frame.attrs["source_metadata"] = metadata
            return frame

    def status(self) -> dict[str, Any]:
        return self.store.status()


history_cache = HistoryCache(QiniuSingleManifestStore())


def get_history_cache_status() -> dict[str, Any]:
    current = history_cache.status()
    if history_cache.enabled and current["state"] == "not_checked":
        try:
            history_cache.store.read_manifest()
        except Exception:
            # The store records the concrete error and degraded state. Status
            # must remain available even while Qiniu is unavailable.
            pass
    return history_cache.status()
