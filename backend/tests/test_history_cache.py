import copy
import hashlib
import json
import threading
from datetime import datetime, timedelta
from io import BytesIO
from zoneinfo import ZoneInfo

import pandas as pd

import data.history_cache as history


class MemoryHistoryStore:
    enabled = True

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.archives: list[bytes] = []
        self.manifest = history.ManifestSnapshot(
            manifest=history._empty_manifest(),
            raw=None,
            etag=None,
            sha256=None,
        )
        self.fail_reads = False
        self.fail_writes = False
        self.state = "ready"
        self.last_error: str | None = None
        self._lock = threading.RLock()

    def read_manifest(self, *, force=False):
        del force
        with self._lock:
            if self.fail_reads:
                raise OSError("manifest unavailable")
            return history.ManifestSnapshot(
                manifest=copy.deepcopy(self.manifest.manifest),
                raw=self.manifest.raw,
                etag=self.manifest.etag,
                sha256=self.manifest.sha256,
            )

    def write_manifest(self, previous, manifest):
        with self._lock:
            if self.fail_writes:
                raise OSError("manifest write failed")
            if previous.raw is not None:
                self.archives.append(previous.raw)
            raw = history._canonical_json(manifest)
            digest = hashlib.sha256(raw).hexdigest()
            self.manifest = history.ManifestSnapshot(
                manifest=copy.deepcopy(manifest),
                raw=raw,
                etag=digest[:16],
                sha256=digest,
            )
            return self.read_manifest(force=True)

    def get_bytes(self, object_key):
        return self.objects[object_key]

    def put_bytes(self, object_key, content, content_type):
        assert content_type == "application/vnd.apache.parquet"
        self.objects[object_key] = content
        return hashlib.md5(content, usedforsecurity=False).hexdigest()

    def record_error(self, exc):
        self.state = "degraded"
        self.last_error = str(exc)

    def status(self):
        return {
            "enabled": True,
            "state": self.state,
            "mode": "memory_test",
            "manifest_etag": self.manifest.etag,
            "manifest_sha256": self.manifest.sha256,
            "entry_count": len(self.manifest.manifest["entries"]),
            "last_error": self.last_error,
        }


def _frame(start_date: str, end_date: str, source_id: str = "test-source") -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "ticker": ["600000"] * len(dates),
            "open": range(len(dates)),
            "close": range(len(dates)),
        }
    )
    frame.attrs["source_metadata"] = {"source_id": source_id}
    return frame


def _fetch_recorder(calls):
    def fetch(start_date, end_date):
        calls.append((start_date, end_date))
        return _frame(start_date, end_date)

    return fetch


def test_first_write_full_hit_and_partial_extension():
    store = MemoryHistoryStore()
    cache = history.HistoryCache(store)
    series = history.HistorySeries("price", "stock", "600000", "qfq")
    calls = []
    fetch = _fetch_recorder(calls)

    initial = cache.get_or_fetch(series, "20250101", "20250105", fetch)
    hit = cache.get_or_fetch(series, "20250102", "20250104", fetch)
    extended = cache.get_or_fetch(series, "20250101", "20250110", fetch)

    assert calls == [("20250101", "20250105"), ("20250106", "20250110")]
    assert initial["date"].tolist() == pd.date_range("2025-01-01", "2025-01-05").strftime("%Y-%m-%d").tolist()
    assert hit["date"].tolist() == ["2025-01-02", "2025-01-03", "2025-01-04"]
    assert len(extended) == 10
    entry = store.manifest.manifest["entries"][series.series_id]
    assert entry["coverage"] == [{"start_date": "2025-01-01", "end_date": "2025-01-10"}]
    assert entry["partitions"]["2025"]["row_count"] == 10
    assert entry["partitions"]["2025"]["object_key"].startswith(
        "data/price/stock/600000/qfq/2025/snapshot-"
    )
    assert len(store.archives) == 1


def test_cross_year_partitions_and_series_dimensions_do_not_mix():
    store = MemoryHistoryStore()
    cache = history.HistoryCache(store)
    stock_qfq = history.HistorySeries("price", "stock", "600000", "qfq")
    stock_hfq = history.HistorySeries("price", "stock", "600000", "hfq")
    etf_nav = history.HistorySeries("nav", "etf", "600000", "none")

    cache.get_or_fetch(stock_qfq, "20241230", "20250102", _fetch_recorder([]))
    cache.get_or_fetch(stock_hfq, "20250101", "20250102", _fetch_recorder([]))
    cache.get_or_fetch(etf_nav, "20250101", "20250102", _fetch_recorder([]))

    entries = store.manifest.manifest["entries"]
    assert set(entries[stock_qfq.series_id]["partitions"]) == {"2024", "2025"}
    assert set(entries) == {stock_qfq.series_id, stock_hfq.series_id, etf_nav.series_id}


def test_recent_three_days_are_fetched_but_never_persisted():
    store = MemoryHistoryStore()
    cache = history.HistoryCache(store)
    series = history.HistorySeries("price", "stock", "600000", "qfq")
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    start = today - timedelta(days=2)
    calls = []
    fetch = _fetch_recorder(calls)

    cache.get_or_fetch(series, start.strftime("%Y%m%d"), today.strftime("%Y%m%d"), fetch)
    cache.get_or_fetch(series, start.strftime("%Y%m%d"), today.strftime("%Y%m%d"), fetch)

    expected = (start.strftime("%Y%m%d"), today.strftime("%Y%m%d"))
    assert calls == [expected, expected]
    assert store.manifest.manifest["entries"] == {}
    assert store.objects == {}


def test_concurrent_updates_preserve_both_manifest_entries():
    store = MemoryHistoryStore()
    cache = history.HistoryCache(store)
    barrier = threading.Barrier(2)
    errors = []

    def run(ticker):
        try:
            def fetch(start_date, end_date):
                barrier.wait(timeout=5)
                return _frame(start_date, end_date)

            cache.get_or_fetch(
                history.HistorySeries("price", "stock", ticker, "qfq"),
                "20250101",
                "20250102",
                fetch,
            )
        except Exception as exc:  # pragma: no cover - assertion reports thread failures
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(ticker,)) for ticker in ("600000", "000001")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert set(store.manifest.manifest["entries"]) == {
        "price:stock:600000:qfq",
        "price:stock:000001:qfq",
    }


def test_manifest_failure_leaves_uploaded_parquet_unreferenced_and_bypasses():
    store = MemoryHistoryStore()
    store.fail_writes = True
    cache = history.HistoryCache(store)
    calls = []

    result = cache.get_or_fetch(
        history.HistorySeries("price", "stock", "600000", "qfq"),
        "20250101",
        "20250102",
        _fetch_recorder(calls),
    )

    assert len(result) == 2
    assert calls == [("20250101", "20250102"), ("20250101", "20250102")]
    assert store.objects
    assert store.manifest.manifest["entries"] == {}
    assert result.attrs["source_metadata"]["cache"] == "bypass"


def test_unavailable_object_store_bypasses_to_upstream():
    store = MemoryHistoryStore()
    store.fail_reads = True
    cache = history.HistoryCache(store)
    calls = []

    result = cache.get_or_fetch(
        history.HistorySeries("price", "stock", "600000", "qfq"),
        "20250101",
        "20250102",
        _fetch_recorder(calls),
    )

    assert len(result) == 2
    assert calls == [("20250101", "20250102")]
    assert result.attrs["source_metadata"]["cache"] == "bypass"
    assert store.status()["state"] == "degraded"


def test_corrupt_parquet_degrades_and_bypasses_cache():
    store = MemoryHistoryStore()
    cache = history.HistoryCache(store)
    series = history.HistorySeries("price", "stock", "600000", "qfq")
    calls = []
    fetch = _fetch_recorder(calls)
    cache.get_or_fetch(series, "20250101", "20250102", fetch)
    partition = store.manifest.manifest["entries"][series.series_id]["partitions"]["2025"]
    store.objects[partition["object_key"]] = b"corrupt"
    cache._frame_cache.clear()

    result = cache.get_or_fetch(series, "20250101", "20250102", fetch)

    assert len(result) == 2
    assert result.attrs["source_metadata"]["cache"] == "bypass"
    assert store.status()["state"] == "degraded"
    assert "SHA-256" in store.status()["last_error"]


def test_qiniu_manifest_write_uses_no_conditional_or_versioning_arguments():
    manifest = history._empty_manifest()
    previous_raw = history._canonical_json(manifest)

    class FakeClient:
        def __init__(self):
            self.objects = {"prefix/manifest.json": previous_raw}
            self.put_calls = []

        def put_object(self, **kwargs):
            self.put_calls.append(kwargs)
            self.objects[kwargs["Key"]] = kwargs["Body"]
            return {"ETag": '"etag"'}

        def get_object(self, **kwargs):
            return {"Body": BytesIO(self.objects[kwargs["Key"]]), "ETag": '"etag"'}

    client = FakeClient()
    store = history.QiniuSingleManifestStore()
    store.enabled = True
    store.bucket = "bucket"
    store.prefix = "prefix"
    store._client = client
    previous = history.ManifestSnapshot(
        manifest=manifest,
        raw=previous_raw,
        etag="etag",
        sha256=hashlib.sha256(previous_raw).hexdigest(),
    )
    updated = copy.deepcopy(manifest)
    updated["updated_at"] = "2025-01-01T00:00:00+00:00"

    store.write_manifest(previous, updated)

    assert len(client.put_calls) == 2
    assert client.put_calls[0]["Key"].startswith("prefix/manifest-archive/manifest-")
    assert client.put_calls[1]["Key"] == "prefix/manifest.json"
    for call in client.put_calls:
        assert "IfMatch" not in call
        assert "IfNoneMatch" not in call
        assert "VersionId" not in call
    assert json.loads(client.objects["prefix/manifest.json"]) == updated
