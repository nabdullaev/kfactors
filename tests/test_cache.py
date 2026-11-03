"""Tests for kfactors.data.cache (offline: the HTTP layer is monkeypatched)."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path

import pytest
import requests

from kfactors.data import cache


class FakeServer:
    """Maps URL -> body; records calls; optional per-URL scripted status sequences."""

    def __init__(self, files=None, script=None):
        self.files = dict(files or {})
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, url, timeout):
        with self.lock:
            self.calls.append(url)
            seq = self.script.get(url)
            step = seq.pop(0) if seq else None
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, int):
            return step, b"", {}
        if isinstance(step, tuple):
            return step
        if url in self.files:
            return 200, self.files[url], {}
        return 404, b"not found", {}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("KFACTORS_DATA", str(tmp_path / "cache"))
    sleeps = []
    monkeypatch.setattr(cache, "_sleep", sleeps.append)
    return tmp_path / "cache", sleeps


def serve(monkeypatch, **kw):
    srv = FakeServer(**kw)
    monkeypatch.setattr(cache, "_http_get", srv)
    return srv


def test_cache_root_env_and_default(tmp_path, monkeypatch):
    monkeypatch.setenv("KFACTORS_DATA", str(tmp_path / "x"))
    assert cache.cache_root() == tmp_path / "x" and (tmp_path / "x").is_dir()
    monkeypatch.delenv("KFACTORS_DATA")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert cache.cache_root() == tmp_path / "home" / ".cache" / "kfactors"


def test_fetch_downloads_once_and_records_manifest(root, monkeypatch):
    body = b"hello,world\n" * 100
    srv = serve(monkeypatch, files={"https://h/a.csv": body})
    p = cache.fetch("https://h/a.csv", "src/a.csv")
    assert p == root[0] / "src" / "a.csv" and p.read_bytes() == body
    p2 = cache.fetch("https://h/a.csv", "src/a.csv")
    assert p2 == p and len(srv.calls) == 1            # cached: no second request
    e = cache.manifest()["src/a.csv"]
    assert e["url"] == "https://h/a.csv" and e["relpath"] == "src/a.csv"
    assert e["sha256"] == hashlib.sha256(body).hexdigest() and e["bytes"] == len(body)
    assert datetime.fromisoformat(e["downloaded_at"]).tzinfo is not None
    assert cache.manifest_entry("src/a.csv") == e
    # manifest on disk is valid JSON keyed by relpath
    on_disk = json.loads((root[0] / "manifest.json").read_text())
    assert set(on_disk) == {"src/a.csv"}


def test_refresh_redownloads(root, monkeypatch):
    srv = serve(monkeypatch, files={"https://h/a": b"v1"})
    cache.fetch("https://h/a", "a")
    srv.files["https://h/a"] = b"v2"
    assert cache.fetch("https://h/a", "a").read_bytes() == b"v1"
    assert cache.fetch("https://h/a", "a", refresh=True).read_bytes() == b"v2"
    assert cache.manifest()["a"]["sha256"] == hashlib.sha256(b"v2").hexdigest()


def test_retry_with_backoff_then_success(root, monkeypatch):
    url = "https://h/flaky"
    srv = serve(monkeypatch, files={url: b"ok"},
                script={url: [503, requests.ConnectionError("reset"), (429, b"", {"Retry-After": "3"})]})
    assert cache.fetch(url, "flaky").read_bytes() == b"ok"
    assert len(srv.calls) == 4
    assert root[1] == [1.0, 2.0, 4.0]       # exponential backoff 1, 2, 4 s


def test_retry_after_header_is_honoured(root, monkeypatch):
    url = "https://h/ra"
    serve(monkeypatch, files={url: b"ok"}, script={url: [(429, b"", {"Retry-After": "7"})]})
    cache.fetch(url, "ra")
    assert root[1] == [7.0]


def test_retries_exhausted_raise(root, monkeypatch):
    url = "https://h/down"
    srv = serve(monkeypatch, script={url: [500] * 10})
    with pytest.raises(cache.DownloadError, match="HTTP 500"):
        cache.fetch(url, "down")
    assert len(srv.calls) == cache.MAX_ATTEMPTS
    assert not (root[0] / "down").exists()
    srv2 = serve(monkeypatch, script={url: [requests.Timeout("t")] * 10})
    with pytest.raises(cache.DownloadError, match="Timeout"):
        cache.fetch(url, "down")
    assert len(srv2.calls) == cache.MAX_ATTEMPTS


def test_404_not_retried_and_nothing_written(root, monkeypatch):
    srv = serve(monkeypatch)
    with pytest.raises(cache.DownloadNotFound) as ei:
        cache.fetch("https://h/missing", "m/missing.zip")
    assert isinstance(ei.value, FileNotFoundError) and ei.value.status == 404
    assert len(srv.calls) == 1 and root[1] == []
    assert not (root[0] / "m" / "missing.zip").exists()
    assert cache.manifest() == {}


def test_other_4xx_not_retried(root, monkeypatch):
    srv = serve(monkeypatch, script={"https://h/f": [403]})
    with pytest.raises(cache.DownloadError, match="403") as ei:
        cache.fetch("https://h/f", "f")
    assert not isinstance(ei.value, cache.DownloadNotFound)
    assert len(srv.calls) == 1


def test_atomic_write_leaves_no_partial_file(root, monkeypatch):
    serve(monkeypatch, files={"https://h/a": b"data"})

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(cache.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        cache.fetch("https://h/a", "d/a.bin")
    d = root[0] / "d"
    assert not (d / "a.bin").exists()
    assert list(d.iterdir()) == []          # temp file cleaned up


@pytest.mark.parametrize("bad", ["../escape", "/abs/path", "a/../../b", ""])
def test_relpath_must_stay_inside_cache(root, bad):
    with pytest.raises(ValueError):
        cache.fetch("https://h/x", bad)


def test_fetch_many_parallel_order_404_and_single_manifest_write(root, monkeypatch):
    files = {f"https://h/{i}": f"body{i}".encode() for i in range(40) if i % 7}
    srv = serve(monkeypatch, files=files)
    writes = []
    orig = cache._record
    monkeypatch.setattr(cache, "_record", lambda entries: (writes.append(len(entries)),
                                                           orig(entries)))
    items = [(f"https://h/{i}", f"many/{i}.txt") for i in range(40)]
    out = cache.fetch_many(items, n_jobs=8)
    for i, p in enumerate(out):
        if i % 7:
            assert isinstance(p, Path) and p.read_bytes() == f"body{i}".encode()
        else:
            assert p is None
    assert writes == [len(files)]
    assert len(cache.manifest()) == len(files)
    n = len(srv.calls)
    out2 = cache.fetch_many(items, n_jobs=8)
    # cached files are not requested again; the 404s are retried (not negative-cached)
    assert len(srv.calls) - n == 40 - len(files)
    assert [p is None for p in out2] == [p is None for p in out]


def test_fetch_many_missing_not_ok_raises_after_recording(root, monkeypatch):
    serve(monkeypatch, files={"https://h/ok": b"1"})
    with pytest.raises(cache.DownloadNotFound):
        cache.fetch_many([("https://h/ok", "ok"), ("https://h/no", "no")], missing_ok=False)
    assert "ok" in cache.manifest()


def test_concurrent_fetch_calls_do_not_lose_manifest_entries(root, monkeypatch):
    files = {f"https://h/{i}": b"x" * i for i in range(30)}
    serve(monkeypatch, files=files)
    threads = [threading.Thread(target=cache.fetch, args=(f"https://h/{i}", f"c/{i}"))
               for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(cache.manifest()) == 30


def test_user_agent_is_browser_like():
    s = cache._session()
    assert s.headers["User-Agent"].startswith("Mozilla/5.0")
    assert "python-requests" not in s.headers["User-Agent"]


def test_corrupt_manifest_is_reported(root):
    root[0].mkdir(parents=True, exist_ok=True)
    (root[0] / "manifest.json").write_text("{not json")
    with pytest.raises(RuntimeError, match="corrupt"):
        cache.manifest()
