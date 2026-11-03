"""Download-once cache with a manifest (url, sha256, bytes, downloaded_at).

Cache root: $KFACTORS_DATA or ~/.cache/kfactors. ``fetch(url, relpath)`` returns a local Path,
downloading only if missing; ``manifest()`` returns the recorded entries. Downloads use a
browser-like User-Agent, retries with backoff, and atomic writes (tmp file + rename).

Design notes
------------
* Every file lives at ``cache_root() / relpath``; ``relpath`` must be relative and may not escape
  the root (``..`` components are rejected).
* ``manifest.json`` (in the cache root) maps ``relpath -> {url, relpath, sha256, bytes,
  downloaded_at}``; ``downloaded_at`` is an ISO-8601 UTC timestamp. It is the provenance record of
  the snapshot a result was computed on; loaders copy the entry into ``Panel.meta``.
* Writes are atomic: data are streamed to a temporary file in the destination directory and moved
  into place with :func:`os.replace`, so an interrupted download never leaves a truncated file at
  ``relpath``. The manifest itself is rewritten atomically under a thread lock and (on POSIX) an
  advisory ``fcntl`` file lock, so concurrent threads/processes do not lose entries.
* HTTP status 404/410 raises :class:`DownloadNotFound` immediately (no retries): for Binance
  Vision this means "month not available". Status 429 and 5xx, timeouts and connection errors are
  retried with exponential backoff (``Retry-After`` honoured, capped); other 4xx raise
  :class:`DownloadError` at once.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import requests

__all__ = [
    "USER_AGENT", "DownloadError", "DownloadNotFound", "cache_root", "fetch", "fetch_many",
    "http_get", "manifest", "manifest_entry",
]

#: Browser-like User-Agent (some academic hosts reject the default python-requests UA).
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
MANIFEST_NAME = "manifest.json"
#: Number of attempts (1 initial + retries) and backoff base in seconds (1, 2, 4, ...).
MAX_ATTEMPTS = 4
BACKOFF_BASE = 1.0
MAX_BACKOFF = 30.0
_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_manifest_lock = threading.Lock()
_tls = threading.local()
_sleep = time.sleep  # indirection so tests can skip real waiting


class DownloadError(RuntimeError):
    """A download failed permanently (non-retryable status or retries exhausted)."""

    def __init__(self, url: str, message: str, status: int | None = None):
        super().__init__(f"{message} [{url}]")
        self.url = url
        self.status = status


class DownloadNotFound(DownloadError, FileNotFoundError):
    """The server answered 404/410: the resource does not exist (e.g. Binance month missing)."""


# --------------------------------------------------------------------------------------------
# paths and manifest
# --------------------------------------------------------------------------------------------
def cache_root() -> Path:
    """Return the cache root directory, creating it if needed.

    ``$KFACTORS_DATA`` if set and non-empty, otherwise ``~/.cache/kfactors``. The value is read on
    every call, so tests (and users) can redirect the cache by changing the environment variable.
    """
    env = os.environ.get("KFACTORS_DATA", "").strip()
    root = Path(env).expanduser() if env else Path.home() / ".cache" / "kfactors"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve(relpath: str) -> Path:
    rp = PurePosixPath(str(relpath).replace("\\", "/"))
    if rp.is_absolute() or not rp.parts or any(part in ("..", "") for part in rp.parts):
        raise ValueError(f"relpath must be a non-empty relative path inside the cache: {relpath!r}")
    return cache_root().joinpath(*rp.parts)


@contextlib.contextmanager
def _locked_manifest():
    """Thread lock + POSIX advisory file lock around a manifest read-modify-write."""
    with _manifest_lock:
        lock_path = cache_root() / (MANIFEST_NAME + ".lock")
        with open(lock_path, "a+") as fh:  # closing the file releases the flock
            try:
                import fcntl  # POSIX only
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - Windows: thread lock only
                pass
            yield


def _read_manifest() -> dict:
    path = cache_root() / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"corrupt cache manifest {path}: {exc}; delete it to rebuild") from exc
    if not isinstance(data, dict):
        raise TypeError(f"corrupt cache manifest {path}: expected a JSON object")
    return data


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _record(entries: list[dict]) -> None:
    if not entries:
        return
    with _locked_manifest():
        man = _read_manifest()
        for e in entries:
            man[e["relpath"]] = e
        payload = json.dumps(man, indent=1, sort_keys=True).encode("utf-8")
        _atomic_write_bytes(cache_root() / MANIFEST_NAME, payload)


def manifest() -> dict:
    """Return all recorded entries: ``{relpath: {url, relpath, sha256, bytes, downloaded_at}}``.

    Returns a fresh copy (mutating it does not affect the cache). Empty dict if nothing was ever
    downloaded into the current cache root.
    """
    with _manifest_lock:
        return _read_manifest()


def manifest_entry(relpath: str) -> dict | None:
    """Return the manifest entry for ``relpath`` (or ``None`` if the file was never recorded)."""
    return manifest().get(str(PurePosixPath(str(relpath).replace("\\", "/"))))


# --------------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------------
def _session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
        _tls.session = s
    return s


def _http_get(url: str, timeout: float) -> tuple[int, bytes, dict]:
    """Single GET (no retries). Returns (status, body, headers). Tests monkeypatch this."""
    r = _session().get(url, timeout=timeout)
    return r.status_code, r.content, dict(r.headers)


def http_get(url: str, *, timeout: float = 60.0, attempts: int = MAX_ATTEMPTS) -> bytes:
    """GET ``url`` with retries and exponential backoff; return the body.

    Parameters
    ----------
    url : str
    timeout : float
        Per-attempt timeout in seconds.
    attempts : int
        Total number of attempts (>= 1).

    Raises
    ------
    DownloadNotFound
        On HTTP 404/410 (never retried).
    DownloadError
        On other non-retryable 4xx, or when all attempts failed.
    """
    last = None
    for attempt in range(max(1, int(attempts))):
        if attempt:
            delay = min(MAX_BACKOFF, BACKOFF_BASE * 2 ** (attempt - 1))
            if isinstance(last, tuple):  # honour Retry-After on 429/503
                ra = last[1].get("Retry-After") or last[1].get("retry-after")
                with contextlib.suppress(TypeError, ValueError):
                    delay = min(MAX_BACKOFF, max(delay, float(ra)))
            _sleep(delay)
        try:
            status, body, headers = _http_get(url, timeout)
        except (requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError) as exc:
            last = exc
            continue
        if 200 <= status < 300:
            return body
        if status in (404, 410):
            raise DownloadNotFound(url, f"HTTP {status}: not found", status)
        if status in _RETRY_STATUS:
            last = (status, headers)
            continue
        raise DownloadError(url, f"HTTP {status}", status)
    if isinstance(last, tuple):
        raise DownloadError(url, f"HTTP {last[0]} after {attempts} attempts", last[0])
    raise DownloadError(url, f"{type(last).__name__}: {last} after {attempts} attempts")


def _entry(url: str, relpath: str, data: bytes) -> dict:
    return {
        "url": url,
        "relpath": str(PurePosixPath(str(relpath).replace("\\", "/"))),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _download(url: str, relpath: str, timeout: float) -> tuple[Path, dict]:
    path = _resolve(relpath)
    data = http_get(url, timeout=timeout)
    _atomic_write_bytes(path, data)
    return path, _entry(url, relpath, data)


# --------------------------------------------------------------------------------------------
# public fetch API
# --------------------------------------------------------------------------------------------
def fetch(url: str, relpath: str, *, refresh: bool = False, timeout: float = 60.0) -> Path:
    """Return the local path of ``url`` cached at ``cache_root()/relpath``; download if missing.

    Parameters
    ----------
    url : str
        Source URL.
    relpath : str
        Destination relative to the cache root (POSIX separators; no ``..``).
    refresh : bool
        Re-download even if the file exists (the manifest entry is replaced).
    timeout : float
        Per-attempt HTTP timeout in seconds.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    DownloadNotFound, DownloadError
        See :func:`http_get`. Nothing is written on failure.
    """
    path = _resolve(relpath)
    if path.exists() and not refresh:
        return path
    path, entry = _download(url, relpath, timeout)
    _record([entry])
    return path


def fetch_many(items, *, refresh: bool = False, timeout: float = 60.0, n_jobs: int = 8,
               missing_ok: bool = True) -> list[Path | None]:
    """Parallel :func:`fetch` for many ``(url, relpath)`` pairs; manifest updated once.

    Only files not yet cached are downloaded (all of them if ``refresh``). Downloads run in a
    thread pool of ``n_jobs`` workers; results keep the input order.

    Parameters
    ----------
    items : iterable of (url, relpath)
    missing_ok : bool
        If True, a 404/410 yields ``None`` in the output; otherwise :class:`DownloadNotFound`
        propagates. Other download errors always propagate (after the successful downloads of the
        batch have been recorded in the manifest).

    Returns
    -------
    list of Path or None
    """
    items = [(str(u), str(r)) for u, r in items]
    out: list[Path | None] = [None] * len(items)
    todo = []
    for i, (url, rel) in enumerate(items):
        path = _resolve(rel)
        if path.exists() and not refresh:
            out[i] = path
        else:
            todo.append(i)
    if not todo:
        return out

    def job(i):
        url, rel = items[i]
        try:
            return i, _download(url, rel, timeout), None
        except DownloadNotFound as exc:
            if missing_ok:
                return i, None, None
            return i, None, exc
        except Exception as exc:
            return i, None, exc

    entries, errors = [], []
    workers = max(1, min(int(n_jobs), len(todo)))
    if workers == 1:
        results = [job(i) for i in todo]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(job, todo))
    for i, res, err in results:
        if res is not None:
            out[i] = res[0]
            entries.append(res[1])
        if err is not None:
            errors.append(err)
    _record(entries)
    if errors:
        raise errors[0]
    return out
