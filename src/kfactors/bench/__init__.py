"""One-command benchmark suite: simulations + real-data k-curves -> leaderboard.

Command line
------------
::

    python -m kfactors.bench {sim,real,all,leaderboard} --config configs/default.yaml \\
        --out results/ [--reps N] [--n-jobs J] [--calibrate] [--panels NAME ...] \\
        [--designs PATTERN ...] [--seed S] [--no-cache] [--fail-fast]

* ``sim`` -- for every design of the simulation grid and every replication: simulate a panel
  (:func:`kfactors.sim.simulate`), apply every rule configuration
  (:func:`kfactors.rules.select_k`) and compute the panel diagnostics
  (:func:`kfactors.report.diagnose`). Writes ``<out>/sim/raw.csv`` (one row per design x rep x
  rule; also ``raw.parquet`` when pyarrow is installed), ``accuracy.csv`` (rule x design:
  exact %, MAE, mean k, ...), ``heatmap.png`` / ``heatmap_mae.png``
  (:func:`kfactors.plots.plot_heatmap`) and ``timings.csv``. With ``--calibrate`` a decision
  table is learnt with :func:`kfactors.report.calibrate_decision_table` and written to
  ``<out>/sim/decision_table.json`` -- the packaged resource is **never** overwritten; the
  command that installs it is printed instead.
* ``real`` -- for every panel of the config: the eigenvalue spectrum of a representative window
  (:func:`kfactors.plots.plot_spectrum`), the rolling ``k_t`` of every rule over walk-forward
  windows (CSV + line plot), :func:`kfactors.report.report` on the latest window (markdown) and
  the k-curve of every listed strategy (:func:`kfactors.evaluate.k_curve` semantics). Strategies
  whose ``weights`` raise :class:`NotImplementedError` (unfinished contributor cards) are
  skipped with a message; other failures are recorded and make the exit code 1.
* ``all`` -- ``sim`` then ``real``. Every command rebuilds ``<out>/leaderboard.csv`` and
  ``<out>/leaderboard.md`` from whatever results exist in ``<out>`` (``leaderboard`` does only
  that).

Everything is deterministic given ``seed``: replication ``r`` of design ``d`` uses the seed
``derive_seed(seed, "sim", d, r)`` (independent of the grid order, of design filters and of
``--n-jobs``), and stochastic rules get their own derived seeds. Intermediate results are
cached under ``<out>/cache/`` (pickles keyed by a hash of every input that affects them,
including the data fingerprint and the kfactors version); ``--no-cache`` disables the cache.

Configuration (YAML)
--------------------
Top level::

    name: default            # free text, copied into the outputs
    seed: 0                  # master seed
    n_jobs: 1                # joblib processes (CLI --n-jobs overrides)
    sim: {...}               # optional
    real: {...}              # optional

``sim`` keys (defaults in brackets):

* ``grid`` [``"default"``] -- a name for :func:`kfactors.sim.design_grid` or a list of
  :class:`kfactors.sim.SimSpec` keyword dicts (a missing ``name`` is generated).
* ``designs`` [null] -- optional list of :mod:`fnmatch` patterns on design names.
* ``reps`` [20] -- replications per design (CLI ``--reps`` overrides).
* ``kmax`` [10], ``standardize`` [true] -- passed to every rule and to ``diagnose``.
* ``rules`` [``"all"``] -- rule labels such as ``"onatski"`` or ``"bai_ng(criterion=IC_p1)"``
  (see :func:`kfactors.report.rule_label`), or ``"all"`` = every registered rule with defaults.
* ``calibrate`` [false] and ``calibrate_options`` [{}] -- forwarded to
  :func:`kfactors.report.calibrate_decision_table` (``max_depth``, ``min_samples_leaf``, ...).

``real`` keys:

* ``rules`` [``"all"``] -- rule labels used for the rolling ``k_t`` and in every k-curve.
* ``n_boot`` [1000], ``alpha`` [0.10], ``mean_block`` [null] -- k-curve bootstrap.
* ``walkforward`` [{}] -- defaults for :class:`kfactors.engine.WalkForwardConfig` (``window``,
  ``rebalance``, ``lag``, ``cost_bps``, ``min_assets``, ``standardize``, ``kmax``, ``start``,
  ``end``, ``rule_kwargs``, ``random_state`` [= seed]).
* ``report`` [{n_boot: 200, kmax: 20}] -- options of :func:`kfactors.report.report`
  (``window`` [= walk-forward window], ``enabled`` [true]).
* ``rolling`` [{}] -- ``window`` [= walk-forward window], ``step`` [= rebalance], ``start``,
  ``end``, ``min_assets``, ``kmax``, ``standardize`` (defaults from ``walkforward``).
* ``spectrum`` [{}] -- ``window`` [= walk-forward window], ``end`` [null = latest feasible].
* ``panels`` -- list of panel blocks::

      - name: jkp_usa_monthly
        source: jkp            # jkp | french | binance | synthetic | csv
        args: {...}            # loader keyword arguments (see below)
        walkforward: {...}     # overrides real.walkforward
        rolling: {...}         # overrides real.rolling
        spectrum: {...}
        report: {...}
        rules: [...]           # overrides real.rules
        experiments:
          - name: tangency_w240
            strategy: kfactors.strategies.sdf:TangencyPCA   # "module:Class"
            strategy_kwargs: {}
            metric: sharpe_net  # any key of kfactors.evaluate.METRICS
            k_grid: "1..40"     # list of ints or "a..b" / "a..b:step" (inclusive)
            walkforward: {...}  # overrides the panel's walkforward
            rules: [...]        # overrides the panel's rules
            n_boot / alpha / mean_block: ...

  Loader ``args``: ``jkp`` -> :func:`kfactors.data.load_jkp`, ``french`` ->
  :func:`kfactors.data.load_french`, ``binance`` -> :func:`kfactors.data.load_binance`;
  ``synthetic`` -> :func:`kfactors.bench.real_bench.synthetic_panel` (offline, a
  :class:`~kfactors.sim.SimSpec` plus ``freq``, ``start``, ``scale``, ``premia``,
  ``late_assets``, ``late_rows``); ``csv`` -> ``path`` (wide CSV, dates x assets),
  ``freq`` and optional ``periods_per_year``.

Rule labels with options (``"parallel(null=circular)"``) are applied inside the walk-forward
engine through a callable ``k`` (same arguments as the engine's own rule path); plain names use
the engine's rule path with ``walkforward.rule_kwargs``.
"""
from __future__ import annotations

import contextlib
import copy
import fnmatch
import functools
import hashlib
import json
import logging
import math
import os
import pickle
import tempfile
import time
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

__all__ = [
    "CACHE_SCHEMA",
    "Cache",
    "build_leaderboard",
    "derive_seed",
    "load_config",
    "main",
    "parse_k_grid",
    "resolve_rules",
    "run_real",
    "run_sim",
    "stable_hash",
    "timed",
]

#: Bumped whenever the content of cached objects changes; part of every cache key.
CACHE_SCHEMA = 1

LOG = logging.getLogger("kfactors.bench")

_TOP_KEYS = {"name", "seed", "n_jobs", "sim", "real"}
_SIM_KEYS = {"grid", "designs", "reps", "kmax", "standardize", "rules", "calibrate",
             "calibrate_options"}
_REAL_KEYS = {"rules", "n_boot", "alpha", "mean_block", "walkforward", "report", "rolling",
              "spectrum", "panels"}
_PANEL_KEYS = {"name", "source", "args", "walkforward", "rolling", "spectrum", "report",
               "rules", "experiments"}
_EXP_KEYS = {"name", "strategy", "strategy_kwargs", "metric", "k_grid", "walkforward",
             "rules", "n_boot", "alpha", "mean_block"}
_SOURCES = ("jkp", "french", "binance", "synthetic", "csv")

_SIM_DEFAULTS = {"grid": "default", "designs": None, "reps": 20, "kmax": 10,
                 "standardize": True, "rules": "all", "calibrate": False,
                 "calibrate_options": {}}
_REAL_DEFAULTS = {"rules": "all", "n_boot": 1000, "alpha": 0.10, "mean_block": None,
                  "walkforward": {}, "report": {"n_boot": 200, "kmax": 20}, "rolling": {},
                  "spectrum": {}, "panels": []}


# ------------------------------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------------------------------
def _check_keys(d: Mapping, allowed: set, where: str) -> None:
    if not isinstance(d, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(d).__name__}")
    extra = sorted(set(d) - allowed)
    if extra:
        raise ValueError(f"unknown key(s) {extra} in {where}; allowed: {sorted(allowed)}")


def parse_k_grid(spec) -> list[int]:
    """Parse a k grid: a list of non-negative ints, or ``"a..b"`` / ``"a..b:step"`` (inclusive).

    Examples
    --------
    >>> parse_k_grid("0..4")
    [0, 1, 2, 3, 4]
    >>> parse_k_grid("1..9:4")
    [1, 5, 9]
    >>> parse_k_grid([3, 1, 2])
    [1, 2, 3]
    """
    if isinstance(spec, str):
        body, _, step = spec.partition(":")
        a, sep, b = body.partition("..")
        if not sep:
            raise ValueError(f"k_grid string must look like 'a..b' or 'a..b:step', got {spec!r}")
        try:
            lo, hi, st = int(a), int(b), int(step) if step else 1
        except ValueError:
            raise ValueError(f"malformed k_grid {spec!r}") from None
        if lo < 0 or hi < lo or st < 1:
            raise ValueError(f"k_grid {spec!r} must satisfy 0 <= a <= b and step >= 1")
        return list(range(lo, hi + 1, st))
    if isinstance(spec, Iterable):
        ks = []
        for k in spec:
            if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
                raise ValueError(f"k_grid entries must be non-negative integers, got {k!r}")
            ks.append(int(k))
        if not ks or len(set(ks)) != len(ks):
            raise ValueError(f"k_grid must be non-empty without duplicates, got {ks}")
        return sorted(ks)
    raise ValueError(f"k_grid must be a list of ints or 'a..b', got {spec!r}")


def resolve_rules(rules) -> list[tuple[str, str, dict]]:
    """Rule configurations as ``(label, name, kwargs)`` triples.

    Parameters
    ----------
    rules : "all" or sequence of str
        ``"all"`` = every rule of :data:`kfactors.RULES` with default options (sorted by name);
        otherwise labels parsed with :func:`kfactors.report.parse_rule_label`. Labels are
        canonicalised with :func:`kfactors.report.rule_label`.

    Raises
    ------
    ValueError
        Unknown rule name, malformed or duplicated label.
    """
    from kfactors.report import parse_rule_label, rule_label
    from kfactors.rules import RULES

    if rules is None or rules == "all":
        return [(name, name, {}) for name in sorted(RULES)]
    if isinstance(rules, str):
        rules = [rules]
    out, seen = [], set()
    for lab in rules:
        if lab == "all":
            raise ValueError("'all' cannot be combined with other rule labels")
        name, kw = parse_rule_label(lab)
        if name not in RULES:
            raise ValueError(f"unknown rule {name!r} in label {lab!r}; available: {sorted(RULES)}")
        canon = rule_label(name, kw)
        if canon in seen:
            raise ValueError(f"duplicated rule label {canon!r}")
        seen.add(canon)
        out.append((canon, name, kw))
    if not out:
        raise ValueError("empty rule list")
    return out


def _merge(*dicts) -> dict:
    out: dict = {}
    for d in dicts:
        if d:
            out.update(copy.deepcopy(dict(d)))
    return out


def load_config(path_or_dict) -> dict:
    """Load and validate a benchmark config (YAML path or dict); fill in defaults.

    Unknown keys raise ``ValueError`` (typos must not be silently ignored). Rule labels, k grids,
    sources and strategy paths are checked for syntax here; importing strategies and loading
    data happen at run time.

    Returns
    -------
    dict
        A new dict with ``name``, ``seed``, ``n_jobs``, and -- if present in the input -- ``sim``
        and ``real`` sections completed with their defaults.
    """
    if isinstance(path_or_dict, (str, os.PathLike)):
        import yaml

        with open(path_or_dict, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        default_name = Path(path_or_dict).stem
    else:
        raw = copy.deepcopy(dict(path_or_dict))
        default_name = "config"
    _check_keys(raw, _TOP_KEYS, "config")
    cfg = {"name": str(raw.get("name", default_name)), "seed": raw.get("seed", 0),
           "n_jobs": raw.get("n_jobs", 1)}
    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int) or cfg["seed"] < 0:
        raise ValueError(f"seed must be a non-negative integer, got {cfg['seed']!r}")
    if isinstance(cfg["n_jobs"], bool) or not isinstance(cfg["n_jobs"], int) or cfg["n_jobs"] == 0:
        raise ValueError(f"n_jobs must be a non-zero integer, got {cfg['n_jobs']!r}")
    if raw.get("sim") is not None:
        _check_keys(raw["sim"], _SIM_KEYS, "sim")
        sim = _merge(_SIM_DEFAULTS, raw["sim"])
        resolve_rules(sim["rules"])
        if isinstance(sim["reps"], bool) or not isinstance(sim["reps"], int) or sim["reps"] < 1:
            raise ValueError(f"sim.reps must be a positive integer, got {sim['reps']!r}")
        if not isinstance(sim["grid"], (str, list)):
            raise ValueError("sim.grid must be a design_grid name or a list of SimSpec dicts")
        cfg["sim"] = sim
    if raw.get("real") is not None:
        _check_keys(raw["real"], _REAL_KEYS, "real")
        real = _merge(_REAL_DEFAULTS, raw["real"])
        resolve_rules(real["rules"])
        names = set()
        panels = []
        for i, p in enumerate(real["panels"] or []):
            where = f"real.panels[{i}]"
            _check_keys(p, _PANEL_KEYS, where)
            if "name" not in p or "source" not in p:
                raise ValueError(f"{where} needs 'name' and 'source'")
            if p["source"] not in _SOURCES:
                raise ValueError(f"{where}.source must be one of {_SOURCES}, got {p['source']!r}")
            if p["name"] in names:
                raise ValueError(f"duplicated panel name {p['name']!r}")
            names.add(p["name"])
            if "rules" in p:
                resolve_rules(p["rules"])
            exps, enames = [], set()
            for j, e in enumerate(p.get("experiments") or []):
                ew = f"{where}.experiments[{j}]"
                _check_keys(e, _EXP_KEYS, ew)
                for key in ("name", "strategy", "k_grid"):
                    if key not in e:
                        raise ValueError(f"{ew} needs {key!r}")
                if e["name"] in enames:
                    raise ValueError(f"duplicated experiment name {e['name']!r} in {where}")
                enames.add(e["name"])
                if ":" not in str(e["strategy"]):
                    raise ValueError(f"{ew}.strategy must be 'module:Class', got {e['strategy']!r}")
                parse_k_grid(e["k_grid"])
                if "rules" in e:
                    resolve_rules(e["rules"])
                exps.append(dict(e))
            pp = dict(p)
            pp["experiments"] = exps
            panels.append(pp)
        real["panels"] = panels
        cfg["real"] = real
    return cfg


def select_designs(names: Iterable[str], patterns) -> list[str]:
    """Names matching any :mod:`fnmatch` pattern (all names when ``patterns`` is empty)."""
    names = list(names)
    if not patterns:
        return names
    if isinstance(patterns, str):
        patterns = [patterns]
    return [n for n in names if any(fnmatch.fnmatchcase(n, pat) for pat in patterns)]


# ------------------------------------------------------------------------------------------
# seeds, hashing, cache, timing
# ------------------------------------------------------------------------------------------
def derive_seed(seed: int, *parts) -> int:
    """Deterministic 32-bit seed from a master seed and labels (order-independent of any grid).

    Uses :class:`numpy.random.SeedSequence` with entropy ``[seed, crc32(part_1), ...]``.
    """
    ent = [int(seed)] + [zlib.crc32(str(p).encode("utf-8")) for p in parts]
    return int(np.random.SeedSequence(ent).generate_state(1, dtype=np.uint32)[0])


def _jsonable(obj):
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return repr(f) if not math.isfinite(f) else f
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return repr(obj)


def stable_hash(obj) -> str:
    """SHA-256 (hex, 20 chars) of a canonical JSON rendering of ``obj`` (dict keys sorted)."""
    s = json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:20]


@functools.lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """SHA-256 (hex, 20 chars) of every ``.py`` file of the installed kfactors package.

    Part of every cache key, so that editing a rule, the engine or a strategy invalidates the
    cached results instead of silently serving stale ones.
    """
    import kfactors

    root = Path(kfactors.__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        h.update(str(f.relative_to(root)).encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()[:20]


def file_fingerprint(module_name: str) -> str:
    """SHA-256 (hex, 20 chars) of the source file of an importable module ('' if unknown)."""
    import importlib.util

    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        return ""
    return hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()[:20]


def to_json(obj, path) -> None:
    """Write ``obj`` as pretty JSON (NaN/inf become null; numpy scalars converted)."""
    def conv(o):
        if isinstance(o, Mapping):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return float(o) if math.isfinite(float(o)) else None
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return o

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(conv(obj), fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")


class Cache:
    """Pickle cache under ``root/<kind>/<key>.pkl`` (atomic writes; disabled -> no-op).

    Only the benchmark's own files are ever unpickled; do not point ``root`` at untrusted data.
    """

    def __init__(self, root, enabled: bool = True):
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.hits = 0
        self.misses = 0

    def path(self, kind: str, key: str) -> Path:
        return self.root / kind / f"{key}.pkl"

    def get(self, kind: str, key: str):
        """Cached object or ``None``."""
        if not self.enabled:
            return None
        p = self.path(kind, key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            with open(p, "rb") as fh:
                obj = pickle.load(fh)
        except Exception as exc:
            LOG.warning("ignoring unreadable cache entry %s (%s)", p, exc)
            self.misses += 1
            return None
        self.hits += 1
        return obj

    def put(self, kind: str, key: str, obj) -> None:
        if not self.enabled:
            return
        p = self.path(kind, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp_", suffix=".pkl")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, p)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


@contextlib.contextmanager
def timed(label: str, timings: list | None = None, **fields):
    """Log the wall time of a block (INFO) and append ``{step, seconds, **fields}`` to timings."""
    t0 = time.perf_counter()
    LOG.info("%s ...", label)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        LOG.info("%s done in %.2f s", label, dt)
        if timings is not None:
            timings.append({"step": label, "seconds": round(dt, 4), **fields})


def new_figure(figsize=(7, 4.5)):
    """A matplotlib Figure that is not registered with pyplot (no GUI, no leak)."""
    from matplotlib.figure import Figure

    return Figure(figsize=figsize, layout="constrained")


# ------------------------------------------------------------------------------------------
# lazy public API (the submodules import this package)
# ------------------------------------------------------------------------------------------
def __getattr__(name):
    if name == "run_sim":
        from kfactors.bench.sim_bench import run_sim
        return run_sim
    if name == "run_real":
        from kfactors.bench.real_bench import run_real
        return run_real
    if name == "build_leaderboard":
        from kfactors.bench.leaderboard import build_leaderboard
        return build_leaderboard
    if name == "main":
        from kfactors.bench.__main__ import main
        return main
    raise AttributeError(f"module 'kfactors.bench' has no attribute {name!r}")
