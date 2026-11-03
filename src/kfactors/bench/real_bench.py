"""Real-data benchmark: spectra, rolling k_t, report() and economic k-curves per panel.

For every panel block of the config (see :mod:`kfactors.bench` for the keys) this module

1. loads the panel (JKP factors, Ken French portfolios, Binance spot, a CSV file, or an offline
   synthetic panel built with :mod:`kfactors.sim`);
2. draws the eigenvalue spectrum of a representative estimation window with the naive
   Marchenko-Pastur and the Laloux et al. (1999, PRL 83(7)) market-adjusted edges
   (:func:`kfactors.plots.plot_spectrum`);
3. computes the number of factors chosen by every rule in rolling walk-forward windows
   (``rolling_k.csv`` + plot); the universe of every window is the engine's (tradable at the
   decision date, complete in the window, non-constant when standardising), so ``k_t`` uses
   rows ``<= t`` only;
4. runs :func:`kfactors.report.report` on the latest window (markdown + tables + figure);
5. for every experiment (strategy x walk-forward settings) runs the leakage-safe engine
   (:class:`kfactors.engine.WalkForward`) for every fixed k of the grid and every rule, and
   builds the k-curve with :func:`kfactors.evaluate.curve_from_results` (oracle k, bootstrap
   oracle confidence set, rule distances). Strategies whose ``weights`` raise
   :class:`NotImplementedError` are reported as *skipped*.

All outputs go to ``<out>/real/<panel>/``; experiment outputs to
``<out>/real/<panel>/experiments/<experiment>/``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import logging
import math
import time
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kfactors import __version__ as KF_VERSION
from kfactors.bench import (
    CACHE_SCHEMA,
    Cache,
    _check_keys,
    _merge,
    code_fingerprint,
    derive_seed,
    file_fingerprint,
    new_figure,
    parse_k_grid,
    resolve_rules,
    stable_hash,
    to_json,
)
from kfactors.data.panel import Panel

LOG = logging.getLogger("kfactors.bench")

_ROLLING_KEYS = {"enabled", "window", "step", "start", "end", "min_assets", "kmax", "standardize"}
_SPECTRUM_KEYS = {"enabled", "window", "end", "top"}
_REPORT_KEYS = {"enabled", "window", "n_boot", "kmax", "mean_block", "boot_n_perm",
                "boot_unique_rows"}


class StrategyNotImplemented(Exception):
    """Raised internally when a strategy's ``weights`` raises NotImplementedError."""


# ------------------------------------------------------------------------------------------
# panels
# ------------------------------------------------------------------------------------------
def synthetic_panel(args: dict, seed: int, name: str = "synthetic") -> Panel:
    """Offline return panel from the factor simulator (for CI, demos and sanity checks).

    Parameters
    ----------
    args : dict
        :class:`kfactors.sim.SimSpec` keywords (``n``, ``p``, ``k``, ``regime``, ``noise``,
        ``strengths``, ``market``, ...) plus

        * ``freq`` (``"M"`` month ends or ``"D"`` business days; default ``"M"``) and
          ``start`` (first date, default ``"2000-01-31"``);
        * ``scale`` (default 0.01): returns are ``scale * X`` (the simulator is on a unit-noise
          scale);
        * ``premia`` (scalar or one value per factor; default 0.1): factor means in units of the
          factor standard deviation, i.e. ``X = (F + premia) L' + E``. Non-zero premia give the
          factor-tangency strategy something to earn, so the k-curve has a meaningful optimum
          near the true k;
        * ``late_assets``, ``late_rows`` (default 0): the first ``late_assets`` columns are
          missing (NaN, not tradable) in the first ``late_rows`` rows, which exercises
          time-varying universes.
    seed : int
        Seed of :func:`kfactors.sim.simulate`.
    name : str
        Panel name (and the design name if ``args`` has none).

    Returns
    -------
    Panel
        ``tradable = returns.notna()``; ``meta`` holds the spec, the true ``k``, the seed and the
        population factor premia.
    """
    from kfactors.sim import SimSpec, simulate

    a = dict(args)
    freq = str(a.pop("freq", "M"))
    start = a.pop("start", "2000-01-31")
    scale = float(a.pop("scale", 0.01))
    premia = a.pop("premia", 0.1)
    late_assets = int(a.pop("late_assets", 0))
    late_rows = int(a.pop("late_rows", 0))
    ppy = a.pop("periods_per_year", None)
    for key in ("strengths", "garch", "hetero_range"):
        if a.get(key) is not None:
            a[key] = tuple(a[key])
    a.setdefault("name", name)
    try:
        spec = SimSpec(**a)
    except TypeError as exc:
        raise ValueError(f"synthetic panel {name!r}: {exc}") from None
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"synthetic panel {name!r}: scale must be > 0, got {scale}")
    data = simulate(spec, seed)
    k = data.F.shape[1]
    prem = np.asarray(premia, dtype=float).ravel()
    if prem.size == 1:
        prem = np.full(k, float(prem[0]))
    if prem.size != k:
        raise ValueError(f"synthetic panel {name!r}: premia has {prem.size} values for k={k}")
    X = data.X + (prem @ data.L.T if k else 0.0)
    R = scale * X
    n, p = R.shape
    if not (0 <= late_assets <= p and 0 <= late_rows < n):
        raise ValueError(f"synthetic panel {name!r}: need 0 <= late_assets <= p and "
                         f"0 <= late_rows < n")
    R[:late_rows, :late_assets] = np.nan
    if freq == "M":
        idx = pd.date_range(start, periods=n, freq="ME")
    elif freq == "D":
        idx = pd.date_range(start, periods=n, freq="B")
    else:
        raise ValueError(f"synthetic panel {name!r}: freq must be 'M' or 'D', got {freq!r}")
    cols = [f"A{j:03d}" for j in range(p)]
    ret = pd.DataFrame(R, index=pd.DatetimeIndex(idx, name="date"), columns=cols)
    meta = {"source": "synthetic", "k_true": int(data.k), "seed": int(seed),
            "spec": dataclasses.asdict(spec), "premia": prem.tolist(), "scale": scale,
            "units": f"simulated simple returns (scale {scale:g})"}
    if ppy is not None:
        meta["periods_per_year"] = int(ppy)
    return Panel(returns=ret, tradable=ret.notna(), name=name, freq=freq, meta=meta)


def _csv_panel(args: dict, name: str) -> Panel:
    a = dict(args)
    if "path" not in a:
        raise ValueError(f"csv panel {name!r} needs args.path")
    df = pd.read_csv(a.pop("path"), index_col=0, parse_dates=True)
    freq = str(a.pop("freq", "D"))
    meta = {"source": "csv"}
    if "periods_per_year" in a:
        meta["periods_per_year"] = int(a.pop("periods_per_year"))
    if a:
        raise ValueError(f"csv panel {name!r}: unknown args {sorted(a)}")
    df = df.sort_index().astype(float)
    df.index = pd.DatetimeIndex(df.index, name="date")
    return Panel(returns=df, tradable=df.notna(), name=name, freq=freq, meta=meta)


def load_panel(pcfg: dict, seed: int) -> Panel:
    """Build the :class:`~kfactors.data.panel.Panel` of a panel block (see module docstring)."""
    src, args, name = pcfg["source"], dict(pcfg.get("args") or {}), pcfg["name"]
    if src == "synthetic":
        return synthetic_panel(args, derive_seed(seed, "panel", name), name=name)
    if src == "csv":
        return _csv_panel(args, name)
    from kfactors.data import load_binance, load_french, load_jkp

    loader = {"jkp": load_jkp, "french": load_french, "binance": load_binance}[src]
    panel = loader(**args)
    panel.name = name
    return panel


def panel_fingerprint(panel: Panel) -> str:
    """SHA-256 (hex, 20 chars) of the returns, tradable mask, dates, asset ids and frequency."""
    h = hashlib.sha256()
    A = panel.returns.to_numpy(dtype=float, copy=True)
    A[np.isnan(A)] = np.nan          # canonical NaN bit pattern
    h.update(np.ascontiguousarray(A).tobytes())
    h.update(np.ascontiguousarray(panel.tradable.to_numpy(dtype=bool)).tobytes())
    h.update(panel.returns.index.asi8.tobytes())
    h.update("\x1f".join(map(str, panel.returns.columns)).encode("utf-8"))
    h.update(f"{panel.freq}|{panel.periods_per_year()}".encode())
    return h.hexdigest()[:20]


# ------------------------------------------------------------------------------------------
# windows (same universe rule as kfactors.engine, step 1 of the timing contract)
# ------------------------------------------------------------------------------------------
def _nan_cumsum(R: np.ndarray) -> np.ndarray:
    return np.vstack([np.zeros((1, R.shape[1]), dtype=np.int64),
                      np.cumsum(np.isnan(R), axis=0, dtype=np.int64)])


def window_universe(R: np.ndarray, T: np.ndarray, cum: np.ndarray, t: int, window: int,
                    standardize: bool) -> np.ndarray:
    """Column positions eligible at decision row ``t`` (uses rows ``t-window+1 .. t`` only).

    Tradable at ``t``, no NaN in the window and -- when standardising -- non-zero window variance
    (tolerance of :func:`kfactors.utils.preprocess`), exactly as in
    :class:`kfactors.engine.WalkForward`.
    """
    s = t - window + 1
    cols = np.flatnonzero(T[t] & ((cum[t + 1] - cum[s]) == 0))
    if standardize and cols.size:
        A = R[s:t + 1, cols]
        tol = 1e-12 * max(1.0, float(np.abs(A).max()))
        cols = cols[A.std(axis=0) > tol]
    return cols


def _row_range(index: pd.DatetimeIndex, window: int, start=None, end=None) -> tuple[int, int]:
    last = len(index) - 1
    if end is not None:
        last = int(index.searchsorted(pd.Timestamp(end), side="right")) - 1
    first = window - 1
    if start is not None:
        first = max(first, int(index.searchsorted(pd.Timestamp(start), side="left")))
    return first, last


def latest_window(panel: Panel, window: int, min_assets: int, standardize: bool,
                  end=None) -> tuple[int, np.ndarray]:
    """Last decision row ``t <= end`` whose universe has at least ``min_assets`` assets.

    Returns
    -------
    (t, cols) : (int, ndarray)

    Raises
    ------
    ValueError
        If no window qualifies.
    """
    R = panel.returns.to_numpy(dtype=float)
    T = panel.tradable.to_numpy(dtype=bool)
    cum = _nan_cumsum(R)
    first, last = _row_range(panel.returns.index, window, None, end)
    for t in range(last, first - 1, -1):
        cols = window_universe(R, T, cum, t, window, standardize)
        if cols.size >= min_assets:
            return t, cols
    raise ValueError(f"panel {panel.name!r}: no window of {window} rows with >= {min_assets} "
                     f"complete tradable assets")


def window_key(X) -> bytes:
    """Content key of a window (shape + float64 values, BLAKE2b-128); equal data -> equal key."""
    A = np.ascontiguousarray(X, dtype=float)
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(A.shape, dtype=np.int64).tobytes())
    h.update(A.tobytes())
    return h.digest()


def _rolling_chunk(R, T, ts, window, standardize, min_assets, rules, kmax, random_state,
                   with_keys: bool = False):
    from kfactors.rules import select_k

    cum = _nan_cumsum(R)
    rows, warn_counts, errors = [], Counter(), Counter()
    for t in ts:
        cols = window_universe(R, T, cum, t, window, standardize)
        row = {"t": int(t), "n_assets": int(cols.size)}
        if cols.size >= min_assets:
            X = R[t - window + 1:t + 1, cols]
            if with_keys:
                row["_key"] = window_key(X)
            for label, name, kw in rules:
                with warnings.catch_warnings(record=True) as wrec:
                    warnings.simplefilter("always")
                    try:
                        row[label] = float(select_k(X, name, kmax=kmax, standardize=standardize,
                                                    random_state=random_state, **kw).k)
                    except Exception as exc:
                        row[label] = math.nan
                        errors[(label, f"{type(exc).__name__}: {exc}"[:160])] += 1
                for w in wrec:
                    warn_counts[(label, str(w.message).splitlines()[0][:120])] += 1
        else:
            for label, _, _ in rules:
                row[label] = math.nan
        rows.append(row)
    return rows, warn_counts, errors


def rolling_k(panel: Panel, rules, *, window: int, step: int = 1, kmax: int = 20,
              standardize: bool = True, min_assets: int = 10, start=None, end=None,
              random_state=0, n_jobs: int = 1,
              return_keys: bool = False) -> tuple[pd.DataFrame, dict]:
    """Number of factors chosen by every rule in rolling windows of ``window`` rows.

    Decision rows are ``first, first + step, ...`` up to the last row ``<= end`` with
    ``first = max(window - 1, first row >= start)`` (the engine's grid when ``step`` equals the
    rebalance period). At every decision the rule sees only the window's rows and universe.

    Parameters
    ----------
    panel : Panel
    rules : list of (label, name, kwargs)
        From :func:`kfactors.bench.resolve_rules`.
    window, step, kmax, standardize, min_assets, start, end
        Walk-forward settings; windows with fewer than ``min_assets`` eligible assets get NaN.
    random_state : int
        Seed of stochastic rules (the same in every window, like the engine).
    n_jobs : int
        joblib processes over chunks of decision rows (results do not depend on it).
    return_keys : bool
        Also return ``info["k_by_key"] = {label: {window_key(window): k}}`` (used to hand the
        precomputed choices to the walk-forward engine, see :class:`PrecomputedRule`).

    Returns
    -------
    (DataFrame, dict)
        Index = decision dates; columns ``n_assets`` and one float column of ``k_t`` per rule
        label (NaN when skipped or when the rule failed). The dict has ``warnings`` and
        ``errors`` as ``{label: {message: count}}``.
    """
    R = panel.returns.to_numpy(dtype=float)
    T = panel.tradable.to_numpy(dtype=bool)
    first, last = _row_range(panel.returns.index, window, start, end)
    ts = list(range(first, last + 1, int(step)))
    if not ts:
        raise ValueError(f"panel {panel.name!r}: no rolling window of {window} rows in range")
    if n_jobs == 1:
        parts = [_rolling_chunk(R, T, ts, window, standardize, min_assets, rules, kmax,
                                random_state, return_keys)]
    else:
        from joblib import Parallel, delayed, effective_n_jobs

        n_chunks = max(1, min(len(ts), 4 * effective_n_jobs(n_jobs)))
        chunks = [c.tolist() for c in np.array_split(np.asarray(ts), n_chunks) if c.size]
        parts = Parallel(n_jobs=n_jobs)(
            delayed(_rolling_chunk)(R, T, c, window, standardize, min_assets, rules, kmax,
                                    random_state, return_keys) for c in chunks)
    rows = [r for part, _, _ in parts for r in part]
    warn, err = Counter(), Counter()
    for _, w, e in parts:
        warn.update(w)
        err.update(e)
    labels = [r[0] for r in rules]
    df = pd.DataFrame(rows, columns=["t", "n_assets"] + labels)
    df.index = pd.DatetimeIndex(panel.returns.index[df.pop("t").to_numpy()], name="date")
    k_by_key: dict = {lab: {} for lab in labels}
    if return_keys:
        for r in rows:
            if "_key" in r:
                for lab in labels:
                    if not math.isnan(r[lab]):
                        k_by_key[lab][r["_key"]] = int(r[lab])

    def nest(c: Counter) -> dict:
        out: dict = {}
        for (lab, msg), v in sorted(c.items()):
            out.setdefault(lab, {})[msg] = int(v)
        return out

    info = {"warnings": nest(warn), "errors": nest(err)}
    if return_keys:
        info["k_by_key"] = k_by_key
    return df, info


def rolling_summary(rk: pd.DataFrame) -> pd.DataFrame:
    """Per rule: windows, mean/median/sd/min/max of ``k_t`` and the number of changes of k."""
    rows = []
    for lab in rk.columns.drop("n_assets"):
        k = rk[lab].dropna()
        rows.append({"rule": lab, "n_windows": int(k.size),
                     "n_missing": int(rk[lab].isna().sum()),
                     "mean": k.mean(), "median": k.median(),
                     "sd": k.std(ddof=1) if k.size > 1 else math.nan,
                     "min": k.min(), "max": k.max(),
                     "n_changes": int((k.diff().fillna(0) != 0).sum())})
    return pd.DataFrame(rows)


def _plot_rolling(rk: pd.DataFrame, title: str, path: Path) -> None:
    import matplotlib as mpl

    labels = list(rk.columns.drop("n_assets"))
    fig = new_figure((11, 6.5))
    ax, ax2 = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1])
    cmap = mpl.colormaps["tab20"]
    styles = ["-", "--", "-.", ":"]
    # rules often agree; a small vertical offset per rule keeps every line visible
    m = len(labels)
    span = 0.3 if m > 1 else 0.0
    for i, lab in enumerate(labels):
        off = (i / (m - 1) - 0.5) * span if m > 1 else 0.0
        ax.step(rk.index, rk[lab] + off, where="post", lw=1.2, color=cmap(i % 20),
                ls=styles[(i // 20) % 4], label=lab)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("k_t chosen by the rule" + (" (lines offset by < 0.15)" if m > 1 else ""))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="x-small", ncol=2 if len(labels) > 8 else 1, loc="upper left")
    ax2.step(rk.index, rk["n_assets"], where="post", color="0.3", lw=1)
    ax2.set_ylabel("assets")
    ax2.set_xlabel("decision date")
    ax2.grid(True, alpha=0.3)
    fig.savefig(path, dpi=110)


# ------------------------------------------------------------------------------------------
# strategies and experiments
# ------------------------------------------------------------------------------------------
def load_strategy(path: str, kwargs: dict | None = None):
    """Instantiate ``"package.module:Class"`` with ``kwargs``."""
    mod, _, attr = str(path).partition(":")
    if not mod or not attr:
        raise ValueError(f"strategy must be 'module:Class', got {path!r}")
    cls = getattr(importlib.import_module(mod), attr)
    return cls(**(kwargs or {}))


class LabeledRule:
    """Callable ``k`` for the engine: ``select_k`` with fixed options (picklable).

    ``engine.run(LabeledRule(...))`` applies the rule inside every window exactly as the
    engine's own rule path does (same ``kmax``, ``standardize`` and ``random_state``), but with
    the rule options of a label such as ``"parallel(null=circular)"``.
    """

    def __init__(self, label: str, name: str, kwargs: dict, kmax: int, standardize: bool,
                 random_state):
        self.label, self.name, self.kwargs = label, name, dict(kwargs)
        self.kmax, self.standardize, self.random_state = int(kmax), bool(standardize), random_state
        self.__name__ = label

    def __call__(self, X) -> int:
        from kfactors.rules import select_k

        return int(select_k(X, self.name, kmax=self.kmax, standardize=self.standardize,
                            random_state=self.random_state, **self.kwargs).k)

    def __repr__(self) -> str:
        return f"LabeledRule({self.label!r})"


class PrecomputedRule(LabeledRule):
    """:class:`LabeledRule` that first looks the window up in a table of precomputed choices.

    ``table`` maps :func:`window_key` of a window to the rule's ``k``; it is filled by
    :func:`rolling_k` (``return_keys=True``) on exactly the engine's decision grid and universe,
    in parallel over windows. A window that is not in the table (never expected) is computed
    directly and counted in ``self.misses`` (per process).
    """

    def __init__(self, label, name, kwargs, kmax, standardize, random_state, table: dict):
        super().__init__(label, name, kwargs, kmax, standardize, random_state)
        self.table = dict(table)
        self.misses = 0

    def __call__(self, X) -> int:
        k = self.table.get(window_key(X))
        if k is None:
            self.misses += 1
            return super().__call__(X)
        return int(k)


def _walkforward_config(wf: dict, seed: int):
    from kfactors.engine import WalkForwardConfig

    names = {f.name for f in dataclasses.fields(WalkForwardConfig)}
    extra = sorted(set(wf) - names)
    if extra:
        raise ValueError(f"unknown walkforward key(s) {extra}; allowed: {sorted(names)}")
    wf = dict(wf)
    wf.setdefault("random_state", seed)
    return WalkForwardConfig(**wf)


def _spec_label(s) -> str:
    if isinstance(s, LabeledRule):
        return s.label
    return f"k={s}" if isinstance(s, int) else str(s)


def _probe_strategy(engine, k0: int) -> None:
    """Run the first executed decision only; NotImplementedError -> StrategyNotImplemented."""
    from kfactors.engine import WalkForward

    sched = engine.schedule()
    ex = sched[sched["executed"]]
    if ex.empty:
        raise ValueError(f"no decision has >= min_assets={engine.config.min_assets} eligible "
                         f"assets (max universe {int(sched['n_assets'].max())})")
    cfg = dataclasses.replace(engine.config, end=str(ex["hold_end"].iloc[0].date()))
    try:
        WalkForward(engine.panel, engine.strategy, cfg).run(k0)
    except NotImplementedError as exc:
        raise StrategyNotImplemented(str(exc) or "weights() not implemented") from None


def run_experiment(panel: Panel, exp: dict, *, rules, wf: dict, seed: int, n_boot: int,
                   alpha: float, mean_block, n_jobs: int = 1) -> dict:
    """Walk-forward k-curve of one strategy on one panel.

    Returns a dict with ``frame`` (:meth:`KCurve.to_frame`), ``curve``, ``rules`` (KCurve
    tables), ``oracle_k``, ``oracle_set``, ``metric``, ``meta`` (KCurve meta without the draw
    matrix), ``k_used`` (decision date x rule label), ``returns`` (holding date x spec, the
    metric's series), ``specs_summary`` (:meth:`BacktestResult.summary` per spec) and
    ``warnings`` (``{message: count}``: rule warnings of the precomputation plus warnings
    raised by in-process engine runs).

    Every rule's ``k`` at every decision is computed once, in parallel over windows, by
    :func:`rolling_k` on the engine's own decision grid and universe, and handed to the engine
    as a :class:`PrecomputedRule` callable (the engine's built-in ``run_many`` parallelises only
    over rules, which leaves the slow ``parallel`` rule on one core). The chosen ``k`` are
    identical to ``engine.run(rule_name)`` (tested).

    Raises
    ------
    StrategyNotImplemented
        If the strategy's ``weights`` raises NotImplementedError.
    """
    from kfactors.engine import WalkForward
    from kfactors.evaluate import METRICS, curve_from_results

    metric = exp.get("metric", "sharpe_net")
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; available: {sorted(METRICS)}")
    ks = parse_k_grid(exp["k_grid"])
    strat = load_strategy(exp["strategy"], exp.get("strategy_kwargs"))
    cfg = _walkforward_config(wf, seed)
    engine = WalkForward(panel, strat, cfg)
    _probe_strategy(engine, ks[0])

    # every rule's k at every decision, in parallel over windows (same grid and universe as
    # the engine; plain rule names get config.rule_kwargs like the engine's own rule path)
    eff = [(label, name, kw if kw else dict(cfg.rule_kwargs.get(name, {})))
           for label, name, kw in rules]
    t0 = time.perf_counter()
    _, rinfo = rolling_k(panel, eff, window=cfg.window, step=cfg.rebalance, kmax=cfg.kmax,
                         standardize=cfg.standardize, min_assets=cfg.min_assets,
                         start=cfg.start, end=cfg.end, random_state=cfg.random_state,
                         n_jobs=n_jobs, return_keys=True)
    LOG.info("    rule choices for %d rules precomputed in %.2f s", len(eff),
             time.perf_counter() - t0)
    spec_of = {label: PrecomputedRule(label, name, kw, cfg.kmax, cfg.standardize,
                                      cfg.random_state, rinfo["k_by_key"][label])
               for label, name, kw in eff}
    specs = list(ks) + list(spec_of.values())
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        try:
            if n_jobs == 1:
                results = {}
                for i, s in enumerate(specs, 1):
                    t0 = time.perf_counter()
                    results[s] = engine.run(s)
                    LOG.info("    [%d/%d] %s: %.2f s", i, len(specs), _spec_label(s),
                             time.perf_counter() - t0)
            else:
                results = engine.run_many(specs, n_jobs=n_jobs)
        except NotImplementedError as exc:
            raise StrategyNotImplemented(str(exc) or "weights() not implemented") from None
    by_label: dict = {k: results[k] for k in ks}
    for label, s in spec_of.items():
        by_label[label] = results[s]
    labels = list(spec_of)
    kc = curve_from_results(by_label, ks, labels, metric=metric, n_boot=n_boot, alpha=alpha,
                            mean_block=mean_block, random_state=seed)
    series = METRICS[metric].series
    k_used = pd.DataFrame({lab: by_label[lab].k_used for lab in labels})
    k_used.index.name = "date"
    rets = pd.DataFrame({_spec_label(s): getattr(by_label[s], series) for s in list(ks) + labels})
    rets.index.name = "date"
    summ = pd.DataFrame([{"spec": _spec_label(s), **by_label[s].summary(),
                          "n_clipped": by_label[s].info.get("n_clipped", 0)}
                         for s in list(ks) + labels])
    meta = {k: v for k, v in kc.meta.items() if k != "draws"}
    warn = Counter(f"{w.category.__name__}: {str(w.message).splitlines()[0][:150]}"
                   for w in wrec)
    for lab, msgs in rinfo["warnings"].items():
        for msg, cnt in msgs.items():
            warn[f"{lab}: {msg}"] += cnt
    misses = sum(s.misses for s in spec_of.values())   # in-process runs only (n_jobs == 1)
    if misses:
        LOG.warning("    %d window(s) were not in the precomputed rule table and were "
                    "recomputed (rule failures?)", misses)
    return {"frame": kc.to_frame(), "curve": kc.curve, "rules": kc.rules,
            "oracle_k": int(kc.oracle_k), "oracle_set": [int(k) for k in kc.oracle_set],
            "metric": kc.metric, "meta": meta, "k_used": k_used, "returns": rets,
            "specs_summary": summ, "warnings": dict(warn), "k_grid": ks,
            "n_table_misses": int(misses)}


def _plot_kcurve(res: dict, title: str, path: Path) -> None:
    from kfactors.plots import plot_k_curve

    ns = SimpleNamespace(curve=res["curve"], rules=res["rules"], oracle_k=res["oracle_k"],
                         oracle_set=res["oracle_set"], metric=res["metric"])
    fig = new_figure((9, 5))
    plot_k_curve(ns, ax=fig.subplots(), title=title)
    fig.savefig(path, dpi=110)


def _write_experiment(res: dict, edir: Path, title: str) -> None:
    res["frame"].to_csv(edir / "kcurve.csv", index=False)
    res["curve"].to_csv(edir / "curve.csv")
    res["rules"].to_csv(edir / "rules.csv")
    res["k_used"].to_csv(edir / "k_used.csv")
    res["returns"].to_csv(edir / "returns.csv")
    res["specs_summary"].to_csv(edir / "specs_summary.csv", index=False)
    _plot_kcurve(res, title, edir / "kcurve.png")


# ------------------------------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------------------------------
class _Status:
    def __init__(self, fail_fast: bool):
        self.rows: list[dict] = []
        self.fail_fast = fail_fast

    def add(self, panel, step, status, seconds=math.nan, message=""):
        self.rows.append({"panel": panel, "step": step, "status": status,
                          "seconds": round(seconds, 3) if math.isfinite(seconds) else seconds,
                          "message": message})

    @property
    def n_failed(self) -> int:
        return sum(r["status"] == "failed" for r in self.rows)


def _step(status: _Status, panel: str, step: str, fn):
    """Run ``fn()``; record ok/failed with timing; re-raise only in fail-fast mode."""
    t0 = time.perf_counter()
    LOG.info("[%s] %s ...", panel, step)
    try:
        out = fn()
    except Exception as exc:
        dt = time.perf_counter() - t0
        msg = f"{type(exc).__name__}: {exc}"
        LOG.error("[%s] %s FAILED after %.2f s: %s", panel, step, dt, msg)
        status.add(panel, step, "failed", dt, msg[:500])
        if status.fail_fast:
            raise
        return None
    dt = time.perf_counter() - t0
    LOG.info("[%s] %s done in %.2f s", panel, step, dt)
    status.add(panel, step, "ok", dt)
    return out


def run_real(cfg: dict, out, *, n_jobs: int | None = None, panels=None, use_cache: bool = True,
             fail_fast: bool = False) -> dict:
    """Run the real-data benchmark of a loaded config (see :mod:`kfactors.bench`).

    Parameters
    ----------
    cfg : dict
        Output of :func:`kfactors.bench.load_config` (needs a ``real`` section).
    out : path
        Results root; files go to ``out/real/<panel>/``, the cache to ``out/cache/``.
    n_jobs : int, optional
        joblib processes for the rolling ``k_t`` and the engine runs (default ``cfg["n_jobs"]``).
    panels : list of str, optional
        Only these panel names.
    use_cache : bool
        Reuse cached rolling ``k_t``, reports and k-curves (keys include the data fingerprint).
    fail_fast : bool
        Re-raise the first failure instead of recording it and continuing.

    Returns
    -------
    dict
        ``{"status": DataFrame (panel, step, status, seconds, message), "n_failed": int,
        "n_skipped": int}``. Skipped = strategies that are not implemented yet.
    """
    if "real" not in cfg:
        raise ValueError("the config has no 'real' section")
    rc = cfg["real"]
    seed = int(cfg["seed"])
    n_jobs = int(cfg["n_jobs"] if n_jobs is None else n_jobs)
    cache = Cache(Path(out) / "cache", enabled=use_cache)
    blocks = rc["panels"]
    if panels:
        unknown = sorted(set(panels) - {p["name"] for p in blocks})
        if unknown:
            raise ValueError(f"unknown panel(s) {unknown}; config has "
                             f"{[p['name'] for p in blocks]}")
        blocks = [p for p in blocks if p["name"] in set(panels)]
    root = Path(out) / "real"
    root.mkdir(parents=True, exist_ok=True)
    status = _Status(fail_fast)
    n_skipped = 0
    t_all = time.perf_counter()

    for pc in blocks:
        name = pc["name"]
        pdir = root / name
        pdir.mkdir(parents=True, exist_ok=True)
        panel = _step(status, name, "load", lambda pc=pc: load_panel(pc, seed))
        if panel is None:
            continue
        fp = panel_fingerprint(panel)
        LOG.info("[%s] %d rows x %d assets, %s .. %s, freq %s (fingerprint %s)", name,
                 *panel.shape, panel.returns.index[0].date(), panel.returns.index[-1].date(),
                 panel.freq, fp)
        to_json({"name": name, "source": pc["source"], "args": pc.get("args") or {},
                 "n_rows": panel.shape[0], "n_assets": panel.shape[1],
                 "first_date": panel.returns.index[0], "last_date": panel.returns.index[-1],
                 "freq": panel.freq, "periods_per_year": panel.periods_per_year(),
                 "fingerprint": fp, "share_missing": float(panel.returns.isna().mean().mean()),
                 "meta": panel.meta}, pdir / "panel.json")

        wf = _merge(dataclasses.asdict(_walkforward_config({}, seed)), rc["walkforward"],
                    pc.get("walkforward"))
        rules = resolve_rules(pc.get("rules", rc["rules"]))
        labels = [r[0] for r in rules]

        # ---- spectrum of a representative window
        sp = _merge(rc["spectrum"], pc.get("spectrum"))
        _check_keys(sp, _SPECTRUM_KEYS, f"{name}.spectrum")
        if sp.get("enabled", True):
            def do_spectrum(sp=sp, panel=panel, wf=wf, pdir=pdir, name=name):
                from kfactors.plots import plot_spectrum, spectrum_edges
                from kfactors.utils import eigvals_desc, preprocess

                win = int(sp.get("window") or wf["window"])
                t, cols = latest_window(panel, win, wf["min_assets"], wf["standardize"],
                                        sp.get("end"))
                X = panel.returns.to_numpy(dtype=float)[t - win + 1:t + 1, cols]
                ev = eigvals_desc(preprocess(X, standardize=wf["standardize"])[0])
                d0, d1 = panel.returns.index[t - win + 1], panel.returns.index[t]
                fig = new_figure((8, 5))
                plot_spectrum(ev, win, cols.size, ax=fig.subplots(), top=sp.get("top", 50),
                              title=f"{name}: {d0.date()} .. {d1.date()} "
                                    f"(n={win}, p={cols.size})")
                fig.savefig(pdir / "spectrum.png", dpi=110)
                to_json({"window_start": d0, "window_end": d1, "n": win, "p": int(cols.size),
                         "standardize": wf["standardize"],
                         "edges": spectrum_edges(ev, win, int(cols.size)),
                         "eigenvalues_top": ev[: min(50, ev.size)]}, pdir / "spectrum.json")

            _step(status, name, "spectrum", do_spectrum)

        # ---- rolling k_t of every rule
        ro = _merge({"window": wf["window"], "step": wf["rebalance"], "start": wf["start"],
                     "end": wf["end"], "min_assets": wf["min_assets"], "kmax": wf["kmax"],
                     "standardize": wf["standardize"]}, rc["rolling"], pc.get("rolling"))
        _check_keys(ro, _ROLLING_KEYS, f"{name}.rolling")
        if ro.pop("enabled", True):
            def do_rolling(ro=ro, panel=panel, rules=rules, pdir=pdir, name=name, fp=fp):
                key = stable_hash({"schema": CACHE_SCHEMA, "v": KF_VERSION,
                                   "code": code_fingerprint(), "fp": fp,
                                   "rolling": ro, "rules": rules, "seed": seed})
                got = cache.get("rolling", key)
                if got is None:
                    got = rolling_k(panel, rules, random_state=derive_seed(seed, "rolling"),
                                    n_jobs=n_jobs, **ro)
                    cache.put("rolling", key, got)
                else:
                    LOG.info("[%s] rolling k_t from cache", name)
                rk, info = got
                rk.to_csv(pdir / "rolling_k.csv")
                rolling_summary(rk).to_csv(pdir / "rolling_k_summary.csv", index=False)
                to_json({**ro, "rules": [r[0] for r in rules], **info},
                        pdir / "rolling_k_info.json")
                if info["errors"]:
                    LOG.warning("[%s] rolling k_t: rule failures %s", name, info["errors"])
                _plot_rolling(rk, f"{name}: k_t by rule (window {ro['window']}, "
                                  f"step {ro['step']})", pdir / "rolling_k.png")

            _step(status, name, "rolling_k", do_rolling)

        # ---- report() on the latest window
        rp = _merge(rc["report"], pc.get("report"))
        _check_keys(rp, _REPORT_KEYS, f"{name}.report")
        if rp.pop("enabled", True):
            def do_report(rp=rp, panel=panel, wf=wf, pdir=pdir, name=name, fp=fp,
                          labels=labels):
                from kfactors.report import report

                win = int(rp.pop("window", None) or wf["window"])
                t, cols = latest_window(panel, win, wf["min_assets"], wf["standardize"])
                X = panel.returns.iloc[t - win + 1:t + 1, cols]
                key = stable_hash({"schema": CACHE_SCHEMA, "v": KF_VERSION,
                                   "code": code_fingerprint(), "fp": fp,
                                   "report": rp, "win": win, "t": t, "rules": labels,
                                   "std": wf["standardize"], "seed": seed})
                rep = cache.get("report", key)
                if rep is None:
                    rep = report(X, rules=labels, standardize=wf["standardize"],
                                 random_state=derive_seed(seed, "report", name), **rp)
                    cache.put("report", key, rep)
                d0, d1 = X.index[0].date(), X.index[-1].date()
                md = (f"<!-- panel {name}, window {d0} .. {d1} (n={win}, p={cols.size}) -->\n\n"
                      + rep.to_markdown())
                (pdir / "report.md").write_text(md, encoding="utf-8")
                rep.rules.to_csv(pdir / "report_rules.csv")
                rep.k_distribution.to_csv(pdir / "report_k_distribution.csv")
                to_json({"window_start": d0, "window_end": d1, "n": win, "p": int(cols.size),
                         "diagnostics": rep.diagnostics, "recommended": rep.recommended,
                         "info": rep.info}, pdir / "report.json")
                fig = new_figure((12, 4.5))
                rep.plot(axes=fig.subplots(1, 2))
                fig.savefig(pdir / "report.png", dpi=110)

            _step(status, name, "report", do_report)

        # ---- k-curves
        for exp in pc.get("experiments") or []:
            ename = exp["name"]
            edir = pdir / "experiments" / ename
            edir.mkdir(parents=True, exist_ok=True)
            ewf = _merge(wf, exp.get("walkforward"))
            erules = resolve_rules(exp.get("rules", pc.get("rules", rc["rules"])))
            opts = {"n_boot": int(exp.get("n_boot", rc["n_boot"])),
                    "alpha": float(exp.get("alpha", rc["alpha"])),
                    "mean_block": exp.get("mean_block", rc["mean_block"])}
            summary = {"panel": name, "experiment": ename, "strategy": exp["strategy"],
                       "strategy_kwargs": exp.get("strategy_kwargs") or {},
                       "metric": exp.get("metric", "sharpe_net"),
                       "k_grid": parse_k_grid(exp["k_grid"]), "rules": [r[0] for r in erules],
                       "walkforward": ewf, **opts, "seed": seed, "fingerprint": fp}
            step = f"kcurve:{ename}"
            t0 = time.perf_counter()
            LOG.info("[%s] %s (%s, %d k values + %d rules) ...", name, step, exp["strategy"],
                     len(summary["k_grid"]), len(erules))
            key = stable_hash({"schema": CACHE_SCHEMA, "v": KF_VERSION, "code": code_fingerprint(),
                               "strategy_src": file_fingerprint(
                                   str(exp["strategy"]).partition(":")[0]), **summary})
            res = cache.get("kcurve", key)
            cached = res is not None
            try:
                if res is None:
                    res = run_experiment(panel, exp, rules=erules, wf=ewf, seed=seed,
                                         n_jobs=n_jobs, **opts)
                    cache.put("kcurve", key, res)
                title = (f"{name} / {ename}: {res['metric']} vs k "
                         f"(oracle k={res['oracle_k']})")
                _write_experiment(res, edir, title)
            except StrategyNotImplemented as exc:
                dt = time.perf_counter() - t0
                msg = (f"strategy {exp['strategy']} is not implemented yet "
                       f"(NotImplementedError: {exc}); experiment skipped")
                LOG.warning("[%s] %s SKIPPED: %s", name, step, msg)
                status.add(name, step, "skipped", dt, msg)
                to_json({**summary, "status": "skipped", "message": msg,
                         "seconds": round(dt, 3)}, edir / "summary.json")
                n_skipped += 1
                continue
            except Exception as exc:
                dt = time.perf_counter() - t0
                msg = f"{type(exc).__name__}: {exc}"
                LOG.error("[%s] %s FAILED after %.2f s: %s", name, step, dt, msg)
                status.add(name, step, "failed", dt, msg[:500])
                to_json({**summary, "status": "failed", "message": msg[:2000],
                         "seconds": round(dt, 3)}, edir / "summary.json")
                if fail_fast:
                    raise
                continue
            dt = time.perf_counter() - t0
            if res["warnings"]:
                top = sorted(res["warnings"].items(), key=lambda kv: -kv[1])[:3]
                LOG.info("[%s] %s: %d warnings in-process, most frequent: %s", name, step,
                         sum(res["warnings"].values()), top)
            meta = res["meta"]
            to_json({**summary, "status": "ok", "message": "", "cached": cached,
                     "seconds": round(dt, 3), "oracle_k": res["oracle_k"],
                     "oracle_set": res["oracle_set"], "metric_name": res["metric"],
                     "n_common": meta.get("n_common"), "common_start": meta.get("common_start"),
                     "common_end": meta.get("common_end"),
                     "periods_per_year": meta.get("periods_per_year"),
                     "strategy_name": meta.get("strategy"), "warnings": res["warnings"]},
                    edir / "summary.json")
            status.add(name, step, "ok", dt, "cached" if cached else "")
            LOG.info("[%s] %s done in %.2f s%s: oracle k=%d, set=%s", name, step, dt,
                     " (cache)" if cached else "", res["oracle_k"], res["oracle_set"])

    st = pd.DataFrame(status.rows, columns=["panel", "step", "status", "seconds", "message"])
    st.to_csv(root / "status.csv", index=False)
    to_json({"config": cfg["name"], "seed": seed, "n_jobs": n_jobs,
             "kfactors_version": KF_VERSION, "seconds": round(time.perf_counter() - t_all, 3),
             "cache_hits": cache.hits, "cache_misses": cache.misses,
             "n_failed": status.n_failed, "n_skipped": n_skipped,
             "panels": [p["name"] for p in blocks]}, root / "run_info.json")
    return {"status": st, "n_failed": status.n_failed, "n_skipped": n_skipped}


__all__ = ["LabeledRule", "latest_window", "load_panel", "load_strategy", "panel_fingerprint",
           "rolling_k", "rolling_summary", "run_experiment", "run_real", "synthetic_panel",
           "window_universe"]
