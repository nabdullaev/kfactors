"""k-curves: what the choice of k costs for a given strategy/loss.

For a strategy and a panel, :func:`k_curve` runs the walk-forward engine for every fixed ``k`` of
a grid and for every selection rule, and scores each run with an economic metric (default: net
annualised Sharpe ratio). Uncertainty is quantified with the stationary block bootstrap of
Politis & Romano (1994, JASA 89(428)) using **one shared index matrix** for all configurations,
so that bootstrap draws are paired across k (the dependence between the return series of
different k is preserved; Ledoit & Wolf 2008, J. Empirical Finance 15(5), make the same point
for Sharpe-ratio comparisons).

* **Oracle k**: the k of the grid with the best realised metric (best in hindsight; not
  attainable ex ante, it is the yardstick).
* **Oracle confidence set**: for every bootstrap draw take the argmax over the grid; order the
  grid by how often it is the argmax and keep the smallest prefix that covers at least
  ``1 - alpha`` of the draws. This is a bootstrap "confidence set for the argmax" in the spirit of
  the model confidence set of Hansen, Lunde & Nason (2011), Econometrica 79(2), and of the
  rank confidence sets of Mogstad, Romano, Shaikh & Wilhelm (2024), REStud 91(1); it is a
  simple percentile construction, not either of those procedures.
* Each rule is scored by ``dist_to_oracle = |k_median - oracle_k|``, by whether its median k
  lies in the oracle set, and by its regret ``metric(oracle_k) - metric(rule)``. The regret
  interval (``regret_low`` / ``regret_high``) is *paired*: percentiles of the per-draw
  differences on the shared bootstrap indices. The ``low`` / ``high`` columns, in contrast, are
  marginal intervals of each configuration's metric; overlapping marginal intervals of
  positively correlated configurations do not mean the difference is insignificant.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kfactors.engine import BacktestResult, WalkForward
from kfactors.metrics import stationary_bootstrap_indices

__all__ = ["METRICS", "KCurve", "MetricSpec", "bootstrap_argmax", "curve_from_results",
           "k_curve", "oracle_confidence_set", "register_metric"]

# --------------------------------------------------------------------------- metric registry


@dataclass(frozen=True)
class MetricSpec:
    """A k-curve metric (higher is better).

    Attributes
    ----------
    name : str
    series : {"net", "gross"}
        Which :class:`~kfactors.engine.BacktestResult` return series it is computed on.
    fn : callable
        ``fn(M, periods_per_year) -> ndarray (B,)`` evaluated on every row of a ``B x T`` matrix
        of returns (row = one bootstrap resample; NaN must be ignored).
    description : str
    """

    name: str
    series: str
    fn: Callable[[np.ndarray, int], np.ndarray]
    description: str = ""


def _row_moments(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NaN-aware row mean, sd (ddof=1) and a degenerate-row mask (fewer than 2 obs or constant)."""
    ok = ~np.isnan(M)
    cnt = ok.sum(axis=1)
    X = np.where(ok, M, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = X.sum(axis=1) / cnt
        dev = np.where(ok, M - mu[:, None], 0.0)
        sd = np.sqrt((dev ** 2).sum(axis=1) / (cnt - 1))
    hi = np.where(ok, M, -np.inf).max(axis=1)
    lo = np.where(ok, M, np.inf).min(axis=1)
    bad = (cnt < 2) | (hi == lo) | ~(sd > 0)
    return mu, sd, bad


def _sharpe_rows(M: np.ndarray, ppy: int) -> np.ndarray:
    """Annualised Sharpe ratio of every row (same definition as :func:`kfactors.metrics.sharpe`:
    mean / sd(ddof=1) * sqrt(ppy); NaN for constant rows or fewer than 2 observations)."""
    mu, sd, bad = _row_moments(M)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = mu / sd * math.sqrt(ppy)
    out[bad] = np.nan
    return out


def _neg_vol_rows(M: np.ndarray, ppy: int) -> np.ndarray:
    """Minus the annualised volatility of every row (sd ddof=1 * sqrt(ppy)).

    NaN for degenerate rows (fewer than 2 observations or constant, e.g. a strategy that holds
    nothing such as ``k = 0`` with ``TangencyPCA``): a zero-volatility flat book is not a
    "least risky" portfolio, and scoring it ``-0.0`` would make it the oracle. The metric is
    meaningful only for strategies that are always invested (e.g. minimum variance with
    ``sum w = 1``); a configuration that is flat part of the time still has a lower vol.
    """
    _mu, sd, bad = _row_moments(M)
    out = -sd * math.sqrt(ppy)
    out[bad] = np.nan
    return out


METRICS: dict[str, MetricSpec] = {
    "sharpe_net": MetricSpec("sharpe_net", "net", _sharpe_rows,
                             "annualised Sharpe ratio of net (after-cost) returns"),
    "sharpe_gross": MetricSpec("sharpe_gross", "gross", _sharpe_rows,
                               "annualised Sharpe ratio of gross (before-cost) returns"),
    "neg_vol": MetricSpec("neg_vol", "net", _neg_vol_rows,
                          "minus annualised volatility of net returns (risk strategies, e.g. "
                          "minimum variance: higher = less risk; NaN for a constant/flat "
                          "series; only meaningful for always-invested strategies)"),
}


def register_metric(name: str, fn: Callable, *, series: str = "net", description: str = "",
                    vectorized: bool = True, overwrite: bool = False) -> MetricSpec:
    """Add a metric to :data:`METRICS` (higher must be better).

    Parameters
    ----------
    name : str
    fn : callable
        If ``vectorized``: ``fn(M: ndarray (B, T), periods_per_year) -> ndarray (B,)``.
        Otherwise ``fn(x: ndarray (T,), periods_per_year) -> float``, applied row by row
        (slower). NaN in the input must be ignored.
    series : {"net", "gross"}
    description : str
    vectorized : bool
    overwrite : bool
        Replacing an existing name raises ``ValueError`` unless True.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("metric name must be a non-empty string")
    if name in METRICS and not overwrite:
        raise ValueError(f"metric {name!r} already registered (pass overwrite=True)")
    if series not in ("net", "gross"):
        raise ValueError(f"series must be 'net' or 'gross', got {series!r}")
    if not callable(fn):
        raise TypeError("fn must be callable")
    if vectorized:
        vfn = fn
    else:
        def vfn(M, ppy, _f=fn):
            return np.array([float(_f(row, ppy)) for row in M], dtype=float)
    spec = MetricSpec(name, series, vfn, description)
    METRICS[name] = spec
    return spec


def _get_metric(metric) -> MetricSpec:
    if isinstance(metric, MetricSpec):
        return metric
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; available: {sorted(METRICS)}")
    return METRICS[metric]


# --------------------------------------------------------------------------- oracle helpers


def bootstrap_argmax(draws: np.ndarray, ks) -> np.ndarray:
    """Argmax k of every bootstrap draw.

    Parameters
    ----------
    draws : ndarray (n_k, n_boot)
        Metric of every grid point (rows, in the order of ``ks``) in every draw.
    ks : sequence of int
        The grid.

    Returns
    -------
    ndarray (n_boot,) of float
        The k with the largest finite value in each draw (ties -> smallest k); NaN when every
        value of the draw is non-finite.
    """
    D = np.asarray(draws, dtype=float)
    ks = np.asarray(list(ks))
    if D.ndim != 2 or D.shape[0] != ks.size:
        raise ValueError(f"draws must have shape (len(ks)={ks.size}, n_boot), got {D.shape}")
    order = np.argsort(ks, kind="stable")
    D, ks = D[order], ks[order]
    Dm = np.where(np.isfinite(D), D, -np.inf)
    arg = np.argmax(Dm, axis=0)           # first max -> smallest k on ties
    out = ks[arg].astype(float)
    out[~np.isfinite(D).any(axis=0)] = np.nan
    return out


def oracle_confidence_set(argmax_k, alpha: float = 0.10, oracle_k: int | None = None
                          ) -> tuple[list[int], pd.Series]:
    """Smallest set of k covering at least ``1 - alpha`` of the bootstrap argmax draws.

    Grid points are added in decreasing order of argmax frequency (ties: closer to ``oracle_k``
    first, then smaller k) until the cumulative frequency reaches ``1 - alpha``.

    Parameters
    ----------
    argmax_k : array-like (n_boot,)
        Output of :func:`bootstrap_argmax`; NaN draws are ignored.
    alpha : float in (0, 1)
    oracle_k : int, optional
        Used only to break frequency ties.

    Returns
    -------
    (list of int, pandas.Series)
        The set (sorted ascending) and the argmax frequencies (index k, sorted by k).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    a = np.asarray(argmax_k, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("no finite bootstrap argmax draws")
    vals, counts = np.unique(a.astype(np.int64), return_counts=True)
    freq = pd.Series(counts / a.size, index=pd.Index(vals, name="k"), name="argmax_freq")
    ref = oracle_k if oracle_k is not None else 0
    order = sorted(range(vals.size), key=lambda i: (-counts[i], abs(vals[i] - ref), vals[i]))
    need = (1.0 - alpha) * a.size
    chosen, acc = [], 0
    for i in order:
        chosen.append(int(vals[i]))
        acc += counts[i]
        if acc >= need - 1e-9:
            break
    return sorted(chosen), freq


# --------------------------------------------------------------------------- KCurve


@dataclass
class KCurve:
    """Economic k-curve of one strategy on one panel.

    Attributes
    ----------
    curve : pandas.DataFrame
        Index ``k`` (the fixed grid); columns ``metric`` (point estimate), ``low``/``high``
        (marginal bootstrap percentile interval of level ``1 - alpha``), ``se`` (bootstrap sd),
        ``argmax_freq`` (share of draws in which this k is the best), ``k_used_mean`` (mean
        k actually used: smaller than k when the engine clipped k to the window rank) and
        ``regret_low`` / ``regret_high`` (paired interval of ``metric(oracle_k) - metric(k)``).
    rules : pandas.DataFrame
        Index ``rule``; columns ``k_mean``, ``k_median`` (of the k chosen across decisions),
        ``metric``, ``low``, ``high``, ``se``, ``dist_to_oracle`` (``|k_median - oracle_k|``),
        ``in_set`` (median k, rounded half up, is in the oracle set), ``regret``
        (``metric(oracle_k) - metric(rule)``) and ``regret_low`` / ``regret_high``: the paired
        ``alpha/2`` and ``1 - alpha/2`` percentiles of the per-draw differences
        ``draws[oracle_k] - draws[rule]`` over the shared bootstrap indices (non-finite draws
        dropped). The interval treats ``oracle_k`` as fixed (it is chosen on the same data, so
        it is conditional on that choice).
    oracle_k : int
        Argmax of ``curve.metric`` (ties -> smallest k).
    oracle_set : list of int
        See :func:`oracle_confidence_set`.
    metric : str
    results : dict
        ``k_spec -> BacktestResult``.
    meta : dict
        ``series``, ``n_boot``, ``alpha``, ``mean_block``, ``random_state``,
        ``periods_per_year``, ``strategy``, ``common_start``, ``common_end``, ``n_common``,
        ``n_valid_draws`` (draws with a finite argmax), ``draws`` (DataFrame spec x n_boot of
        bootstrap metric values) and ``n_dropped`` (rows outside the common range, per spec).
    """

    curve: pd.DataFrame          # index k: metric, low, high, ..., regret_low, regret_high
    rules: pd.DataFrame          # index rule: k_mean, ..., in_set, regret, regret_low, regret_high
    oracle_k: int
    oracle_set: list[int]
    metric: str
    results: dict                # k_spec -> BacktestResult
    meta: dict

    def plot(self, ax=None):
        """Plot the k-curve with bootstrap bands, the oracle confidence set and rule markers.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created if None.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on (use ``.figure.savefig(...)``). Never calls ``plt.show``.

        See Also
        --------
        kfactors.plots.plot_k_curve
        """
        from kfactors.plots import plot_k_curve
        return plot_k_curve(self, ax=ax)

    def to_frame(self) -> pd.DataFrame:
        """Long table, one row per configuration (fixed k first, then rules).

        Columns: ``kind`` ("fixed" | "rule"), ``spec`` (k or rule name), ``k`` (k for fixed,
        median k for rules), ``k_mean``, ``metric``, ``low``, ``high``, ``se``,
        ``dist_to_oracle``, ``in_set``, ``is_oracle``, ``regret``, ``regret_low``,
        ``regret_high`` (paired interval; NaN if the KCurve predates these columns), plus the
        ``metric`` name in column ``metric_name``.
        """
        best = float(self.curve.loc[self.oracle_k, "metric"])
        oset = set(self.oracle_set)
        rows = []
        for k, r in self.curve.iterrows():
            rows.append({"kind": "fixed", "spec": int(k), "k": float(k),
                         "k_mean": float(r["k_used_mean"]) if "k_used_mean" in r else float(k),
                         "metric": r["metric"], "low": r["low"], "high": r["high"],
                         "se": r.get("se", np.nan),
                         "dist_to_oracle": float(abs(int(k) - self.oracle_k)),
                         "in_set": int(k) in oset, "is_oracle": int(k) == self.oracle_k,
                         "regret": best - r["metric"],
                         "regret_low": r.get("regret_low", np.nan),
                         "regret_high": r.get("regret_high", np.nan)})
        for name, r in self.rules.iterrows():
            rows.append({"kind": "rule", "spec": name, "k": r["k_median"], "k_mean": r["k_mean"],
                         "metric": r["metric"], "low": r["low"], "high": r["high"],
                         "se": r.get("se", np.nan), "dist_to_oracle": r["dist_to_oracle"],
                         "in_set": bool(r["in_set"]), "is_oracle": False,
                         "regret": r["regret"], "regret_low": r.get("regret_low", np.nan),
                         "regret_high": r.get("regret_high", np.nan)})
        out = pd.DataFrame(rows, columns=["kind", "spec", "k", "k_mean", "metric", "low", "high",
                                          "se", "dist_to_oracle", "in_set", "is_oracle",
                                          "regret", "regret_low", "regret_high"])
        out["metric_name"] = self.metric
        return out


def _check_grid(k_grid) -> list[int]:
    ks = []
    for k in k_grid:
        if isinstance(k, bool) or not isinstance(k, numbers.Integral) or k < 0:
            raise ValueError(f"k_grid must contain non-negative integers, got {k!r}")
        ks.append(int(k))
    if not ks:
        raise ValueError("k_grid is empty")
    if len(set(ks)) != len(ks):
        raise ValueError(f"k_grid has duplicates: {ks}")
    return sorted(ks)


def curve_from_results(results: dict, k_grid: Iterable[int], rules: Iterable[str] = (),
                       metric: str = "sharpe_net", n_boot: int = 1000, alpha: float = 0.10,
                       mean_block: float | None = None, random_state=0) -> KCurve:
    """Build a :class:`KCurve` from already computed backtests (used by :func:`k_curve`).

    Parameters
    ----------
    results : dict
        ``{k_spec: BacktestResult}``; must contain every k of ``k_grid`` and every rule.
    k_grid, rules, metric, n_boot, alpha, mean_block, random_state
        As in :func:`k_curve`.

    Notes
    -----
    All series are restricted to the dates common to every result (the intersection of the
    holding dates), and one stationary-bootstrap index matrix
    (:func:`kfactors.metrics.stationary_bootstrap_indices`, mean block ``n ** (1/3)`` by
    default) is drawn for that range and applied to every configuration. Regret intervals are
    computed from the paired per-draw differences against ``oracle_k``.
    """
    ks = _check_grid(k_grid)
    rules = list(rules)
    for r in rules:
        if not isinstance(r, str):
            raise TypeError(f"rules must be rule names (str), got {r!r}")
    if len(set(rules)) != len(rules):
        raise ValueError(f"rules has duplicates: {rules}")
    spec = _get_metric(metric)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if isinstance(n_boot, bool) or not isinstance(n_boot, numbers.Integral) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    names = list(ks) + rules
    missing = [s for s in names if s not in results]
    if missing:
        raise KeyError(f"results lack specs {missing}")
    res = [results[s] for s in names]
    for s, r in zip(names, res):
        if not isinstance(r, BacktestResult):
            raise TypeError(f"results[{s!r}] is not a BacktestResult")
    ppys = {r.periods_per_year for r in res}
    if len(ppys) != 1:
        raise ValueError(f"results have different periods_per_year: {sorted(ppys)}")
    ppy = ppys.pop()

    series = [getattr(r, spec.series) for r in res]
    common = series[0].index
    for s in series[1:]:
        common = common.intersection(s.index)
    common = common.sort_values()
    n = len(common)
    if n < 2:
        raise ValueError(f"the results share only {n} holding dates; need >= 2")
    M = np.vstack([s.reindex(common).to_numpy(dtype=float) for s in series])  # n_spec x n
    mb = max(1.0, n ** (1.0 / 3.0)) if mean_block is None else float(mean_block)
    idx = stationary_bootstrap_indices(n, int(n_boot), mb, random_state)

    point = spec.fn(M, ppy)
    draws = np.empty((len(names), idx.shape[0]))
    chunk = max(1, int(2_000_000 // max(n, 1)))
    for i in range(len(names)):
        x = M[i]
        for a in range(0, idx.shape[0], chunk):
            draws[i, a:a + chunk] = spec.fn(x[idx[a:a + chunk]], ppy)
    with np.errstate(invalid="ignore"):
        fin = np.where(np.isfinite(draws), draws, np.nan)
        q = np.full((len(names), 2), np.nan)
        se = np.full(len(names), np.nan)
        for i in range(len(names)):
            f = fin[i][np.isfinite(fin[i])]
            if f.size:
                q[i] = np.quantile(f, [alpha / 2.0, 1.0 - alpha / 2.0])
                se[i] = f.std(ddof=1) if f.size > 1 else np.nan

    nk = len(ks)
    kpoint = point[:nk]
    if not np.isfinite(kpoint).any():
        raise ValueError(f"metric {spec.name!r} is not finite for any k of the grid "
                         "(e.g. the strategy holds no positions)")
    oracle_k = int(ks[int(np.argmax(np.where(np.isfinite(kpoint), kpoint, -np.inf)))])
    argmax = bootstrap_argmax(draws[:nk], ks)
    oracle_set, freq = oracle_confidence_set(argmax, alpha, oracle_k)

    # paired regret intervals: per-draw metric(oracle_k) - metric(spec) on the shared indices
    rq = np.full((len(names), 2), np.nan)
    with np.errstate(invalid="ignore"):
        diff = draws[ks.index(oracle_k)][None, :] - draws
    for i in range(len(names)):
        f = diff[i][np.isfinite(diff[i])]
        if f.size:
            rq[i] = np.quantile(f, [alpha / 2.0, 1.0 - alpha / 2.0])

    curve = pd.DataFrame({
        "metric": kpoint, "low": q[:nk, 0], "high": q[:nk, 1], "se": se[:nk],
        "argmax_freq": [float(freq.get(k, 0.0)) for k in ks],
        "k_used_mean": [float(results[k].k_used.mean()) for k in ks],
        "regret_low": rq[:nk, 0], "regret_high": rq[:nk, 1],
    }, index=pd.Index(ks, name="k"))

    oset = set(oracle_set)
    best = float(kpoint[ks.index(oracle_k)])
    rrows = []
    for j, r in enumerate(rules):
        i = nk + j
        kk = results[r].k_used.to_numpy(dtype=float)
        k_med = float(np.median(kk)) if kk.size else np.nan
        k_mean = float(kk.mean()) if kk.size else np.nan
        rrows.append({
            "k_mean": k_mean, "k_median": k_med, "metric": point[i], "low": q[i, 0],
            "high": q[i, 1], "se": se[i],
            "dist_to_oracle": abs(k_med - oracle_k),
            "in_set": bool(np.isfinite(k_med) and math.floor(k_med + 0.5) in oset),
            "regret": best - point[i], "regret_low": rq[i, 0], "regret_high": rq[i, 1],
        })
    rules_df = pd.DataFrame(rrows, index=pd.Index(rules, name="rule"),
                            columns=["k_mean", "k_median", "metric", "low", "high", "se",
                                     "dist_to_oracle", "in_set", "regret", "regret_low",
                                     "regret_high"])
    if not rules:
        rules_df["in_set"] = rules_df["in_set"].astype(bool)

    meta = {
        "series": spec.series, "n_boot": int(idx.shape[0]), "alpha": float(alpha),
        "mean_block": mb, "random_state": random_state, "periods_per_year": ppy,
        "strategy": res[0].strategy, "common_start": common[0], "common_end": common[-1],
        "n_common": n, "n_valid_draws": int(np.isfinite(argmax).sum()),
        "draws": pd.DataFrame(draws, index=pd.Index(names, name="spec", dtype=object)),
        "n_dropped": {s: int(len(x) - n) for s, x in zip(names, series)},
    }
    return KCurve(curve=curve, rules=rules_df, oracle_k=oracle_k, oracle_set=oracle_set,
                  metric=spec.name, results={s: results[s] for s in names}, meta=meta)


def k_curve(engine: WalkForward, k_grid: Iterable[int] = range(1, 21),
            rules: Iterable[str] = ("mp", "mp_market", "bai_ng", "ahn_horenstein", "onatski",
                                    "gavish_donoho", "parallel"),
            metric: str = "sharpe_net", n_boot: int = 1000, alpha: float = 0.10,
            mean_block: float | None = None, random_state=0, n_jobs: int = 1) -> KCurve:
    """Run the engine on every fixed k and every rule; bootstrap with shared stationary-block
    indices over the common date range; oracle = argmax of the metric over k_grid; oracle_set =
    smallest set of k covering >= 1-alpha of bootstrap argmax draws (ordered by frequency).

    Parameters
    ----------
    engine : WalkForward
    k_grid : iterable of int
        Fixed numbers of factors (non-negative, unique). Values above a window's rank are
        clipped by the engine (see ``curve.k_used_mean``).
    rules : iterable of str
        Rule names from :data:`kfactors.RULES` (options via ``engine.config.rule_kwargs``).
    metric : str
        A key of :data:`METRICS`: ``"sharpe_net"`` (default), ``"sharpe_gross"``, ``"neg_vol"``
        (risk strategies), or one added with :func:`register_metric`. Higher is better.
    n_boot : int
        Number of stationary-bootstrap resamples (shared by all configurations).
    alpha : float
        The oracle set covers ``>= 1 - alpha`` of the argmax draws; the intervals ``low``/
        ``high`` are the ``alpha/2`` and ``1 - alpha/2`` percentiles.
    mean_block : float, optional
        Mean block length; default ``max(1, n ** (1/3))`` for ``n`` common dates.
    random_state : None, int or Generator
        Seed of the bootstrap indices.
    n_jobs : int
        Passed to :meth:`WalkForward.run_many`.

    Returns
    -------
    KCurve
    """
    ks = _check_grid(k_grid)
    rules = list(rules)
    _get_metric(metric)
    results = engine.run_many(list(ks) + rules, n_jobs=n_jobs)
    return curve_from_results(results, ks, rules, metric=metric, n_boot=n_boot, alpha=alpha,
                              mean_block=mean_block, random_state=random_state)
