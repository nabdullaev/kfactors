"""Performance metrics and inference for return series (pandas Series or 1-D arrays).

Conventions
-----------
* Inputs are simple returns in decimals, as pandas Series, lists or 1-D arrays (an ``(n, 1)``
  array is accepted and flattened). ``inf`` raises ``ValueError``.
* The point metrics (:func:`sharpe`, :func:`annual_return`, :func:`annual_vol`,
  :func:`max_drawdown`, :func:`probabilistic_sharpe_ratio`, :func:`deflated_sharpe_ratio`)
  **drop NaN** before computing.
* :func:`bootstrap_metric` does *not* drop NaN (so that a shared index matrix stays aligned
  across series); the metric function receives resampled arrays that may contain NaN, which
  all metrics in this module ignore. :func:`sharpe_diff_ci` aligns the two series on common
  dates and drops rows where either is NaN.

References
----------
* Politis, D. N. & Romano, J. P. (1994). The stationary bootstrap. JASA 89(428), 1303-1313.
* Bailey, D. H. & Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier.
  Journal of Risk 15(2), 3-44 (probabilistic Sharpe ratio, PSR).
* Bailey, D. H. & Lopez de Prado, M. (2014). The deflated Sharpe ratio. Journal of Portfolio
  Management 40(5), 94-107 (DSR, expected maximum Sharpe ratio).
* Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2017). The probability of
  backtest overfitting. Journal of Computational Finance 20(4), 39-69 (CSCV, PBO).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

from kfactors.utils import check_random_state

__all__ = ["annual_return", "annual_vol", "bootstrap_metric", "cscv_logits",
           "deflated_sharpe_ratio", "expected_max_sharpe", "max_drawdown", "pbo_cscv",
           "probabilistic_sharpe_ratio", "sharpe", "sharpe_diff_ci",
           "stationary_bootstrap_indices"]

# --------------------------------------------------------------------------- helpers


def _as_1d(r) -> np.ndarray:
    """Float 1-D array from Series/list/array; ``(n, 1)`` is flattened; inf raises."""
    x = np.asarray(getattr(r, "values", r), dtype=float)
    if x.ndim == 2 and 1 in x.shape:
        x = x.ravel()
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D return series, got shape {x.shape}")
    if np.any(np.isinf(x)):
        raise ValueError("return series contains inf")
    return x


def _clean(r) -> np.ndarray:
    x = _as_1d(r)
    return x[~np.isnan(x)]


def _check_ppy(periods_per_year) -> float:
    ppy = float(periods_per_year)
    if not ppy > 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    return ppy


def _per_period_sharpe(x: np.ndarray) -> float:
    """mean / std (ddof=1) of a NaN-free array; NaN if len < 2 or zero dispersion."""
    if x.size < 2 or np.ptp(x) == 0.0:
        return float("nan")
    sd = x.std(ddof=1)
    if not sd > 0:
        return float("nan")
    return float(x.mean() / sd)


def _rowwise_sharpe(M: np.ndarray) -> np.ndarray:
    """Per-period Sharpe of each row of M (NaN-aware, ddof=1)."""
    ok = ~np.isnan(M)
    cnt = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(ok, M, 0.0).sum(axis=1) / cnt
        dev = np.where(ok, M - mu[:, None], 0.0)
        sd = np.sqrt((dev**2).sum(axis=1) / (cnt - 1))
        out = mu / sd
    out[(cnt < 2) | ~(sd > 0)] = np.nan
    return out


# ----------------------------------------------------------------------- point metrics


def sharpe(r, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio mean/std*sqrt(ppy) (ddof=1); NaN if std == 0 or len < 2.

    Parameters
    ----------
    r : Series or array-like
        Periodic (excess) returns; NaN are dropped.
    periods_per_year : int
        Annualisation factor (252 daily, 12 monthly, 365 crypto daily).

    Returns
    -------
    float
        ``mean(r) / std(r, ddof=1) * sqrt(periods_per_year)``; no risk-free adjustment is made
        (pass excess returns if needed).
    """
    ppy = _check_ppy(periods_per_year)
    return _per_period_sharpe(_clean(r)) * math.sqrt(ppy)


def annual_return(r, periods_per_year: int = 252) -> float:
    """Geometric (compound) annualised return.

    ``prod(1 + r) ** (ppy / T) - 1`` over the ``T`` non-NaN observations, computed through
    ``log1p`` for numerical stability. A period with ``r <= -1`` wipes the equity out and the
    result is ``-1``. NaN for an empty series.
    """
    ppy = _check_ppy(periods_per_year)
    x = _clean(r)
    if x.size == 0:
        return float("nan")
    if np.any(x <= -1.0):
        return -1.0
    return float(np.expm1(np.sum(np.log1p(x)) * ppy / x.size))


def annual_vol(r, periods_per_year: int = 252) -> float:
    """Annualised volatility ``std(r, ddof=1) * sqrt(ppy)``; NaN if fewer than 2 observations."""
    ppy = _check_ppy(periods_per_year)
    x = _clean(r)
    if x.size < 2:
        return float("nan")
    return float(x.std(ddof=1) * math.sqrt(ppy))


def max_drawdown(r) -> float:
    """Maximum drawdown of the compounded equity curve (negative number, 0 if none).

    The equity curve starts at 1 before the first return (so a loss in the first period counts)
    and is ``cumprod(1 + r)``; ``1 + r`` is floored at 0 (limited liability), hence the result
    lies in ``[-1, 0]``. NaN are dropped (equivalent to a zero return). NaN for an empty series.
    """
    x = _clean(r)
    if x.size == 0:
        return float("nan")
    equity = np.concatenate([[1.0], np.cumprod(np.maximum(1.0 + x, 0.0))])
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0  # peak >= 1 > 0
    return float(min(dd.min(), 0.0))


# --------------------------------------------------------------------------- bootstrap


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float, random_state=None) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix of shape (n_boot, n).

    Politis & Romano (1994), Sec. 2: each resample is a concatenation of blocks
    ``(I, I+1, ..., I+L-1)`` with uniform random starts ``I ~ U{0..n-1}`` and independent
    geometric lengths ``L ~ Geom(1/mean_block)`` (mean ``mean_block``); indices wrap around
    circularly (``mod n``). Equivalently, position ``t`` starts a new block with probability
    ``q = 1/mean_block`` and otherwise continues with ``(previous + 1) mod n``.
    ``mean_block = 1`` gives the iid bootstrap.

    Parameters
    ----------
    n : int
        Series length (>= 1).
    n_boot : int
        Number of resamples (>= 1).
    mean_block : float
        Expected block length, ``>= 1``.
    random_state : None | int | Generator

    Returns
    -------
    ndarray of int64, shape (n_boot, n)
    """
    n, n_boot = int(n), int(n_boot)
    if n < 1 or n_boot < 1:
        raise ValueError(f"n and n_boot must be >= 1, got n={n}, n_boot={n_boot}")
    mean_block = float(mean_block)
    if not mean_block >= 1.0:
        raise ValueError(f"mean_block must be >= 1, got {mean_block}")
    rng = check_random_state(random_state)
    starts = rng.integers(0, n, size=(n_boot, n))
    new_block = rng.random((n_boot, n)) < 1.0 / mean_block
    new_block[:, 0] = True
    pos = np.arange(n)
    last_start = np.maximum.accumulate(np.where(new_block, pos, 0), axis=1)
    offset = pos[None, :] - last_start
    first = np.take_along_axis(starts, last_start, axis=1)
    return ((first + offset) % n).astype(np.int64)


def _default_block(n: int) -> float:
    return max(1.0, float(n) ** (1.0 / 3.0))


def _check_indices(indices, n: int) -> np.ndarray:
    idx = np.asarray(indices)
    if idx.ndim != 2 or idx.shape[1] != n:
        raise ValueError(f"indices must have shape (n_boot, {n}), got {idx.shape}")
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("indices must be integers")
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ValueError(f"indices must lie in [0, {n - 1}]")
    return idx


def _percentile_ci(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def bootstrap_metric(r, fn, n_boot: int = 1000, mean_block: float | None = None,
                     alpha: float = 0.05, random_state=None, indices=None) -> dict:
    """Bootstrap CI of fn(r). Returns {"estimate", "low", "high", "draws"}. mean_block default
    ~ n**(1/3). ``indices`` (n_boot x n) may be supplied to share resamples across series.

    Percentile interval (Efron) on stationary-bootstrap resamples (Politis & Romano 1994).

    Parameters
    ----------
    r : Series or array-like (n,)
        Return series. NaN are kept (to keep ``indices`` aligned); ``fn`` must ignore them.
    fn : callable
        ``fn(1-D ndarray) -> float``, e.g. ``sharpe`` or ``lambda x: sharpe(x, 12)``.
    n_boot : int
        Number of resamples (ignored when ``indices`` is given).
    mean_block : float, optional
        Expected block length; default ``max(1, n ** (1/3))``.
    alpha : float
        Two-sided level; the interval is the ``alpha/2`` and ``1 - alpha/2`` quantiles of the
        finite draws.
    random_state : None | int | Generator
    indices : int array (n_boot, n), optional
        Pre-computed resample indices (e.g. from :func:`stationary_bootstrap_indices`) shared
        by several series; overrides ``n_boot``, ``mean_block`` and ``random_state``.

    Returns
    -------
    dict
        ``estimate`` = ``fn(r)``, ``low``/``high`` percentile bounds, ``draws`` (n_boot,).
    """
    x = _as_1d(r)
    n = x.size
    if n == 0:
        raise ValueError("empty return series")
    if indices is None:
        mb = _default_block(n) if mean_block is None else mean_block
        indices = stationary_bootstrap_indices(n, n_boot, mb, random_state)
    else:
        indices = _check_indices(indices, n)
    estimate = float(fn(x))
    draws = np.array([fn(x[i]) for i in indices], dtype=float)
    lo, hi = _percentile_ci(draws, alpha)
    return {"estimate": estimate, "low": lo, "high": hi, "draws": draws}


def _align_pair(r1, r2) -> tuple[np.ndarray, np.ndarray]:
    """Align two series on common dates (pandas) or positions (arrays); drop NaN rows."""
    if hasattr(r1, "index") and hasattr(r2, "index"):
        import pandas as pd

        s1 = r1.iloc[:, 0] if isinstance(r1, pd.DataFrame) else r1
        s2 = r2.iloc[:, 0] if isinstance(r2, pd.DataFrame) else r2
        if not (s1.index.is_unique and s2.index.is_unique):
            raise ValueError("series indices must be unique to align on dates")
        df = pd.concat([s1.rename("a"), s2.rename("b")], axis=1, join="inner")
        a, b = _as_1d(df["a"]), _as_1d(df["b"])
    else:
        a, b = _as_1d(r1), _as_1d(r2)
        if a.size != b.size:
            raise ValueError(f"arrays must have equal length to pair, got {a.size} and {b.size}")
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def sharpe_diff_ci(r1, r2, periods_per_year: int = 252, n_boot: int = 2000,
                   mean_block: float | None = None, alpha: float = 0.05, random_state=None) -> dict:
    """Paired stationary-bootstrap CI for sharpe(r1) - sharpe(r2) on aligned dates.

    The two series are aligned on their common index (pandas inputs; positional for arrays),
    rows where either is NaN are dropped, and both series are resampled with the **same**
    stationary-bootstrap indices (Politis & Romano 1994), which preserves their
    cross-correlation (paired test, in the spirit of Ledoit & Wolf 2008, J. Empirical Finance
    15(5), but with a percentile rather than studentised interval).

    Returns
    -------
    dict
        ``estimate``, ``low``, ``high``, ``draws`` (annualised Sharpe differences), ``n``
        (aligned observations) and ``p_value``: the two-sided bootstrap p-value
        ``min(1, 2 * min(P*(d <= 0), P*(d >= 0)))`` for H0: equal Sharpe ratios (percentile
        inversion; approximate).
    """
    ppy = _check_ppy(periods_per_year)
    a, b = _align_pair(r1, r2)
    n = a.size
    if n < 2:
        raise ValueError(f"need at least 2 aligned non-NaN observations, got {n}")
    mb = _default_block(n) if mean_block is None else mean_block
    idx = stationary_bootstrap_indices(n, n_boot, mb, random_state)
    scale = math.sqrt(ppy)
    estimate = (_per_period_sharpe(a) - _per_period_sharpe(b)) * scale
    draws = np.empty(idx.shape[0])
    chunk = 256
    for s in range(0, idx.shape[0], chunk):
        sl = idx[s:s + chunk]
        draws[s:s + chunk] = (_rowwise_sharpe(a[sl]) - _rowwise_sharpe(b[sl])) * scale
    lo, hi = _percentile_ci(draws, alpha)
    finite = draws[np.isfinite(draws)]
    if finite.size:
        p = min(1.0, 2.0 * min(np.mean(finite <= 0.0), np.mean(finite >= 0.0)))
    else:
        p = float("nan")
    return {"estimate": float(estimate), "low": lo, "high": hi, "draws": draws, "n": int(n),
            "p_value": float(p)}


# ------------------------------------------------------------------------ PSR and DSR


def probabilistic_sharpe_ratio(r, sr_benchmark: float = 0.0) -> float:
    """Bailey & Lopez de Prado PSR on per-period (non-annualised) Sharpe with skew/kurtosis.

    Bailey & Lopez de Prado (2012), probabilistic Sharpe ratio::

        PSR(SR*) = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2) )

    with ``SR = mean / std(ddof=1)`` the per-period Sharpe ratio, ``g3`` the sample skewness and
    ``g4`` the (non-excess, normal = 3) sample kurtosis, both with population (biased) moments
    (``scipy.stats.skew`` / ``kurtosis(fisher=False)`` defaults), and ``T`` the number of
    non-NaN observations.

    Parameters
    ----------
    r : Series or array-like
        Returns; NaN dropped.
    sr_benchmark : float
        Benchmark ``SR*`` in **per-period** units (divide an annual Sharpe by ``sqrt(ppy)``).

    Returns
    -------
    float
        Probability in [0, 1]; NaN if ``T < 3``, zero dispersion, or the variance term
        ``1 - g3 SR + (g4 - 1)/4 SR^2`` is not positive (possible only for extreme skew).
    """
    x = _clean(r)
    t = x.size
    sr = _per_period_sharpe(x)
    if t < 3 or not np.isfinite(sr):
        return float("nan")
    g3 = float(stats.skew(x, bias=True))
    g4 = float(stats.kurtosis(x, fisher=False, bias=True))
    var_term = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    if not var_term > 0:
        return float("nan")
    z = (sr - float(sr_benchmark)) * math.sqrt(t - 1) / math.sqrt(var_term)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_trials_var: float) -> float:
    """Expected maximum Sharpe ratio among ``n_trials`` independent zero-skill trials.

    Bailey & Lopez de Prado (2014), expected maximum of N Sharpe estimates (via the
    asymptotic distribution of the maximum of N Gaussians)::

        SR0 = sqrt(V) [ (1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)) ]

    with ``gamma`` the Euler-Mascheroni constant and ``V`` the cross-trial variance of the
    Sharpe estimates. For ``N = 1`` there is no selection and ``SR0 = 0`` (the formula's first
    term diverges at ``N = 1``).
    """
    if isinstance(n_trials, bool) or int(n_trials) != n_trials or n_trials < 1:
        raise ValueError(f"n_trials must be an integer >= 1, got {n_trials}")
    if not sr_trials_var >= 0:
        raise ValueError(f"sr_trials_var must be >= 0, got {sr_trials_var}")
    n = int(n_trials)
    if n == 1:
        return 0.0
    g = float(np.euler_gamma)
    return float(math.sqrt(sr_trials_var) * ((1.0 - g) * stats.norm.ppf(1.0 - 1.0 / n)
                                             + g * stats.norm.ppf(1.0 - 1.0 / (n * math.e))))


def deflated_sharpe_ratio(r, n_trials: int, sr_trials_var: float) -> float:
    """DSR: PSR against the expected maximum Sharpe of n_trials with variance sr_trials_var
    (per-period units).

    Bailey & Lopez de Prado (2014): ``DSR = PSR(SR0)`` with
    ``SR0 = expected_max_sharpe(n_trials, sr_trials_var)``. ``sr_trials_var`` is the variance
    across the ``n_trials`` tried configurations of their **per-period** Sharpe ratios.
    Non-increasing in ``n_trials``; equals ``PSR(0)`` for ``n_trials = 1``.
    """
    sr0 = expected_max_sharpe(n_trials, sr_trials_var)
    return probabilistic_sharpe_ratio(r, sr_benchmark=sr0)


# ---------------------------------------------------------------------------- PBO/CSCV


def _default_metric(M: np.ndarray) -> np.ndarray:
    """Per-period Sharpe of each column (NaN-aware, ddof=1)."""
    return _rowwise_sharpe(M.T)


def cscv_logits(returns_matrix, n_splits: int = 16, metric=None) -> np.ndarray:
    """Logits ``lambda_c`` of the combinatorially symmetric cross-validation (CSCV).

    Bailey, Borwein, Lopez de Prado & Zhu (2017), Sec. 2:

    1. Split the T x N matrix of per-period returns (rows = time, columns = configurations)
       into ``S = n_splits`` contiguous, equally sized row blocks. If ``T`` is not a multiple
       of ``S``, the **first** ``T mod S`` (oldest) rows are discarded.
    2. For each of the ``C(S, S/2)`` ways to pick ``S/2`` blocks as the in-sample set ``J``
       (the complement ``J̄`` is out-of-sample), compute the metric per configuration on ``J``
       and on ``J̄``.
    3. Take ``n* = argmax`` in-sample; let ``omega = rank(OOS metric of n*) / (N + 1)`` with
       ranks ``1..N`` ascending (average ranks on ties) and ``lambda = log(omega / (1-omega))``.

    Parameters
    ----------
    returns_matrix : array-like or DataFrame (T, N)
        ``N >= 2`` configurations. NaN allowed with the default metric (ignored).
    n_splits : int
        Even number ``S >= 2`` of blocks; each block needs ``>= 2`` rows.
    metric : callable, optional
        ``metric(rows x N ndarray) -> (N,) ndarray`` (higher is better). Default: per-period
        Sharpe ratio (mean / std, ddof=1) per column. Non-finite metric values rank lowest.

    Returns
    -------
    ndarray (C(S, S/2),)
    """
    M = np.asarray(getattr(returns_matrix, "values", returns_matrix), dtype=float)
    if M.ndim != 2:
        raise ValueError(f"returns_matrix must be 2-D (T x N), got shape {M.shape}")
    T, N = M.shape
    if N < 2:
        raise ValueError("need at least 2 configurations (columns)")
    if np.any(np.isinf(M)):
        raise ValueError("returns_matrix contains inf")
    S = int(n_splits)
    if S != n_splits or S < 2 or S % 2:
        raise ValueError(f"n_splits must be an even integer >= 2, got {n_splits}")
    size = T // S
    if size < 2:
        raise ValueError(f"T={T} too short for n_splits={S} (need >= 2 rows per block)")
    M = M[T - size * S:]
    blocks = M.reshape(S, size, N)
    combos = np.array(list(itertools.combinations(range(S), S // 2)))
    ind = np.zeros((combos.shape[0], S), dtype=bool)
    ind[np.arange(combos.shape[0])[:, None], combos] = True

    if metric is None:
        # Fast path: Sharpe from per-block sufficient statistics (count, sum, sum of squares).
        ok = ~np.isnan(blocks)
        b0 = ok.sum(axis=1).astype(float)  # S x N
        b1 = np.where(ok, blocks, 0.0).sum(axis=1)
        b2 = np.where(ok, blocks**2, 0.0).sum(axis=1)

        def _from_stats(w: np.ndarray) -> np.ndarray:
            c, s1, s2 = w @ b0, w @ b1, w @ b2
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = s1 / c
                var = np.maximum(s2 - c * mu**2, 0.0) / (c - 1)
                out = mu / np.sqrt(var)
            out[(c < 2) | ~(var > 0)] = np.nan
            return out

        wis = ind.astype(float)
        R_is, R_oos = _from_stats(wis), _from_stats(1.0 - wis)
    else:
        R_is = np.empty((combos.shape[0], N))
        R_oos = np.empty((combos.shape[0], N))
        for c in range(combos.shape[0]):
            R_is[c] = np.asarray(metric(blocks[ind[c]].reshape(-1, N)), dtype=float)
            R_oos[c] = np.asarray(metric(blocks[~ind[c]].reshape(-1, N)), dtype=float)

    R_is = np.where(np.isfinite(R_is), R_is, -np.inf)
    R_oos = np.where(np.isfinite(R_oos), R_oos, -np.inf)
    best = np.argmax(R_is, axis=1)
    ranks = stats.rankdata(R_oos, method="average", axis=1)
    omega = ranks[np.arange(ranks.shape[0]), best] / (N + 1.0)
    return np.log(omega / (1.0 - omega))


def pbo_cscv(returns_matrix, n_splits: int = 16, metric=None) -> float:
    """Probability of backtest overfitting via combinatorially symmetric CV (T x configs).

    Bailey, Borwein, Lopez de Prado & Zhu (2017): the share of CSCV splits in
    which the in-sample best configuration ranks at or below the out-of-sample median,
    ``PBO = #{lambda_c <= 0} / C(S, S/2)``. See :func:`cscv_logits` for the procedure,
    the parameters and the default metric (per-period Sharpe ratio).

    Notes
    -----
    ``C(16, 8) = 12870`` splits; the default metric uses per-block sufficient statistics, so
    the cost is ``O(C(S, S/2) * S * N)`` regardless of ``T``.
    """
    lam = cscv_logits(returns_matrix, n_splits=n_splits, metric=metric)
    return float(np.mean(lam <= 0.0))
