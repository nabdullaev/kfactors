"""Permutation / parallel-analysis rules.

References
----------
Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis.
    *Psychometrika* 30(2), 179-185.
Buja, A., Eyuboglu, N. (1992). Remarks on parallel analysis. *Multivariate Behav. Res.* 27(4),
    509-540.  (column-wise permutation null)
Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations.
    *Ann. Statist.* 17(3), 1217-1241.  (moving-block bootstrap)
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import linalg

from kfactors.rules.base import KResult, RuleWarning, _spectrum, register
from kfactors.utils import check_random_state

NULLS = ("iid", "circular", "block")


def _top_eigvals(Zb: np.ndarray, r: int) -> np.ndarray:
    """Top ``r`` eigenvalues (descending) of ``Zb'Zb / n`` via the smaller Gram matrix."""
    n, p = Zb.shape
    G = Zb.T @ Zb if p <= n else Zb @ Zb.T
    d = G.shape[0]
    r = min(r, d)
    ev = linalg.eigh(G / n, eigvals_only=True, subset_by_index=[d - r, d - 1])
    return ev[::-1]


def _circular_index(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """Row index (``n x p``) of an independent circular shift, optionally time-reversed, per column.

    There are ``2n`` such transforms (``n`` shifts x reversed or not; reversal keeps the
    autocovariances of a stationary series). They are assigned to the columns in a balanced random
    way: all distinct when ``p <= 2n``, otherwise each one is used ``floor(p / 2n)`` or
    ``ceil(p / 2n)`` times. Columns that share a transform keep their co-movement in the null.
    """
    reps = math.ceil(p / (2 * n))
    u = np.concatenate([rng.permutation(2 * n) for _ in range(reps)])[:p]
    u = u[rng.permutation(p)]
    shift, reverse = u % n, u >= n
    idx = (np.arange(n)[:, None] + shift[None, :]) % n
    idx[:, reverse] = (n - 1) - idx[:, reverse]
    return idx


def _null_panel(Z: np.ndarray, null: str, block_size: int, rng: np.random.Generator,
                col_mean: np.ndarray, col_sd: np.ndarray) -> np.ndarray:
    """One draw from the null of no cross-sectional dependence (columns resampled independently)."""
    n, p = Z.shape
    if null == "iid":
        return rng.permuted(Z, axis=0)              # independent permutation of every column
    if null == "circular":
        return np.take_along_axis(Z, _circular_index(n, p, rng), axis=0)
    # moving-block bootstrap per column
    nb = math.ceil(n / block_size)
    starts = rng.integers(0, n - block_size + 1, size=(nb, p))
    idx = (starts[:, None, :] + np.arange(block_size)[None, :, None]).reshape(nb * block_size, p)
    Zb = np.take_along_axis(Z, idx[:n], axis=0)
    # resampling changes column moments: restore each column's mean and standard deviation so the
    # null panel has exactly the same marginal pre-processing as Z (as permutations do for free)
    Zb = Zb - Zb.mean(axis=0)
    sd = Zb.std(axis=0)
    ok = sd > 0
    Zb[:, ok] *= col_sd[ok] / sd[ok]
    return Zb + col_mean


def _devolatilize(Z: np.ndarray) -> np.ndarray:
    """Divide every row by its cross-sectional RMS, then restore column means and sds.

    A volatility path shared by all series (market-wide volatility clustering) multiplies whole
    rows; dividing by the row RMS removes it, so the null resamplings (which shift rows of
    different columns against each other) no longer desynchronise it.
    """
    rms = np.sqrt(np.mean(Z**2, axis=1))
    rms = np.where(rms > 0, rms, 1.0)
    Zd = Z / rms[:, None]
    Zd = Zd - Zd.mean(axis=0)
    sd, sd_new = Z.std(axis=0), Zd.std(axis=0)
    ok = sd_new > 0
    Zd[:, ok] *= sd[ok] / sd_new[ok]
    return Zd + Z.mean(axis=0)


@register("parallel")
def parallel(Z, *, kmax: int, null: str = "iid", n_perm: int = 100, quantile: float = 0.95,
             block_size: int | None = None, devolatilize: bool = False,
             random_state=None) -> KResult:
    """Horn (1965) parallel analysis with a choice of null resampling scheme.

    Each null panel resamples every column of ``Z`` independently, which destroys
    cross-sectional (factor) dependence while keeping each column's marginal distribution:

    * ``"iid"`` - independent random permutation of each column (Buja & Eyuboglu 1992);
      also destroys serial dependence.
    * ``"circular"`` - independent random circular shift of each column, time-reversed with
      probability 1/2 (reversal keeps the autocovariances of a stationary series); preserves each
      column's autocorrelation up to the wrap-around point. The ``2n`` transforms are assigned
      to columns in a balanced way, all distinct when ``p <= 2n``. For ``p > 2n`` columns must
      share a transform and keep their co-movement in the null, which inflates the null
      quantiles; a :class:`RuleWarning` is issued and ``info["max_shared"]`` records the
      largest number of columns sharing one transform. Even with distinct transforms the
      circular null is conservative for ``p`` of the order of ``n`` and above when a strong
      factor is present (shifted copies of one factor series have the spread of its
      periodogram, not of a Marchenko-Pastur bulk): at 250 x 500 with a market factor the
      second null quantile was ~15% above the ``"iid"`` one, so weak factors can be missed.
    * ``"block"`` - moving-block bootstrap of each column (Kunsch 1989), block length
      ``block_size`` (default ``ceil(sqrt(n))``; development versions before 2026-11-03 used
      ``ceil(n ** (1/3))``, which was far too short when ``p >= n``: 14-20 spurious factors on
      AR(0.5) noise at 250 x 250 / 250 x 500). Columns are re-centred and rescaled to their
      original standard deviation. Blocks cut the autocorrelation at their joins and understate
      low-frequency power, which drives the top eigenvalues when ``p >= n``; even the longer
      default still over-counts on AR(0.5) noise at 250 x 500. Prefer ``"circular"`` there.

    None of the nulls preserves a volatility path shared by all series (market-wide volatility
    clustering): resampling columns independently desynchronises it, the null spectrum is too
    narrow and all three nulls over-count (``kmax`` on 500 x 200 noise with a common AR(0.98)
    log-volatility of sd 0.5). ``devolatilize=True`` divides every row of ``Z`` by its
    cross-sectional RMS (then restores column means and sds) before both the observed and the
    null spectra are computed; on such panels it gave 0 at ``k = 0`` and the true ``k`` in our
    checks, and left planted i.i.d. panels unchanged. It is off by default for backward
    compatibility. With it, ``criterion``, ``threshold`` and ``info["tested_eigenvalues"]``
    refer to the de-volatilised panel, while ``eigenvalues`` stays the spectrum of ``S``.

    Only the top ``kmax + 1`` eigenvalues are compared. With ``q_j`` the ``quantile`` of the
    ``n_perm`` null ``j``-th eigenvalues, ``k`` is the number of *leading consecutive* ranks
    with ``lambda_j > q_j`` (stop at the first failure), capped at ``kmax``.
    Null eigenvalues are computed from the ``min(n, p)``-dimensional Gram matrix, so ``p > n``
    costs ``O(n^2 p)`` per draw. Deterministic given ``random_state``.

    Returns
    -------
    KResult
        ``criterion`` = ``lambda_j - q_j`` for ranks ``1..kmax+1``; ``threshold`` = ``q``
        (array, ranks ``1..kmax+1``); ``info``: ``null``, ``n_perm``, ``quantile``,
        ``block_size``, ``devolatilize``, ``tested_eigenvalues``, ``null_eigenvalues``
        (``n_perm x (kmax+1)``), ``max_shared`` (circular only), ``k_uncapped``.
    """
    if null not in NULLS:
        raise ValueError(f"null must be one of {NULLS}, got {null!r}")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if int(n_perm) < 1:
        raise ValueError(f"n_perm must be >= 1, got {n_perm}")
    Z = np.asarray(Z, dtype=float)
    ev, n, p, _m, kmax = _spectrum(Z, kmax)
    if block_size is None:
        block_size = math.ceil(math.sqrt(n))
    block_size = int(block_size)
    if not 1 <= block_size <= n:
        raise ValueError(f"block_size must be in [1, n={n}], got {block_size}")
    rng = check_random_state(random_state)
    r = kmax + 1
    if devolatilize:
        Z = _devolatilize(Z)
        obs = _top_eigvals(Z, r)
    else:
        obs = ev[:r]
    extra = {}
    if null == "circular":
        extra["max_shared"] = math.ceil(p / (2 * n))
        if p > 2 * n:
            warnings.warn(f"parallel(null='circular'): p = {p} > 2n = {2 * n}, so up to "
                          f"{extra['max_shared']} columns share each shift and keep their "
                          "co-movement in the null; the null quantiles are inflated (conservative)",
                          RuleWarning, stacklevel=2)
    col_mean, col_sd = Z.mean(axis=0), Z.std(axis=0)
    null_ev = np.zeros((int(n_perm), r))
    for b in range(int(n_perm)):
        e = _top_eigvals(_null_panel(Z, null, block_size, rng, col_mean, col_sd), r)
        null_ev[b, : e.size] = e
    q = np.quantile(null_ev, quantile, axis=0)
    obs = np.concatenate([obs, np.zeros(r - obs.size)])
    above = obs > q
    k_unc = int(r if above.all() else np.argmin(above))
    k = min(k_unc, kmax)
    return KResult(k=k, method="parallel", eigenvalues=ev, criterion=obs - q, threshold=q,
                   info={"kmax_effective": kmax, "null": null, "n_perm": int(n_perm),
                         "quantile": float(quantile), "block_size": block_size,
                         "devolatilize": bool(devolatilize), "tested_eigenvalues": obs,
                         "null_eigenvalues": null_ev, "k_uncapped": k_unc,
                         "capped": k_unc > kmax, **extra})
