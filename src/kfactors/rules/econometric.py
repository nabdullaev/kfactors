"""Econometric factor-number estimators: Bai-Ng, Ahn-Horenstein, Onatski.

All three work with the eigenvalues ``lambda_1 >= ... >= lambda_p`` of ``S = Z'Z / n`` of the
pre-processed panel (``n`` = T time periods, ``p`` = N cross-sectional units in the papers'
notation). Only ``m = min(n - 1, p)`` eigenvalues are non-trivial after centring.

References
----------
Bai, J., Ng, S. (2002). Determining the number of factors in approximate factor models.
    *Econometrica* 70(1), 191-221.
Ahn, S. C., Horenstein, A. R. (2013). Eigenvalue ratio test for the number of factors.
    *Econometrica* 81(3), 1203-1227.
Onatski, A. (2010). Determining the number of factors from empirical distribution of
    eigenvalues. *Review of Economics and Statistics* 92(4), 1004-1016.
"""
from __future__ import annotations

import warnings

import numpy as np

from kfactors.rules.base import KResult, RuleWarning, _spectrum, register

RANK_TOL = 1e-12  # relative size below which eigenvalues / residual variances are round-off
BAI_NG_CRITERIA = ("IC_p1", "IC_p2", "IC_p3", "PC_p1", "PC_p2", "PC_p3")
MIN_NONTRIVIAL = 3  # bai_ng / ahn_horenstein need m = min(n-1, p) >= 3 non-trivial eigenvalues


def _degenerate(method: str, ev: np.ndarray, m: int, kmax: int) -> KResult:
    """``k = 0`` with a :class:`RuleWarning` for panels with fewer than ``MIN_NONTRIVIAL``
    non-trivial eigenvalues (no residual left to compare a factor against)."""
    # stacklevel 3 points at the caller of the rule (select_k re-emits it at its own caller)
    warnings.warn(f"{method}: needs min(n-1, p) >= {MIN_NONTRIVIAL} non-trivial eigenvalues, "
                  f"got {m}; returning k = 0", RuleWarning, stacklevel=3)
    return KResult(k=0, method=method, eigenvalues=ev, criterion=None, threshold=None,
                   info={"kmax_effective": kmax, "degenerate": True})


def bai_ng_criteria(ev: np.ndarray, n: int, p: int, kmax: int) -> dict[str, np.ndarray]:
    """All six Bai & Ng (2002) criteria for ``k = 0..kmax`` from the eigenvalues of ``Z'Z/n``.

    ``V(k) = (N T)^{-1} SSR(k) = sum_{j > k} lambda_j / p`` is the mean squared residual after
    ``k`` principal components (Bai & Ng 2002, Eq. (6)). With ``N = p``, ``T = n``,
    ``C^2_{NT} = min(N, T)`` and ``sigma2_hat = V(kmax)`` (Eq. (9), (10)):

    * ``IC_p1 = ln V(k) + k (N+T)/(NT) ln(NT/(N+T))``
    * ``IC_p2 = ln V(k) + k (N+T)/(NT) ln C^2_{NT}``
    * ``IC_p3 = ln V(k) + k ln(C^2_{NT}) / C^2_{NT}``
    * ``PC_p*`` = same penalties with ``V(k)`` instead of ``ln V(k)`` and the penalty scaled by
      ``sigma2_hat``.

    ``V`` is floored at ``RANK_TOL * V(0)``: residual variances below that are round-off of an
    exactly low-rank panel, and comparing their logarithms would select ``kmax`` spuriously.

    Convention: ``T = n`` (the number of rows) is used in the penalties although column-centring
    leaves ``n - 1`` degrees of freedom, as in the paper (which assumes no centring) and in
    ``statsmodels``; using ``n - 1`` instead changed no selected ``k`` in our checks.
    """
    N, T = p, n
    V = np.array([ev[k:].sum() for k in range(kmax + 1)]) / N
    V = np.maximum(V, RANK_TOL * V[0])
    k = np.arange(kmax + 1)
    c = (N + T) / (N * T)
    C2 = min(N, T)
    g = {"p1": c * np.log(1.0 / c), "p2": c * np.log(C2), "p3": np.log(C2) / C2}
    sigma2 = V[kmax]
    out = {}
    for key, gk in g.items():
        out[f"IC_{key}"] = np.log(V) + k * gk
        out[f"PC_{key}"] = V + k * sigma2 * gk
    out["V"] = V
    return out


@register("bai_ng")
def bai_ng(Z, *, kmax: int, criterion: str = "IC_p2") -> KResult:
    """Bai & Ng (2002) information criteria; ``k = argmin_{0 <= k <= kmax_bn} criterion(k)``.

    ``kmax_bn = min(kmax, max(1, m // 2))`` with ``m = min(n - 1, p)``. Bai & Ng's theory keeps
    ``kmax`` fixed while ``min(N, T)`` grows. When ``kmax`` is close to ``m``, ``V(kmax)`` is the
    sum of the last one or two eigenvalues: ``ln V(k)`` then falls faster than the penalty and the
    argmin is ``kmax`` even on pure noise (e.g. 120 x 20 Gaussian noise with ``kmax`` clipped to
    18 returned 18 on most seeds). Halving the usable range removes this; it binds only for
    ``m < 2 kmax`` (fewer than 40 non-trivial eigenvalues at the default ``kmax = 20``).
    ``info["kmax_bn"]`` records the value used. Panels with ``m < 3`` return ``k = 0`` with a
    :class:`RuleWarning` and ``info["degenerate"] = True``.

    See :func:`bai_ng_criteria` for the formulas. For ``IC_p1..3`` the argmin coincides with
    ``statsmodels.multivariate.pca.PCA(Z, ncomp=kmax, standardize=False, demean=False).ic``
    (statsmodels uses ``ln SSR(k)``, which differs from ``ln V(k)`` by a constant).
    ``PC_p*`` criteria depend on the scale of the data through ``sigma2_hat = V(kmax)`` and on
    ``kmax`` itself (Bai & Ng 2002, Sec. 5).

    Parameters
    ----------
    criterion : {"IC_p1", "IC_p2", "IC_p3", "PC_p1", "PC_p2", "PC_p3"}

    Returns
    -------
    KResult
        ``criterion`` = chosen criterion over ``k = 0..kmax_bn``; ``info``: ``criterion``,
        ``kmax_bn``, ``V`` (residual variance path), ``sigma2_hat = V(kmax_bn)``, ``all`` (dict
        of all six paths), ``degenerate``.
    """
    if criterion not in BAI_NG_CRITERIA:
        raise ValueError(f"criterion must be one of {BAI_NG_CRITERIA}, got {criterion!r}")
    ev, n, p, m, kmax = _spectrum(Z, kmax)
    if m < MIN_NONTRIVIAL:
        return _degenerate("bai_ng", ev, m, kmax)
    kmax_bn = min(kmax, max(1, m // 2))
    crit = bai_ng_criteria(ev, n, p, kmax_bn)
    path = crit[criterion]
    k = int(np.argmin(path))
    return KResult(k=k, method="bai_ng", eigenvalues=ev, criterion=path, threshold=None,
                   info={"kmax_effective": kmax, "kmax_bn": kmax_bn, "criterion": criterion,
                         "V": crit["V"], "sigma2_hat": float(crit["V"][kmax_bn]),
                         "all": {name: crit[name] for name in BAI_NG_CRITERIA},
                         "degenerate": False})


def ahn_horenstein_stats(ev: np.ndarray, n: int, p: int, kmax: int) -> tuple[np.ndarray, ...]:
    """ER and GR statistics of Ahn & Horenstein (2013) for ``k = 0..kmax``.

    ``ER(k) = mu_k / mu_{k+1}``; ``GR(k) = ln(V(k-1)/V(k)) / ln(V(k)/V(k+1))
    = ln(1 + mu*_k) / ln(1 + mu*_{k+1})`` with ``V(k) = sum_{j > k} mu_j`` and
    ``mu*_k = mu_k / V(k)`` (AH 2013, Eq. (5)-(6)). ``k = 0`` is admitted through the mock
    eigenvalue ``mu_0 = V(0) / ln(m)``, ``m = min(n, p)`` (AH 2013, p. 1207: the paper's
    ``min(N, T)`` without a centring correction, although ``V(0)`` sums only the
    ``min(n - 1, p)`` non-trivial eigenvalues; ``min(n - 1, p)`` changed no selected ``k`` in our
    checks).
    Eigenvalue scale is irrelevant (ratios), so ``S = Z'Z/n`` is used directly; only the
    ``min(n-1, p)`` non-trivial eigenvalues enter, floored at ``RANK_TOL * lambda_1``.

    Returns
    -------
    (er, gr, mu) : ndarrays; ``mu[k]`` for ``k = 0..kmax+1`` with ``mu[0]`` the mock eigenvalue.
    """
    m = min(n, p)
    # eigenvalues below RANK_TOL * lambda_1 are round-off (exactly low-rank panel); flooring them
    # avoids spurious x / 0 ratios among numerically-zero eigenvalues
    ev = np.maximum(ev[: min(n - 1, p)], RANK_TOL * ev[0])
    V = np.array([ev[k:].sum() for k in range(kmax + 3)])      # V[k] = sum_{j > k} mu_j
    mu = np.concatenate(([V[0] / np.log(m)], ev[: kmax + 1]))   # mu[0] mock, mu[k] = lambda_k
    ks = np.arange(kmax + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        er = mu[ks] / mu[ks + 1]
        mu_star = mu / V[: kmax + 2]                             # mu*_k, k = 0..kmax+1
        gr = np.log1p(mu_star[ks]) / np.log1p(mu_star[ks + 1])
    return er, gr, mu


@register("ahn_horenstein")
def ahn_horenstein(Z, *, kmax: int, stat: str = "ER") -> KResult:
    """Ahn & Horenstein (2013) eigenvalue ratio (ER) or growth ratio (GR) estimator.

    ``k = argmax_{0 <= k <= kmax} ER(k)`` (or ``GR(k)``); see :func:`ahn_horenstein_stats`.
    Undefined ratios (``0/0`` on degenerate tiny panels) are treated as ``-inf``. Panels with
    ``m = min(n - 1, p) < 3`` return ``k = 0`` with a :class:`RuleWarning` and
    ``info["degenerate"] = True`` (the ratio would only compare the two available eigenvalues).

    Parameters
    ----------
    stat : {"ER", "GR"}

    Returns
    -------
    KResult
        ``criterion`` = chosen statistic over ``k = 0..kmax``; ``info``: ``stat``, ``er``,
        ``gr``, ``mock_eigenvalue``, ``degenerate``.
    """
    if stat not in ("ER", "GR"):
        raise ValueError(f"stat must be 'ER' or 'GR', got {stat!r}")
    ev, n, p, m, kmax = _spectrum(Z, kmax)
    if m < MIN_NONTRIVIAL:
        return _degenerate("ahn_horenstein", ev, m, kmax)
    er, gr, mu = ahn_horenstein_stats(ev, n, p, kmax)
    path = er if stat == "ER" else gr
    k = int(np.argmax(np.where(np.isnan(path), -np.inf, path)))
    return KResult(k=k, method="ahn_horenstein", eigenvalues=ev, criterion=path, threshold=None,
                   info={"kmax_effective": kmax, "stat": stat, "er": er, "gr": gr,
                         "mock_eigenvalue": float(mu[0]), "degenerate": False})


@register("onatski")
def onatski(Z, *, kmax: int, max_iter: int = 20) -> KResult:
    """Onatski (2010) edge-distribution (ED) estimator, iterated calibration of ``delta``.

    Algorithm (Onatski 2010, Sec. IV, steps 1-4), with ``r_max = kmax``:

    1. ``j = r_max + 1``.
    2. OLS of ``(lambda_j, ..., lambda_{j+4})`` on a constant and
       ``((j-1)^{2/3}, ..., (j+3)^{2/3})``; ``delta = 2 |slope|``.
    3. ``k(delta) = max{i <= r_max : lambda_i - lambda_{i+1} >= delta}``, or 0 if none.
    4. ``j = k(delta) + 1``; repeat 2-3 until ``k`` stops changing (at most ``max_iter`` times;
       a :class:`RuleWarning` is raised if it does not converge, and the last ``k`` is kept).

    The regression needs ``lambda_{r_max+5}`` to be non-trivial, so ``r_max`` is further clipped
    to ``m - 5`` (``m = min(n-1, p)``); ``info["rmax"]`` records the value used. If ``m < 6`` the
    estimator is undefined: a :class:`RuleWarning` is issued and ``k = 0`` is returned with
    ``info["degenerate"] = True``.

    Returns
    -------
    KResult
        ``criterion`` = eigenvalue gaps ``lambda_i - lambda_{i+1}``, ``i = 1..rmax``;
        ``threshold`` = final ``delta``; ``info``: ``rmax``, ``iterations``, ``converged``,
        ``path`` (k after each iteration), ``deltas``, ``degenerate``.
    """
    ev, _n, _p, m, kmax = _spectrum(Z, kmax)
    rmax = min(kmax, m - 5)
    if rmax < 1:
        warnings.warn(f"onatski: needs min(n-1, p) >= 6 non-trivial eigenvalues, got {m}; "
                      "returning k = 0", RuleWarning, stacklevel=2)
        return KResult(k=0, method="onatski", eigenvalues=ev, criterion=None, threshold=None,
                       info={"kmax_effective": kmax, "rmax": 0, "iterations": 0,
                             "converged": False, "path": [], "deltas": [], "degenerate": True})
    lam = ev
    gaps = lam[:rmax] - lam[1: rmax + 1]          # gaps[i-1] = lambda_i - lambda_{i+1}
    j = rmax + 1
    k_hat, path, deltas, converged = None, [], [], False
    for _ in range(max_iter):
        idx = np.arange(j, j + 5)                 # 1-based ranks j..j+4
        y = lam[idx - 1]
        x = (idx - 1.0) ** (2.0 / 3.0)
        xc = x - x.mean()
        slope = float(xc @ (y - y.mean()) / (xc @ xc))
        delta = 2.0 * abs(slope)
        hits = np.flatnonzero(gaps >= delta)
        k_new = int(hits.max() + 1) if hits.size else 0
        path.append(k_new)
        deltas.append(delta)
        if k_new == k_hat:
            converged = True
            break
        k_hat, j = k_new, k_new + 1
    if not converged:
        warnings.warn(f"onatski: no convergence in {max_iter} iterations (path {path}); "
                      "returning the last estimate", RuleWarning, stacklevel=2)
    return KResult(k=int(k_hat if converged else path[-1]), method="onatski", eigenvalues=ev,
                   criterion=gaps, threshold=float(deltas[-1]),
                   info={"kmax_effective": kmax, "rmax": rmax, "iterations": len(path),
                         "converged": converged, "path": path, "deltas": deltas,
                         "degenerate": False})
