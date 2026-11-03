"""Random-matrix-theory rules: Marchenko-Pastur edge, Laloux market-adjusted edge, Gavish-Donoho.

Conventions (shared by all three rules)
---------------------------------------
``Z`` is column-centred, so ``Z'Z`` is a Wishart matrix with ``n - 1`` (not ``n``) degrees of
freedom under i.i.d. Gaussian noise. With ``S = Z'Z / n`` and ``gamma = p / (n - 1)``,
the ``m = min(n - 1, p)`` non-trivial eigenvalues of ``S`` under pure noise of variance
``sigma^2`` follow ``c * sigma^2 * max(1, gamma) * MP(beta)`` with ``c = (n - 1) / n`` and
``beta = min(gamma, 1 / gamma) <= 1``, where ``MP(beta)`` is the unit-variance
Marchenko-Pastur law on ``[(1 - sqrt(beta))^2, (1 + sqrt(beta))^2]``
(Marchenko & Pastur 1967; for ``gamma > 1`` use the companion matrix ``ZZ'``). In all regimes
the upper edge of the noise spectrum of ``S`` is ``c * sigma^2 * (1 + sqrt(gamma))^2``.

References
----------
Marchenko, V. A., Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random
    matrices. *Math. USSR-Sbornik* 1(4), 457-483.
Laloux, L., Cizeau, P., Bouchaud, J.-P., Potters, M. (1999). Noise dressing of financial
    correlation matrices. *Phys. Rev. Lett.* 83(7), 1467-1470.
Gavish, M., Donoho, D. L. (2014). The optimal hard threshold for singular values is 4/sqrt(3).
    *IEEE Trans. Inf. Theory* 60(8), 5040-5053.
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
from scipy import integrate, optimize

from kfactors.rules.base import KResult, RuleWarning, _spectrum, register


# ------------------------------------------------------------------------------------------------
# Marchenko-Pastur helpers
# ------------------------------------------------------------------------------------------------
def mp_cdf(x: float, beta: float) -> float:
    """CDF of the unit-variance Marchenko-Pastur law with ratio ``beta`` in ``(0, 1]``.

    Density ``f(t) = sqrt((b - t)(t - a)) / (2 pi beta t)`` on ``[a, b]``,
    ``a = (1 - sqrt(beta))^2``, ``b = (1 + sqrt(beta))^2`` (Marchenko & Pastur 1967). The
    substitution ``t = a + u^2`` removes the square-root singularities (including the
    ``t^{-1/2}`` pole at ``a = 0`` when ``beta = 1``), so ``quad`` integrates a smooth function.
    """
    if not 0.0 < beta <= 1.0:
        raise ValueError(f"beta must be in (0, 1], got {beta}")
    a, b = (1.0 - np.sqrt(beta)) ** 2, (1.0 + np.sqrt(beta)) ** 2
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0

    def g(u):
        return 2.0 * u * u * np.sqrt(max(b - a - u * u, 0.0)) / (a + u * u)

    val, _ = integrate.quad(g, 0.0, np.sqrt(x - a), epsabs=1e-13, epsrel=1e-12, limit=200)
    return float(val / (2.0 * np.pi * beta))


@lru_cache(maxsize=256)
def mp_median(beta: float) -> float:
    """Median of the unit-variance Marchenko-Pastur law with ratio ``beta`` in ``(0, 1]``.

    Solved to ~1e-12 with Brent's method on :func:`mp_cdf` (``mu_beta`` in Gavish & Donoho
    2014, Eq. (13)). ``mp_median(1) = 0.6528...``.
    """
    beta = float(beta)
    a, b = (1.0 - np.sqrt(beta)) ** 2, (1.0 + np.sqrt(beta)) ** 2
    return float(optimize.brentq(lambda x: mp_cdf(x, beta) - 0.5, a, b, xtol=1e-14, rtol=1e-13))


def gd_lambda_star(beta: float) -> float:
    """Optimal hard threshold for known noise, Gavish & Donoho (2014) Eq. (11).

    ``lambda*(beta) = sqrt(2 (beta + 1) + 8 beta / (beta + 1 + sqrt(beta^2 + 14 beta + 1)))``;
    ``lambda*(1) = 4 / sqrt(3)``.
    """
    return float(np.sqrt(2.0 * (beta + 1.0)
                         + 8.0 * beta / (beta + 1.0 + np.sqrt(beta**2 + 14.0 * beta + 1.0))))


def gd_omega(beta: float) -> float:
    """Exact unknown-noise coefficient ``omega(beta) = lambda*(beta) / sqrt(mu_beta)``.

    Gavish & Donoho (2014), Eq. (13)-(14); ``mu_beta`` is the Marchenko-Pastur median.
    ``omega(1) = 2.858...``.
    """
    return gd_lambda_star(beta) / np.sqrt(mp_median(beta))


def gd_omega_approx(beta: float) -> float:
    """Cubic approximation ``0.56 b^3 - 0.95 b^2 + 1.82 b + 1.43`` (Gavish & Donoho 2014, Eq. 5)."""
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


def _ratios(n: int, p: int) -> tuple[float, float, float]:
    """(gamma = p / (n-1), beta = min(gamma, 1/gamma), c = (n-1)/n) for a centred panel."""
    gamma = p / (n - 1)
    return gamma, min(gamma, 1.0 / gamma), (n - 1) / n


# ------------------------------------------------------------------------------------------------
# Rules
# ------------------------------------------------------------------------------------------------
@register("mp")
def mp(Z, *, kmax: int, noise: str = "mean", max_iter: int = 100) -> KResult:
    """Number of eigenvalues above the Marchenko-Pastur upper edge.

    ``k = #{j : lambda_j > edge}`` (capped at ``kmax``), ``edge = s (1 + sqrt(beta))^2`` where
    ``s`` is the scale of the noise bulk (see module conventions; ``s = c sigma^2 max(1, gamma)``).

    Noise estimation
    ----------------
    ``noise="mean"`` (default; development versions before 2026-11-03 defaulted to ``"median"``):
    ``s = max(1, gamma) * trace(S) / p``, i.e. the noise variance is the average variance (equal
    to 1 for standardised data, giving the classic edge ``(1 + sqrt(gamma))^2`` used for
    correlation matrices by Laloux et al. 1999 and Plerou et al. 2002). Strong factors inflate it,
    so it is conservative (weak factors next to a strong one can be missed), but it does not
    over-count when the idiosyncratic variances differ across series.

    ``noise="median"`` (opt-in): median matching on the ``m`` non-trivial eigenvalues, iterated
    to exclude outliers: start with the bulk = all non-trivial eigenvalues, set
    ``s = median(bulk) / mp_median(beta)``, recompute the edge, redefine the bulk as the
    eigenvalues ``<= edge`` and repeat until the bulk no longer changes (at most ``max_iter``
    iterations; a :class:`RuleWarning` is raised otherwise). Medians are robust to the few spikes
    produced by factors (cf. the median-based noise estimate of Gavish & Donoho 2014, Eq. (15)).
    It is exact only for homoskedastic, light-tailed, serially independent noise. Heterogeneous
    idiosyncratic variances (e.g. a standardised panel with dispersed factor exposures, or a
    covariance panel with dispersed idiosyncratic volatilities), heavy tails or serial dependence
    widen the bulk beyond the MP law; the median then places the edge inside the bulk and the rule
    over-counts badly (often up to ``kmax`` on return-like panels).

    Returns
    -------
    KResult
        ``criterion`` = ``lambda_j / edge`` for ranks ``j = 1..kmax+1``; ``threshold`` = edge;
        ``info``: ``sigma2`` (noise variance per entry implied by ``s``), ``gamma``, ``beta``,
        ``noise``, ``iterations``, ``converged``, ``k_uncapped``.
    """
    if noise not in ("median", "mean"):
        raise ValueError(f"noise must be 'median' or 'mean', got {noise!r}")
    ev, n, p, m, kmax = _spectrum(Z, kmax)
    gamma, beta, c = _ratios(n, p)
    lam = ev[:m]
    stretch = (1.0 + np.sqrt(beta)) ** 2
    iterations, converged = 0, True
    if noise == "mean":
        s = max(1.0, gamma) * ev.sum() / p
        edge = s * stretch
    else:
        med = mp_median(beta)
        bulk = np.ones(m, dtype=bool)
        converged = False
        for iterations in range(1, max_iter + 1):
            s = float(np.median(lam[bulk])) / med
            edge = s * stretch
            new_bulk = lam <= edge
            if np.array_equal(new_bulk, bulk):
                converged = True
                break
            bulk = new_bulk
        if not converged:
            warnings.warn(f"mp: median matching did not converge in {max_iter} iterations",
                          RuleWarning, stacklevel=2)
    k_unc = int(np.sum(lam > edge))
    k = min(k_unc, kmax)
    sigma2 = s / (c * max(1.0, gamma))
    return KResult(k=k, method="mp", eigenvalues=ev, criterion=ev[: kmax + 1] / edge,
                   threshold=float(edge),
                   info={"kmax_effective": kmax, "noise": noise, "sigma2": float(sigma2),
                         "gamma": gamma, "beta": beta, "iterations": iterations,
                         "converged": converged, "k_uncapped": k_unc, "capped": k_unc > kmax})


@register("mp_market")
def mp_market(Z, *, kmax: int, market: str = "auto") -> KResult:
    """Laloux et al. (1999) market-adjusted Marchenko-Pastur edge.

    The largest eigenvalue (the "market") is removed from the noise variance:
    ``sigma2 = (trace(S) - lambda_1) / p`` (for a correlation matrix ``1 - lambda_1 / p``,
    Laloux et al. 1999, p. 1468), ``edge = sigma2 (1 + sqrt(gamma))^2`` with
    ``gamma = p / (n - 1)`` (centred data, see module conventions; Laloux et al. use
    ``Q = T / N``, i.e. ``gamma = 1 / Q``). The edge formula holds for ``gamma > 1`` as well,
    because ``trace(S) / p`` estimates ``c sigma^2`` in both regimes.
    ``k = #{j : lambda_j > edge}``, capped at ``kmax``.

    Parameters
    ----------
    market : {"auto", "always"}
        ``"auto"`` (default): remove ``lambda_1`` only if it lies above the plain MP edge
        ``(trace(S) / p) (1 + sqrt(gamma))^2``, i.e. only if there is a market mode to remove;
        otherwise the edge is the plain one (same as ``mp(noise="mean")``). Under pure noise
        ``lambda_1`` sits at the edge, and always removing it lowers ``sigma2`` by about
        ``(1 + sqrt(gamma))^2 / p``, which made the rule count noise eigenvalues for small ``p``
        (mean ``k`` 3.25 at 1000 x 8). ``"always"``: always remove ``lambda_1`` (Laloux et al.
        1999 as written; the behaviour before this option existed).

    Notes
    -----
    The method assumes that, once the market mode is removed, the residual variance is the same
    for every series. With dispersed market betas or idiosyncratic volatilities (typical of
    stock returns) the residual bulk is wider than the MP law and the rule over-counts (Laloux
    et al. themselves note eigenvalues remaining above the edge and fit a smaller ``sigma2``).
    For example at 250 x 300 with one market factor, betas U(0.5, 1.5) and idiosyncratic sd
    lognormal(0, 0.5) it returns 4-6 where onatski, bai_ng, ahn_horenstein and
    ``mp(noise="mean")`` return 1.

    Returns
    -------
    KResult
        ``criterion`` = ``lambda_j / edge`` (ranks ``1..kmax+1``), ``threshold`` = edge,
        ``info``: ``sigma2``, ``gamma``, ``market``, ``market_removed``, ``k_uncapped``.
    """
    if market not in ("auto", "always"):
        raise ValueError(f"market must be 'auto' or 'always', got {market!r}")
    ev, n, p, m, kmax = _spectrum(Z, kmax)
    gamma, _beta, _c = _ratios(n, p)
    stretch = (1.0 + np.sqrt(gamma)) ** 2
    removed = market == "always" or bool(ev[0] > ev.sum() / p * stretch)
    sigma2 = (ev.sum() - ev[0]) / p if removed else ev.sum() / p
    edge = sigma2 * stretch
    k_unc = int(np.sum(ev[:m] > edge))
    k = min(k_unc, kmax)
    return KResult(k=k, method="mp_market", eigenvalues=ev, criterion=ev[: kmax + 1] / edge,
                   threshold=float(edge),
                   info={"kmax_effective": kmax, "sigma2": float(sigma2), "gamma": gamma,
                         "market": market, "market_removed": removed,
                         "k_uncapped": k_unc, "capped": k_unc > kmax})


@register("gavish_donoho")
def gavish_donoho(Z, *, kmax: int) -> KResult:
    """Gavish & Donoho (2014) optimal hard threshold for singular values, unknown noise level.

    Singular values ``y_j = sqrt(n lambda_j)`` of ``Z`` (``j = 1..m``, the non-trivial ones,
    ``m = min(n - 1, p)``) are kept if ``y_j > omega(beta) * median(y)``,
    ``beta = m / max(n - 1, p)``, with the exact ``omega(beta) = lambda*(beta) / sqrt(mu_beta)``
    (Eq. (11), (13)-(15)); ``k`` = number kept, capped at ``kmax``. This reproduces
    ``optht.optht(beta, sv=y, sigma=None)`` (which uses the same exact formula with a numerically
    approximated MP median) except that it returns 0 instead of raising when nothing survives.
    The aspect ratio uses ``n - 1`` because centring removes one degree of freedom.

    Returns
    -------
    KResult
        ``criterion`` = ``y_j / cutoff`` for ranks ``1..kmax+1``; ``threshold`` is the cutoff in
        eigenvalue units of ``S`` (``cutoff^2 / n``); ``info``: ``sv_cutoff``, ``omega``,
        ``omega_approx`` (Eq. 5 cubic), ``beta``, ``median_sv``, ``k_uncapped``.
    """
    ev, n, p, m, kmax = _spectrum(Z, kmax)
    beta = m / max(n - 1, p)
    sv = np.sqrt(n * ev[:m])
    omega = gd_omega(beta)
    med = float(np.median(sv))
    cutoff = omega * med
    k_unc = int(np.sum(sv > cutoff))
    k = min(k_unc, kmax)
    crit = sv[: kmax + 1] / cutoff  # m >= kmax + 1 always (effective_kmax)
    return KResult(k=k, method="gavish_donoho", eigenvalues=ev, criterion=crit,
                   threshold=float(cutoff**2 / n),
                   info={"kmax_effective": kmax, "sv_cutoff": float(cutoff),
                         "omega": float(omega), "omega_approx": float(gd_omega_approx(beta)),
                         "beta": float(beta), "median_sv": med, "k_uncapped": k_unc,
                         "capped": k_unc > kmax})
