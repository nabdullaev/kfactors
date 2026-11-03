"""Simulator of factor panels with a known number of factors.

Model
-----
``X = F L' + E`` with ``X`` of shape ``n x p`` (rows = time, columns = assets), ``F`` of shape
``n x k`` (factors, unit variance, uncorrelated), ``L`` of shape ``p x k`` (loadings) and ``E``
of shape ``n x p`` (idiosyncratic noise). The population covariance of a row of ``X`` is

    Sigma = L L' + diag(v),

where ``v_j`` is the *marginal* variance of noise column ``j`` (1 for every noise family except
``"hetero"``). Serial dependence (AR(1) noise, GARCH noise, AR(1) factors) changes the joint law
of the rows but not this marginal covariance, because every process is started from (or burnt
in to) its stationary distribution with the stated unconditional variance.

Scale
-----
The panel is on a **unit-noise scale**, not a returns-like scale: idiosyncratic variances are 1
(or ``U(a, b)`` for ``"hetero"``) and factor variances are 1. Multiply ``X`` by e.g. ``0.01`` to
obtain daily-return magnitudes; eigenvalue-based rules applied to standardised data are scale
invariant, so nothing in the benchmark depends on the scale.

Regimes
-------
* ``"pervasive"`` (Bai & Ng, 2002, *Econometrica* 70(1), DGP of Section 6): loadings
  ``L_jk ~ N(0, s_k^2 / 25)`` i.i.d., so the k-th population spike is ``~ p s_k^2 / 25`` and
  grows linearly in ``p`` (strong factors, ``L'L / p -> diag(s_k^2/25)``).
* ``"spiked"`` (Johnstone, 2001, *Ann. Statist.* 29(2); Baik, Ben Arous & Peche, 2005,
  *Ann. Probab.* 33(5)): ``L = Q diag(s)`` with ``Q`` a uniformly random ``p x k`` orthonormal
  frame, so the population spikes are exactly ``s_k^2`` above the noise level and stay bounded.
  With unit noise a spike is detectable from the sample eigenvalues (BBP phase transition)
  iff ``s_k^2 > sqrt(p / n)``.

``design_grid`` returns the named benchmark grids used by tests and ``kfactors.bench``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from kfactors.utils import check_random_state

__all__ = ["DEFAULT_SHAPES", "DEFAULT_STRENGTHS", "GARCH_BURN_IN", "NOISE_VARIANTS",
           "SIGMA_MAX_P", "WEAK_RATIOS", "SimData", "SimSpec", "design_grid", "simulate"]

#: Default factor strengths s_k; entries beyond the fifth repeat the last value (1.2).
DEFAULT_STRENGTHS: tuple[float, ...] = (5.0, 3.0, 2.0, 1.5, 1.2)

#: Burn-in length of the GARCH(1,1) noise recursion.
GARCH_BURN_IN: int = 200

#: Above this p the dense population covariance is not materialised (SimData.Sigma is None).
SIGMA_MAX_P: int = 2000


@dataclass
class SimSpec:
    """Specification of one simulated design.

    Parameters
    ----------
    n, p : int
        Number of observations (rows) and series (columns).
    k : int
        True number of factors, *including* the market factor when ``market`` is set.
        ``k = 0`` gives a pure-noise panel.
    regime : {"pervasive", "spiked"}
        ``"pervasive"``: ``L_jk ~ N(0, s_k^2 / 25)`` so spikes grow with p (Bai-Ng setting).
        ``"spiked"``: ``L = Q diag(s)`` with ``Q`` p x k orthonormal, bounded population spikes
        ``s_k^2`` (RMT / BBP setting).
    strengths : sequence of float, optional
        ``s_k`` for the non-market factors (there are ``k - 1`` of them if ``market`` is set,
        otherwise ``k``). Default ``(5, 3, 2, 1.5, 1.2, 1.2, ...)`` truncated to that number.
        A longer user sequence is truncated; a shorter one raises.
    market : float, optional
        If set, an extra dominant first factor with positive loadings ("betas")
        ``b_j proportional to 1 + 0.3 z_j`` is added; ``||b||^2`` equals ``market * p``
        (pervasive) or ``market`` (spiked), and ``b`` is an exact eigenvector of ``L L'`` with
        that eigenvalue (the other loading columns are made orthogonal to it). It counts
        towards ``k``.
    noise : {"gauss", "t", "ar1", "garch", "hetero"}
        Idiosyncratic noise family (see :func:`simulate`).
    df : float
        Degrees of freedom of Student-t noise (must be > 2); rescaled to unit variance.
    phi : float
        AR(1) coefficient of ``"ar1"`` noise, ``|phi| < 1``; unit stationary variance.
    garch : (omega, alpha, beta)
        GARCH(1,1) parameters; the series is rescaled to unit unconditional variance
        ``omega / (1 - alpha - beta)`` (the default already has it equal to 1).
    hetero_range : (a, b)
        Idiosyncratic variances ``v_j ~ U(a, b)`` for ``"hetero"`` noise.
    factor_ar : float
        AR(1) coefficient of the factors (0 = i.i.d.), ``|factor_ar| < 1``; unit variance.
    name : str
        Descriptive identifier of the design.
    """

    n: int
    p: int
    k: int = 3
    regime: Literal["pervasive", "spiked"] = "pervasive"
    # pervasive: loadings L_jk ~ N(0, s_k^2 / 25) so spikes grow with p (Bai-Ng setting).
    # spiked: L = Q diag(s) with Q p x k orthonormal -> bounded population spikes s_k^2 (RMT/BBP).
    strengths: Sequence[float] | None = None   # s_k; default (5, 3, 2, 1.5, ...) truncated to k
    market: float | None = None
    # if set, an extra dominant first factor is added whose population eigenvalue equals
    # market * p (pervasive) or market (spiked); counts towards k (k includes it).
    noise: Literal["gauss", "t", "ar1", "garch", "hetero"] = "gauss"
    df: float = 3.0                 # t noise, rescaled to unit variance
    phi: float = 0.5                # ar1 coefficient (unit stationary variance)
    garch: tuple[float, float, float] = (0.05, 0.10, 0.85)  # omega, alpha, beta (unit uncond. var)
    hetero_range: tuple[float, float] = (0.5, 3.0)          # idiosyncratic variances U(a, b)
    factor_ar: float = 0.0          # AR(1) coefficient of factors (0 = iid)
    name: str = ""


@dataclass
class SimData:
    """One simulated panel.

    Attributes
    ----------
    X : ndarray (n, p)
        Observed panel ``F L' + E``.
    F : ndarray (n, k)
        Factors (unit variance, mutually independent).
    L : ndarray (p, k)
        Loadings; column 0 is the market beta vector when ``spec.market`` is set.
    E : ndarray (n, p)
        Idiosyncratic noise.
    k : int
        True number of factors.
    Sigma : ndarray (p, p) or None
        Population (marginal) covariance of a row of ``X``: ``L L' + diag(noise_var)``;
        ``None`` when ``p > 2000``.
    spec : SimSpec
        The specification that generated the panel.
    noise_var : ndarray (p,)
        Marginal idiosyncratic variances (the diagonal added in ``Sigma``).
    """

    X: np.ndarray        # n x p
    F: np.ndarray        # n x k
    L: np.ndarray        # p x k
    E: np.ndarray        # n x p
    k: int
    Sigma: np.ndarray    # p x p population covariance of X (rows), or None if p > 2000
    spec: SimSpec = field(repr=False, default=None)
    noise_var: np.ndarray | None = field(repr=False, default=None)


# --------------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------------

_NOISES = ("gauss", "t", "ar1", "garch", "hetero")
_REGIMES = ("pervasive", "spiked")


def _resolve_strengths(spec: SimSpec) -> np.ndarray:
    """Strengths of the non-market factors (length k - [market is set])."""
    m = spec.k - (1 if spec.market is not None else 0)
    if spec.strengths is None:
        s = [DEFAULT_STRENGTHS[min(i, len(DEFAULT_STRENGTHS) - 1)] for i in range(m)]
    else:
        s = [float(v) for v in spec.strengths]
        if len(s) < m:
            raise ValueError(f"strengths has {len(s)} entries but {m} non-market factors "
                             f"are required (k={spec.k}, market={spec.market})")
        s = s[:m]
    s = np.asarray(s, dtype=float)
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError(f"strengths must be finite and > 0, got {s.tolist()}")
    return s


def _validate(spec: SimSpec) -> None:
    if not isinstance(spec, SimSpec):
        raise TypeError(f"expected a SimSpec, got {type(spec).__name__}")
    if int(spec.n) != spec.n or int(spec.p) != spec.p or int(spec.k) != spec.k:
        raise ValueError("n, p and k must be integers")
    if spec.n < 2 or spec.p < 2:
        raise ValueError(f"need n >= 2 and p >= 2, got n={spec.n}, p={spec.p}")
    if spec.k < 0:
        raise ValueError(f"k must be >= 0, got {spec.k}")
    if spec.k > spec.p:
        raise ValueError(f"k={spec.k} exceeds p={spec.p}")
    if spec.regime not in _REGIMES:
        raise ValueError(f"regime must be one of {_REGIMES}, got {spec.regime!r}")
    if spec.noise not in _NOISES:
        raise ValueError(f"noise must be one of {_NOISES}, got {spec.noise!r}")
    if spec.market is not None:
        if spec.k < 1:
            raise ValueError("market factor requires k >= 1 (k includes the market factor)")
        if not np.isfinite(spec.market) or spec.market <= 0:
            raise ValueError(f"market must be > 0, got {spec.market}")
    if spec.noise == "t" and not (np.isfinite(spec.df) and spec.df > 2):
        raise ValueError(f"t noise needs df > 2 for a finite variance, got df={spec.df}")
    if spec.noise == "ar1" and not abs(spec.phi) < 1:
        raise ValueError(f"ar1 noise needs |phi| < 1, got {spec.phi}")
    if spec.noise == "garch":
        omega, alpha, beta = (float(v) for v in spec.garch)
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            raise ValueError("garch needs omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1; "
                             f"got {spec.garch}")
    if spec.noise == "hetero":
        a, b = (float(v) for v in spec.hetero_range)
        if not (0 < a <= b < np.inf):
            raise ValueError(f"hetero_range must satisfy 0 < a <= b, got {spec.hetero_range}")
    if not abs(spec.factor_ar) < 1:
        raise ValueError(f"factor_ar must satisfy |factor_ar| < 1, got {spec.factor_ar}")


# --------------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------------

def _market_direction(p: int, rng: np.random.Generator) -> np.ndarray:
    """Positive-mean beta profile u_j = 1 + 0.3 z_j, normalised to unit length."""
    u = 1.0 + 0.3 * rng.standard_normal(p)
    return u / np.linalg.norm(u)


def _loadings(spec: SimSpec, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = spec.p
    has_mkt = spec.market is not None
    m = s.size
    if spec.regime == "pervasive":
        # Bai & Ng (2002), Section 6: L_jk i.i.d. N(0, s_k^2 / 25).
        G = rng.standard_normal((p, m)) * (s / 5.0)
        if not has_mkt:
            return G
        u = _market_direction(p, rng)
        G = G - np.outer(u, u @ G)          # make u an exact eigenvector of L L'
        b = np.sqrt(spec.market * p) * u
        return np.column_stack([b, G])
    # spiked: L = Q diag(s), Q uniformly random orthonormal frame (QR of a Gaussian matrix,
    # sign-corrected so that the distribution is Haar; Mezzadri, 2007, Notices AMS 54(5)).
    ncol = m + (1 if has_mkt else 0)
    if ncol == 0:
        return np.zeros((p, 0))
    G = rng.standard_normal((p, ncol))
    if has_mkt:
        G[:, 0] = _market_direction(p, rng)
    Q, R = np.linalg.qr(G)
    Q = Q * np.sign(np.where(np.diag(R) == 0, 1.0, np.diag(R)))
    if has_mkt:
        # first column is u / ||u|| exactly (up to sign, fixed positive by the R correction)
        scales = np.concatenate([[np.sqrt(spec.market)], s])
    else:
        scales = s
    return Q * scales


def _ar1(n: int, p: int, a: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary unit-variance AR(1): x_t = a x_{t-1} + sqrt(1 - a^2) e_t, x_0 ~ N(0, 1)."""
    Z = rng.standard_normal((n, p))
    if a == 0.0:
        return Z
    out = np.empty((n, p))
    out[0] = Z[0]
    c = np.sqrt(1.0 - a * a)
    for t in range(1, n):
        out[t] = a * out[t - 1] + c * Z[t]
    return out


def _garch(n: int, p: int, params, rng: np.random.Generator) -> np.ndarray:
    """GARCH(1,1) (Bollerslev, 1986, J. Econometrics 31(3), eq. 1-2) with Gaussian innovations.

    e_t = sigma_t z_t, sigma_t^2 = omega + alpha e_{t-1}^2 + beta sigma_{t-1}^2, started at the
    unconditional variance, run for GARCH_BURN_IN + n steps, burn-in discarded, then divided by
    sqrt(omega / (1 - alpha - beta)) so the unconditional variance is exactly 1.
    """
    omega, alpha, beta = (float(v) for v in params)
    uvar = omega / (1.0 - alpha - beta)
    T = GARCH_BURN_IN + n
    Z = rng.standard_normal((T, p))
    out = np.empty((T, p))
    s2 = np.full(p, uvar)
    for t in range(T):
        e = np.sqrt(s2) * Z[t]
        out[t] = e
        s2 = omega + alpha * e * e + beta * s2
    return out[GARCH_BURN_IN:] / np.sqrt(uvar)


def _noise(spec: SimSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n, p = spec.n, spec.p
    var = np.ones(p)
    if spec.noise == "gauss":
        E = rng.standard_normal((n, p))
    elif spec.noise == "t":
        # Var(t_df) = df / (df - 2) -> rescale to unit variance.
        E = rng.standard_t(spec.df, size=(n, p)) * np.sqrt((spec.df - 2.0) / spec.df)
    elif spec.noise == "ar1":
        E = _ar1(n, p, float(spec.phi), rng)
    elif spec.noise == "garch":
        E = _garch(n, p, spec.garch, rng)
    else:  # hetero
        a, b = (float(v) for v in spec.hetero_range)
        var = rng.uniform(a, b, size=p)
        E = rng.standard_normal((n, p)) * np.sqrt(var)
    return E, var


# --------------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------------

def simulate(spec: SimSpec, random_state=None) -> SimData:
    """Draw one panel ``X = F L' + E`` from ``spec``. Deterministic given ``random_state``.

    Parameters
    ----------
    spec : SimSpec
        Design (see :class:`SimSpec`).
    random_state : None, int or numpy.random.Generator
        Seed; normalised by :func:`kfactors.utils.check_random_state`. Draw order is fixed:
        loadings, then factors, then noise.

    Returns
    -------
    SimData

    Notes
    -----
    Loadings (see the module docstring for the two regimes):

    * pervasive: ``L_jk ~ N(0, s_k^2 / 25)`` (Bai & Ng, 2002, Section 6);
    * spiked: ``L = Q diag(s)``, ``Q`` Haar-distributed ``p x k`` orthonormal frame, so the
      eigenvalues of ``L L'`` are exactly ``s_k^2`` (Johnstone, 2001, spiked covariance model).

    Factors are Gaussian with unit variance; with ``factor_ar = a != 0`` each factor is a
    stationary AR(1) ``f_t = a f_{t-1} + sqrt(1 - a^2) u_t`` with ``f_0 ~ N(0, 1)``.

    Noise families (all with the stated *marginal* variance, so ``Sigma`` is exact):

    * ``"gauss"``: i.i.d. N(0, 1);
    * ``"t"``: i.i.d. Student-t(df) times ``sqrt((df - 2) / df)`` (unit variance; excess
      kurtosis ``6 / (df - 4)`` for ``df > 4``, infinite for ``2 < df <= 4``);
    * ``"ar1"``: independent stationary AR(1) columns with coefficient ``phi``, unit variance;
    * ``"garch"``: independent GARCH(1,1) columns (Bollerslev, 1986) with Gaussian innovations,
      200 burn-in steps, rescaled to unit unconditional variance;
    * ``"hetero"``: Gaussian with column variances ``v_j ~ U(a, b)``.

    The population covariance is ``Sigma = L L' + diag(v)``; it is returned only for
    ``p <= 2000`` (``None`` otherwise) to bound memory.
    """
    _validate(spec)
    rng = check_random_state(random_state)
    s = _resolve_strengths(spec)
    n, p, k = int(spec.n), int(spec.p), int(spec.k)

    L = _loadings(spec, s, rng)
    if k > 0:
        F = _ar1(n, k, float(spec.factor_ar), rng)
    else:
        F = np.zeros((n, 0))
    E, var = _noise(spec, rng)
    X = F @ L.T + E

    Sigma = None
    if p <= SIGMA_MAX_P:
        Sigma = L @ L.T
        Sigma[np.diag_indices(p)] += var
    return SimData(X=X, F=F, L=L, E=E, k=k, Sigma=Sigma, spec=spec, noise_var=var)


# --------------------------------------------------------------------------------------------
# design grids
# --------------------------------------------------------------------------------------------

#: (n, p) shapes of the default grid: p / n in {0.4, 1.0, 2.5} for n in {100, 250}.
DEFAULT_SHAPES: tuple[tuple[int, int], ...] = (
    (100, 40), (100, 100), (100, 250),
    (250, 100), (250, 250), (250, 625),
)

#: Noise variants of the default grid: tag -> SimSpec keyword overrides.
NOISE_VARIANTS: dict[str, dict] = {
    "gauss": {"noise": "gauss"},
    "t3": {"noise": "t", "df": 3.0},
    "ar1-0.5": {"noise": "ar1", "phi": 0.5},
    "ar1-0.9": {"noise": "ar1", "phi": 0.9},
    "garch": {"noise": "garch"},
    "hetero": {"noise": "hetero"},
}

#: BBP ratios r of the weak-factor family: weakest spike s_4^2 = r * sqrt(p / n).
WEAK_RATIOS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)


def _default_grid() -> list[SimSpec]:
    specs: list[SimSpec] = []
    # 1) base: regime x noise x shape, k = 4 with strengths (5, 3, 2, 1.5)  -> 2*6*6 = 72
    for regime in _REGIMES:
        for tag, kw in NOISE_VARIANTS.items():
            for n, p in DEFAULT_SHAPES:
                specs.append(SimSpec(n=n, p=p, k=4, regime=regime,
                                     name=f"{regime}_{tag}_n{n}_p{p}", **kw))
    # 2) dominant market factor + 3 weaker factors (3, 2, 1.5); n = 250  -> 2*3*3 = 18
    #    pervasive: market eigenvalue p (vs 0.36p, 0.16p, 0.09p);
    #    spiked: market spike 100 (vs 9, 4, 2.25).
    market_value = {"pervasive": 1.0, "spiked": 100.0}
    for regime in _REGIMES:
        for tag in ("gauss", "t3", "hetero"):
            for n, p in DEFAULT_SHAPES[3:]:
                specs.append(SimSpec(n=n, p=p, k=4, regime=regime,
                                     market=market_value[regime], strengths=(3.0, 2.0, 1.5),
                                     name=f"market_{regime}_{tag}_n{n}_p{p}",
                                     **NOISE_VARIANTS[tag]))
    # 3) weak factor near the BBP threshold (spiked, gauss): spikes (25, 9, 4, r*sqrt(p/n))
    #    -> 4*3 = 12; r < 1 is undetectable from the spectrum in the large-(n, p) limit.
    for r in WEAK_RATIOS:
        for n, p in DEFAULT_SHAPES[3:]:
            s4 = float(np.sqrt(r * np.sqrt(p / n)))
            specs.append(SimSpec(n=n, p=p, k=4, regime="spiked", strengths=(5.0, 3.0, 2.0, s4),
                                 noise="gauss", name=f"weak_bbp{r:g}_n{n}_p{p}"))
    # 4) null designs, k = 0 (false-positive control)  -> 3*3 = 9
    for tag in ("gauss", "t3", "garch"):
        for n, p in DEFAULT_SHAPES[3:]:
            specs.append(SimSpec(n=n, p=p, k=0, name=f"null_{tag}_n{n}_p{p}",
                                 **NOISE_VARIANTS[tag]))
    return specs


def _small_grid() -> list[SimSpec]:
    return [
        SimSpec(n=100, p=40, k=4, regime="pervasive", name="pervasive_gauss_n100_p40"),
        SimSpec(n=100, p=250, k=4, regime="spiked", name="spiked_gauss_n100_p250"),
        SimSpec(n=100, p=100, k=4, regime="pervasive", noise="t", df=3.0,
                name="pervasive_t3_n100_p100"),
        SimSpec(n=100, p=100, k=4, regime="spiked", noise="ar1", phi=0.5,
                name="spiked_ar1-0.5_n100_p100"),
        SimSpec(n=100, p=100, k=4, regime="pervasive", noise="garch",
                name="pervasive_garch_n100_p100"),
        SimSpec(n=100, p=250, k=4, regime="pervasive", noise="hetero",
                name="pervasive_hetero_n100_p250"),
        SimSpec(n=100, p=100, k=4, regime="pervasive", market=1.0, strengths=(3.0, 2.0, 1.5),
                name="market_pervasive_gauss_n100_p100"),
        SimSpec(n=100, p=100, k=0, name="null_gauss_n100_p100"),
    ]


_GRIDS = {"default": _default_grid, "small": _small_grid}


def design_grid(name: str = "default") -> list[SimSpec]:
    """Named benchmark grids of :class:`SimSpec` (fresh objects on every call).

    Parameters
    ----------
    name : {"default", "small"}

    Returns
    -------
    list of SimSpec
        Every spec has a unique descriptive ``name``.

    Notes
    -----
    ``"default"`` (111 designs; shapes ``(n, p)`` in ``DEFAULT_SHAPES``, i.e. ``p / n`` in
    {0.4, 1.0, 2.5} for ``n`` in {100, 250}):

    1. **base** (72): regime {pervasive, spiked} x noise {gauss, t(3), ar1(0.5), ar1(0.9),
       garch(0.05, 0.10, 0.85), hetero U(0.5, 3)} x 6 shapes; ``k = 4``, strengths
       (5, 3, 2, 1.5). Names ``"{regime}_{noise}_n{n}_p{p}"``.
    2. **market** (18): a dominant market factor plus 3 factors with strengths (3, 2, 1.5),
       ``k = 4``; market eigenvalue ``p`` (pervasive, ``market=1``) or 100 (spiked);
       noise {gauss, t(3), hetero}; n = 250, p in {100, 250, 625}.
       Names ``"market_{regime}_{noise}_n{n}_p{p}"``.
    3. **weak** (12): spiked, gauss, ``k = 4``, spikes (25, 9, 4, ``r sqrt(p/n)``) with
       ``r`` in {0.5, 1, 1.5, 2} around the BBP threshold (Baik, Ben Arous & Peche, 2005);
       n = 250, p in {100, 250, 625}. Names ``"weak_bbp{r}_n{n}_p{p}"``. For ``r <= 1`` the
       fourth factor is asymptotically invisible in the spectrum: rules that return 3 are
       "right" about detectability but "wrong" about the truth -- by design.
    4. **null** (9): ``k = 0`` pure noise {gauss, t(3), garch}; n = 250, p in {100, 250, 625}.
       Names ``"null_{noise}_n{n}_p{p}"``.

    ``"small"`` (8 designs, all ``n = 100``): one representative per family/noise for CI.
    """
    try:
        builder = _GRIDS[name]
    except KeyError:
        raise ValueError(f"unknown design grid {name!r}; available: {sorted(_GRIDS)}") from None
    specs = builder()
    names = [s.name for s in specs]
    if len(set(names)) != len(names):  # pragma: no cover - guarded by tests
        raise RuntimeError(f"duplicate design names in grid {name!r}")
    return specs
