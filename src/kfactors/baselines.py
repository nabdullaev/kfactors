"""Card C (teammate): baseline k-selectors built on established third-party libraries.

These wrappers let us cross-check :mod:`kfactors.rules` against independent implementations
(scikit-learn, statsmodels, optht, horns) and add them to the benchmark leaderboard as
"what a practitioner would do with off-the-shelf tools". Every function takes an ``n x p``
returns matrix (rows = time, columns = assets; ndarray or DataFrame, no NaN) and returns an
``int`` number of factors, or ``None`` when the method cannot be applied (the reason is reported
with a :class:`BaselineUnavailableWarning`).

Pre-processing convention (must match ``kfactors.select_k`` so results are comparable): when
``standardize=True`` (default) use ``Z, _, _ = kfactors.utils.preprocess(X, standardize=True)``
(column-centred, divided by the ddof=0 standard deviation, i.e. the correlation matrix); with
``standardize=False`` only centre the columns. ``preprocess`` already raises ``ValueError`` on
NaN/inf, fewer than 3 rows or 2 columns and zero-variance columns.

References
----------
* Jolliffe, I. T. (2002), *Principal Component Analysis*, 2nd ed., Springer, Sec. 6.1.1
  (cumulative percentage of variance).
* Minka, T. P. (2000), "Automatic choice of dimensionality for PCA", NIPS 13 (Laplace
  approximation used by ``sklearn.decomposition.PCA(n_components="mle")``).
* Bai, J. & Ng, S. (2002), "Determining the number of factors in approximate factor models",
  Econometrica 70(1), 191-221 (``statsmodels.multivariate.pca.PCA(...).ic``).
* Horn, J. L. (1965), "A rationale and test for the number of factors in factor analysis",
  Psychometrika 30(2), 179-185 (``horns.parallel_analysis``).
* Gavish, M. & Donoho, D. L. (2014), "The optimal hard threshold for singular values is
  4/sqrt(3)", IEEE Trans. Inf. Theory 60(8), 5040-5053 (``optht.optht``).

TODO (Card C): implement the five functions below. See ``docs/tasks/card_C_baselines.md`` and
``tests/teammates/test_card_C_baselines.py``.
"""
from __future__ import annotations


class BaselineUnavailableWarning(UserWarning):
    """Emitted (via :func:`warnings.warn`) when a baseline returns ``None``; the message says why.

    Already implemented: use it as
    ``warnings.warn("sklearn_mle needs n >= p ...", BaselineUnavailableWarning, stacklevel=2)``.
    """


def sklearn_variance_share(X, share: float = 0.9, *, standardize: bool = True) -> int:
    """Smallest k explaining more than ``share`` of the variance, via scikit-learn.

    ``sklearn.decomposition.PCA(n_components=share, svd_solver="full").fit(Z).n_components_``.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n, p)
        Complete panel (no NaN).
    share : float
        Target explained-variance share, ``0 < share < 1`` (scikit-learn's float convention).
    standardize : bool, default True
        Correlation (True) or covariance (False) PCA, see the module docstring.

    Returns
    -------
    int
        ``k >= 1``. Agrees with
        ``kfactors.select_k(X, "variance_share", share=share).info["k_uncapped"]`` (up to exact
        ties: scikit-learn uses ``cumulative share > share``, kfactors ``>=``).

    Raises
    ------
    ValueError
        If ``share`` is not strictly between 0 and 1, or ``X`` is invalid.
    """
    raise NotImplementedError("Card C: sklearn_variance_share is not implemented yet "
                              "(see docs/tasks/card_C_baselines.md)")


def sklearn_mle(X, *, standardize: bool = True) -> int | None:
    """Minka (2000) Bayesian model selection for PCA, via scikit-learn.

    ``sklearn.decomposition.PCA(n_components="mle", svd_solver="full").fit(Z).n_components_``.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n, p)
    standardize : bool, default True

    Returns
    -------
    int or None
        The MLE dimension. ``None`` when ``n < p``: scikit-learn only supports
        ``n_components="mle"`` if ``n_samples >= n_features``. In that case emit a
        :class:`BaselineUnavailableWarning` whose message contains ``"n < p"`` and the actual
        shape, then return ``None`` (never let scikit-learn's ``ValueError`` escape).

    Raises
    ------
    ValueError
        If ``X`` is invalid (NaN, too small, zero-variance column).
    """
    raise NotImplementedError("Card C: sklearn_mle is not implemented yet "
                              "(see docs/tasks/card_C_baselines.md)")


def statsmodels_bai_ng(X, criterion: str = "IC_p2", kmax: int = 20, *,
                       standardize: bool = True) -> int:
    """Bai & Ng (2002) information criterion computed by statsmodels.

    ``kmax`` is first clipped exactly like kfactors does:
    ``kmax_eff = kfactors.rules.base.effective_kmax(n, p, kmax)``. Then
    ``pc = statsmodels.multivariate.pca.PCA(pd.DataFrame(A), ncomp=kmax_eff,
    standardize=standardize, demean=True)`` with ``A = kfactors.utils.as_2d_array(X)`` and
    ``k = int(pc.ic[criterion].to_numpy().argmin())``. Passing a DataFrame makes ``pc.ic`` a
    DataFrame with columns ``IC_p1, IC_p2, IC_p3`` (with an ndarray input it is a plain
    ``(kmax_eff + 1) x 3`` array); row ``r`` is ``k = r``, starting at 0. statsmodels does its
    own standardisation (ddof=0, identical to kfactors), so pass the raw data, not ``Z``.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n, p)
    criterion : {"IC_p1", "IC_p2", "IC_p3"}
        statsmodels only computes the three ``IC_p`` criteria (not ``PC_p``).
    kmax : int, default 20
    standardize : bool, default True

    Returns
    -------
    int
        ``0 <= k <= kmax_eff``. Must equal
        ``kfactors.select_k(X, "bai_ng", criterion=criterion, kmax=kmax,
        standardize=standardize).k``.

    Raises
    ------
    ValueError
        Unknown ``criterion``, ``kmax < 1`` or invalid ``X`` (validate ``X`` with
        ``kfactors.utils.as_2d_array`` first so NaN raises ``ValueError``).
    """
    raise NotImplementedError("Card C: statsmodels_bai_ng is not implemented yet "
                              "(see docs/tasks/card_C_baselines.md)")


def horns_parallel(X, n_sim: int = 1000, quantile: float = 0.95) -> int | None:
    """Horn (1965) parallel analysis via the optional ``horns`` package.

    ``int(horns.parallel_analysis(Xa, simulations=n_sim, analysis_type="pca",
    quantile=quantile))`` where ``Xa = kfactors.utils.as_2d_array(X)`` (horns computes the
    correlation matrix itself, so no standardisation here).

    Parameters
    ----------
    X : array-like or DataFrame, shape (n, p)
    n_sim : int, default 1000
        Number of simulated null data sets (``simulations`` in horns), ``>= 1``.
    quantile : float, default 0.95
        Null eigenvalue quantile, ``0 < quantile < 1`` (Glorfeld 1995 recommends 0.95).

    Returns
    -------
    int or None
        Number of components retained by horns. ``None`` if ``import horns`` fails
        (``ImportError``): emit a :class:`BaselineUnavailableWarning` whose message contains
        ``"horns"`` and ``"not installed"`` and return ``None``.

    Raises
    ------
    ValueError
        Invalid ``n_sim``/``quantile`` or ``X``.

    Notes
    -----
    ``horns`` is optional and **not installable in the project environment**: horns 0.6.0
    requires ``numpy<2`` and ``numba``, while kfactors uses numpy 2.x. horns uses numba's
    internal random generator, so its output is not reproducible with a seed. Import it inside
    the function (``import horns``), never at module level, so ``kfactors.baselines`` imports
    fine without it. Our own seeded implementation is ``kfactors.select_k(X, "parallel")``.
    """
    raise NotImplementedError("Card C: horns_parallel is not implemented yet "
                              "(see docs/tasks/card_C_baselines.md)")


def optht_threshold(X, *, standardize: bool = True) -> int:
    """Gavish & Donoho (2014) optimal hard threshold (unknown noise) via ``optht``.

    With ``Z`` pre-processed as in the module docstring, ``n, p = Z.shape``,
    ``m = min(n - 1, p)`` (centring removes one degree of freedom, so only ``m`` singular values
    are non-trivial), ``sv = np.linalg.svd(Z, compute_uv=False)[:m]`` and
    ``beta = m / max(n - 1, p)``, return ``int(optht.optht(beta, sv=sv, sigma=None))``.
    ``optht`` raises ``ValueError`` when no singular value exceeds the cutoff; return ``0`` in
    that case.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n, p)
    standardize : bool, default True

    Returns
    -------
    int
        ``0 <= k <= m``. Equals
        ``kfactors.select_k(X, "gavish_donoho", standardize=standardize).info["k_uncapped"]``
        unless a singular value lies within ~0.05% of the cutoff (optht approximates the
        Marchenko-Pastur median numerically).

    Raises
    ------
    ValueError
        Invalid ``X``.
    """
    raise NotImplementedError("Card C: optht_threshold is not implemented yet "
                              "(see docs/tasks/card_C_baselines.md)")
