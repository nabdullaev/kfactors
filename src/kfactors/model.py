"""Statistical (PCA) factor model fitted on one window.

The model is the classical approximate factor model estimated by principal components
(Bai & Ng 2002, Econometrica 70(1), Sec. 3; Bai 2003, Econometrica 71(1)):

    Xc = F B' + E,        Cov(X) = B Cov(F) B' + D,

where ``Xc`` is the column-centred n x p window, ``F`` (n x k) are the principal-component
factor scores, ``B`` (p x k) the loadings and ``E`` the idiosyncratic residuals. Eigenportfolios
and the residual regression follow Avellaneda & Lee (2010), "Statistical arbitrage in the US
equities market", Quantitative Finance 10(7) (Sec. 2: eigenportfolios and the regression of
stock returns on eigenportfolio returns).
"""
from __future__ import annotations

import warnings

import numpy as np

from kfactors.utils import as_2d_array, preprocess

__all__ = ["FactorModel"]

_IDIO_FLOOR = 1e-12


def _check_k(k) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be a non-negative integer, got {k!r}")
    k = int(k)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    return k


class FactorModel:
    """PCA factor model with k factors.

    fit(X): X n x p complete (no NaN). With standardize=True PCA runs on standardised columns
    (correlation matrix); loadings/covariance are returned in original units.
    k = 0 is allowed (no factors: covariance is diagonal of sample variances).

    Fitted attributes: mean_ (p,), scale_ (p,), eigenvalues_ (min(n,p),) of Z'Z/n descending,
    eigenvectors_ (p x k), factors_ (n x k) = Z @ eigenvectors_, loadings_ (p x k) in original
    units such that Xc ≈ factors_ @ loadings_.T, residuals_ (n x p) = Xc - factors_ @ loadings_.T,
    idio_var_ (p,) = residual variance (ddof=0, floored at 1e-12), explained_variance_ratio_ (k,).

    Parameters
    ----------
    k : int
        Number of factors, ``0 <= k <= rank`` where ``rank = min(n - 1, p)`` is the rank of the
        column-centred window. ``k > rank`` raises ``ValueError`` in :meth:`fit`.
    standardize : bool, default True
        If True, PCA is run on ``Z = (X - mean) / sd`` (sd with ddof=0), i.e. on the sample
        correlation matrix, as in Avellaneda & Lee (2010). If False, on ``Z = X - mean``
        (sample covariance matrix).

    Attributes
    ----------
    mean_, scale_ : ndarray (p,)
        Column means and scales (``scale_`` is all ones when ``standardize=False``).
    eigenvalues_ : ndarray (min(n, p),)
        Eigenvalues of ``S = Z'Z / n``, descending (computed as squared singular values of ``Z``
        divided by ``n``). Because of centring, the eigenvalue at position ``n`` (1-based) is
        numerically zero when ``n <= p``.
    eigenvectors_ : ndarray (p, k)
        Leading eigenvectors of ``S``. Sign convention (for determinism): in every column the
        entry with the largest absolute value is positive (first such entry on exact ties).
    factors_ : ndarray (n, k)
        Factor scores ``Z @ eigenvectors_``; uncorrelated with variances ``eigenvalues_[:k]``.
    loadings_ : ndarray (p, k)
        Loadings in original units, ``scale_[:, None] * eigenvectors_``, so that
        ``Xc ≈ factors_ @ loadings_.T``.
    residuals_ : ndarray (n, p)
        ``Xc - factors_ @ loadings_.T`` (original units).
    idio_var_ : ndarray (p,)
        Residual variances (ddof=0), floored at ``1e-12``.
    explained_variance_ratio_ : ndarray (k,)
        ``eigenvalues_[:k] / trace(S)``.
    n_samples_, n_features_, rank_ : int
        Window shape and ``min(n - 1, p)``.

    Notes
    -----
    The decomposition uses the thin SVD ``Z = U diag(s) V'`` (``numpy.linalg.svd`` with
    ``full_matrices=False``), whose cost is ``O(n p min(n, p))`` and which works directly on the
    smaller of the two Gram matrices implicitly, so ``p > n`` panels (e.g. 3000 stocks x 252
    days) are cheap. Eigen-decomposing ``Z'Z`` (p x p) is never done.

    Since ``F'F/n = diag(eigenvalues_[:k])`` and ``Z = F V' + Z (I - V V')``, the model-implied
    covariance with the ``"diag"`` residual matches the sample variances on the diagonal for
    every ``k`` and equals the full sample covariance (ddof=0, plus the 1e-12 floor) at
    ``k = rank``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).standard_normal((100, 10)) * 0.01
    >>> m = FactorModel(k=2).fit(X)
    >>> m.loadings_.shape, m.covariance().shape
    ((10, 2), (10, 10))
    """

    def __init__(self, k: int, standardize: bool = True):
        self.k = k
        self.standardize = standardize

    # ------------------------------------------------------------------ fitting
    def fit(self, X) -> FactorModel:
        """Fit the PCA factor model on a complete n x p window.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n, p)
            Complete window (no NaN/inf), ``n >= 3``, ``p >= 2``.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``X`` is not a complete 2-D array, has a zero-variance column (when
            ``standardize=True``), or ``k > min(n - 1, p)``.
        TypeError
            If ``k`` is not an integer.
        """
        k = _check_k(self.k)
        A = as_2d_array(X)
        Z, mean, scale = preprocess(A, standardize=self.standardize)
        n, p = Z.shape
        rank = min(n - 1, p)
        if k > rank:
            raise ValueError(
                f"k={k} exceeds the rank of the centred window min(n-1, p) = {rank} "
                f"(n={n}, p={p})"
            )

        # Thin SVD: Z = U diag(s) Vt, eigenvalues of Z'Z/n are s^2/n (length min(n, p)).
        _, s, Vt = np.linalg.svd(Z, full_matrices=False)
        eigenvalues = s**2 / n
        tol = (s[0] if s.size else 0.0) * max(n, p) * np.finfo(float).eps
        numerical_rank = int(np.sum(s > tol))
        if k > numerical_rank:
            warnings.warn(
                f"k={k} exceeds the numerical rank ({numerical_rank}) of the window; the "
                "trailing eigenvectors span a null space and are not identified",
                RuntimeWarning,
                stacklevel=2,
            )

        V = Vt[:k].T.copy()  # p x k
        if k > 0:
            idx = np.argmax(np.abs(V), axis=0)
            signs = np.sign(V[idx, np.arange(k)])
            signs[signs == 0] = 1.0
            V *= signs

        F = Z @ V  # n x k
        B = scale[:, None] * V  # p x k, original units
        Xc = A - mean
        resid = Xc - F @ B.T
        idio = np.maximum(resid.var(axis=0), _IDIO_FLOOR)
        total = float(np.sum(Z**2) / n)

        self.mean_ = mean
        self.scale_ = scale
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = V
        self.factors_ = F
        self.loadings_ = B
        self.residuals_ = resid
        self.idio_var_ = idio
        self.explained_variance_ratio_ = (
            eigenvalues[:k] / total if total > 0 else np.zeros(k)
        )
        self.n_samples_ = n
        self.n_features_ = p
        self.rank_ = rank
        self.k_ = k
        self._X_fit = A
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "loadings_"):
            raise RuntimeError("FactorModel is not fitted; call fit(X) first")

    # --------------------------------------------------------------- covariance
    def covariance(self, residual: str = "diag") -> np.ndarray:
        """B Cov(F) B' + D in original units. residual: "diag" (D = diag(idio_var_)) or
        "ledoit_wolf" (sklearn LedoitWolf on residuals_).

        Parameters
        ----------
        residual : {"diag", "ledoit_wolf"}
            ``"diag"``: strict factor model, ``D = diag(idio_var_)``.
            ``"ledoit_wolf"``: ``D`` is the Ledoit & Wolf (2004, J. Multivariate Analysis
            88(2)) shrinkage of the residual covariance towards a scaled identity
            (``sklearn.covariance.LedoitWolf``); its diagonal is floored at 1e-12.

        Returns
        -------
        ndarray (p, p)
            Symmetric positive semi-definite covariance in original (unstandardised) units.
            ``Cov(F)`` is the ddof=0 sample covariance of ``factors_`` (equal to
            ``diag(eigenvalues_[:k])`` up to rounding).
        """
        self._check_fitted()
        if residual not in ("diag", "ledoit_wolf"):
            raise ValueError(f"residual must be 'diag' or 'ledoit_wolf', got {residual!r}")
        F, B = self.factors_, self.loadings_
        n = F.shape[0]
        Fc = F - F.mean(axis=0)
        cov_f = Fc.T @ Fc / n
        cov_f = 0.5 * (cov_f + cov_f.T)
        common = B @ cov_f @ B.T
        if residual == "diag":
            D = np.diag(self.idio_var_)
        else:
            from sklearn.covariance import LedoitWolf

            D = LedoitWolf().fit(self.residuals_).covariance_
            d = np.diag(D)
            D = D + np.diag(np.maximum(d, _IDIO_FLOOR) - d)
        S = common + D
        return 0.5 * (S + S.T)

    # ------------------------------------------------------------ eigenportfolios
    def eigenportfolios(self) -> np.ndarray:
        """Avellaneda-Lee eigenportfolio weights p x k: Q_ij = v_ij / sigma_i (sigma = scale_).

        Avellaneda & Lee (2010), Sec. 2: the amount invested in asset i by eigenportfolio j is
        ``Q_i^(j) = v_i^(j) / sigma_i``, so the eigenportfolio return is
        ``F_j = sum_i Q_i^(j) R_i``. With ``standardize=False`` (``scale_ = 1``), ``Q = V``.

        Returns
        -------
        ndarray (p, k)
        """
        self._check_fitted()
        return self.eigenvectors_ / self.scale_[:, None]

    def residual_returns(self, X=None) -> tuple[np.ndarray, np.ndarray]:
        """Regress each asset's returns on the k eigenportfolio returns (with intercept) over X
        (default: the fitted window). Returns (residuals n x p, betas p x k).

        Implements the Avellaneda & Lee (2010) residual regression:
        ``R_i = beta_0i + sum_j beta_ij F_j + eps_i`` with ``F = X @ Q`` (Q from
        :meth:`eigenportfolios`), estimated by OLS separately for every asset over the rows of
        ``X``. On the fitted window the slopes equal ``loadings_`` (the factors are orthogonal
        and ``X @ Q`` differs from ``factors_`` only by a constant).

        Parameters
        ----------
        X : array-like (m, p), optional
            Complete returns with the same columns as the fitted window. Must have
            ``m >= k + 1`` rows. Default: the fitted window.

        Returns
        -------
        residuals : ndarray (m, p)
            OLS residuals ``eps_i`` (mean zero per column, orthogonal to ``X @ Q``).
        betas : ndarray (p, k)
            Slopes on the eigenportfolio returns (intercepts are not returned).
        """
        self._check_fitted()
        A = self._X_fit if X is None else as_2d_array(X)
        m, p = A.shape
        if p != self.n_features_:
            raise ValueError(f"X has {p} columns, the model was fitted on {self.n_features_}")
        k = self.k_
        if m < k + 1:
            raise ValueError(f"need at least k + 1 = {k + 1} rows to regress, got {m}")
        Fq = A @ self.eigenportfolios()  # m x k
        design = np.column_stack([np.ones(m), Fq])
        coef, _, rk, _ = np.linalg.lstsq(design, A, rcond=None)
        if rk < k + 1:
            warnings.warn(
                f"eigenportfolio design matrix is rank deficient ({rk} < {k + 1}); "
                "minimum-norm betas returned",
                RuntimeWarning,
                stacklevel=2,
            )
        resid = A - design @ coef
        betas = coef[1:].T.copy()
        return resid, betas
