"""scikit-learn compatible estimators around :func:`kfactors.rules.select_k`.

Both estimators pass :func:`sklearn.utils.estimator_checks.check_estimator` without exemptions.
Degenerate inputs are handled explicitly rather than rejected:

* a single feature (``p = 1``) has no cross-section to share a factor, so ``k_ = 0`` by
  definition (``result_.info["degenerate"]`` says so);
* fewer than 3 samples raise ``ValueError`` (via ``validate_data``);
* rules that are undefined on very small panels (e.g. Onatski with ``min(n-1, p) < 6``) return
  ``k = 0`` with a :class:`kfactors.rules.base.RuleWarning`.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from kfactors.rules.base import KResult, select_k
from kfactors.utils import preprocess


def _seed(random_state):
    """Accept legacy ``np.random.RandomState`` (sklearn convention) besides int/Generator/None."""
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(np.iinfo(np.int32).max))
    return random_state


def _fit_rule(est, X) -> tuple[KResult, np.ndarray, np.ndarray, np.ndarray]:
    """Validate X, run the rule; return (result, Z, mean, scale) with Z the pre-processed data."""
    rule_kwargs = dict(est.rule_kwargs or {})
    for name in ("random_state", "kmax", "standardize", "method"):
        if name in rule_kwargs:
            raise ValueError(f"rule_kwargs must not contain {name!r}; set the estimator "
                             f"parameter {name}=... instead")
    X = validate_data(est, X, dtype=np.float64, ensure_min_samples=3)
    n, p = X.shape
    if p == 1:
        mean = X.mean(axis=0)
        sd = X.std(axis=0)
        scale = np.where(sd > 0, sd, 1.0) if est.standardize else np.ones(1)
        Z = (X - mean) / scale
        res = KResult(k=0, method=est.method, eigenvalues=np.array([float((Z**2).mean())]),
                      info={"kmax_effective": 0, "n": n, "p": 1,
                            "degenerate": "single feature: no cross-section, k = 0 by definition"})
        return res, Z, mean, scale
    res = select_k(X, est.method, kmax=est.kmax, standardize=est.standardize,
                   random_state=_seed(est.random_state), **rule_kwargs)
    Z, mean, scale = preprocess(X, standardize=est.standardize)
    return res, Z, mean, scale


class FactorCountSelector(BaseEstimator):
    """Estimate the number of factors of a panel with a registered rule.

    Parameters
    ----------
    method : str, default="onatski"
        Rule name in :data:`kfactors.rules.RULES`.
    kmax : int, default=20
        Maximum number of factors (clipped by :func:`kfactors.rules.base.effective_kmax`).
    standardize : bool, default=True
        Use the correlation (True) or covariance (False) matrix.
    random_state : None, int, Generator or RandomState
        Seed for stochastic rules (``parallel``).
    rule_kwargs : dict or None
        Extra keyword arguments for the rule (e.g. ``{"criterion": "IC_p3"}``). The keys
        ``random_state``, ``kmax``, ``standardize`` and ``method`` are estimator parameters and
        raise ``ValueError`` in ``fit`` if given here.

    Attributes
    ----------
    k_ : int
        Selected number of factors.
    result_ : KResult
        Full rule output.
    n_features_in_ : int
    feature_names_in_ : ndarray of str, only when X has string column names.
    """

    def __init__(self, method="onatski", kmax=20, standardize=True, random_state=None,
                 rule_kwargs=None):
        self.method = method
        self.kmax = kmax
        self.standardize = standardize
        self.random_state = random_state
        self.rule_kwargs = rule_kwargs

    def fit(self, X, y=None):
        """Select ``k`` on ``X`` (n x p, rows = time). ``y`` is ignored."""
        self.result_, _, _, _ = _fit_rule(self, X)
        self.k_ = int(self.result_.k)
        return self


class SelectedPCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """PCA whose number of components is chosen by a factor-number rule.

    Components are the leading eigenvectors of ``S = Z'Z / n`` of the pre-processed data
    (centred, and standardised with ``ddof=0`` if ``standardize``), with the sign convention of
    ``sklearn.utils.extmath.svd_flip(..., u_based_decision=False)`` (largest absolute loading
    positive) for determinism.

    Parameters
    ----------
    Same as :class:`FactorCountSelector`.

    Attributes
    ----------
    k_, n_components_ : int
        Selected number of components (may be 0).
    result_ : KResult
    components_ : ndarray, shape (k_, p)
        Orthonormal loadings in the pre-processed space.
    mean_, scale_ : ndarray, shape (p,)
        Column means and scales used for pre-processing (scale is 1 if not standardising).
    explained_variance_ : ndarray, shape (k_,)
        Eigenvalues of ``S`` for the kept components.
    explained_variance_ratio_ : ndarray, shape (k_,)
        Share of ``trace(S)``.
    """

    def __init__(self, method="onatski", kmax=20, standardize=True, random_state=None,
                 rule_kwargs=None):
        self.method = method
        self.kmax = kmax
        self.standardize = standardize
        self.random_state = random_state
        self.rule_kwargs = rule_kwargs

    def fit(self, X, y=None):
        """Select ``k`` and fit the ``k`` leading principal components. ``y`` is ignored."""
        res, Z, mean, scale = _fit_rule(self, X)
        n, _p = Z.shape
        k = int(res.k)
        _, s, vt = np.linalg.svd(Z, full_matrices=False)
        # sign convention: the largest-|.| loading of every component is positive
        signs = np.sign(vt[np.arange(vt.shape[0]), np.argmax(np.abs(vt), axis=1)])
        vt = vt * np.where(signs == 0, 1.0, signs)[:, None]
        ev = s**2 / n
        total = float((Z**2).sum() / n)
        self.result_ = res
        self.k_ = self.n_components_ = k
        self.components_ = vt[:k].copy()
        self.mean_, self.scale_ = mean, scale
        self.explained_variance_ = ev[:k].copy()
        self.explained_variance_ratio_ = (ev[:k] / total) if total > 0 else np.zeros(k)
        return self

    @property
    def _n_features_out(self):
        return self.n_components_

    def transform(self, X):
        """Principal-component scores ``((X - mean_) / scale_) @ components_.T``, shape (n, k_)."""
        check_is_fitted(self, "components_")
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return ((X - self.mean_) / self.scale_) @ self.components_.T
