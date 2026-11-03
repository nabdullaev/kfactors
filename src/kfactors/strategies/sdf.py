"""Reference strategy (core): factor-zoo SDF via PCA tangency portfolio.

On the window: factors f = model.factors_ (n x k); b = (Cov(f) + gamma*I)^-1 mean(f);
asset weights in original units w = (eigenvectors_ @ b) / scale_; normalised to sum |w| = 1.
k = 0 -> all-zero weights.

Background
----------
Kozak, Nagel & Santosh (2018), "Interpreting factor models", Journal of Finance 73(3), and
Kozak, Nagel & Santosh (2020), "Shrinking the cross-section", Journal of Financial Economics
135(2), argue that the stochastic discount factor (SDF) of a large cross-section of
(anomaly) portfolios is well approximated by a few leading principal components. The SDF
coefficients on the PC portfolios are the mean-variance (tangency) weights

    b = Sigma_F^{-1} mu_F,

with an optional L2 (ridge) penalty ``gamma`` (KNS 2020, Sec. 2, the L2 estimator
``b = (Sigma + gamma I)^{-1} mu``, here applied to the k PC portfolios only).

Implementation notes
--------------------
``model.factors_`` are the PC scores of the *centred* window, so their sample mean is exactly 0.
The tangency portfolio needs the factor portfolios' **mean returns**, so the factor returns used
here are the (uncentred) eigenportfolio returns of Avellaneda & Lee (2010),

    F = X Q,    Q = eigenvectors_ / scale_  (= model.eigenportfolios()),

which differ from ``factors_`` only by the constant ``mean_ @ Q``: ``Cov(F) = Cov(factors_) =
diag(eigenvalues_[:k])`` (ddof = 0, same units as the model) and ``mean(F) = mean_ @ Q``. The
asset weights ``w = Q b = (eigenvectors_ @ b) / scale_`` therefore give the portfolio return
``X w = F b``. Finally ``w`` is scaled to gross exposure ``sum |w| = 1``; this does not change
the Sharpe ratio. With ``k = rank`` (all PCs, ``n > p``) and ``gamma = 0`` this reproduces the
full-sample tangency portfolio ``Sigma^{-1} mu`` (normalised), a known-answer check used in the
tests. ``gamma`` is in the units of the factor variances (correlation units when the model is
standardised).
"""
from __future__ import annotations

import numbers

import numpy as np
import pandas as pd

from kfactors.strategy import Context, Strategy

__all__ = ["TangencyPCA"]


class TangencyPCA(Strategy):
    """Tangency (maximum-Sharpe) portfolio of the first k PCA factor portfolios.

    Parameters
    ----------
    gamma : float, default 0.0
        Ridge penalty added to the factor covariance, ``gamma >= 0``.

    Notes
    -----
    Returns weights indexed by ``window.returns.columns`` with ``sum |w| = 1``; all zeros when
    ``k = 0`` or when every estimated factor mean is zero up to round-off
    (``|mean(F_j)| <= 1e-12 * sd(F_j)``). Long-short, unconstrained.
    See the module docstring for the references and the derivation.
    """

    name = "sdf_pca_tangency"

    def __init__(self, gamma: float = 0.0):
        if isinstance(gamma, bool) or not isinstance(gamma, numbers.Real) or not (
                np.isfinite(gamma) and gamma >= 0):
            raise ValueError(f"gamma must be a finite number >= 0, got {gamma!r}")
        self.gamma = float(gamma)

    def weights(self, window, model, context: Context) -> pd.Series:
        cols = window.returns.columns
        k = model.eigenvectors_.shape[1]
        if k == 0:
            return pd.Series(0.0, index=cols)
        X = window.returns.to_numpy(dtype=float)
        Q = model.eigenportfolios()                 # p x k
        F = X @ Q                                   # uncentred factor-portfolio returns
        mu = F.mean(axis=0)
        Fc = F - mu
        cov = Fc.T @ Fc / F.shape[0]                # ddof = 0, = diag(eigenvalues_[:k])
        # factor means indistinguishable from zero (relative round-off) -> no signal, no bet;
        # normalising a round-off-sized b would give an arbitrary direction
        if np.all(np.abs(mu) <= 1e-12 * np.sqrt(np.maximum(np.diag(cov), 0.0))):
            return pd.Series(0.0, index=cols)
        A = 0.5 * (cov + cov.T) + self.gamma * np.eye(k)
        try:
            b = np.linalg.solve(A, mu)
        except np.linalg.LinAlgError:
            b = np.linalg.pinv(A) @ mu
        w = Q @ b
        gross = np.abs(w).sum()
        if not np.isfinite(gross):
            raise FloatingPointError("non-finite tangency weights (singular factor covariance)")
        if gross == 0.0:
            return pd.Series(0.0, index=cols)
        return pd.Series(w / gross, index=cols)
