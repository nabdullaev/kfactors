"""Card A (teammate): global minimum-variance portfolio built on the factor-model covariance.

This strategy is the **risk evaluator** of the k-curve: for each number of factors ``k`` the
walk-forward engine fits a :class:`~kfactors.model.FactorModel` with ``k`` factors, this strategy
turns its covariance estimate into the global minimum-variance (GMV) portfolio, and the
out-of-sample volatility of that portfolio tells us how good the risk model with ``k`` factors
is. A covariance matrix that under-fits (``k`` too small) or over-fits (``k`` too large) gives a
GMV portfolio with higher realised volatility.

Background
----------
The GMV portfolio solves ``min_w w' S w`` subject to ``1' w = 1`` (fully invested, shorting
allowed). The Lagrangian solution is the classical closed form (Merton 1972, "An analytic
derivation of the efficient portfolio frontier", J. Financial and Quantitative Analysis 7(4),
1851-1872)::

    w = S^{-1} 1 / (1' S^{-1} 1).

With ``p >= n`` the sample covariance is singular, so ``S`` must be a structured estimate. Here
``S = B Cov(F) B' + D`` comes from the PCA factor model (Fan, Liao & Mincheva 2013, "Large
covariance estimation by thresholding principal orthogonal complements", JRSS-B 75(4), use the
same structure), which is positive definite for every ``k`` because ``D`` has a strictly
positive diagonal. Its quality as a risk model depends on ``k``, which is exactly what the
k-curve measures (see also Ledoit & Wolf 2004 and Clarke, de Silva & Thorley 2011 for
minimum-variance portfolios built on factor covariances).

TODO (Card A): implement :meth:`MinVariance.weights`. See ``docs/tasks/card_A_minvar.md`` and the
acceptance tests in ``tests/teammates/test_card_A_minvar.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kfactors.strategy import Context, Strategy

_RESIDUALS = ("diag", "ledoit_wolf")


class MinVariance(Strategy):
    """Global minimum-variance (GMV) portfolio on the factor-model covariance.

    Parameters
    ----------
    residual : {"diag", "ledoit_wolf"}, default "diag"
        Which idiosyncratic covariance the factor model uses, passed unchanged to
        :meth:`kfactors.model.FactorModel.covariance`. ``"diag"`` = strict factor model,
        ``"ledoit_wolf"`` = Ledoit-Wolf shrunk residual covariance.

    Raises
    ------
    ValueError
        If ``residual`` is not one of ``"diag"``, ``"ledoit_wolf"`` (raised in ``__init__``,
        already implemented).

    Examples
    --------
    >>> from kfactors.engine import WalkForward, WalkForwardConfig
    >>> engine = WalkForward(panel, MinVariance(), WalkForwardConfig(window=120))  # doctest: +SKIP
    >>> result = engine.run(3)                                                    # doctest: +SKIP
    >>> result.summary()["ann_vol"]                                               # doctest: +SKIP
    """

    name = "min_variance"

    def __init__(self, residual: str = "diag"):
        if residual not in _RESIDUALS:
            raise ValueError(f"residual must be one of {_RESIDUALS}, got {residual!r}")
        self.residual = residual

    def weights(self, window, model, context: Context) -> pd.Series:
        """GMV weights ``w = S^{-1} 1 / (1' S^{-1} 1)`` with ``S = model.covariance(residual)``.

        Parameters
        ----------
        window : kfactors.data.Panel
            Past data only (rows up to and including ``context.date``) for the current
            universe. ``window.returns`` is a complete ``n x p`` DataFrame (no NaN).
        model : kfactors.model.FactorModel
            Already fitted on ``window.returns`` with ``context.k`` factors. Do **not** refit it.
        context : kfactors.strategy.Context
            Decision date, ``k`` and scratch state (not needed by this strategy).

        Returns
        -------
        pandas.Series
            Index exactly ``window.returns.columns`` (same order), float dtype, all finite,
            ``weights.sum() == 1`` (up to rounding). Short positions (negative weights) are
            allowed. ``name`` should be ``self.name``.

        Raises
        ------
        ValueError
            If the solution is not finite or ``1' S^{-1} 1`` is not positive (cannot happen for
            a positive definite ``S``, but never return NaN weights silently).

        Notes
        -----
        Step-by-step plan (TODO, Card A):

        1. ``S = model.covariance(residual=self.residual)``  -> ``p x p`` numpy array.
        2. ``ones = np.ones(p)``; solve the linear system ``S x = ones`` with
           ``np.linalg.solve`` (do **not** compute ``np.linalg.inv(S)``; solving is faster and
           more accurate).
        3. ``w = x / x.sum()``; check ``np.isfinite(w).all()`` and ``x.sum() > 0``.
        4. Return ``pd.Series(w, index=window.returns.columns, name=self.name)``.
        """
        S = model.covariance(residual=self.residual)

        x = np.linalg.solve(S, np.ones(S.shape[0]))
        denominator = x.sum()

        if not np.isfinite(x).all() or not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Invalid GMV portfolio weights")

        w = x / denominator

        return pd.Series(w, index=window.returns.columns, name=self.name)
