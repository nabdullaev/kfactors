"""Card B (teammate): Avellaneda & Lee (2010) residual mean-reversion (s-score) stat-arb.

This strategy is the **trading evaluator** of the k-curve: the residuals of a PCA factor model
with ``k`` factors are treated as mean-reverting, and the trading rule bets on their reversion.
If ``k`` is too small the "residuals" still contain common factor moves (which do not
mean-revert); if ``k`` is too large the factors absorb the idiosyncratic moves we want to trade.

Background (Avellaneda, M. & Lee, J.-H. (2010), "Statistical arbitrage in the US equities
market", Quantitative Finance 10(7), 761-782; Sec. 2-4 and the estimation appendix)
----------------------------------------------------------------------------------------------
1. On the last ``L`` (= 60) rows, regress every asset's returns on the ``k`` eigenportfolio
   returns: ``R_i = beta_0i + sum_j beta_ij F_j + eps_i``. This is
   :meth:`kfactors.model.FactorModel.residual_returns`.
2. The cumulative residual ``X_i(t) = sum_{s<=t} eps_i(s)`` is modelled as an
   Ornstein-Uhlenbeck (OU) process, i.e. in discrete time an AR(1):
   ``X(t+1) = a + b X(t) + zeta(t+1)``.
3. OU parameters: ``kappa = -ln(b)`` (speed of mean reversion per period, AL multiply by 252 to
   annualise), mean-reversion time ``tau = 1/kappa`` periods, equilibrium mean
   ``m = a / (1 - b)`` and equilibrium standard deviation
   ``sigma_eq = sqrt(Var(zeta) / (1 - b^2))``.
4. s-score ``s = (X(L) - m) / sigma_eq``: how many equilibrium standard deviations the asset
   is away from its equilibrium. AL use a "modified" s-score with ``m`` centred across assets
   (``m_i - mean(m)``) because the regression intercept makes ``X(L) = 0``.
5. Trade only assets with fast mean reversion: ``tau < L / 2`` (AL: ``kappa > 252/30``).
6. Bang-bang rule (AL thresholds): open long if ``s < -1.25``, open short if ``s > +1.25``,
   close a long if ``s > -0.50``, close a short if ``s < +0.75``.
7. Every residual position is hedged with the eigenportfolios: ``+1`` dollar of asset ``i``
   comes with ``-beta_ij`` dollars of eigenportfolio ``j``, so the book is factor-neutral.

The module is split into small functions that can be implemented and tested one by one:
:func:`fit_ar1` -> :func:`s_scores` -> :func:`update_positions` -> :func:`hedge_weights` ->
:meth:`ResidualReversion.weights`.

TODO (Card B): implement the five functions/methods that raise ``NotImplementedError``. See
``docs/tasks/card_B_statarb.md`` and ``tests/teammates/test_card_B_statarb.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kfactors.strategy import Context, Strategy

#: Columns of the DataFrame returned by :func:`s_scores`, in this order.
S_SCORE_COLUMNS = ("a", "b", "kappa", "tau", "m", "sigma_eq", "s", "eligible")

#: Minimum number of rows of the residual regression window.
MIN_ESTIMATION_ROWS = 10


def fit_ar1(x) -> tuple[float, float, float]:
    """Least-squares AR(1) fit ``x[t+1] = a + b * x[t] + zeta[t+1]``.

    Parameters
    ----------
    x : array-like, shape (L,)
        One series (e.g. a cumulative residual), ``L >= 3``, all finite.

    Returns
    -------
    a, b, var_zeta : float
        OLS intercept and slope of ``x[1:]`` on ``[1, x[:-1]]`` and the variance of the
        regression residuals ``zeta`` with ``ddof=0`` (i.e. ``mean(zeta**2)``). If ``x[:-1]`` is
        constant (slope not identified) return ``(nan, nan, nan)``.

    Raises
    ------
    ValueError
        If ``x`` is not 1-D, has fewer than 3 elements or contains NaN/inf.

    Notes
    -----
    Avellaneda & Lee (2010), appendix: the OU parameters are obtained from this regression.
    Closed form (simple linear regression)::

        x0, x1 = x[:-1], x[1:]
        b = sum((x0 - mean(x0)) * (x1 - mean(x1))) / sum((x0 - mean(x0))**2)
        a = mean(x1) - b * mean(x0)
        zeta = x1 - (a + b * x0)

    TODO (Card B, step 1).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("x must be a finite 1-D array with at least 3 elements")

    x0, x1 = x[:-1], x[1:]
    x0_centered = x0 - x0.mean()
    denominator = np.sum(x0_centered ** 2)
    if denominator == 0:
        return (np.nan, np.nan, np.nan)

    b = float(np.sum(x0_centered * (x1 - x1.mean())) / denominator)
    a = float(x1.mean() - b * x0.mean())
    zeta = x1 - (a + b * x0)
    return a, b, float(np.mean(zeta ** 2))


def s_scores(residuals, *, max_tau: float, center: bool = True) -> pd.DataFrame:
    """Per-asset OU parameters and s-scores from a block of residual returns.

    Parameters
    ----------
    residuals : array-like, shape (L, p)
        Residual returns (rows = time, oldest first), e.g. the first output of
        ``model.residual_returns(...)``. All finite, ``L >= 3``.
    max_tau : float
        Maximum mean-reversion time in periods (rows). Assets with ``tau >= max_tau`` are not
        eligible. AL use ``max_tau = L / 2`` (30 trading days for ``L = 60``).
    center : bool, default True
        If True, replace ``m`` by ``m - mean(m[eligible])`` (the AL "modified s-score"; only
        eligible assets enter the mean, and nothing is centred when no asset is eligible).

    Returns
    -------
    pandas.DataFrame
        Index ``RangeIndex(p)`` (column positions of ``residuals``, also when a DataFrame is
        passed), columns exactly :data:`S_SCORE_COLUMNS` (in this order):

        * ``a, b`` : AR(1) coefficients of ``X = cumsum(residuals, axis=0)`` (:func:`fit_ar1`).
        * ``kappa`` : ``-ln(b)`` per period (NaN unless ``0 < b < 1``).
        * ``tau`` : ``1 / kappa`` periods (NaN unless ``0 < b < 1``).
        * ``m`` : ``a / (1 - b)``, centred if ``center`` (NaN unless ``0 < b < 1``).
        * ``sigma_eq`` : ``sqrt(var_zeta / (1 - b**2))`` (NaN unless ``0 < b < 1``).
        * ``s`` : ``(X[-1] - m) / sigma_eq`` for eligible assets, **NaN for ineligible ones**.
        * ``eligible`` : bool, ``0 < b < 1`` and ``var_zeta > 0`` and ``tau < max_tau``.

    Raises
    ------
    ValueError
        If ``residuals`` is not 2-D, has fewer than 3 rows, contains NaN/inf, or
        ``max_tau <= 0``.

    Notes
    -----
    TODO (Card B, step 2): ``X = np.cumsum(residuals, axis=0)``; loop over columns calling
    :func:`fit_ar1`; compute the columns above with numpy (watch out for ``log`` of
    ``b <= 0`` and division by ``1 - b`` when ``b >= 1``: those assets are simply ineligible).
    """
    residuals = np.asarray(residuals, dtype=float)
    if (residuals.ndim != 2 or residuals.shape[0] < 3
            or not np.isfinite(residuals).all() or not np.isfinite(max_tau)
            or max_tau <= 0):
        raise ValueError("residuals must be a finite 2-D array and max_tau must be positive")

    cumulative = np.cumsum(residuals, axis=0)
    p = residuals.shape[1]
    values = np.full((p, 7), np.nan, dtype=float)
    eligible = np.zeros(p, dtype=bool)

    for i in range(p):
        a, b, var_zeta = fit_ar1(cumulative[:, i])
        values[i, :2] = a, b
        if 0 < b < 1:
            kappa = -np.log(b)
            tau = 1 / kappa
            m = a / (1 - b)
            sigma_eq = np.sqrt(var_zeta / (1 - b ** 2))
            values[i, 2:6] = kappa, tau, m, sigma_eq
            eligible[i] = var_zeta > 0 and tau < max_tau

    if center and eligible.any():
        values[:, 4] = values[:, 4] - values[eligible, 4].mean()

    values[eligible, 6] = ((cumulative[-1, eligible] - values[eligible, 4])
                           / values[eligible, 5])
    result = pd.DataFrame(values, columns=S_SCORE_COLUMNS[:-1])
    result["eligible"] = eligible
    return result


def update_positions(s: pd.Series, previous: pd.Series | None = None, *,
                     s_open: float = 1.25, s_close_long: float = 0.50,
                     s_close_short: float = 0.75) -> pd.Series:
    """Avellaneda-Lee bang-bang rule: new residual positions in {-1, 0, +1}.

    Parameters
    ----------
    s : pandas.Series
        Today's s-scores indexed by asset id; NaN = not tradable today.
    previous : pandas.Series or None
        Yesterday's positions (values in {-1, 0, 1}) indexed by asset id. Assets missing from
        ``previous`` (or ``previous=None``) start flat (0). Assets in ``previous`` but not in
        ``s`` are dropped (they left the universe, i.e. the position is closed).
    s_open, s_close_long, s_close_short : float
        AL thresholds (Sec. 4): ``s_bo = s_so = 1.25``, ``s_sc = 0.50``, ``s_bc = 0.75``.
        Must satisfy ``s_open > 0``, ``s_close_long >= 0``, ``s_close_short >= 0``,
        ``s_close_long < s_open``, ``s_close_short < s_open``.

    Returns
    -------
    pandas.Series
        Same index as ``s`` (same order), integer dtype, values in {-1, 0, 1}. All
        inequalities are **strict**. For each asset with previous position ``q``:

        * ``s`` is NaN -> 0 (forced exit / stay flat).
        * ``q == 0``: ``+1`` if ``s < -s_open``; ``-1`` if ``s > s_open``; else 0.
        * ``q == +1`` (long): ``-1`` if ``s > s_open`` (close and reverse); else ``0`` if
          ``s > -s_close_long`` (close the long); else stay ``+1``.
        * ``q == -1`` (short): ``+1`` if ``s < -s_open``; else ``0`` if ``s < s_close_short``
          (close the short); else stay ``-1``.

    Raises
    ------
    ValueError
        On invalid thresholds or ``previous`` values outside {-1, 0, 1}.

    Notes
    -----
    TODO (Card B, step 3). Vectorise with numpy boolean masks or write a clear loop; both are
    fine (p is at most a few thousand).
    """
    if not (s_open > 0 and s_close_long >= 0 and s_close_short >= 0
            and s_close_long < s_open and s_close_short < s_open):
        raise ValueError("invalid Avellaneda-Lee thresholds")
    if not isinstance(s, pd.Series):
        raise ValueError("s must be a pandas Series")

    if previous is None:
        previous = pd.Series(0, index=s.index, dtype=int)
    else:
        if not isinstance(previous, pd.Series):
            raise ValueError("previous must be a pandas Series or None")
        if not previous.isin((-1, 0, 1)).all():
            raise ValueError("previous positions must be in {-1, 0, 1}")
        previous = previous.reindex(s.index, fill_value=0).astype(int)

    positions = np.zeros(len(s), dtype=int)
    for j, score in enumerate(s.to_numpy(dtype=float)):
        if not np.isfinite(score):
            continue
        old = previous.iloc[j]
        if old == 0:
            positions[j] = 1 if score < -s_open else -1 if score > s_open else 0
        elif old == 1:
            positions[j] = -1 if score > s_open else 0 if score > -s_close_long else 1
        else:
            positions[j] = 1 if score < -s_open else 0 if score < s_close_short else -1
    return pd.Series(positions, index=s.index, dtype=int)


def hedge_weights(positions, eigenportfolios, betas) -> np.ndarray:
    """Factor-hedged, gross-normalised portfolio weights.

    Parameters
    ----------
    positions : array-like, shape (p,)
        Residual positions ``q`` in {-1, 0, 1}.
    eigenportfolios : ndarray, shape (p, k)
        ``Q = model.eigenportfolios()`` (column ``j`` = dollars in each asset per unit of
        eigenportfolio ``j``). ``k = 0`` (shape ``(p, 0)``) means no hedge.
    betas : ndarray, shape (p, k)
        Slopes of each asset on the eigenportfolio returns (second output of
        ``model.residual_returns``).

    Returns
    -------
    ndarray, shape (p,)
        ``h = q - Q @ (betas.T @ q)`` divided by ``sum(|h|)`` so that the gross exposure
        ``sum(|w|)`` is exactly 1. If ``q`` is all zero (or ``h`` is numerically zero:
        ``sum(|h|) <= 1e-12``) return all zeros. Always finite.

    Raises
    ------
    ValueError
        On shape mismatch.

    Notes
    -----
    Why it hedges (AL 2010, Sec. 2): asset ``i`` earns ``beta_0i + sum_j beta_ij F_j + eps_i``;
    holding ``q_i`` of asset ``i`` and ``-q_i beta_ij`` units of eigenportfolio ``j`` earns
    ``q_i (beta_0i + eps_i)``, so the book has zero exposure to every factor ``F_j`` on the
    regression window.

    TODO (Card B, step 4).
    """
    q = np.asarray(positions, dtype=float)
    Q = np.asarray(eigenportfolios, dtype=float)
    B = np.asarray(betas, dtype=float)
    if q.ndim != 1 or Q.ndim != 2 or B.ndim != 2:
        raise ValueError("positions must be 1-D and factor matrices must be 2-D")
    p = len(q)
    if Q.shape != B.shape or Q.shape[0] != p:
        raise ValueError("positions, eigenportfolios and betas have incompatible shapes")

    hedged = q - Q @ (B.T @ q)
    gross = np.abs(hedged).sum()
    if gross <= 1e-12:
        return np.zeros(p, dtype=float)
    return hedged / gross


class ResidualReversion(Strategy):
    """Avellaneda & Lee (2010) s-score residual mean-reversion strategy.

    Parameters
    ----------
    estimation_window : int, default 60
        ``L``: number of most recent window rows used for the residual regression and the OU
        fit (AL: 60 trading days). If the window is shorter, all its rows are used.
    s_open, s_close_long, s_close_short : float
        AL thresholds, see :func:`update_positions`.
    max_tau : float or None, default None
        Maximum mean-reversion time in rows; ``None`` = ``L_eff / 2`` where ``L_eff`` is the
        number of rows actually used (AL: 30 days for ``L = 60``).
    center : bool, default True
        Use the AL modified (cross-sectionally centred) s-score, see :func:`s_scores`.

    Raises
    ------
    TypeError
        If ``estimation_window`` is not an integer (bools are rejected).
    ValueError
        If ``estimation_window < MIN_ESTIMATION_ROWS``; if a threshold is not finite or the
        thresholds violate the conditions of :func:`update_positions`; or if ``max_tau`` is not
        ``None`` and not a finite number ``> 0``. Parameters are validated here (already
        implemented by the core team), so a bad configuration fails at construction time
        instead of at the first decision inside the engine.

    Notes
    -----
    State: the residual positions ``q`` (pd.Series of int indexed by asset id) are stored in
    ``context.state["positions"]`` after every call and read back at the next decision
    (``context.state`` persists over one walk-forward run). The last s-score table is stored in
    ``context.state["s_scores"]`` for diagnostics. A fresh :class:`Context` means "start flat".
    """

    name = "residual_reversion"

    def __init__(self, estimation_window: int = 60, s_open: float = 1.25,
                 s_close_long: float = 0.50, s_close_short: float = 0.75,
                 max_tau: float | None = None, center: bool = True):
        if isinstance(estimation_window, bool) or not isinstance(estimation_window,
                                                                 (int, np.integer)):
            raise TypeError(f"estimation_window must be an int, got {estimation_window!r}")
        if estimation_window < MIN_ESTIMATION_ROWS:
            raise ValueError(f"estimation_window must be >= {MIN_ESTIMATION_ROWS}, "
                             f"got {estimation_window}")
        thresholds = {"s_open": s_open, "s_close_long": s_close_long,
                      "s_close_short": s_close_short}
        for label, value in thresholds.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{label} must be finite, got {value!r}")
        if not (s_open > 0 and s_close_long >= 0 and s_close_short >= 0
                and s_close_long < s_open and s_close_short < s_open):
            raise ValueError("thresholds must satisfy s_open > 0, 0 <= s_close_long < s_open "
                             f"and 0 <= s_close_short < s_open, got {thresholds}")
        if max_tau is not None and not (np.isfinite(float(max_tau)) and max_tau > 0):
            raise ValueError(f"max_tau must be None or a finite number > 0, got {max_tau!r}")
        self.estimation_window = int(estimation_window)
        self.s_open = s_open
        self.s_close_long = s_close_long
        self.s_close_short = s_close_short
        self.max_tau = max_tau
        self.center = center

    def weights(self, window, model, context: Context) -> pd.Series:
        """Hedged s-score positions for the decision date ``context.date``.

        Parameters
        ----------
        window : kfactors.data.Panel
            Complete past window (no NaN) for the current universe.
        model : kfactors.model.FactorModel
            Fitted on ``window.returns`` with ``context.k`` factors. Do not refit.
        context : kfactors.strategy.Context
            ``context.state`` carries the positions between decisions.

        Returns
        -------
        pandas.Series
            Index exactly ``window.returns.columns``; finite; gross exposure ``sum(|w|)`` is 1
            when at least one residual position is open and 0 otherwise.

        Raises
        ------
        ValueError
            If ``L_eff = min(estimation_window, n) < MIN_ESTIMATION_ROWS`` or
            ``model.k_ + 2 > L_eff`` (too few rows to regress on ``k`` factors).

        Notes
        -----
        Step-by-step plan (TODO, Card B, step 5):

        1. ``R = window.returns.to_numpy()``; ``L_eff = min(self.estimation_window, len(R))``;
           validate as above.
        2. ``resid, betas = model.residual_returns(R[-L_eff:])`` (only past rows!).
        3. ``max_tau = L_eff / 2 if self.max_tau is None else self.max_tau`` (test ``is None``,
           not truthiness), then
           ``scores = s_scores(resid, max_tau=max_tau, center=self.center)``.
        4. ``s = pd.Series(scores["s"].to_numpy(), index=window.returns.columns)``.
        5. ``q = update_positions(s, context.state.get("positions"), s_open=..., ...)``.
        6. Save ``context.state["positions"] = q`` and ``context.state["s_scores"] = scores``.
        7. ``w = hedge_weights(q.to_numpy(), model.eigenportfolios(), betas)``.
        8. Return ``pd.Series(w, index=window.returns.columns, name=self.name)``.
        """
        returns = window.returns.to_numpy()
        effective_length = min(self.estimation_window, len(returns))
        if effective_length < MIN_ESTIMATION_ROWS:
            raise ValueError("too few rows for residual reversion")
        if model.k_ + 2 > effective_length:
            raise ValueError("too few rows for the requested number of factors")

        residuals, betas = model.residual_returns(returns[-effective_length:])
        max_tau = effective_length / 2 if self.max_tau is None else self.max_tau
        scores = s_scores(residuals, max_tau=max_tau, center=self.center)
        columns = window.returns.columns
        score_series = pd.Series(scores["s"].to_numpy(), index=columns)
        positions = update_positions(
            score_series,
            context.state.get("positions"),
            s_open=self.s_open,
            s_close_long=self.s_close_long,
            s_close_short=self.s_close_short,
        )
        context.state["positions"] = positions
        context.state["s_scores"] = scores
        weights = hedge_weights(positions.to_numpy(), model.eigenportfolios(), betas)
        return pd.Series(weights, index=columns, name=self.name)
