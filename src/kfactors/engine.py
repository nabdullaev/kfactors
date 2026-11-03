"""Leakage-safe walk-forward engine (see docs/design.md, 'Engine timing').

The engine turns a :class:`~kfactors.strategy.Strategy` plug-in into a realised return series
for a given choice of the number of factors ``k``: a fixed integer, a selection rule from
:data:`kfactors.RULES` applied inside every estimation window, or a user callable.

Timing contract (row positions ``t`` of ``panel.returns``)
-----------------------------------------------------------
At every decision row ``t`` (``t0, t0 + rebalance, t0 + 2*rebalance, ...``):

1. **Universe** = assets with ``tradable[t] == True`` and no NaN in rows
   ``t - window + 1 .. t``. When ``config.standardize`` is True, assets whose window variance is
   numerically zero are also dropped (they cannot be standardised; the same tolerance as
   :func:`kfactors.utils.preprocess`). The test only uses rows ``<= t``.
2. The strategy receives the window ``rows t - window + 1 .. t`` x universe as a
   :class:`~kfactors.data.panel.Panel`; it never sees rows ``> t``.
3. ``k``: a fixed int (clipped to the window rank ``min(window - 1, n_assets)``), a rule name
   (``select_k`` on the window, in-window standardisation) or ``callable(window ndarray) -> int``
   (also clipped to the rank).
4. ``model = FactorModel(k, standardize).fit(window)``.
5. ``w_t = strategy.weights(window, model, context)``.
6. ``w_t`` earns rows ``t + 1 + lag .. t + rebalance + lag``. By default the weights are held
   constant (no drift, ``config.drift=False``): this is the usual simplification, but holding
   constant *NAV weights* implicitly re-balances the book on every row and that intra-period
   turnover is **not charged**, so with ``rebalance > 1`` net returns are optimistic relative to
   ``rebalance = 1``. With ``config.drift=True`` the weights drift with returns (buy-and-hold
   inside the period, ``w <- w * (1 + r) / (1 + w'r)``, cash earning 0) and the next turnover
   is measured against the drifted book.
   A NaN return of a held asset is an **exit at the last price**: the asset contributes 0 from
   its first NaN row to the end of that holding period, even if it has returns again later in
   the period (no free re-entry; it can only be bought back at the next decision, which pays
   the cost). The number of such exits is ``info["n_nan_exits"]``; there is no delisting
   return haircut.
7. Turnover ``sum_j |w_t,j - w_prev,j|`` (an asset missing on one side counts as 0; ``w_prev``
   is the drifted book when ``drift=True``) is charged ``cost_bps / 1e4 * turnover`` on the
   first holding row only.

Decisions with fewer than ``min_assets`` eligible assets are *skipped*. Before the first executed
decision a skip produces no output (warm-up). After it, a skip means the book is flat (all
weights 0) for that holding period: the period earns 0 and pays the cost of closing the previous
book. Because consecutive holding periods tile the rows exactly, the realised series is
contiguous from the first executed holding row to ``min(config.end, last row)``.

The walk-forward (rolling-origin) evaluation protocol follows Tashman (2000), "Out-of-sample
tests of forecasting accuracy", International Journal of Forecasting 16(4); the transaction-cost
model (linear in turnover) and the no-drift convention are the standard simplifications used in
e.g. DeMiguel, Garlappi & Uppal (2009), RFS 22(5), Sec. 3.

The strategy window carries only static descriptors of the panel in ``window.meta`` (see
:data:`WINDOW_META_KEYS`); keys that describe the whole sample (``end``, ``ticker_splits``,
``symbol``, ``first_complete_date``, ...) are dropped because they contain information from after
the decision date. Strategies must not derive signals from ``window.meta``.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kfactors import metrics as _m
from kfactors.data.panel import Panel
from kfactors.model import FactorModel
from kfactors.rules import RULES, select_k
from kfactors.strategy import Context, Strategy

__all__ = ["WINDOW_META_KEYS", "BacktestResult", "WalkForward", "WalkForwardConfig"]

#: ``Panel.meta`` keys passed on to the strategy window (static descriptors only). Everything
#: else (sample end, ticker splits, symbol maps, first complete date, counts, ...) may encode
#: information from after the decision date and is not exposed to the strategy.
WINDOW_META_KEYS = ("periods_per_year", "source", "units", "licence", "citation", "region",
                    "weighting", "quote", "dataset", "jkp_freq", "french_freq")

#: ``select_k`` arguments set by the engine itself; they cannot appear in ``rule_kwargs``.
_RESERVED_RULE_KWARGS = ("kmax", "standardize", "random_state")


@dataclass
class WalkForwardConfig:
    """Walk-forward settings.

    Attributes
    ----------
    window : int
        Estimation window length in rows (>= 3). The default rule ``onatski`` needs
        ``min(window - 1, n_assets) >= 6``.
    rebalance : int
        A decision is taken every ``rebalance`` rows; weights are held for ``rebalance`` rows.
    lag : int
        Execution delay in rows: weights decided at ``t`` first earn row ``t + 1 + lag``.
    cost_bps : float
        Linear transaction cost in basis points per unit of turnover (one-way, ``sum |dw|``),
        charged on the first holding row of every decision.
    min_assets : int
        Decisions with fewer eligible assets are skipped (>= 2).
    standardize : bool
        PCA on the correlation (True) or covariance (False) matrix of the window; also passed to
        the selection rules.
    kmax : int
        Upper bound passed to the selection rules (clipped inside every rule).
    start : str or None
        First decision date (the first row with date >= start, but never before row
        ``window - 1``). Default: row ``window - 1``. A naive date on a tz-aware panel index is
        interpreted in the index time zone (as in :meth:`Panel.subset`).
    end : str or None
        Last date used: rows after ``end`` are ignored entirely (no window or holding row
        reaches past it). Same time-zone handling as ``start``.
    rule_kwargs : dict
        Per-rule keyword arguments ``{rule_name: {...}}`` forwarded to
        :func:`kfactors.rules.select_k`. Keys must be rule names in :data:`kfactors.RULES`
        and values dicts; ``kmax``, ``standardize`` and ``random_state`` are set by the engine
        and are rejected inside them (``ValueError``).
    random_state : int or None
        Seed forwarded to stochastic rules (``parallel``). The same seed is used in every
        window, so a rule's choice at ``t`` depends on rows ``<= t`` only. A numpy
        ``Generator`` is accepted for backward compatibility but makes rule choices depend on
        the call order (not reproducible across ``run_many(n_jobs != 1)``).
    drift : bool
        False (default): weights are held constant within a holding period (implicit,
        uncharged re-balancing on every row). True: weights drift with returns inside the
        period and the next turnover is measured against the drifted book. Added option; the
        default keeps the original behaviour.
    """

    window: int = 120
    rebalance: int = 1          # decide every `rebalance` rows
    lag: int = 0                # execution delay in rows
    cost_bps: float = 0.0       # per unit turnover, charged on the first holding row
    min_assets: int = 10
    standardize: bool = True
    kmax: int = 20
    start: str | None = None    # first decision date (default: first feasible row)
    end: str | None = None
    rule_kwargs: dict = field(default_factory=dict)  # per-rule kwargs, {rule_name: {...}}
    random_state: int | None = 0
    drift: bool = False         # let weights drift with returns inside a holding period


def _validate_config(c: WalkForwardConfig) -> None:
    def as_int(name, lo):
        v = getattr(c, name)
        if isinstance(v, bool) or not isinstance(v, numbers.Integral):
            raise TypeError(f"config.{name} must be an integer, got {v!r}")
        if v < lo:
            raise ValueError(f"config.{name} must be >= {lo}, got {v}")

    as_int("window", 3)
    as_int("rebalance", 1)
    as_int("lag", 0)
    as_int("min_assets", 2)
    as_int("kmax", 1)
    if not (isinstance(c.cost_bps, numbers.Real) and math.isfinite(c.cost_bps)
            and c.cost_bps >= 0):
        raise ValueError(f"config.cost_bps must be a finite number >= 0, got {c.cost_bps!r}")
    if not isinstance(c.rule_kwargs, dict):
        raise TypeError("config.rule_kwargs must be a dict {rule_name: {kwarg: value}}")
    for rule, kw in c.rule_kwargs.items():
        if rule not in RULES:
            raise ValueError(f"config.rule_kwargs has an unknown rule {rule!r}; "
                             f"available rules: {sorted(RULES)}")
        if not isinstance(kw, dict):
            raise TypeError(f"config.rule_kwargs[{rule!r}] must be a dict, got "
                            f"{type(kw).__name__}")
        bad = [a for a in _RESERVED_RULE_KWARGS if a in kw]
        if bad:
            raise ValueError(f"config.rule_kwargs[{rule!r}] sets {bad}; these are taken from "
                             f"the config fields (config.kmax, config.standardize, "
                             f"config.random_state)")
    rs = c.random_state
    if rs is not None and (isinstance(rs, bool) or not isinstance(
            rs, (numbers.Integral, np.random.Generator))):
        raise TypeError(f"config.random_state must be None or an int, got {rs!r}")
    if not isinstance(c.drift, (bool, np.bool_)):
        raise TypeError(f"config.drift must be a bool, got {c.drift!r}")


@dataclass
class BacktestResult:
    """Outcome of one walk-forward run.

    Attributes
    ----------
    gross, net : pandas.Series
        Realised portfolio returns indexed by holding date (contiguous rows). ``net`` = ``gross``
        minus ``cost_bps / 1e4 * turnover`` on the first holding row of every decision.
    weights : pandas.DataFrame
        Decision date x assets (every asset that was ever in an executed universe); a flat
        (skipped) decision after the first trade is an all-zero row.
    turnover : pandas.Series
        ``sum |w_t - w_prev|`` indexed by execution (first holding) date.
    k_used : pandas.Series
        Number of factors actually used, by executed decision date (int).
    k_spec : object
        The ``k`` given to :meth:`WalkForward.run` (int, rule name or callable).
    strategy : str
        ``strategy.name``.
    config : WalkForwardConfig
    periods_per_year : int
    skipped : pandas.DatetimeIndex
        Decision dates with fewer than ``min_assets`` eligible assets (warm-up skips included).
    universe_size : pandas.Series
        Number of eligible assets at every decision date (executed or skipped).
    info : dict
        ``n_decisions``, ``n_executed``, ``n_skipped``, ``n_clipped`` (fixed/callable k clipped to
        the window rank), ``k_source`` and ``n_nan_exits`` (held positions that met a NaN
        return inside their holding period and were exited at the last price).
    """

    gross: pd.Series            # realised gross portfolio returns indexed by holding date
    net: pd.Series
    weights: pd.DataFrame       # decision date x assets
    turnover: pd.Series         # by execution (first holding) date
    k_used: pd.Series           # by decision date
    k_spec: object              # int or rule name/callable given to run()
    strategy: str
    config: WalkForwardConfig
    periods_per_year: int
    skipped: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    universe_size: pd.Series = field(default_factory=lambda: pd.Series(dtype="int64"))
    info: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """sharpe_gross, sharpe_net, ann_return, ann_vol, max_drawdown, mean_turnover,
        k_mean, k_median, n_periods.

        ``ann_return`` (geometric), ``ann_vol`` and ``max_drawdown`` are computed on the **net**
        series with :mod:`kfactors.metrics`; Sharpe ratios are annualised with
        ``periods_per_year`` and contain no risk-free adjustment. ``mean_turnover`` is the mean
        turnover per decision. Also reports ``n_decisions`` and ``n_skipped``.
        """
        ppy = self.periods_per_year
        k = self.k_used.to_numpy(dtype=float)
        return {
            "sharpe_gross": _m.sharpe(self.gross, ppy),
            "sharpe_net": _m.sharpe(self.net, ppy),
            "ann_return": _m.annual_return(self.net, ppy),
            "ann_vol": _m.annual_vol(self.net, ppy),
            "max_drawdown": _m.max_drawdown(self.net),
            "mean_turnover": float(self.turnover.mean()) if len(self.turnover) else float("nan"),
            "k_mean": float(k.mean()) if k.size else float("nan"),
            "k_median": float(np.median(k)) if k.size else float("nan"),
            "n_periods": int(self.net.size),
            "n_decisions": int(self.info.get("n_decisions", len(self.universe_size))),
            "n_skipped": len(self.skipped),
        }


@dataclass(frozen=True)
class _Decision:
    t: int                  # decision row
    cols: np.ndarray        # universe column positions (int, increasing)
    hold: np.ndarray        # holding row positions (non-empty, contiguous)


def _check_k_spec(k):
    """Validate a k spec early; returns ('fixed'|'rule'|'callable', normalised spec)."""
    if isinstance(k, bool):
        raise TypeError("k must be an int, a rule name or a callable, not bool")
    if isinstance(k, numbers.Integral):
        if k < 0:
            raise ValueError(f"fixed k must be >= 0, got {k}")
        return "fixed", int(k)
    if isinstance(k, str):
        if k not in RULES:
            raise ValueError(f"unknown rule {k!r}; available rules: {sorted(RULES)}")
        return "rule", k
    if callable(k):
        return "callable", k
    raise TypeError(f"k must be an int, a rule name or a callable, got {type(k).__name__}")


def _as_k_int(v, src: str) -> int:
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        if isinstance(v, numbers.Real) and float(v).is_integer():
            v = int(v)
        else:
            raise TypeError(f"{src} must return an integer number of factors, got {v!r}")
    v = int(v)
    if v < 0:
        raise ValueError(f"{src} returned a negative number of factors ({v})")
    return v


def _callable_name(fn) -> str:
    return "callable:" + getattr(fn, "__name__", type(fn).__name__)


def _rule_ks_task(engine: WalkForward, rule: str, plan: list[_Decision], fp) -> tuple:
    """Worker: k chosen by ``rule`` at every executed decision (used by run_many)."""
    return rule, {d.t: engine._rule_k(rule, d, fp) for d in plan
                  if d.cols.size >= engine.config.min_assets}


def _run_task(engine: WalkForward, k, plan: list[_Decision]) -> BacktestResult:
    return engine._run_plan(k, plan)


class WalkForward:
    """Walk-forward backtest of a strategy for a given number-of-factors specification.

    Parameters
    ----------
    panel : Panel
        Returns and point-in-time tradability mask.
    strategy : Strategy
        Plug-in returning weights from a past-only window and a fitted :class:`FactorModel`.
        Per-run state must live in ``context.state`` (a fresh dict for every run).
    config : WalkForwardConfig, optional

    Examples
    --------
    >>> from kfactors.strategies import TangencyPCA                       # doctest: +SKIP
    >>> wf = WalkForward(panel, TangencyPCA(), WalkForwardConfig(window=120))  # doctest: +SKIP
    >>> res = wf.run("onatski"); res.summary()["sharpe_net"]             # doctest: +SKIP
    """

    def __init__(self, panel: Panel, strategy: Strategy, config: WalkForwardConfig | None = None):
        self.panel = panel
        self.strategy = strategy
        self.config = config or WalkForwardConfig()
        self._rule_cache: dict = {}

    # ----------------------------------------------------------------------- planning
    def _row_bounds(self) -> tuple[int, int]:
        """(first decision row, last usable row)."""
        c = self.config
        idx = self.panel.returns.index
        n = len(idx)
        last = n - 1
        if c.end is not None:
            last = int(idx.searchsorted(_as_index_ts(c.end, idx), side="right")) - 1
        first = c.window - 1
        if c.start is not None:
            first = max(first, int(idx.searchsorted(_as_index_ts(c.start, idx), side="left")))
        return first, last

    def _plan(self) -> list[_Decision]:
        """Decision rows, universes and holding rows (uses no row after each decision)."""
        c = self.config
        _validate_config(c)
        if not isinstance(self.panel, Panel):
            raise TypeError("panel must be a kfactors.data.panel.Panel")
        R, T = self._arrays()
        first, last = self._row_bounds()
        isnan = np.isnan(R)
        cum = np.vstack([np.zeros((1, R.shape[1]), dtype=np.int64),
                         np.cumsum(isnan, axis=0, dtype=np.int64)])
        plan = []
        for t in range(first, last - c.lag, c.rebalance):
            s = t - c.window + 1
            ok = T[t] & ((cum[t + 1] - cum[s]) == 0)
            cols = np.flatnonzero(ok)
            if c.standardize and cols.size:
                A = R[s:t + 1, cols]
                # same zero-variance tolerance as kfactors.utils.preprocess
                tol = 1e-12 * max(1.0, float(np.abs(A).max()))
                cols = cols[A.std(axis=0) > tol]
            hold = np.arange(t + 1 + c.lag, min(t + c.rebalance + c.lag, last) + 1)
            plan.append(_Decision(t, cols, hold))
        if not plan:
            raise ValueError(
                f"no feasible decision: need window={c.window} rows before the first decision, "
                f"lag={c.lag} and at least one holding row within the panel/config.end "
                f"(panel has {self.panel.shape[0]} rows)")
        return plan

    def schedule(self) -> pd.DataFrame:
        """Decision schedule: one row per decision date with ``t`` (row position),
        ``n_assets`` (eligible universe), ``executed`` (n_assets >= min_assets),
        ``hold_start`` and ``hold_end`` (holding dates)."""
        idx = self.panel.returns.index
        plan = self._plan()
        return pd.DataFrame(
            {"t": [d.t for d in plan],
             "n_assets": [int(d.cols.size) for d in plan],
             "executed": [bool(d.cols.size >= self.config.min_assets) for d in plan],
             "hold_start": idx[[d.hold[0] for d in plan]],
             "hold_end": idx[[d.hold[-1] for d in plan]]},
            index=pd.DatetimeIndex(idx[[d.t for d in plan]], name="date"))

    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(returns float64, tradable bool) as numpy arrays, cached per panel object."""
        p = self.panel
        key = (id(p), id(p.returns), id(p.tradable), p.returns.shape)
        if getattr(self, "_arrays_key", None) != key:
            self._arrays_cache = (p.returns.to_numpy(dtype=float),
                                  p.tradable.to_numpy(dtype=bool))
            self._arrays_key = key
        return self._arrays_cache

    def _window(self, d: _Decision) -> Panel:
        """Rows ``t - window + 1 .. t`` x universe as a Panel (copies; never rows > t)."""
        s = d.t - self.config.window + 1
        p = self.panel
        R, T = self._arrays()
        idx = p.returns.index[s:d.t + 1]
        cols = p.returns.columns[d.cols]
        # The window is a row/column slice of an already validated Panel, so the (O(n p))
        # validation in Panel.__post_init__ is skipped for speed; every invariant it checks
        # (float64 returns, bool tradable, sorted unique dates, unique columns, tradable only
        # where returns are observed) is inherited from the parent panel.
        w = object.__new__(Panel)
        w.returns = pd.DataFrame(R[s:d.t + 1, d.cols], index=idx, columns=cols)
        w.tradable = pd.DataFrame(T[s:d.t + 1, d.cols], index=idx, columns=cols)
        # only static descriptors: sample-level meta (end, ticker_splits, ...) is post-t info
        w.name, w.freq = p.name, p.freq
        w.meta = {key: p.meta[key] for key in WINDOW_META_KEYS if key in p.meta}
        return w

    # ----------------------------------------------------------------------- k resolution
    def _fingerprint(self, rule: str):
        c = self.config
        kw = c.rule_kwargs.get(rule, {})
        return (c.window, bool(c.standardize), int(c.kmax), c.random_state,
                repr(sorted(kw.items())))

    def _rule_k(self, rule: str, d: _Decision, fp=None) -> int:
        fp = self._fingerprint(rule) if fp is None else fp
        key = (fp, rule, d.t, d.cols.tobytes())
        k = self._rule_cache.get(key)
        if k is None:
            c = self.config
            s = d.t - c.window + 1
            X = self._arrays()[0][s:d.t + 1, d.cols]
            res = select_k(X, rule, kmax=c.kmax, standardize=c.standardize,
                           random_state=c.random_state, **c.rule_kwargs.get(rule, {}))
            k = int(res.k)
            self._rule_cache[key] = k
        return k

    # ----------------------------------------------------------------------- running
    def _run_plan(self, k, plan: list[_Decision]) -> BacktestResult:
        kind, spec = _check_k_spec(k)
        c = self.config
        panel = self.panel
        idx = panel.returns.index
        cols_all = panel.returns.columns
        R = self._arrays()[0]
        n, p = R.shape
        k_source = {"fixed": "fixed", "rule": spec}.get(kind) or _callable_name(spec)
        fp = self._fingerprint(spec) if kind == "rule" else None
        cost = c.cost_bps / 1e4

        state: dict = {}
        held = np.zeros(p)          # book just before the next execution (drifted if c.drift)
        known = held                # book as known at the next decision row (rows <= t only)
        started = False
        n_nan_exits = 0
        gross = np.full(n, np.nan)
        net = np.full(n, np.nan)
        turn_dates, turn_vals = [], []
        w_dates, w_rows = [], []
        k_dates, k_vals = [], []
        skipped, usize = [], []
        ever = np.zeros(p, dtype=bool)
        n_clipped = 0

        for d in plan:
            date = idx[d.t]
            usize.append(int(d.cols.size))
            if d.cols.size < c.min_assets:
                skipped.append(date)
                if not started:
                    continue
                new = np.zeros(p)
            else:
                window = self._window(d)
                rank = min(c.window - 1, int(d.cols.size))
                if kind == "fixed":
                    kk = spec
                elif kind == "rule":
                    kk = self._rule_k(spec, d, fp)
                else:
                    kk = _as_k_int(spec(window.returns.to_numpy(copy=True)), k_source)
                if kk > rank:
                    kk = rank
                    n_clipped += 1
                model = FactorModel(kk, standardize=c.standardize).fit(window.returns.to_numpy())
                prev = None
                if started:
                    nz = np.flatnonzero(known)
                    prev = pd.Series(known[nz], index=cols_all[nz], dtype=float)
                ctx = Context(date=date, k=kk, k_source=k_source, previous_weights=prev,
                              state=state)
                w = self.strategy.weights(window, model, ctx)
                new = np.zeros(p)
                new[d.cols] = _validate_weights(w, window.returns.columns, date)
                ever[d.cols] = True
                k_dates.append(date)
                k_vals.append(kk)
                started = True
            to = float(np.abs(new - held).sum())
            Rh = R[d.hold]
            nan_h = np.isnan(Rh)
            # exit at the last price: an asset contributes 0 from its first NaN row onward
            Rh = np.where(np.cumsum(nan_h, axis=0) == 0, Rh, 0.0)
            n_nan_exits += int(np.count_nonzero(nan_h.any(axis=0) & (new != 0)))
            if c.drift:
                g, held, known = _drifted_returns(Rh, new, d.hold, d.t + c.rebalance)
            else:
                g = Rh @ new
                held = known = new
            gross[d.hold] = g
            net[d.hold] = g
            net[d.hold[0]] -= cost * to
            turn_dates.append(idx[d.hold[0]])
            turn_vals.append(to)
            w_dates.append(date)
            w_rows.append(new)

        if not started:
            raise ValueError(
                f"no decision had >= min_assets={c.min_assets} eligible assets "
                f"(max universe size {max(usize) if usize else 0}); lower min_assets or window")
        rec = ~np.isnan(gross)
        hidx = idx[rec]
        W = pd.DataFrame(np.vstack(w_rows)[:, ever], index=pd.DatetimeIndex(w_dates, name="date"),
                         columns=cols_all[ever])
        return BacktestResult(
            gross=pd.Series(gross[rec], index=hidx, name="gross"),
            net=pd.Series(net[rec], index=hidx, name="net"),
            weights=W,
            turnover=pd.Series(turn_vals, index=pd.DatetimeIndex(turn_dates, name="date"),
                               name="turnover", dtype=float),
            k_used=pd.Series(k_vals, index=pd.DatetimeIndex(k_dates, name="date"), name="k",
                             dtype="int64"),
            k_spec=k,
            strategy=getattr(self.strategy, "name", type(self.strategy).__name__),
            config=c,
            periods_per_year=panel.periods_per_year(),
            skipped=pd.DatetimeIndex(skipped, name="date"),
            universe_size=pd.Series(usize, index=pd.DatetimeIndex(idx[[d.t for d in plan]],
                                                                  name="date"),
                                    name="n_assets", dtype="int64"),
            info={"n_decisions": len(plan), "n_executed": len(k_vals),
                  "n_skipped": len(skipped), "n_clipped": n_clipped, "k_source": k_source,
                  "n_nan_exits": n_nan_exits},
        )

    def run(self, k: int | str | Callable) -> BacktestResult:
        """k: fixed int, a rule name from kfactors.RULES, or callable(window_returns ndarray)->int.

        Parameters
        ----------
        k : int, str or callable
            * ``int >= 0``: fixed number of factors, clipped to the window rank
              ``min(window - 1, n_assets)`` (the clipped value is recorded in ``k_used``).
            * ``str``: rule name; ``select_k(window, rule, kmax=config.kmax,
              standardize=config.standardize, random_state=config.random_state,
              **config.rule_kwargs.get(rule, {}))`` at every decision. Results are cached per
              (rule settings, decision row, universe) on this engine.
            * ``callable``: ``f(window_returns: ndarray n x p) -> int``, clipped like a fixed k.

        Returns
        -------
        BacktestResult

        Raises
        ------
        TypeError, ValueError
            Invalid ``k`` or config; a strategy returning invalid weights; no feasible or no
            executed decision.
        """
        _check_k_spec(k)
        return self._run_plan(k, self._plan())

    def run_many(self, ks: Iterable[int | str], n_jobs: int = 1) -> dict:
        """{k_spec: BacktestResult}. Rule-selected k per date is computed once per (rule, date).

        Parameters
        ----------
        ks : iterable of int, str or callable
            Specs as in :meth:`run`; duplicates are run once.
        n_jobs : int
            joblib parallelism (``-1`` = all cores). With ``n_jobs != 1`` the rule choices are
            computed in parallel over rules, then the runs in parallel over specs (loky
            processes; the strategy is pickled, so it must not rely on mutable state shared
            between runs). Results are identical to ``n_jobs=1``.

        Returns
        -------
        dict
            ``{k_spec: BacktestResult}`` in the order of first appearance.
        """
        specs = list(dict.fromkeys(ks))
        for k in specs:
            _check_k_spec(k)
        plan = self._plan()
        rules = [k for k in specs if isinstance(k, str)]
        if n_jobs == 1:
            for r in rules:
                _rule_ks_task(self, r, plan, self._fingerprint(r))
            return {k: self._run_plan(k, plan) for k in specs}

        from joblib import Parallel, delayed

        todo = [r for r in rules if any(
            (self._fingerprint(r), r, d.t, d.cols.tobytes()) not in self._rule_cache
            for d in plan if d.cols.size >= self.config.min_assets)]
        if todo:
            out = Parallel(n_jobs=n_jobs)(
                delayed(_rule_ks_task)(self, r, plan, self._fingerprint(r)) for r in todo)
            by_t = {d.t: d for d in plan}
            for r, ks_t in out:
                fp = self._fingerprint(r)
                for t, kk in ks_t.items():
                    self._rule_cache[(fp, r, t, by_t[t].cols.tobytes())] = kk
        res = Parallel(n_jobs=n_jobs)(delayed(_run_task)(self, k, plan) for k in specs)
        return dict(zip(specs, res))


def _as_index_ts(x, idx: pd.DatetimeIndex) -> pd.Timestamp:
    """``pd.Timestamp(x)`` made comparable with ``idx`` (naive dates are localised to a tz-aware
    index, as in :meth:`Panel.subset`)."""
    t = pd.Timestamp(x)
    if getattr(idx, "tz", None) is not None and t.tz is None:
        t = t.tz_localize(idx.tz)
    return t


def _drifted_returns(Rh: np.ndarray, w0: np.ndarray, hold: np.ndarray, t_next: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Buy-and-hold inside one holding period (weights as fractions of NAV, cash earns 0).

    Returns the portfolio return of every holding row, the drifted book after the last row and
    the drifted book after the last row ``<= t_next`` (the next decision row; the undrifted
    ``w0`` if no holding row is ``<= t_next``). If the NAV is wiped out (``1 + w'r <= 0``) the
    book is flat for the rest of the period.
    """
    w = w0.copy()
    g = np.empty(Rh.shape[0])
    known = w0
    for i, r in enumerate(Rh):
        g[i] = w @ r
        nav = 1.0 + g[i]
        w = w * (1.0 + r) / nav if nav > 0 else np.zeros_like(w)
        if hold[i] <= t_next:
            known = w
    return g, w, known


def _validate_weights(w, columns: pd.Index, date) -> np.ndarray:
    """Weights from a strategy -> float array aligned with ``columns`` (missing = 0)."""
    if isinstance(w, dict):
        w = pd.Series(w, dtype=float)
    if isinstance(w, pd.Series):
        if w.index.has_duplicates:
            raise ValueError(f"{date}: strategy weights have duplicated asset ids")
        extra = w.index.difference(columns)
        if len(extra):
            raise ValueError(f"{date}: strategy returned weights for assets outside the "
                             f"universe/window, e.g. {list(extra[:5])}")
        v = w.reindex(columns).to_numpy(dtype=float, copy=True)  # missing assets -> NaN
        v[~columns.isin(w.index)] = 0.0
    else:
        v = np.asarray(w, dtype=float)
        if v.shape != (len(columns),):
            raise TypeError(f"{date}: strategy must return a pd.Series indexed by asset ids "
                            f"(or an array of length {len(columns)}), got shape {v.shape}")
        v = v.copy()
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{date}: strategy returned non-finite weights")
    return v
