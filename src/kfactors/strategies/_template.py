"""Template strategy plug-in -- copy this file to start a new strategy.

How to write a strategy
-----------------------
1. Copy this file to ``src/kfactors/strategies/<your_name>.py`` and rename the class.
2. Set a unique ``name`` (used as the label in results and plots).
3. Put hyper-parameters in ``__init__`` (store them as attributes, validate them there).
4. Implement :meth:`weights`. You receive

   * ``window`` -- a :class:`kfactors.data.panel.Panel` with the estimation window only:
     rows ``t - window + 1 .. t`` (``t`` = decision date = ``context.date``) and the assets of
     the current universe (tradable at ``t`` and complete in the window). There is **no data
     after the decision date** in it: anything you compute is automatically leakage-free.
   * ``model`` -- a :class:`kfactors.model.FactorModel` with ``context.k`` factors already fitted
     on ``window.returns`` (``loadings_``, ``factors_``, ``residuals_``, ``idio_var_``,
     ``covariance()``, ``eigenportfolios()``, ``residual_returns()``). ``context.k`` may be 0.
   * ``context`` -- a :class:`kfactors.strategy.Context`: ``date``, ``k``, ``k_source``
     (``"fixed"``, a rule name or ``"callable:<name>"``), ``previous_weights`` (the book held
     before this decision, non-zero positions only; ``None`` at the first decision) and
     ``state`` (a dict that persists across the decisions of one run -- use it, not ``self``,
     for anything that changes over time, so that runs stay independent and parallel-safe).

5. Return a ``pd.Series`` of weights indexed by (a subset of) ``window.returns.columns``.
   Assets you leave out get weight 0. Weights must be finite. The engine does not rescale them:
   ``sum |w|`` is your gross exposure, and returns/costs are computed on these weights as they
   are (held constant for ``rebalance`` rows, no drift).
6. Add tests: at least (a) weights are finite and indexed by window columns, (b) a known-answer
   case, (c) the engine leakage test passes with your strategy (see ``tests/test_engine.py``).

Running it::

    from kfactors.engine import WalkForward, WalkForwardConfig
    from kfactors.evaluate import k_curve
    wf = WalkForward(panel, EqualWeight(), WalkForwardConfig(window=120, cost_bps=10))
    res = wf.run(5)                 # fixed k
    res = wf.run("onatski")         # k chosen by a rule inside every window
    curve = k_curve(wf, k_grid=range(0, 11), rules=("onatski", "bai_ng"))

This template is the simplest possible plug-in: the equally weighted (1/N) portfolio, which
ignores the factor model. DeMiguel, Garlappi & Uppal (2009), "Optimal versus naive
diversification", Review of Financial Studies 22(5), show that it is a hard benchmark to beat
out of sample. Its k-curve is flat by construction, which makes it a useful sanity check.
"""
from __future__ import annotations

import pandas as pd

from kfactors.data.panel import Panel
from kfactors.model import FactorModel
from kfactors.strategy import Context, Strategy

__all__ = ["EqualWeight"]


class EqualWeight(Strategy):
    """Equally weighted portfolio over the current universe (1/N), long-only.

    Parameters
    ----------
    gross : float, default 1.0
        Gross exposure; every asset gets ``gross / N``.

    Notes
    -----
    Ignores ``model`` and ``context.k`` (the k-curve is flat). It counts the decisions in
    ``context.state["n_decisions"]`` only to illustrate per-run state.
    """

    name = "equal_weight"

    def __init__(self, gross: float = 1.0):
        if not gross > 0:
            raise ValueError(f"gross must be > 0, got {gross!r}")
        self.gross = float(gross)

    def weights(self, window: Panel, model: FactorModel, context: Context) -> pd.Series:
        context.state["n_decisions"] = context.state.get("n_decisions", 0) + 1
        cols = window.returns.columns
        return pd.Series(self.gross / len(cols), index=cols)
