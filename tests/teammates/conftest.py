"""Shared fixtures for the teammate acceptance tests (Cards A, B, C).

Every acceptance test **skips** (instead of failing) while the corresponding card still raises
``NotImplementedError``. As soon as a function is implemented, its tests run for real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kfactors.data.panel import Panel


def _panel(R, *, prefix: str = "A", start: str = "2021-01-01", freq: str = "D",
           name: str = "synthetic") -> Panel:
    """Wrap an ``n x p`` array of simple returns into a fully tradable daily :class:`Panel`."""
    R = np.asarray(R, dtype=float)
    n, p = R.shape
    idx = pd.bdate_range(start, periods=n)
    cols = [f"{prefix}{j:02d}" for j in range(p)]
    returns = pd.DataFrame(R, index=idx, columns=cols)
    tradable = pd.DataFrame(True, index=idx, columns=cols)
    return Panel(returns, tradable, name=name, freq=freq)


@pytest.fixture
def make_panel():
    """Factory fixture: ``make_panel(R, prefix="A")`` -> Panel with business-day dates."""
    return _panel


@pytest.fixture
def factor_returns():
    """Factory: ``factor_returns(n, p, k, seed)`` -> n x p returns with k strong factors.

    ``R = F B' + E`` with ``F ~ N(0, 0.01^2)``, loadings ``B ~ U(0.5, 1.5)`` for the first
    factor (a "market") and ``N(0, 0.5^2)`` for the others, idiosyncratic ``E ~ N(0, s_i^2)``
    with heterogeneous ``s_i ~ U(0.005, 0.015)``. Deterministic in ``seed``.
    """
    def make(n: int = 150, p: int = 30, k: int = 3, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        F = rng.normal(0.0, 0.01, (n, k))
        B = rng.normal(0.0, 0.5, (p, k))
        if k:
            B[:, 0] = rng.uniform(0.5, 1.5, p)
        s = rng.uniform(0.005, 0.015, p)
        return F @ B.T + rng.normal(0.0, 1.0, (n, p)) * s
    return make


@pytest.fixture
def run_engine():
    """Run ``WalkForward(panel, strategy, config).run(k)``; skip if the core engine is a stub.

    Card tests that go through the engine need ``kfactors.engine`` (implemented by the core
    team). If it still raises ``NotImplementedError`` the test is skipped with a clear message.
    """
    def run(panel, strategy, k, **config):
        from kfactors.engine import WalkForward, WalkForwardConfig
        cfg = WalkForwardConfig(**config)
        try:
            return WalkForward(panel, strategy, cfg).run(k)
        except NotImplementedError as exc:
            msg = str(exc)
            if "Card" in msg:   # the card itself (strategy) is still a stub
                pytest.skip(f"card not implemented yet: {msg}")
            pytest.skip(f"core walk-forward engine not implemented yet: {msg!r}")
    return run
