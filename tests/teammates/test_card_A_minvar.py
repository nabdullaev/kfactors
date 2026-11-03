"""Acceptance tests for Card A: ``kfactors.strategies.minvar.MinVariance``.

All tests skip while ``MinVariance.weights`` raises ``NotImplementedError``.
Run: ``.venv/bin/python -m pytest tests/teammates/test_card_A_minvar.py -v``
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kfactors.model import FactorModel
from kfactors.sim import SimSpec, simulate
from kfactors.strategies.minvar import MinVariance
from kfactors.strategy import Context, Strategy


@pytest.fixture(autouse=True)
def _skip_until_implemented(make_panel):
    """Skip the whole module while the card is a stub (any other error still fails)."""
    R = np.random.default_rng(0).normal(0, 0.01, (20, 4))
    window = make_panel(R)
    model = FactorModel(1).fit(R)
    try:
        MinVariance().weights(window, model, Context(date=window.returns.index[-1], k=1))
    except NotImplementedError:
        pytest.skip("Card A not implemented yet: MinVariance.weights raises NotImplementedError")
    except Exception:
        pass


def _gmv(window, k, residual="diag"):
    model = FactorModel(k).fit(window.returns.to_numpy())
    ctx = Context(date=window.returns.index[-1], k=k)
    return MinVariance(residual).weights(window, model, ctx), model


# --------------------------------------------------------------------------- interface
def test_is_a_strategy_with_name():
    assert issubclass(MinVariance, Strategy)
    assert isinstance(MinVariance.name, str) and MinVariance.name


def test_invalid_residual_raises():
    with pytest.raises(ValueError):
        MinVariance(residual="full")


def test_returns_series_on_window_columns(make_panel, factor_returns):
    window = make_panel(factor_returns(n=80, p=12, k=2, seed=1))
    w, _ = _gmv(window, 2)
    assert isinstance(w, pd.Series)
    assert list(w.index) == list(window.returns.columns)
    assert np.issubdtype(w.dtype, np.floating)
    assert np.isfinite(w.to_numpy()).all()


# --------------------------------------------------------------------------- closed form
@pytest.mark.parametrize("residual", ["diag", "ledoit_wolf"])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_matches_closed_form_gmv(make_panel, factor_returns, k, residual):
    """w = S^-1 1 / 1'S^-1 1 with S = model.covariance(residual) (Merton 1972)."""
    window = make_panel(factor_returns(n=100, p=15, k=3, seed=2))
    w, model = _gmv(window, k, residual)
    S = model.covariance(residual=residual)
    Sinv1 = np.linalg.inv(S) @ np.ones(S.shape[0])     # independent route: explicit inverse
    expected = Sinv1 / Sinv1.sum()
    np.testing.assert_allclose(w.to_numpy(), expected, rtol=1e-8, atol=1e-10)


def test_weights_sum_to_one(make_panel, factor_returns):
    for seed in range(3):
        window = make_panel(factor_returns(n=60, p=25, k=2, seed=seed))
        w, _ = _gmv(window, 2)
        assert w.sum() == pytest.approx(1.0, abs=1e-10)


def test_tiny_known_answer():
    """Hand-checkable 2-asset case through a stub model with a known covariance.

    S = [[4, 1], [1, 2]] (variances 4 and 2, covariance 1): S^-1 1 = [1, 3] / 7, so the GMV
    weights are [0.25, 0.75].
    """
    class StubModel:
        def covariance(self, residual="diag"):
            return np.array([[4.0, 1.0], [1.0, 2.0]])

    idx = pd.bdate_range("2021-01-01", periods=5)
    from kfactors.data.panel import Panel
    R = pd.DataFrame(np.zeros((5, 2)), index=idx, columns=["X", "Y"])
    window = Panel(R, pd.DataFrame(True, index=idx, columns=["X", "Y"]))
    w = MinVariance().weights(window, StubModel(), Context(date=idx[-1], k=1))
    np.testing.assert_allclose(w.to_numpy(), [0.25, 0.75], atol=1e-12)
    assert list(w.index) == ["X", "Y"]


def test_k0_diag_is_inverse_variance(make_panel, factor_returns):
    """With k = 0 and a diagonal residual covariance the GMV is the inverse-variance portfolio."""
    window = make_panel(factor_returns(n=90, p=10, k=2, seed=3))
    w, _ = _gmv(window, 0)
    var = window.returns.to_numpy().var(axis=0)  # ddof=0, as FactorModel
    expected = (1 / var) / (1 / var).sum()
    np.testing.assert_allclose(w.to_numpy(), expected, rtol=1e-8)


def test_p_greater_than_n(make_panel, factor_returns):
    """p > n: the sample covariance is singular, the factor covariance is not."""
    window = make_panel(factor_returns(n=40, p=120, k=2, seed=4))
    w, model = _gmv(window, 2)
    S = model.covariance()
    assert np.isfinite(w.to_numpy()).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    # first-order condition of the GMV problem: S w is proportional to 1
    Sw = S @ w.to_numpy()
    np.testing.assert_allclose(Sw, np.full_like(Sw, Sw.mean()), rtol=1e-6)


def test_does_not_mutate_inputs(make_panel, factor_returns):
    window = make_panel(factor_returns(n=60, p=8, k=1, seed=5))
    before = window.returns.copy()
    model = FactorModel(1).fit(window.returns.to_numpy())
    ev = model.eigenvectors_.copy()
    MinVariance().weights(window, model, Context(date=window.returns.index[-1], k=1))
    pd.testing.assert_frame_equal(window.returns, before)
    np.testing.assert_array_equal(model.eigenvectors_, ev)


def test_deterministic(make_panel, factor_returns):
    window = make_panel(factor_returns(n=70, p=20, k=2, seed=6))
    w1, _ = _gmv(window, 2, "ledoit_wolf")
    w2, _ = _gmv(window, 2, "ledoit_wolf")
    pd.testing.assert_series_equal(w1, w2)


# --------------------------------------------------------------------------- economics
def test_true_k_gives_lowest_population_risk(make_panel):
    """Risk-evaluator sanity check on a planted 3-factor model (kfactors.sim, n=120, p=100).

    The population variance w' Sigma w of the GMV portfolio is lowest near the true k = 3:
    k = 0 ignores correlations, k = 50 over-fits noise. (Reference run: variance ratio to the
    true GMV is about 2.2 (k=0), 1.04 (k=3), 2.3 (k=50).)
    """
    d = simulate(SimSpec(n=120, p=100, k=3, noise="hetero"), random_state=0)
    window = make_panel(d.X)
    ones = np.ones(d.Sigma.shape[0])
    w_opt = np.linalg.solve(d.Sigma, ones)
    w_opt /= w_opt.sum()
    v_opt = w_opt @ d.Sigma @ w_opt
    ratio = {}
    for k in (0, 3, 50):
        w, _ = _gmv(window, k)
        ratio[k] = float(w.to_numpy() @ d.Sigma @ w.to_numpy() / v_opt)
    assert ratio[3] < 1.2
    assert ratio[3] < ratio[0] - 0.5
    assert ratio[3] < ratio[50] - 0.5


# --------------------------------------------------------------------------- engine
def test_runs_through_walk_forward(make_panel, factor_returns, run_engine):
    panel = make_panel(factor_returns(n=200, p=20, k=2, seed=7))
    res = run_engine(panel, MinVariance(), 2, window=60, rebalance=5, min_assets=5)
    W = res.weights.fillna(0.0)
    assert len(W) > 0
    assert np.isfinite(W.to_numpy()).all()
    np.testing.assert_allclose(W.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
    assert np.isfinite(res.gross.to_numpy()).all()
    # the decision at the first date equals a direct call on the same window
    t = panel.returns.index.get_loc(W.index[0])
    window = panel.window(t, 60)
    w_direct, _ = _gmv(window, 2)
    np.testing.assert_allclose(W.iloc[0][w_direct.index].to_numpy(), w_direct.to_numpy(),
                               rtol=1e-8, atol=1e-12)


def test_walk_forward_rule_selected_k(make_panel, factor_returns, run_engine):
    panel = make_panel(factor_returns(n=160, p=20, k=2, seed=8))
    res = run_engine(panel, MinVariance("ledoit_wolf"), "bai_ng", window=60, rebalance=10,
                     min_assets=5)
    W = res.weights.fillna(0.0)
    np.testing.assert_allclose(W.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
    assert (res.k_used >= 0).all()
