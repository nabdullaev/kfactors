"""Tests for the reference strategy TangencyPCA (kfactors.strategies.sdf) and the template."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kfactors.data.panel import Panel
from kfactors.engine import WalkForward, WalkForwardConfig
from kfactors.model import FactorModel
from kfactors.strategies import TangencyPCA
from kfactors.strategies._template import EqualWeight
from kfactors.strategy import Context


def window_panel(n=300, p=5, seed=0, mean=0.002):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p)) * 0.5 + np.eye(p)
    X = rng.standard_normal((n, p)) @ A.T * 0.01 + mean * rng.uniform(0.2, 1.0, p)
    idx = pd.bdate_range("2010-01-01", periods=n)
    R = pd.DataFrame(X, index=idx, columns=[f"s{j}" for j in range(p)])
    return Panel(R, R.notna(), freq="D")


def ctx(k):
    return Context(date=pd.Timestamp("2011-01-01"), k=k)


@pytest.mark.parametrize("standardize", [True, False])
def test_full_rank_matches_markowitz_tangency(standardize):
    """k = p (n > p), gamma = 0: w ∝ Sigma^-1 mu (sample moments, ddof 0), sum |w| = 1."""
    win = window_panel()
    X = win.returns.to_numpy()
    m = FactorModel(5, standardize=standardize).fit(X)
    w = TangencyPCA().weights(win, m, ctx(5))
    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False, ddof=0)
    ref = np.linalg.solve(S, mu)
    ref /= np.abs(ref).sum()
    np.testing.assert_allclose(w.to_numpy(), ref, rtol=1e-8, atol=1e-12)
    assert list(w.index) == list(win.returns.columns)


def test_k1_closed_form():
    """k = 1: w ∝ sign(mean(F1)) * v1 / sigma (a single eigenportfolio)."""
    win = window_panel(seed=1)
    X = win.returns.to_numpy()
    m = FactorModel(1).fit(X)
    w = TangencyPCA().weights(win, m, ctx(1)).to_numpy()
    q = m.eigenvectors_[:, 0] / X.std(axis=0)
    f_mean = (X @ q).mean()
    ref = np.sign(f_mean) * q / np.abs(q).sum()
    np.testing.assert_allclose(w, ref, rtol=1e-10)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_gross_exposure_one_and_portfolio_is_factor_combination(k):
    win = window_panel(seed=2, n=80, p=12)
    X = win.returns.to_numpy()
    m = FactorModel(k).fit(X)
    w = TangencyPCA(gamma=0.1).weights(win, m, ctx(k))
    assert np.abs(w).sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(w))
    # X w lies in the span of the k eigenportfolio returns
    F = X @ m.eigenportfolios()
    coef, *_ = np.linalg.lstsq(F, X @ w.to_numpy(), rcond=None)
    np.testing.assert_allclose(F @ coef, X @ w.to_numpy(), atol=1e-12)


def test_k_zero_and_zero_mean_give_zero_weights():
    win = window_panel()
    m0 = FactorModel(0).fit(win.returns.to_numpy())
    w0 = TangencyPCA().weights(win, m0, ctx(0))
    assert (w0 == 0).all() and list(w0.index) == list(win.returns.columns)
    Xc = win.returns - win.returns.mean()
    win_c = Panel(Xc, Xc.notna(), freq="D")
    m = FactorModel(3).fit(Xc.to_numpy())
    w = TangencyPCA().weights(win_c, m, ctx(3))
    assert (w == 0).all()


def test_large_gamma_limit_is_proportional_to_factor_means():
    """gamma -> inf: b ∝ mean(F), w ∝ Q mean(F)."""
    win = window_panel(seed=3, p=6)
    X = win.returns.to_numpy()
    m = FactorModel(3).fit(X)
    w = TangencyPCA(gamma=1e12).weights(win, m, ctx(3)).to_numpy()
    Q = m.eigenportfolios()
    ref = Q @ (X @ Q).mean(axis=0)
    ref /= np.abs(ref).sum()
    np.testing.assert_allclose(w, ref, rtol=1e-6)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True, "x"])
def test_gamma_validation(bad):
    with pytest.raises(ValueError):
        TangencyPCA(gamma=bad)


def test_tangency_in_engine_p_greater_than_n():
    win = window_panel(n=200, p=40, seed=4)
    cfg = WalkForwardConfig(window=30, rebalance=5, min_assets=5)
    res = WalkForward(win, TangencyPCA(), cfg).run(3)
    gross = res.weights.abs().sum(axis=1)
    np.testing.assert_allclose(gross.to_numpy(), 1.0)
    assert np.isfinite(res.net).all()


def test_equal_weight_template():
    win = window_panel(n=60, p=4)
    st = EqualWeight(gross=2.0)
    c = ctx(0)
    w = st.weights(win, FactorModel(0).fit(win.returns.to_numpy()), c)
    np.testing.assert_allclose(w.to_numpy(), 0.5)
    assert c.state["n_decisions"] == 1
    with pytest.raises(ValueError):
        EqualWeight(gross=0)
    # flat k-curve by construction: the result does not depend on k
    wf = WalkForward(win, EqualWeight(), WalkForwardConfig(window=10, min_assets=2))
    r1, r3 = wf.run(1), wf.run(3)
    pd.testing.assert_series_equal(r1.net, r3.net)
