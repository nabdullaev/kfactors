"""Acceptance tests for Card B: ``kfactors.strategies.statarb`` (Avellaneda & Lee 2010).

Each test skips while the function it exercises raises ``NotImplementedError``, so the card can
be implemented step by step: fit_ar1 -> s_scores -> update_positions -> hedge_weights ->
ResidualReversion.weights.
Run: ``.venv/bin/python -m pytest tests/teammates/test_card_B_statarb.py -v``
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kfactors.model import FactorModel
from kfactors.strategies import statarb
from kfactors.strategies.statarb import (
    S_SCORE_COLUMNS,
    ResidualReversion,
    fit_ar1,
    hedge_weights,
    s_scores,
    update_positions,
)
from kfactors.strategy import Context, Strategy

L = 60  # Avellaneda-Lee estimation window


def call(fn, *args, **kwargs):
    """Call ``fn``; skip the test if that step of the card is not implemented yet."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"Card B step not implemented yet: {exc}")


def ou_path(n, b, a=0.0, sd=1.0, x0=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = a + b * x[t - 1] + rng.normal(0.0, sd)
    return x


def planted_panel(make_panel, *, n=100, p=12, b=0.7, sd=0.004, c=4.0, seed=0, msd=0.02):
    """One market factor + OU residuals; asset A00 is planted CHEAP, A01 RICH.

    Over the last ``L`` rows, A00's cumulative residual sits ``c`` equilibrium standard
    deviations above its usual level and drops back on the last day (so at the decision date it
    is far below its window equilibrium -> s << -1.25 -> go long); A01 is the mirror image.
    """
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, msd, n)
    beta = rng.uniform(0.5, 1.5, p)
    X = np.zeros((n + 1, p))
    for t in range(n):
        X[t + 1] = b * X[t] + rng.normal(0.0, sd, p)
    sigma_eq = sd / np.sqrt(1 - b**2)
    X[n - L + 1:n, 0] += c * sigma_eq
    X[n - L + 1:n, 1] -= c * sigma_eq
    R = np.outer(f, beta) + np.diff(X, axis=0)
    return make_panel(R)


def decide(window, k, strategy=None, context=None):
    model = FactorModel(k).fit(window.returns.to_numpy())
    ctx = context or Context(date=window.returns.index[-1], k=k)
    w = call((strategy or ResidualReversion()).weights, window, model, ctx)
    return w, ctx, model


# =========================================================================== step 1: fit_ar1
def test_fit_ar1_matches_least_squares():
    x = ou_path(200, b=0.8, a=0.3, sd=0.5, seed=1)
    a, b, v = call(fit_ar1, x)
    design = np.column_stack([np.ones(199), x[:-1]])
    coef, *_ = np.linalg.lstsq(design, x[1:], rcond=None)
    zeta = x[1:] - design @ coef
    assert a == pytest.approx(coef[0], abs=1e-10)
    assert b == pytest.approx(coef[1], abs=1e-10)
    assert v == pytest.approx(np.mean(zeta**2), rel=1e-10)      # ddof = 0


def test_fit_ar1_exact_noise_free():
    x = np.empty(30)
    x[0] = 5.0
    for t in range(1, 30):
        x[t] = 0.2 + 0.5 * x[t - 1]
    a, b, v = call(fit_ar1, x)
    assert (a, b) == (pytest.approx(0.2, abs=1e-9), pytest.approx(0.5, abs=1e-9))
    assert v == pytest.approx(0.0, abs=1e-12)


def test_fit_ar1_recovers_parameters():
    x = ou_path(20000, b=0.9, a=0.1, sd=1.0, seed=2)
    a, b, v = call(fit_ar1, x)
    assert b == pytest.approx(0.9, abs=0.01)
    assert a / (1 - b) == pytest.approx(1.0, abs=0.25)          # equilibrium mean a/(1-b)
    assert v == pytest.approx(1.0, abs=0.05)


def test_fit_ar1_edge_cases():
    out = call(fit_ar1, np.array([1.0, 1.0, 1.0, 2.0]))         # x[:-1] constant
    assert all(np.isnan(out))
    for bad in (np.array([1.0, 2.0]), np.array([1.0, np.nan, 2.0, 3.0]), np.ones((4, 2))):
        with pytest.raises(ValueError):
            call(fit_ar1, bad)


# =========================================================================== step 2: s_scores
def _independent_scores(E, max_tau, center):
    """Straightforward re-derivation of the AL formulas used to cross-check s_scores."""
    X = np.cumsum(E, axis=0)
    rows = []
    for i in range(E.shape[1]):
        x0, x1 = X[:-1, i], X[1:, i]
        b, a = np.polyfit(x0, x1, 1)
        vz = np.mean((x1 - a - b * x0) ** 2)
        ok = 0 < b < 1
        kappa = -np.log(b) if ok else np.nan
        m = a / (1 - b) if ok else np.nan
        seq = np.sqrt(vz / (1 - b * b)) if ok else np.nan
        rows.append((a, b, kappa, 1 / kappa, m, seq, ok and vz > 0 and 1 / kappa < max_tau))
    df = pd.DataFrame(rows, columns=["a", "b", "kappa", "tau", "m", "sigma_eq", "eligible"])
    if center and df.eligible.any():
        df["m"] = df.m - df.m[df.eligible].mean()
    df["s"] = np.where(df.eligible, (X[-1] - df.m) / df.sigma_eq, np.nan)
    return df


@pytest.mark.parametrize("center", [True, False])
def test_s_scores_match_formulas(center):
    rng = np.random.default_rng(3)
    p = 8
    E = np.diff(np.column_stack([ou_path(L + 1, b=bb, sd=0.01, seed=10 + j)
                                 for j, bb in enumerate(np.linspace(0.2, 0.95, p))]), axis=0)
    E = E + rng.normal(0, 1e-4, E.shape)
    out = call(s_scores, E, max_tau=L / 2, center=center)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(S_SCORE_COLUMNS)
    assert list(out.index) == list(range(p))
    ref = _independent_scores(E, L / 2, center)
    for col in ("a", "b", "kappa", "tau", "m", "sigma_eq", "s"):
        np.testing.assert_allclose(out[col].to_numpy(dtype=float), ref[col].to_numpy(dtype=float),
                                   rtol=1e-7, atol=1e-10, err_msg=col)
    np.testing.assert_array_equal(out["eligible"].to_numpy(dtype=bool), ref["eligible"])


def test_s_scores_tau_filter_and_nan_for_ineligible():
    fast = np.diff(ou_path(L + 1, b=0.6, sd=0.01, seed=4))      # cumulative residual = OU
    walk = np.random.default_rng(5).normal(0, 0.01, L)          # cumulative = random walk
    alt = np.diff(ou_path(L + 1, b=-0.5, sd=0.01, seed=6))      # oscillating: b < 0
    E = np.column_stack([fast, walk, alt])
    out = call(s_scores, E, max_tau=L / 2)
    assert 0 < out.loc[0, "b"] < 1 and out.loc[0, "tau"] < L / 2
    assert bool(out.loc[0, "eligible"]) and np.isfinite(out.loc[0, "s"])
    # random walk: b close to 1, mean-reversion time above L/2 -> filtered out
    assert out.loc[1, "tau"] > L / 2
    assert not out.loc[1, "eligible"] and np.isnan(out.loc[1, "s"])
    # b <= 0: kappa = -ln(b) undefined -> NaN parameters, not eligible
    assert out.loc[2, "b"] < 0 and np.isnan(out.loc[2, "kappa"]) and np.isnan(out.loc[2, "m"])
    assert not out.loc[2, "eligible"] and np.isnan(out.loc[2, "s"])
    # a tiny max_tau switches everything off
    out2 = call(s_scores, E, max_tau=0.01)
    assert not out2["eligible"].any() and out2["s"].isna().all()
    # a trending cumulative residual has b >= 1 and is never eligible
    trend = np.full(L, 0.01) + 1e-6 * np.arange(L) ** 2
    out3 = call(s_scores, np.column_stack([trend, fast]), max_tau=30)
    assert out3.loc[0, "b"] >= 1 and not out3.loc[0, "eligible"]
    assert np.isnan(out3.loc[0, "s"]) and bool(out3.loc[1, "eligible"])


def test_s_scores_centering_uses_eligible_mean():
    E = np.diff(np.column_stack([ou_path(L + 1, b=0.5, a=0.002 * j, sd=0.01, seed=20 + j)
                                 for j in range(6)]), axis=0)
    c = call(s_scores, E, max_tau=L / 2, center=True)
    u = call(s_scores, E, max_tau=L / 2, center=False)
    el = c["eligible"].to_numpy(dtype=bool)
    assert el.sum() >= 2
    assert c.loc[el, "m"].mean() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(c["m"] - u["m"], -u.loc[el, "m"].mean(), atol=1e-12)


def test_s_scores_validation():
    with pytest.raises(ValueError):
        call(s_scores, np.ones(10), max_tau=30)
    with pytest.raises(ValueError):
        call(s_scores, np.full((10, 2), np.nan), max_tau=30)
    with pytest.raises(ValueError):
        call(s_scores, np.random.default_rng(0).normal(size=(10, 2)), max_tau=0)


# =========================================================================== step 3: positions
CASES = [
    # (previous, s, expected)  with s_open=1.25, s_close_long=0.50, s_close_short=0.75
    (0, -1.30, 1), (0, -1.25, 0), (0, 1.30, -1), (0, 1.25, 0), (0, 0.0, 0), (0, np.nan, 0),
    (1, -0.60, 1), (1, -0.50, 1), (1, -0.40, 0), (1, 1.30, -1), (1, -3.0, 1), (1, np.nan, 0),
    (-1, 0.80, -1), (-1, 0.75, -1), (-1, 0.70, 0), (-1, -1.30, 1), (-1, 3.0, -1),
    (-1, np.nan, 0),
]


@pytest.mark.parametrize("prev, s, expected", CASES)
def test_update_positions_avellaneda_lee_thresholds(prev, s, expected):
    out = call(update_positions, pd.Series([s], index=["X"]), pd.Series([prev], index=["X"]))
    assert out.loc["X"] == expected


def test_update_positions_alignment_and_dtype():
    s = pd.Series([-2.0, 2.0, 0.0, -0.6], index=["A", "B", "C", "D"])
    prev = pd.Series([1, -1, 1], index=["D", "Z", "C"])          # Z left the universe
    out = call(update_positions, s, prev)
    assert list(out.index) == ["A", "B", "C", "D"]
    assert np.issubdtype(out.dtype, np.integer)
    assert out.tolist() == [1, -1, 0, 1]
    assert call(update_positions, s, None).tolist() == [1, -1, 0, 0]


def test_update_positions_custom_thresholds_and_validation():
    s = pd.Series([-0.9, 0.9], index=["A", "B"])
    out = call(update_positions, s, None, s_open=0.8, s_close_long=0.1, s_close_short=0.2)
    assert out.tolist() == [1, -1]
    with pytest.raises(ValueError):
        call(update_positions, s, None, s_open=0.5, s_close_long=0.6)
    with pytest.raises(ValueError):
        call(update_positions, s, pd.Series([2, 0], index=["A", "B"]))


# =========================================================================== step 4: hedging
def test_hedge_weights_formula_and_gross():
    rng = np.random.default_rng(6)
    p, k = 10, 3
    Q, B = rng.normal(size=(p, k)), rng.normal(size=(p, k))
    q = np.array([1, -1, 0, 0, 1, 0, 0, 0, 0, -1], dtype=float)
    w = call(hedge_weights, q, Q, B)
    h = q - Q @ (B.T @ q)
    np.testing.assert_allclose(w, h / np.abs(h).sum(), atol=1e-12)
    assert np.abs(w).sum() == pytest.approx(1.0)


def test_hedge_weights_k0_and_flat():
    q = np.array([1.0, 0.0, -1.0, 0.0])
    w = call(hedge_weights, q, np.zeros((4, 0)), np.zeros((4, 0)))
    np.testing.assert_allclose(w, [0.5, 0.0, -0.5, 0.0])
    w0 = call(hedge_weights, np.zeros(4), np.ones((4, 1)), np.ones((4, 1)))
    np.testing.assert_array_equal(w0, np.zeros(4))
    with pytest.raises(ValueError):
        call(hedge_weights, q, np.ones((3, 1)), np.ones((4, 1)))


# =========================================================================== step 5: strategy
def test_is_a_strategy_with_al_defaults():
    st = ResidualReversion()
    assert issubclass(ResidualReversion, Strategy) and st.name
    assert (st.estimation_window, st.s_open, st.s_close_long, st.s_close_short) == (60, 1.25,
                                                                                    0.5, 0.75)


@pytest.mark.parametrize("kwargs", [
    {"max_tau": 0}, {"max_tau": -1.0}, {"max_tau": float("nan")},
    {"estimation_window": 9}, {"s_open": 0.0}, {"s_open": float("inf")},
    {"s_close_long": -0.1}, {"s_close_short": 2.0}, {"s_close_long": 1.25},
])
def test_constructor_validates_parameters(kwargs):
    """__init__ is core code (no skip): invalid settings fail at construction, not in the engine.

    In particular ``max_tau=0`` must not silently fall back to the ``L_eff / 2`` default.
    """
    with pytest.raises(ValueError):
        ResidualReversion(**kwargs)


def test_constructor_type_checks_and_accepts_valid_values():
    with pytest.raises(TypeError):
        ResidualReversion(estimation_window=60.0)
    with pytest.raises(TypeError):
        ResidualReversion(estimation_window=True)
    st = ResidualReversion(estimation_window=np.int64(10), max_tau=5, s_close_long=0.0)
    assert (st.estimation_window, st.max_tau, st.s_close_long) == (10, 5, 0.0)


@pytest.mark.parametrize("seed", range(5))
def test_planted_cheap_long_rich_short(make_panel, seed):
    """With the true k = 1 the strategy buys the cheap asset and shorts the rich one."""
    window = planted_panel(make_panel, seed=seed)
    w, ctx, _ = decide(window, 1)
    q = ctx.state["positions"]
    assert q.loc["A00"] == 1 and q.loc["A01"] == -1
    assert w.loc["A00"] > 0 and w.loc["A01"] < 0
    assert np.abs(w).sum() == pytest.approx(1.0, abs=1e-10)


def test_weights_contract(make_panel):
    window = planted_panel(make_panel, seed=0)
    w, ctx, _ = decide(window, 1)
    assert isinstance(w, pd.Series)
    assert list(w.index) == list(window.returns.columns)
    assert np.isfinite(w.to_numpy()).all()
    # state: positions in {-1,0,1} on the window columns, s-score table stored
    q = ctx.state["positions"]
    assert list(q.index) == list(window.returns.columns)
    assert set(np.unique(q.to_numpy())) <= {-1, 0, 1}
    sc = ctx.state["s_scores"]
    assert list(sc.columns) == list(S_SCORE_COLUMNS)
    # thresholds respected when starting flat: long iff s < -1.25, short iff s > 1.25
    s = sc["s"].to_numpy(dtype=float)
    np.testing.assert_array_equal(q.to_numpy() == 1, np.nan_to_num(s, nan=0.0) < -1.25)
    np.testing.assert_array_equal(q.to_numpy() == -1, np.nan_to_num(s, nan=0.0) > 1.25)


def test_book_is_factor_neutral_on_estimation_window(make_panel):
    """Hedging with eigenportfolios removes all exposure to the k factors (AL 2010, Sec. 2)."""
    window = planted_panel(make_panel, seed=1)
    for k in (1, 3):
        w, _, model = decide(window, k)
        R = window.returns.to_numpy()[-L:]
        F = R @ model.eigenportfolios()
        port = R @ w.to_numpy()
        cov = ((port - port.mean())[:, None] * (F - F.mean(axis=0))).mean(axis=0)
        scale = port.std() * F.std(axis=0)
        np.testing.assert_allclose(cov / scale, 0.0, atol=1e-8)


def test_uses_only_last_L_rows(make_panel):
    """Rows older than the estimation window do not change the residual signal."""
    window = planted_panel(make_panel, seed=2)
    k = 1
    model = FactorModel(k).fit(window.returns.to_numpy())
    st = ResidualReversion()
    w1 = call(st.weights, window, model, Context(date=window.returns.index[-1], k=k))
    # same model (eigenportfolios), corrupted old rows: only R[-L:] may be used for residuals
    R2 = window.returns.copy()
    R2.iloc[: len(R2) - L] = R2.iloc[: len(R2) - L] * -3.0 + 0.05
    window2 = type(window)(R2, window.tradable, name="corrupt", freq="D")
    w2 = call(st.weights, window2, model, Context(date=window.returns.index[-1], k=k))
    pd.testing.assert_series_equal(w1, w2, check_names=False)


def test_state_is_carried_between_decisions(make_panel):
    window = planted_panel(make_panel, seed=0)
    model = FactorModel(1).fit(window.returns.to_numpy())
    st = ResidualReversion()
    ctx = Context(date=window.returns.index[-1], k=1)
    w1 = call(st.weights, window, model, ctx)
    w_flat = call(st.weights, window, model, Context(date=ctx.date, k=1))
    pd.testing.assert_series_equal(w1, w_flat)                  # deterministic
    w2 = call(st.weights, window, model, ctx)                   # same s, previous = own output
    pd.testing.assert_series_equal(w1, w2)                      # open positions are kept
    s = ctx.state["s_scores"]["s"].to_numpy(dtype=float)
    cols = window.returns.columns
    # everybody was long yesterday: stay long unless s > -0.5 (exit) or s > 1.25 (reverse)
    c_long = Context(date=ctx.date, k=1, state={"positions": pd.Series(1, index=cols)})
    call(st.weights, window, model, c_long)
    exp_long = np.where(np.isnan(s), 0, np.where(s > 1.25, -1, np.where(s > -0.5, 0, 1)))
    np.testing.assert_array_equal(c_long.state["positions"].to_numpy(), exp_long)
    # everybody was short yesterday: stay short unless s < 0.75 (exit) or s < -1.25 (reverse)
    c_short = Context(date=ctx.date, k=1, state={"positions": pd.Series(-1, index=cols)})
    call(st.weights, window, model, c_short)
    exp_short = np.where(np.isnan(s), 0, np.where(s < -1.25, 1, np.where(s < 0.75, 0, -1)))
    np.testing.assert_array_equal(c_short.state["positions"].to_numpy(), exp_short)
    assert (exp_long != 0).sum() > (ctx.state["positions"] == 1).sum()   # state mattered


def test_no_positions_gives_zero_weights(make_panel):
    """Pure white-noise residual returns => random-walk cumulative residuals => (almost) no
    eligible asset; with an impossible threshold nothing is traded and all weights are 0."""
    R = np.random.default_rng(9).normal(0, 0.01, (80, 10))
    window = make_panel(R)
    w, ctx, _ = decide(window, 1, ResidualReversion(s_open=50.0))
    assert (w == 0).all() and (ctx.state["positions"] == 0).all()


def test_short_window_and_large_k_raise(make_panel):
    R = np.random.default_rng(10).normal(0, 0.01, (8, 5))
    window = make_panel(R)
    with pytest.raises(ValueError):
        decide(window, 1)
    window = make_panel(np.random.default_rng(11).normal(0, 0.01, (40, 30)))
    with pytest.raises(ValueError):
        decide(window, 19, ResidualReversion(estimation_window=20))


def test_module_exposes_min_rows():
    assert statarb.MIN_ESTIMATION_ROWS == 10


# =========================================================================== engine
def ou_market_panel(make_panel, n=400, p=15, b=0.6, sd=0.006, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, 0.015, n)
    beta = rng.uniform(0.5, 1.5, p)
    X = np.zeros((n + 1, p))
    for t in range(n):
        X[t + 1] = b * X[t] + rng.normal(0.0, sd, p)
    return make_panel(np.outer(f, beta) + np.diff(X, axis=0))


def test_runs_through_walk_forward(make_panel, run_engine):
    panel = ou_market_panel(make_panel)
    cfg = {"window": 100, "rebalance": 1, "min_assets": 5}
    res = run_engine(panel, ResidualReversion(), 1, **cfg)
    W = res.weights.fillna(0.0)
    assert len(W) > 100
    assert np.isfinite(W.to_numpy()).all()
    gross = W.abs().sum(axis=1).to_numpy()
    assert np.all(np.isclose(gross, 1.0) | np.isclose(gross, 0.0))
    assert (gross > 0.5).mean() > 0.5                          # it actually trades
    # deterministic
    res2 = run_engine(panel, ResidualReversion(), 1, **cfg)
    pd.testing.assert_frame_equal(res.weights, res2.weights)
    # planted mean-reverting residuals are profitable before costs with the true k = 1, and
    # ignoring the market factor (k = 0) destroys most of the signal (reference run: annualised
    # gross Sharpe about 10.9 with k = 1 and 0.7 with k = 0).
    sr1 = res.summary()["sharpe_gross"]
    sr0 = run_engine(panel, ResidualReversion(), 0, **cfg).summary()["sharpe_gross"]
    assert sr1 > 5.0
    assert sr1 > sr0 + 3.0
