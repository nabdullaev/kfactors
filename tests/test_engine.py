"""Tests for the walk-forward engine (kfactors.engine): timing contract, leakage, universe,
skipping, k resolution, run_many and summary."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import kfactors.engine as engine_mod
from kfactors import metrics
from kfactors.data.panel import Panel
from kfactors.engine import BacktestResult, WalkForward, WalkForwardConfig
from kfactors.model import FactorModel
from kfactors.rules import select_k
from kfactors.strategies import TangencyPCA
from kfactors.strategies._template import EqualWeight
from kfactors.strategy import Strategy

# --------------------------------------------------------------------------- helpers


def make_panel(n=120, p=8, seed=0, k=2, freq="D", nan_frac=0.0):
    """Factor-structured returns with an optional random NaN pattern (tradable = observed)."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((p, k))
    F = rng.standard_normal((n, k)) * 2.0 + 0.1
    X = (F @ B.T + rng.standard_normal((n, p))) * 0.01
    if nan_frac:
        X[rng.random((n, p)) < nan_frac] = np.nan
    idx = pd.bdate_range("2001-01-01", periods=n)
    R = pd.DataFrame(X, index=idx, columns=[f"a{j}" for j in range(p)])
    return Panel(R, R.notna(), name="test", freq=freq)


class Recorder(Strategy):
    """Equal weight; records what the engine passed in."""

    name = "recorder"

    def __init__(self):
        self.calls = []

    def weights(self, window, model, context):
        self.calls.append({
            "date": context.date, "last": window.returns.index[-1],
            "first": window.returns.index[0], "cols": list(window.returns.columns),
            "k": context.k, "model_k": model.eigenvectors_.shape[1], "src": context.k_source,
            "prev": None if context.previous_weights is None else context.previous_weights.copy(),
            "state_n": context.state.get("n", 0),
        })
        context.state["n"] = context.state.get("n", 0) + 1
        cols = window.returns.columns
        return pd.Series(1.0 / len(cols), index=cols)


class Alternating(Strategy):
    """Decision j (0-based, counted in context.state) -> [1, 0] if j even else [0.5, -0.5]."""

    name = "alternating"

    def weights(self, window, model, context):
        j = context.state.get("j", 0)
        context.state["j"] = j + 1
        cols = window.returns.columns
        w = [1.0, 0.0] if j % 2 == 0 else [0.5, -0.5]
        return pd.Series(w + [0.0] * (len(cols) - 2), index=cols)


class DataDependent(Strategy):
    """Weights depend on every number of the window (via the model and window means)."""

    name = "data_dependent"

    def weights(self, window, model, context):
        X = window.returns.to_numpy()
        w = X.mean(axis=0) / X.std(axis=0)
        if model.eigenvectors_.shape[1]:
            w = w + model.eigenportfolios()[:, 0] * model.eigenvalues_[0]
        w = w + 1e-3 * len(context.state)
        context.state[str(context.date)] = 1
        return pd.Series(w / np.abs(w).sum(), index=window.returns.columns)


def reference_backtest(panel, weights_by_t, window, rebalance, lag, cost_bps, last=None):
    """Naive row-by-row implementation of the timing contract, independent of the engine.

    ``weights_by_t``: {decision row t: pd.Series} (only executed decisions). Returns
    (gross, net, turnover) dicts keyed by row.
    """
    R = panel.returns.to_numpy()
    n = R.shape[0] if last is None else last + 1
    gross, net, turnover = {}, {}, {}
    prev = pd.Series(dtype=float)
    for t in sorted(weights_by_t):
        w = weights_by_t[t]
        first_row = t + 1 + lag
        rows = [h for h in range(first_row, t + rebalance + lag + 1) if h < n]
        assets = prev.index.union(w.index)
        to = float((w.reindex(assets, fill_value=0) - prev.reindex(assets, fill_value=0))
                   .abs().sum())
        turnover[first_row] = to
        exited = set()                  # NaN return = exit at the last price until next decision
        for h in rows:
            g = 0.0
            for a, wa in w.items():
                r = R[h, panel.returns.columns.get_loc(a)]
                if np.isnan(r):
                    exited.add(a)
                g += 0.0 if a in exited else wa * r
            gross[h] = g
            net[h] = g - (cost_bps / 1e4 * to if h == first_row else 0.0)
        prev = w
    return gross, net, turnover


# --------------------------------------------------------------------------- timing


def tiny_panel():
    ra = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    rb = [0.03, -0.01, 0.02, 0.00, 0.01, -0.02, 0.04, 0.05]
    idx = pd.bdate_range("2020-01-06", periods=8)
    R = pd.DataFrame({"a": ra, "b": rb}, index=idx)
    return Panel(R, R.notna(), name="tiny", freq="D")


def test_timing_hand_computed_lag1_rebalance2_costs():
    """window=3, rebalance=2, lag=1, 100 bps. Decisions at rows 2 and 4 (row 6 would need a
    holding row 8 > 7). w(2) = [1, 0] earns rows 4, 5; w(4) = [.5, -.5] earns rows 6, 7."""
    pan = tiny_panel()
    cfg = WalkForwardConfig(window=3, rebalance=2, lag=1, cost_bps=100, min_assets=2)
    res = WalkForward(pan, Alternating(), cfg).run(0)
    d = pan.returns.index
    assert list(res.weights.index) == [d[2], d[4]]
    np.testing.assert_allclose(res.weights.to_numpy(), [[1.0, 0.0], [0.5, -0.5]])
    assert list(res.gross.index) == [d[4], d[5], d[6], d[7]]
    # row4: 1*.04 ; row5: 1*.05 ; row6: .5*.06 - .5*.04 ; row7: .5*.07 - .5*.05
    np.testing.assert_allclose(res.gross.to_numpy(), [0.04, 0.05, 0.01, 0.01], atol=1e-15)
    # turnover: |1| + 0 = 1 on row 4 ; |.5-1| + |-.5-0| = 1 on row 6 ; cost 1% each
    assert list(res.turnover.index) == [d[4], d[6]]
    np.testing.assert_allclose(res.turnover.to_numpy(), [1.0, 1.0])
    np.testing.assert_allclose(res.net.to_numpy(), [0.03, 0.05, 0.0, 0.01], atol=1e-15)
    assert list(res.k_used) == [0, 0]


def test_timing_hand_computed_lag0_rebalance1():
    """window=3, rebalance=1, lag=0: decisions rows 2..6, each earns the next row only."""
    pan = tiny_panel()
    cfg = WalkForwardConfig(window=3, rebalance=1, lag=0, cost_bps=0, min_assets=2)
    res = WalkForward(pan, Alternating(), cfg).run(1)
    d = pan.returns.index
    assert list(res.weights.index) == list(d[2:7])
    assert list(res.gross.index) == list(d[3:8])
    ra, rb = pan.returns["a"].to_numpy(), pan.returns["b"].to_numpy()
    expected = [ra[3], 0.5 * (ra[4] - rb[4]), ra[5], 0.5 * (ra[6] - rb[6]), ra[7]]
    np.testing.assert_allclose(res.gross.to_numpy(), expected, atol=1e-15)
    np.testing.assert_allclose(res.net.to_numpy(), res.gross.to_numpy())
    np.testing.assert_allclose(res.turnover.to_numpy(), [1.0, 1.0, 1.0, 1.0, 1.0])


@pytest.mark.parametrize("lag", [0, 1, 2])
@pytest.mark.parametrize("rebalance", [1, 3])
@pytest.mark.parametrize("cost_bps", [0.0, 25.0])
def test_timing_matches_reference(lag, rebalance, cost_bps):
    pan = make_panel(n=70, p=6, seed=3, nan_frac=0.05)
    cfg = WalkForwardConfig(window=12, rebalance=rebalance, lag=lag, cost_bps=cost_bps,
                            min_assets=2)
    res = WalkForward(pan, DataDependent(), cfg).run(2)
    idx = pan.returns.index
    wb = {idx.get_loc(dt): row[row != 0] for dt, row in res.weights.iterrows()}
    g, nt, to = reference_backtest(pan, wb, 12, rebalance, lag, cost_bps)
    rows = sorted(g)
    assert list(res.gross.index) == list(idx[rows])
    np.testing.assert_allclose(res.gross.to_numpy(), [g[h] for h in rows], atol=1e-14)
    np.testing.assert_allclose(res.net.to_numpy(), [nt[h] for h in rows], atol=1e-14)
    np.testing.assert_allclose(res.turnover.to_numpy(), [to[h] for h in sorted(to)], atol=1e-14)
    assert list(res.turnover.index) == list(idx[sorted(to)])
    # decision rows on the grid window-1, window-1+rebalance, ...; last needs a holding row
    t_rows = [idx.get_loc(dt) for dt in res.weights.index]
    assert t_rows[0] == 11 and np.all(np.diff(t_rows) == rebalance)
    assert t_rows[-1] + 1 + lag <= 69 < t_rows[-1] + rebalance + 1 + lag
    # holding periods tile the rows: contiguous realised series
    assert np.all(np.diff([idx.get_loc(x) for x in res.gross.index]) == 1)


def test_nan_in_holding_rows_exits_until_next_decision():
    """A NaN return of a held asset is an exit at the last price: it contributes 0 on that row
    *and* on the later rows of the same holding period (no free re-entry, LA-3)."""
    pan = tiny_panel()
    R = pan.returns.copy()
    R.iloc[4, 0] = np.nan          # asset a missing on a holding row (after decision row 2)
    T = R.notna()
    pan2 = Panel(R, T, freq="D")
    cfg = WalkForwardConfig(window=3, rebalance=2, lag=1, cost_bps=0, min_assets=2)
    res = WalkForward(pan2, Alternating(), cfg).run(0)
    # decision at row 2 (window rows 0..2 complete) earns rows 4, 5 with w = [1, 0]: a is
    # exited on row 4, so row 5 (a = +5%) is not earned either
    assert res.gross.iloc[0] == 0.0
    assert res.gross.iloc[1] == 0.0
    assert res.info["n_nan_exits"] == 1


def test_nan_gap_inside_holding_period_does_not_re_enter():
    """Reviewer repro (LA-3): NaN on row 31, +50% on row 32, same holding period."""
    X = np.random.default_rng(0).standard_normal((40, 4)) * 0.01
    X[31, 0], X[32, 0] = np.nan, 0.5
    R = pd.DataFrame(X, index=pd.date_range("2024", periods=40), columns=list("abcd"))
    T = R.notna()
    T.iloc[31:, 0] = False
    res = WalkForward(Panel(R, T), EqualWeight(),
                      WalkForwardConfig(window=30, rebalance=5, min_assets=2)).run(1)
    # decision row 29 holds rows 30..34 at 1/4 each; a exits on row 31
    np.testing.assert_allclose(res.gross.iloc[2], X[32, 1:].sum() / 4, atol=1e-15)
    np.testing.assert_allclose(res.gross.iloc[0], X[30].sum() / 4, atol=1e-15)
    assert res.info["n_nan_exits"] == 1
    # the next decision (row 34) drops a (NaN in its window): the exit is charged there
    assert "a" not in res.weights.columns[res.weights.iloc[1] != 0]


# --------------------------------------------------------------------------- leakage


def corrupt_after(panel, t, seed):
    """Rows > t replaced by huge noise with NaN holes and a scrambled tradable mask."""
    rng = np.random.default_rng(seed)
    R = panel.returns.copy()
    n, p = R.shape
    noise = rng.standard_normal((n - t - 1, p)) * 1e3
    noise[rng.random(noise.shape) < 0.3] = np.nan
    R.iloc[t + 1:] = noise
    T = panel.tradable.copy()
    T.iloc[t + 1:] = rng.random((n - t - 1, p)) < 0.5
    T = T & R.notna()
    return Panel(R, T, name=panel.name, freq=panel.freq, meta=dict(panel.meta))


@pytest.mark.parametrize("k", [3, "bai_ng", "parallel", "onatski"])
@pytest.mark.parametrize("strategy", [TangencyPCA, DataDependent])
def test_leakage_corrupting_future_rows_leaves_weights_unchanged(k, strategy):
    pan = make_panel(n=110, p=12, seed=7, k=3, nan_frac=0.02)
    cfg = WalkForwardConfig(window=40, rebalance=3, lag=1, min_assets=5,
                            rule_kwargs={"parallel": {"n_perm": 20}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = WalkForward(pan, strategy(), cfg).run(k)
        rng = np.random.default_rng(0)
        dates = base.weights.index
        for t_date in rng.choice(dates[:-1], size=4, replace=False):
            t = pan.returns.index.get_loc(t_date)
            res = WalkForward(corrupt_after(pan, t, seed=t), strategy(), cfg).run(k)
            upto = base.weights.index[base.weights.index <= t_date]
            a = base.weights.loc[upto]
            b = res.weights.reindex(index=upto, columns=a.columns, fill_value=0.0)
            np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
            pd.testing.assert_series_equal(base.k_used.loc[:t_date], res.k_used.loc[:t_date])


def test_strategy_never_sees_rows_after_decision():
    pan = make_panel(n=60, p=5, seed=1)
    rec = Recorder()
    cfg = WalkForwardConfig(window=10, rebalance=2, lag=1, min_assets=2)
    WalkForward(pan, rec, cfg).run(1)
    idx = pan.returns.index
    for c in rec.calls:
        assert c["last"] == c["date"]
        assert idx.get_loc(c["first"]) == idx.get_loc(c["date"]) - 9


# --------------------------------------------------------------------------- universe


def test_universe_tradable_mask_and_window_nans():
    pan = make_panel(n=40, p=6, seed=2)
    R, T = pan.returns.copy(), pan.tradable.copy()
    idx = R.index
    T.iloc[20, 0] = False                  # a0 not tradable on row 20 only
    R.iloc[15, 1] = np.nan                 # a1 missing on row 15 -> out while 15 in window
    T.iloc[15, 1] = False
    R.iloc[25:, 2] = np.nan                # a2 delisted after row 24
    T.iloc[25:, 2] = False
    pan2 = Panel(R, T, freq="D")
    rec = Recorder()
    cfg = WalkForwardConfig(window=5, rebalance=1, min_assets=2)
    WalkForward(pan2, rec, cfg).run(1)
    by_date = {c["date"]: c["cols"] for c in rec.calls}
    for t in range(4, 39):
        cols = by_date[idx[t]]
        assert ("a0" in cols) == (t != 20)
        assert ("a1" in cols) == (not 15 <= t <= 19)
        assert ("a2" in cols) == (t <= 24)
        assert cols == list(pan2.window(t, 5).returns.columns)


def test_zero_variance_asset_dropped_only_when_standardizing():
    pan = make_panel(n=40, p=5, seed=4)
    R = pan.returns.copy()
    R["a0"] = 0.001                        # constant return: cannot be standardised
    pan2 = Panel(R, R.notna(), freq="D")
    for std, present in [(True, False), (False, True)]:
        rec = Recorder()
        WalkForward(pan2, rec, WalkForwardConfig(window=10, min_assets=2, standardize=std)).run(2)
        assert all(("a0" in c["cols"]) == present for c in rec.calls)


def test_min_assets_skipping_warmup_and_flat_period():
    pan = make_panel(n=40, p=4, seed=5)
    R, T = pan.returns.copy(), pan.tradable.copy()
    R.iloc[:12, 2:] = np.nan               # a2, a3 enter at row 12
    T.iloc[:12, 2:] = False
    T.iloc[25, 1:] = False                 # row 25: only a0 tradable -> skip (flat)
    pan2 = Panel(R, T, freq="D")
    cfg = WalkForwardConfig(window=5, rebalance=1, min_assets=3, cost_bps=10)
    res = WalkForward(pan2, EqualWeight(), cfg).run(1)
    idx = pan2.returns.index
    # warm-up: windows need rows 12..16 complete for a2/a3 -> first decision at row 16
    assert res.weights.index[0] == idx[16]
    assert res.gross.index[0] == idx[17]
    assert idx[4] in res.skipped and idx[15] in res.skipped and idx[25] in res.skipped
    assert res.info["n_skipped"] == len(res.skipped) == 13
    # mid-run skip: flat book, zero return, liquidation turnover charged
    assert (res.weights.loc[idx[25]] == 0).all()
    assert res.gross.loc[idx[26]] == 0.0
    assert res.turnover.loc[idx[26]] == pytest.approx(1.0)
    assert res.net.loc[idx[26]] == pytest.approx(-10 / 1e4)
    assert idx[25] not in res.k_used.index
    assert res.turnover.loc[idx[27]] == pytest.approx(1.0)   # re-entry from flat
    assert res.universe_size.loc[idx[25]] == 1


def test_no_executed_decision_raises():
    pan = make_panel(n=30, p=4)
    with pytest.raises(ValueError, match="min_assets"):
        WalkForward(pan, EqualWeight(), WalkForwardConfig(window=5, min_assets=5)).run(1)


# --------------------------------------------------------------------------- k resolution


def test_fixed_k_is_clipped_to_window_rank():
    pan = make_panel(n=40, p=10, seed=6)
    rec = Recorder()
    res = WalkForward(pan, rec, WalkForwardConfig(window=5, min_assets=2)).run(10)
    assert set(res.k_used) == {4}                      # min(window - 1, p) = 4
    assert all(c["k"] == c["model_k"] == 4 for c in rec.calls)
    assert res.info["n_clipped"] == len(res.k_used)
    res2 = WalkForward(pan, EqualWeight(), WalkForwardConfig(window=20, min_assets=2)).run(10)
    assert set(res2.k_used) == {10} and res2.info["n_clipped"] == 0


def test_callable_k():
    pan = make_panel(n=50, p=9, seed=8)
    shapes = []

    def third(X):
        shapes.append(X.shape)
        return X.shape[1] // 3

    rec = Recorder()
    res = WalkForward(pan, rec, WalkForwardConfig(window=10, min_assets=2)).run(third)
    assert set(res.k_used) == {3}
    assert all(s == (10, 9) for s in shapes)
    assert rec.calls[0]["src"] == "callable:third"
    assert set(WalkForward(pan, EqualWeight(), WalkForwardConfig(window=10, min_assets=2))
               .run(lambda X: 50).k_used) == {9}           # clipped to rank min(9, 9)
    assert set(WalkForward(pan, EqualWeight(), WalkForwardConfig(window=10, min_assets=2))
               .run(lambda X: 2.0).k_used) == {2}
    with pytest.raises(TypeError):
        WalkForward(pan, EqualWeight(), WalkForwardConfig(window=10, min_assets=2)).run(
            lambda X: 2.5)
    with pytest.raises(ValueError):
        WalkForward(pan, EqualWeight(), WalkForwardConfig(window=10, min_assets=2)).run(
            lambda X: -1)


@pytest.mark.parametrize("rule,kw", [("bai_ng", {"criterion": "IC_p1"}),
                                     ("parallel", {"n_perm": 15}), ("mp", {})])
def test_rule_k_matches_select_k_on_window(rule, kw):
    pan = make_panel(n=60, p=12, seed=9, k=3)
    cfg = WalkForwardConfig(window=30, rebalance=4, min_assets=5, kmax=6, random_state=3,
                            rule_kwargs={rule: kw})
    rec = Recorder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = WalkForward(pan, rec, cfg).run(rule)
        for c in rec.calls:
            t = pan.returns.index.get_loc(c["date"])
            X = pan.returns.iloc[t - 29:t + 1][c["cols"]].to_numpy()
            exp = select_k(X, rule, kmax=6, standardize=True, random_state=3, **kw).k
            assert res.k_used.loc[c["date"]] == exp == c["k"]
            assert c["src"] == rule


def test_invalid_specs_and_config():
    pan = make_panel(n=40, p=5)
    wf = WalkForward(pan, EqualWeight(), WalkForwardConfig(window=10, min_assets=2))
    with pytest.raises(TypeError):
        wf.run(True)
    with pytest.raises(ValueError):
        wf.run(-1)
    with pytest.raises(ValueError, match="unknown rule"):
        wf.run("nope")
    with pytest.raises(TypeError):
        wf.run(2.5)
    for bad in [{"window": 2}, {"rebalance": 0}, {"lag": -1}, {"min_assets": 1},
                {"cost_bps": -1.0}, {"cost_bps": float("nan")}]:
        with pytest.raises(ValueError):
            WalkForward(pan, EqualWeight(), WalkForwardConfig(**{"min_assets": 2, **bad})).run(1)
    with pytest.raises(ValueError, match="no feasible decision"):
        WalkForward(pan, EqualWeight(), WalkForwardConfig(window=40, min_assets=2)).run(1)


class BadWeights(Strategy):
    def __init__(self, kind):
        self.kind = kind

    def weights(self, window, model, context):
        cols = window.returns.columns
        if self.kind == "outside":
            return pd.Series(1.0, index=list(cols) + ["zzz"])
        if self.kind == "nan":
            return pd.Series(np.nan, index=cols)
        if self.kind == "shape":
            return np.ones(len(cols) + 1)
        return pd.Series([1.0, 1.0], index=[cols[0], cols[0]])


@pytest.mark.parametrize("kind,exc", [("outside", ValueError), ("nan", ValueError),
                                      ("shape", TypeError), ("dup", ValueError)])
def test_invalid_weights_raise(kind, exc):
    pan = make_panel(n=30, p=4)
    with pytest.raises(exc):
        WalkForward(pan, BadWeights(kind), WalkForwardConfig(window=5, min_assets=2)).run(1)


def test_partial_weights_and_array_weights_accepted():
    class Partial(Strategy):
        def weights(self, window, model, context):
            return pd.Series({window.returns.columns[1]: 0.5})

    class Arr(Strategy):
        def weights(self, window, model, context):
            return np.full(window.returns.shape[1], 0.25)

    pan = make_panel(n=30, p=4)
    cfg = WalkForwardConfig(window=5, min_assets=2)
    r1 = WalkForward(pan, Partial(), cfg).run(1)
    assert (r1.weights["a1"] == 0.5).all() and (r1.weights.drop(columns="a1") == 0).all().all()
    r2 = WalkForward(pan, Arr(), cfg).run(1)
    r3 = WalkForward(pan, EqualWeight(), cfg).run(1)
    np.testing.assert_allclose(r2.gross, r3.gross)


# --------------------------------------------------------------------------- context


def test_context_previous_weights_and_state():
    pan = make_panel(n=40, p=5, seed=3)
    rec = Recorder()
    wf = WalkForward(pan, rec, WalkForwardConfig(window=10, rebalance=5, min_assets=2))
    res = wf.run(2)
    assert rec.calls[0]["prev"] is None
    for i, c in enumerate(rec.calls[1:], start=1):
        prev_row = res.weights.iloc[i - 1]
        pd.testing.assert_series_equal(c["prev"], prev_row[prev_row != 0], check_names=False)
        assert c["state_n"] == i
        assert c["src"] == "fixed"
    n_calls = len(rec.calls)
    wf.run(2)                                       # fresh state for a new run
    assert rec.calls[n_calls]["state_n"] == 0


def test_start_end_restrict_dates_and_ignore_later_rows():
    pan = make_panel(n=80, p=5, seed=11)
    idx = pan.returns.index
    cfg = WalkForwardConfig(window=10, rebalance=2, lag=1, min_assets=2,
                            start=str(idx[30].date()), end=str(idx[60].date()))
    res = WalkForward(pan, DataDependent(), cfg).run(2)
    assert res.weights.index[0] == idx[30]
    assert res.gross.index[-1] == idx[60]
    assert res.weights.index[-1] <= idx[58]
    res2 = WalkForward(corrupt_after(pan, 60, 1), DataDependent(), cfg).run(2)
    pd.testing.assert_series_equal(res.net, res2.net)
    sched = WalkForward(pan, DataDependent(), cfg).schedule()
    assert list(sched.index) == list(res.weights.index)
    assert sched["executed"].all() and (sched["hold_start"] == idx[32:61:2]).all()


# --------------------------------------------------------------------------- run_many


def test_run_many_equals_individual_runs_and_caches_rules(monkeypatch):
    pan = make_panel(n=70, p=10, seed=12, k=3)
    cfg = WalkForwardConfig(window=25, rebalance=3, min_assets=4,
                            rule_kwargs={"parallel": {"n_perm": 10}})
    calls = []
    real = engine_mod.select_k

    def counting(X, method, **kw):
        calls.append(method)
        return real(X, method, **kw)

    monkeypatch.setattr(engine_mod, "select_k", counting)

    def half(X):
        return X.shape[1] // 2

    specs = [0, 2, "bai_ng", "parallel", half, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf = WalkForward(pan, TangencyPCA(), cfg)
        many = wf.run_many(specs)
        n_dec = len(many[0].k_used)
        assert calls.count("bai_ng") == n_dec and calls.count("parallel") == n_dec
        wf.run("bai_ng")                            # served from the cache
        assert calls.count("bai_ng") == n_dec
        assert list(many) == [0, 2, "bai_ng", "parallel", half]
        for k in many:
            single = WalkForward(pan, TangencyPCA(), cfg).run(k)
            for attr in ("gross", "net", "turnover", "k_used"):
                pd.testing.assert_series_equal(getattr(many[k], attr), getattr(single, attr))
            pd.testing.assert_frame_equal(many[k].weights, single.weights)
            assert many[k].k_spec is k


def test_run_many_parallel_matches_sequential():
    pan = make_panel(n=60, p=8, seed=13, k=2)
    cfg = WalkForwardConfig(window=20, rebalance=4, min_assets=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seq = WalkForward(pan, TangencyPCA(), cfg).run_many([1, 3, "bai_ng", "onatski"])
        par = WalkForward(pan, TangencyPCA(), cfg).run_many([1, 3, "bai_ng", "onatski"],
                                                            n_jobs=2)
    for k in seq:
        pd.testing.assert_series_equal(seq[k].net, par[k].net)
        pd.testing.assert_series_equal(seq[k].k_used, par[k].k_used)


# --------------------------------------------------------------------------- summary


def test_summary_keys_and_values():
    pan = make_panel(n=80, p=6, seed=14)
    res = WalkForward(pan, TangencyPCA(), WalkForwardConfig(window=20, rebalance=2,
                                                            cost_bps=5, min_assets=2)).run(2)
    s = res.summary()
    for key in ("sharpe_gross", "sharpe_net", "ann_return", "ann_vol", "max_drawdown",
                "mean_turnover", "k_mean", "k_median", "n_periods"):
        assert key in s
    assert s["sharpe_net"] == pytest.approx(metrics.sharpe(res.net, 252))
    assert s["sharpe_gross"] == pytest.approx(metrics.sharpe(res.gross, 252))
    assert s["ann_vol"] == pytest.approx(metrics.annual_vol(res.net, 252))
    assert s["ann_return"] == pytest.approx(metrics.annual_return(res.net, 252))
    assert s["max_drawdown"] == pytest.approx(metrics.max_drawdown(res.net))
    assert s["mean_turnover"] == pytest.approx(res.turnover.mean())
    assert s["k_mean"] == s["k_median"] == 2.0
    assert s["n_periods"] == len(res.net) == 60
    assert s["sharpe_net"] < s["sharpe_gross"]
    assert isinstance(res, BacktestResult) and res.periods_per_year == 252
    assert res.strategy == "sdf_pca_tangency" and res.k_spec == 2


def test_periods_per_year_from_panel_meta():
    pan = make_panel(n=40, p=4)
    pan.meta["periods_per_year"] = 365
    res = WalkForward(pan, EqualWeight(), WalkForwardConfig(window=10, min_assets=2)).run(1)
    assert res.periods_per_year == 365


def test_model_passed_to_strategy_is_fitted_on_window():
    pan = make_panel(n=40, p=6, seed=15)

    class Check(Strategy):
        def weights(self, window, model, context):
            ref = FactorModel(context.k).fit(window.returns.to_numpy())
            np.testing.assert_allclose(model.loadings_, ref.loadings_)
            return pd.Series(0.1, index=window.returns.columns)

    WalkForward(pan, Check(), WalkForwardConfig(window=12, rebalance=5, min_assets=2)).run(3)


# --------------------------------------------------------------------------- review fixes


def drift_reference(panel, weights_by_t, rebalance, lag, cost_bps):
    """Naive buy-and-hold reference: weights drift inside a holding period (cash earns 0), NaN =
    exit at the last price; turnover is measured against the drifted book."""
    R = panel.returns.to_numpy()
    cols = list(panel.returns.columns)
    n = R.shape[0]
    gross, net, turnover = {}, {}, {}
    book = {}
    for t in sorted(weights_by_t):
        w = {a: float(v) for a, v in weights_by_t[t].items()}
        first_row = t + 1 + lag
        to = sum(abs(w.get(a, 0.0) - book.get(a, 0.0)) for a in set(w) | set(book))
        turnover[first_row] = to
        cur, exited = dict(w), set()
        for h in range(first_row, min(t + rebalance + lag, n - 1) + 1):
            r = {}
            for a in cur:
                x = R[h, cols.index(a)]
                if np.isnan(x):
                    exited.add(a)
                r[a] = 0.0 if a in exited else x
            g = sum(cur[a] * r[a] for a in cur)
            cur = {a: cur[a] * (1.0 + r[a]) / (1.0 + g) for a in cur}
            gross[h] = g
            net[h] = g - (cost_bps / 1e4 * to if h == first_row else 0.0)
        book = cur
    return gross, net, turnover


@pytest.mark.parametrize("lag", [0, 2])
@pytest.mark.parametrize("rebalance", [1, 4])
def test_drift_option_matches_buy_and_hold_reference(lag, rebalance):
    """LA-2: drift=True lets weights drift with returns and charges turnover against the
    drifted book; drift=False (default) is unchanged."""
    pan = make_panel(n=70, p=6, seed=3, nan_frac=0.03)
    kw = dict(window=12, rebalance=rebalance, lag=lag, cost_bps=30.0, min_assets=2)
    res = WalkForward(pan, DataDependent(), WalkForwardConfig(drift=True, **kw)).run(2)
    base = WalkForward(pan, DataDependent(), WalkForwardConfig(**kw)).run(2)
    pd.testing.assert_frame_equal(res.weights, base.weights)       # targets do not change
    idx = pan.returns.index
    wb = {idx.get_loc(dt): row[row != 0] for dt, row in res.weights.iterrows()}
    g, nt, to = drift_reference(pan, wb, rebalance, lag, 30.0)
    rows = sorted(g)
    assert list(res.gross.index) == list(idx[rows])
    np.testing.assert_allclose(res.gross.to_numpy(), [g[h] for h in rows], atol=1e-13)
    np.testing.assert_allclose(res.net.to_numpy(), [nt[h] for h in rows], atol=1e-13)
    np.testing.assert_allclose(res.turnover.to_numpy(), [to[h] for h in sorted(to)], atol=1e-13)
    # with the drifted book as w_prev the extra (otherwise free) re-balancing is charged
    assert res.turnover.iloc[1:].sum() != pytest.approx(base.turnover.iloc[1:].sum())


class Sticky(Strategy):
    """Half old book, half equal weight: weights depend on context.previous_weights."""

    name = "sticky"

    def weights(self, window, model, context):
        cols = window.returns.columns
        w = pd.Series(1.0 / len(cols), index=cols)
        if context.previous_weights is not None:
            prev = context.previous_weights.reindex(cols, fill_value=0.0)
            w = 0.5 * w + 0.5 * prev
        return w


def test_drift_previous_weights_use_no_future_rows():
    """With drift and lag > 0 the book passed as previous_weights at t is drifted through rows
    <= t only: corrupting rows > t leaves every w_<=t unchanged."""
    pan = make_panel(n=90, p=8, seed=11)
    cfg = WalkForwardConfig(window=20, rebalance=3, lag=2, min_assets=4, drift=True)
    base = WalkForward(pan, Sticky(), cfg).run(1)
    for t_date in base.weights.index[[1, 5, 12]]:
        t = pan.returns.index.get_loc(t_date)
        res = WalkForward(corrupt_after(pan, t, seed=t), Sticky(), cfg).run(1)
        upto = base.weights.index[base.weights.index <= t_date]
        a = base.weights.loc[upto]
        b = res.weights.reindex(index=upto, columns=a.columns, fill_value=0.0)
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_window_meta_has_no_sample_level_keys():
    """LA-4: sample-describing meta (end, ticker_splits, symbol, ...) is not passed on."""
    pan = make_panel(n=40, p=5)
    pan.meta.update({"end": "2001-02-23", "ticker_splits": [{"asset": "a0_2"}],
                     "symbol": {"a0": "A0USDT"}, "first_complete_date": "2001-01-01",
                     "n_assets": 5, "periods_per_year": 260, "source": "unit test",
                     "licence": "CC0"})
    seen = []

    class Peek(Strategy):
        def weights(self, window, model, context):
            seen.append(dict(window.meta))
            return pd.Series(0.2, index=window.returns.columns)

    res = WalkForward(pan, Peek(), WalkForwardConfig(window=10, min_assets=2)).run(1)
    assert seen and all(m == {"periods_per_year": 260, "source": "unit test",
                              "licence": "CC0"} for m in seen)
    assert set(seen[0]) <= set(engine_mod.WINDOW_META_KEYS)
    assert res.periods_per_year == 260 and "end" in pan.meta     # parent panel untouched


def test_start_end_on_tz_aware_index():
    """LA-5: naive start/end strings work on a UTC index (as in Panel.subset)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=60, tz="UTC")
    R = pd.DataFrame(rng.standard_normal((60, 8)) * 0.01, index=idx, columns=list("abcdefgh"))
    pan = Panel(R, R.notna())
    cfg = WalkForwardConfig(window=20, min_assets=7, start="2024-02-01", end="2024-02-20")
    res = WalkForward(pan, TangencyPCA(), cfg).run(1)
    assert res.weights.index[0] == pd.Timestamp("2024-02-01", tz="UTC")
    assert res.gross.index[-1] == pd.Timestamp("2024-02-20", tz="UTC")
    aware = WalkForwardConfig(window=20, min_assets=7, start=pd.Timestamp("2024-02-01", tz="UTC"))
    assert WalkForward(pan, TangencyPCA(), aware).run(1).weights.index[0] == res.weights.index[0]


def test_rule_kwargs_and_random_state_validated():
    """LA-7: unknown rule names, reserved keys and non-integer seeds are rejected."""
    pan = make_panel(n=40, p=6)

    def run(**kw):
        cfg = WalkForwardConfig(window=10, min_assets=2, **kw)
        return WalkForward(pan, EqualWeight(), cfg).run(1)

    with pytest.raises(ValueError, match="unknown rule 'paralel'"):
        run(rule_kwargs={"paralel": {"n_perm": 5}})
    with pytest.raises(ValueError, match="random_state"):
        run(rule_kwargs={"parallel": {"random_state": 1}})
    with pytest.raises(ValueError, match="kmax"):
        run(rule_kwargs={"bai_ng": {"kmax": 3}})
    with pytest.raises(TypeError, match="must be a dict"):
        run(rule_kwargs={"parallel": 5})
    for bad in ("0", 1.5, True):
        with pytest.raises(TypeError, match="random_state"):
            run(random_state=bad)
    with pytest.raises(TypeError, match="drift"):
        run(drift="yes")
    for ok in (None, 3, np.int64(3), np.random.default_rng(0)):
        run(random_state=ok, rule_kwargs={"parallel": {"n_perm": 5}})


def test_public_api_all():
    """U3: __all__ lists the public names (used by mkdocstrings and star imports)."""
    import kfactors.evaluate as ev
    import kfactors.metrics as me
    import kfactors.model as mo
    import kfactors.sim as si
    import kfactors.strategy as st
    for mod in (engine_mod, ev, me, mo, si, st):
        assert mod.__all__ and all(hasattr(mod, n) for n in mod.__all__), mod.__name__
    assert {"WalkForward", "WalkForwardConfig", "BacktestResult"} <= set(engine_mod.__all__)
