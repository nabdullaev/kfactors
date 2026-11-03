"""Tests for k-curves (kfactors.evaluate): metric registry, oracle / oracle set, shared bootstrap,
rules table, and an end-to-end k-curve with a planted economic optimum."""
from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from kfactors import metrics
from kfactors.data.panel import Panel
from kfactors.engine import BacktestResult, WalkForward, WalkForwardConfig
from kfactors.evaluate import (
    METRICS,
    KCurve,
    bootstrap_argmax,
    curve_from_results,
    k_curve,
    oracle_confidence_set,
    register_metric,
)
from kfactors.strategies import TangencyPCA
from kfactors.strategies._template import EqualWeight

# --------------------------------------------------------------------------- helpers


def fake_result(net, k_used, ppy=252, gross=None, spec=None):
    """A BacktestResult built directly from a return series (no engine)."""
    net = pd.Series(net, dtype=float)
    gross = net if gross is None else pd.Series(gross, index=net.index, dtype=float)
    kd = pd.DatetimeIndex(net.index[: len(k_used)])
    return BacktestResult(
        gross=gross, net=net, weights=pd.DataFrame(index=kd),
        turnover=pd.Series(0.0, index=kd), k_used=pd.Series(k_used, index=kd, dtype="int64"),
        k_spec=spec, strategy="fake", config=WalkForwardConfig(), periods_per_year=ppy)


def synthetic_results(n=500, best=3, seed=0):
    """Fixed-k results whose mean return peaks sharply at k = best; common noise across k."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    common = rng.standard_normal(n) * 0.01
    res = {}
    for k in range(1, 7):
        mean = 0.002 - 0.0015 * abs(k - best)
        x = mean + common + 0.001 * rng.standard_normal(n)
        res[k] = fake_result(pd.Series(x, index=idx), [k] * 50, spec=k)
    return res, idx


def planted_panel(seed=3, n=1500, p=30, sr=0.3, n_unpriced=5):
    """Market factor (zero premium) + one priced factor + unpriced strong factors.

    Loadings are orthogonal (QR) and scaled so that PC1 = market, PC2 = priced factor (factor
    sd 3 and 2.5), PC3..7 = unpriced factors (sd 1.2). The k-factor tangency portfolio therefore
    has, by construction: k = 0 -> no position; k = 1 -> only the zero-premium market (Sharpe ~ 0);
    k = 2 -> the priced factor (the economic optimum); k > 2 -> extra factors whose estimated
    means are pure noise (each adds ~1/window to the portfolio's squared-Sharpe denominator).
    With window 40 the oracle is 2 in 8/10 seeds at n=1500 (9/10 at n=3000), and k = 1 never
    enters the oracle set; seed 3 is one of them.
    """
    rng = np.random.default_rng(seed)
    K = 2 + n_unpriced
    B = rng.standard_normal((p, K))
    B[:, 0] = 1 + 0.3 * rng.standard_normal(p)
    Qm, _ = np.linalg.qr(B)
    B = Qm * np.sqrt(p)
    fsd = np.array([3.0, 2.5] + [1.2] * n_unpriced)
    mu = np.array([0.0, sr] + [0.0] * n_unpriced) * fsd
    F = rng.standard_normal((n, K)) * fsd + mu
    X = (F @ B.T + rng.standard_normal((n, p))) * 0.01
    idx = pd.bdate_range("2000-01-03", periods=n)
    R = pd.DataFrame(X, index=idx, columns=[f"a{j}" for j in range(p)])
    return Panel(R, R.notna(), name="planted", freq="D")


# --------------------------------------------------------------------------- metrics


def test_registry_metrics_match_kfactors_metrics():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((4, 200)) * 0.01 + 0.001
    M[1, :5] = np.nan
    for i in range(4):
        assert METRICS["sharpe_net"].fn(M, 252)[i] == pytest.approx(metrics.sharpe(M[i], 252))
        assert METRICS["neg_vol"].fn(M, 12)[i] == pytest.approx(-metrics.annual_vol(M[i], 12))
    assert METRICS["sharpe_gross"].series == "gross" and METRICS["neg_vol"].series == "net"
    const = np.full((1, 10), 0.01)
    assert np.isnan(METRICS["sharpe_net"].fn(const, 252)[0])


def test_register_metric_scalar_and_duplicate():
    try:
        register_metric("_test_mean", lambda x, ppy: np.nanmean(x) * ppy, vectorized=False)
        M = np.arange(6.0).reshape(2, 3)
        np.testing.assert_allclose(METRICS["_test_mean"].fn(M, 2), [2.0, 8.0])
        with pytest.raises(ValueError):
            register_metric("_test_mean", lambda x, ppy: 0.0)
        with pytest.raises(ValueError):
            register_metric("_bad", lambda M, ppy: M, series="weird")
    finally:
        METRICS.pop("_test_mean", None)


# --------------------------------------------------------------------------- oracle set


def test_bootstrap_argmax_ties_and_nan():
    D = np.array([[1.0, 2.0, np.nan, 0.0],
                  [1.0, 3.0, np.nan, 0.0],
                  [0.5, 3.0, np.nan, np.nan]])
    out = bootstrap_argmax(D, [1, 2, 3])
    np.testing.assert_array_equal(out[[0, 1, 3]], [1.0, 2.0, 1.0])   # tie -> smallest k
    assert np.isnan(out[2])
    # grid given unsorted: rows follow ks order
    np.testing.assert_array_equal(bootstrap_argmax(D[::-1], [3, 2, 1])[[0, 1]], [1.0, 2.0])


def test_oracle_confidence_set_hand_cases():
    a = np.array([2] * 50 + [3] * 30 + [4] * 15 + [5] * 5, dtype=float)
    assert oracle_confidence_set(a, 0.10, 2)[0] == [2, 3, 4]      # 95% >= 90%
    assert oracle_confidence_set(a, 0.20, 2)[0] == [2, 3]         # 80% >= 80%
    assert oracle_confidence_set(a, 0.50, 2)[0] == [2]
    s, freq = oracle_confidence_set(np.append(a, np.nan), 0.10, 2)
    assert s == [2, 3, 4] and freq.loc[2] == pytest.approx(0.5) and freq.sum() == pytest.approx(1)
    # frequency ties broken by distance to the oracle, then by smaller k
    b = np.array([4] * 40 + [3] * 30 + [5] * 30, dtype=float)
    assert oracle_confidence_set(b, 0.35, 4)[0] == [3, 4]
    assert oracle_confidence_set(b, 0.35, 6)[0] == [4, 5]
    with pytest.raises(ValueError):
        oracle_confidence_set([np.nan], 0.1)
    with pytest.raises(ValueError):
        oracle_confidence_set(a, 1.5)


def test_curve_from_synthetic_results_oracle_rules_and_shared_bootstrap():
    res, idx = synthetic_results()
    res["good_rule"] = fake_result(res[3].net, [3, 3, 4, 3, 2], spec="good_rule")
    res["bad_rule"] = fake_result(res[6].net, [6, 6, 5, 6], spec="bad_rule")
    kc = curve_from_results(res, range(1, 7), ["good_rule", "bad_rule"], n_boot=400,
                            random_state=5)
    assert kc.oracle_k == 3 and kc.oracle_set == [3]
    assert list(kc.curve.index) == [1, 2, 3, 4, 5, 6]
    assert kc.curve.loc[3, "argmax_freq"] == pytest.approx(1.0)
    r = kc.rules
    assert r.loc["good_rule", "k_median"] == 3 and bool(r.loc["good_rule", "in_set"])
    assert r.loc["good_rule", "dist_to_oracle"] == 0
    assert r.loc["good_rule", "regret"] == pytest.approx(0.0)
    assert r.loc["bad_rule", "k_median"] == 6 and not r.loc["bad_rule", "in_set"]
    assert r.loc["bad_rule", "dist_to_oracle"] == 3
    assert r.loc["bad_rule", "k_mean"] == pytest.approx(5.75)
    # point estimate = metrics.sharpe; draws = bootstrap_metric with the same shared indices
    for k in (1, 3):
        assert kc.curve.loc[k, "metric"] == pytest.approx(metrics.sharpe(res[k].net, 252))
    idxm = metrics.stationary_bootstrap_indices(len(idx), 400, len(idx) ** (1 / 3), 5)
    bm = metrics.bootstrap_metric(res[2].net, lambda x: metrics.sharpe(x, 252), indices=idxm,
                                  alpha=0.10)
    np.testing.assert_allclose(kc.meta["draws"].loc[2].to_numpy(), bm["draws"], rtol=1e-10)
    assert kc.curve.loc[2, "low"] == pytest.approx(bm["low"])
    assert kc.curve.loc[2, "high"] == pytest.approx(bm["high"])
    # rule rows share the same resamples as their twin fixed k
    np.testing.assert_allclose(kc.meta["draws"].loc["good_rule"], kc.meta["draws"].loc[3])
    assert kc.meta["n_common"] == len(idx) and kc.meta["n_valid_draws"] == 400


def test_common_date_range_and_validation():
    res, idx = synthetic_results(n=300)
    res[1] = fake_result(res[1].net.iloc[20:], [1] * 10, spec=1)   # starts later
    kc = curve_from_results(res, range(1, 7), [], n_boot=50)
    assert kc.meta["common_start"] == idx[20] and kc.meta["n_common"] == 280
    assert kc.meta["n_dropped"][2] == 20 and kc.meta["n_dropped"][1] == 0
    assert kc.rules.empty and list(kc.rules.columns)[:2] == ["k_mean", "k_median"]
    with pytest.raises(KeyError):
        curve_from_results(res, range(1, 8), [], n_boot=10)
    res[2] = fake_result(res[2].net, [2], ppy=12)
    with pytest.raises(ValueError, match="periods_per_year"):
        curve_from_results(res, range(1, 7), [], n_boot=10)
    with pytest.raises(ValueError):
        curve_from_results(res, [1, 1], [], n_boot=10)
    with pytest.raises(ValueError, match="unknown metric"):
        curve_from_results(res, range(1, 7), [], metric="nope")


def test_neg_vol_metric_prefers_low_volatility():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=400)
    res = {k: fake_result(pd.Series(rng.standard_normal(400) * 0.01 * (1 + abs(k - 2)),
                                    index=idx), [k], spec=k) for k in range(5)}
    kc = curve_from_results(res, range(5), [], metric="neg_vol", n_boot=200)
    assert kc.oracle_k == 2 and kc.metric == "neg_vol"
    assert kc.curve.loc[2, "metric"] == pytest.approx(-metrics.annual_vol(res[2].net, 252))


def test_to_frame_and_plot():
    res, _ = synthetic_results()
    res["r"] = fake_result(res[4].net, [4, 4, 5], spec="r")
    kc = curve_from_results(res, range(1, 7), ["r"], n_boot=100)
    df = kc.to_frame()
    assert len(df) == 7 and list(df["kind"]) == ["fixed"] * 6 + ["rule"]
    assert df.loc[df["spec"] == 3, "is_oracle"].item()
    assert df.loc[df["kind"] == "rule", "k"].item() == 4
    assert (df["metric_name"] == "sharpe_net").all()
    assert isinstance(kc, KCurve)
    ax = kc.plot()
    assert ax.get_ylabel() == "sharpe_net"
    import matplotlib.pyplot as plt
    plt.close("all")


# --------------------------------------------------------------------------- end to end


def test_k_curve_planted_optimum():
    pan = planted_panel()
    wf = WalkForward(pan, TangencyPCA(), WalkForwardConfig(window=40, rebalance=5,
                                                            min_assets=5, kmax=10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kc = k_curve(wf, k_grid=range(9), rules=("bai_ng", "onatski"), n_boot=300,
                     random_state=0)
    c = kc.curve
    assert kc.oracle_k == 2
    assert 2 in kc.oracle_set and 1 not in kc.oracle_set and 8 not in kc.oracle_set
    assert np.isnan(c.loc[0, "metric"])              # k = 0: no position at all
    assert c.loc[2, "metric"] > c.loc[1, "high"]     # missing the priced factor is costly
    assert c.loc[2, "metric"] > c.loc[8, "metric"] + 0.5
    # the rules find the 7 *statistical* factors, not the 2 that matter economically
    for rule in ("bai_ng", "onatski"):
        assert kc.rules.loc[rule, "k_median"] == 7
        assert not kc.rules.loc[rule, "in_set"]
        assert kc.rules.loc[rule, "dist_to_oracle"] == 5
        assert kc.rules.loc[rule, "regret"] > 0
    assert set(kc.results) == set(range(9)) | {"bai_ng", "onatski"}
    np.testing.assert_allclose(c["k_used_mean"].to_numpy(), np.arange(9))


def test_k_curve_flat_for_equal_weight_and_deterministic():
    pan = planted_panel(n=300, p=12, n_unpriced=2)
    wf = WalkForward(pan, EqualWeight(), WalkForwardConfig(window=30, rebalance=5, min_assets=5))
    a = k_curve(wf, k_grid=[1, 2, 3], rules=("bai_ng",), n_boot=100, random_state=1)
    b = k_curve(wf, k_grid=[1, 2, 3], rules=("bai_ng",), n_boot=100, random_state=1)
    pd.testing.assert_frame_equal(a.curve, b.curve)
    assert a.curve["metric"].nunique() == 1
    assert a.oracle_k == 1                            # ties -> smallest k
    assert a.oracle_set == [1]


# --------------------------------------------------------------------------- review fixes


def test_neg_vol_flat_series_is_nan_not_the_oracle():
    """LA-1: a flat (all-zero) series has vol 0 but is not 'least risky'; it scores NaN."""
    from kfactors.evaluate import _neg_vol_rows
    rng = np.random.default_rng(0)
    out = _neg_vol_rows(np.vstack([np.zeros(100), np.full(100, 0.01),
                                   rng.standard_normal(100) * 0.01, [np.nan] * 99 + [0.1]]), 252)
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[3])
    assert np.isfinite(out[2]) and out[2] < 0
    idx = pd.bdate_range("2015-01-01", periods=300)
    res = {0: fake_result(pd.Series(0.0, index=idx), [0], spec=0)}
    for k in (1, 2, 3):
        res[k] = fake_result(pd.Series(rng.standard_normal(300) * 0.01 * k, index=idx), [k],
                             spec=k)
    kc = curve_from_results(res, range(4), [], metric="neg_vol", n_boot=100)
    assert kc.oracle_k == 1 and np.isnan(kc.curve.loc[0, "metric"])
    assert kc.curve.loc[0, "argmax_freq"] == 0.0


def test_neg_vol_tangency_k0_not_oracle_end_to_end():
    """Reviewer repro: TangencyPCA with k = 0 holds nothing; under neg_vol it must not win."""
    pan = planted_panel(n=300, p=12, n_unpriced=2)
    wf = WalkForward(pan, TangencyPCA(), WalkForwardConfig(window=40, rebalance=5,
                                                            min_assets=5))
    kc = k_curve(wf, k_grid=range(5), rules=(), metric="neg_vol", n_boot=100)
    assert kc.oracle_k != 0 and np.isnan(kc.curve.loc[0, "metric"])


def test_regret_intervals_are_paired():
    """LA-6: regret_low/high are percentiles of draws[oracle_k] - draws[spec] (shared indices)."""
    res, _ = synthetic_results()
    res["good_rule"] = fake_result(res[3].net, [3, 3, 4], spec="good_rule")
    res["bad_rule"] = fake_result(res[5].net, [5, 5, 6], spec="bad_rule")
    kc = curve_from_results(res, range(1, 7), ["good_rule", "bad_rule"], n_boot=400,
                            alpha=0.2, random_state=3)
    D = kc.meta["draws"]
    for spec, row in [(5, kc.curve.loc[5]), ("bad_rule", kc.rules.loc["bad_rule"])]:
        diff = (D.loc[kc.oracle_k] - D.loc[spec]).to_numpy()
        lo, hi = np.quantile(diff, [0.1, 0.9])
        assert row["regret_low"] == pytest.approx(lo) and row["regret_high"] == pytest.approx(hi)
    # the oracle itself and its exact twin have a degenerate [0, 0] regret interval
    assert kc.curve.loc[kc.oracle_k, ["regret_low", "regret_high"]].tolist() == [0.0, 0.0]
    assert kc.rules.loc["good_rule", ["regret_low", "regret_high"]].tolist() == [0.0, 0.0]
    # common noise: the paired interval is much tighter than the marginal intervals suggest
    rb = kc.rules.loc["bad_rule"]
    assert rb["regret_low"] > 0
    assert (rb["regret_high"] - rb["regret_low"]) < 0.5 * (
        kc.curve.loc[3, "high"] - kc.curve.loc[3, "low"])
    df = kc.to_frame()
    assert {"regret_low", "regret_high"} <= set(df.columns)
    assert df.set_index("spec").loc["bad_rule", "regret_low"] == pytest.approx(rb["regret_low"])
