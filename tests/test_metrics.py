"""Tests for kfactors.metrics (known answers, Monte Carlo checks, brute-force cross-checks)."""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from kfactors import metrics as M

PHI = NormalDist().cdf
PHI_INV = NormalDist().inv_cdf


# ------------------------------------------------------------------------ point metrics


def test_sharpe_known_answer_and_nan_handling():
    r = [0.01, -0.01, 0.02, 0.0]
    mean = 0.02 / 4
    sd = math.sqrt(((0.005) ** 2 + 0.015**2 + 0.015**2 + 0.005**2) / 3)
    assert M.sharpe(r, 252) == pytest.approx(mean / sd * math.sqrt(252), rel=1e-12)
    assert M.sharpe(np.array(r), 12) == pytest.approx(mean / sd * math.sqrt(12), rel=1e-12)
    s = pd.Series([np.nan, 0.01, -0.01, np.nan, 0.02, 0.0])
    assert M.sharpe(s, 252) == pytest.approx(M.sharpe(r, 252), rel=1e-12)
    assert math.isnan(M.sharpe([0.01] * 10))  # zero dispersion
    assert math.isnan(M.sharpe([0.01]))
    assert math.isnan(M.sharpe([]))
    with pytest.raises(ValueError):
        M.sharpe([0.01, np.inf])
    with pytest.raises(ValueError):
        M.sharpe(np.zeros((3, 2)))


def test_annual_return_geometric():
    r = pd.Series([0.01] * 24)
    assert M.annual_return(r, 12) == pytest.approx(1.01**12 - 1, rel=1e-12)
    r2 = [0.10, -0.05, 0.02]
    expected = (1.10 * 0.95 * 1.02) ** (252 / 3) - 1
    assert M.annual_return(r2, 252) == pytest.approx(expected, rel=1e-10)
    assert M.annual_return([0.1, np.nan, -0.05, 0.02], 252) == pytest.approx(expected, rel=1e-10)
    assert M.annual_return([0.1, -1.0, 0.2]) == -1.0
    assert math.isnan(M.annual_return([np.nan]))


def test_annual_vol():
    r = [0.01, -0.01, 0.02, 0.0]
    assert M.annual_vol(r, 252) == pytest.approx(np.std(r, ddof=1) * math.sqrt(252), rel=1e-12)
    assert math.isnan(M.annual_vol([0.01]))


def test_max_drawdown_known_answers():
    assert M.max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)
    assert M.max_drawdown([0.01, 0.02, 0.0]) == 0.0
    assert M.max_drawdown([-0.2, 0.1]) == pytest.approx(-0.2)  # first-period loss counts
    # peak 1.1*1.1 = 1.21, trough 1.21*0.9*0.8 = 0.8712
    assert M.max_drawdown([0.1, 0.1, -0.1, -0.2, 0.5]) == pytest.approx(0.8712 / 1.21 - 1)
    assert M.max_drawdown(pd.Series([0.1, np.nan, -0.5])) == pytest.approx(-0.5)
    assert M.max_drawdown([0.1, -1.5]) == -1.0  # limited liability
    assert math.isnan(M.max_drawdown([]))


# --------------------------------------------------------------------------- bootstrap


def test_stationary_bootstrap_indices_structure():
    n, nb, L = 1000, 200, 10.0
    idx = M.stationary_bootstrap_indices(n, nb, L, random_state=0)
    assert idx.shape == (nb, n) and idx.dtype == np.int64
    assert idx.min() >= 0 and idx.max() < n
    np.testing.assert_array_equal(idx, M.stationary_bootstrap_indices(n, nb, L, random_state=0))
    # a "break" is any step that is not (previous + 1) mod n; P(break) = (1/L)(1 - 1/n)
    breaks = np.mean(idx[:, 1:] != (idx[:, :-1] + 1) % n)
    assert breaks == pytest.approx((1 / L) * (1 - 1 / n), abs=0.004)
    # wrap-around is used and the marginal distribution is uniform over 0..n-1
    assert np.any((idx[:, :-1] == n - 1) & (idx[:, 1:] == 0))
    counts = np.bincount(idx.ravel(), minlength=n)
    assert counts.mean() == nb
    assert counts.std() < 3 * math.sqrt(nb * L)  # clustered but unbiased
    # mean_block = 1 -> iid
    iid = M.stationary_bootstrap_indices(n, 50, 1.0, random_state=1)
    assert np.mean(iid[:, 1:] != (iid[:, :-1] + 1) % n) == pytest.approx(1 - 1 / n, abs=0.003)
    with pytest.raises(ValueError):
        M.stationary_bootstrap_indices(n, 10, 0.5)
    with pytest.raises(ValueError):
        M.stationary_bootstrap_indices(0, 10, 2.0)


def test_bootstrap_shared_indices_and_fields():
    r = np.random.default_rng(0).standard_normal(300) * 0.01
    a = M.bootstrap_metric(r, M.sharpe, n_boot=200, mean_block=5, random_state=3)
    idx = M.stationary_bootstrap_indices(300, 200, 5, random_state=3)
    b = M.bootstrap_metric(pd.Series(r), M.sharpe, indices=idx)
    assert set(a) >= {"estimate", "low", "high", "draws"}
    assert a["estimate"] == pytest.approx(M.sharpe(r))
    np.testing.assert_array_equal(a["draws"], b["draws"])
    assert a["low"] == b["low"] and a["high"] == b["high"]
    assert a["low"] < a["estimate"] < a["high"]
    np.testing.assert_allclose(np.quantile(a["draws"], [0.025, 0.975]), [a["low"], a["high"]])
    with pytest.raises(ValueError):
        M.bootstrap_metric(r, M.sharpe, indices=idx[:, :100])
    with pytest.raises(ValueError):
        M.bootstrap_metric([], M.sharpe)


def test_bootstrap_ci_coverage_iid_normal():
    """95% percentile CI for the Sharpe ratio covers the true value ~95% of the time."""
    rng = np.random.default_rng(2024)
    mu, sig, n, ppy = 0.0005, 0.01, 500, 252
    true_sr = mu / sig * math.sqrt(ppy)
    reps, hits = 200, 0
    for _ in range(reps):
        r = mu + sig * rng.standard_normal(n)
        ci = M.bootstrap_metric(r, M.sharpe, n_boot=400, random_state=rng)
        hits += ci["low"] <= true_sr <= ci["high"]
    cover = hits / reps
    # binomial sd at 0.95 with 200 reps is ~0.015; allow block-bootstrap undercoverage
    assert 0.88 <= cover <= 0.99, cover


def test_sharpe_diff_ci_paired_symmetry_and_identity():
    rng = np.random.default_rng(5)
    a = 0.0006 + 0.01 * rng.standard_normal(400)
    b = 0.5 * a + 0.01 * rng.standard_normal(400)
    d1 = M.sharpe_diff_ci(a, b, n_boot=500, random_state=11)
    d2 = M.sharpe_diff_ci(b, a, n_boot=500, random_state=11)
    assert d1["estimate"] == pytest.approx(M.sharpe(a) - M.sharpe(b), rel=1e-12)
    assert d2["estimate"] == pytest.approx(-d1["estimate"], rel=1e-12)
    np.testing.assert_allclose(d2["draws"], -d1["draws"], atol=1e-12)
    assert d2["low"] == pytest.approx(-d1["high"], abs=1e-12)
    assert d2["high"] == pytest.approx(-d1["low"], abs=1e-12)
    assert d1["p_value"] == pytest.approx(d2["p_value"])
    # draws are paired: the vectorised Sharpe equals the scalar one on the shared indices
    idx = M.stationary_bootstrap_indices(400, 500, 400 ** (1 / 3), random_state=11)
    ref = [M.sharpe(a[i]) - M.sharpe(b[i]) for i in idx[:20]]
    np.testing.assert_allclose(d1["draws"][:20], ref, rtol=1e-10)
    same = M.sharpe_diff_ci(a, a.copy(), n_boot=200, random_state=0)
    assert same["estimate"] == 0.0 and same["low"] == 0.0 and same["high"] == 0.0


def test_sharpe_diff_ci_detects_difference():
    rng = np.random.default_rng(9)
    common = 0.01 * rng.standard_normal(2000)
    good = 0.002 + common + 0.002 * rng.standard_normal(2000)
    bad = common + 0.002 * rng.standard_normal(2000)
    d = M.sharpe_diff_ci(good, bad, n_boot=500, random_state=0)
    assert d["low"] > 0 and d["p_value"] < 0.01 and d["n"] == 2000


def test_sharpe_diff_ci_aligns_dates_and_drops_nan():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=300)
    s1 = pd.Series(0.01 * rng.standard_normal(300), index=dates)
    s2 = pd.Series(0.01 * rng.standard_normal(300), index=dates).iloc[20:]  # shorter
    s1.iloc[50] = np.nan
    s2.iloc[100] = np.nan
    d = M.sharpe_diff_ci(s1, s2, n_boot=100, random_state=0)
    j = pd.concat([s1, s2], axis=1, join="inner").dropna()
    assert d["n"] == len(j) == 278
    assert d["estimate"] == pytest.approx(M.sharpe(j.iloc[:, 0]) - M.sharpe(j.iloc[:, 1]))
    # shuffled order on the second series gives the same answer (alignment by date)
    d_shuf = M.sharpe_diff_ci(s1, s2.sample(frac=1.0, random_state=0), n_boot=100,
                              random_state=0)
    assert d_shuf["estimate"] == pytest.approx(d["estimate"], rel=1e-12)
    with pytest.raises(ValueError):
        M.sharpe_diff_ci(np.zeros(10), np.zeros(11))


# ------------------------------------------------------------------------ PSR and DSR


def test_psr_hand_computed_symmetric():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 0.01
    # mean 0.03, sd(ddof=1) = sqrt(2.5)*0.01 -> SR^2 = 3.6; skew 0; m2 = 2, m4 = 6.8 -> g4 = 1.7
    sr = math.sqrt(3.6)
    z = sr * math.sqrt(4) / math.sqrt(1 + (1.7 - 1) / 4 * 3.6)
    assert M.probabilistic_sharpe_ratio(x) == pytest.approx(PHI(z), rel=1e-10)
    z2 = (sr - 0.5) * 2 / math.sqrt(1.63)
    assert M.probabilistic_sharpe_ratio(pd.Series(x), 0.5) == pytest.approx(PHI(z2), rel=1e-10)


def test_psr_hand_computed_skewed():
    x = np.array([-1.0, 0.0, 0.0, 0.0, 3.0]) * 0.01
    # deviations (units of 0.01): -1.4, -0.4 x3, 2.6
    m2 = (1.96 + 3 * 0.16 + 6.76) / 5
    m3 = (-2.744 - 3 * 0.064 + 17.576) / 5
    m4 = (3.8416 + 3 * 0.0256 + 45.6976) / 5
    g3, g4 = m3 / m2**1.5, m4 / m2**2
    sr = 0.4 / math.sqrt(9.2 / 4)
    srb = 0.1
    z = (sr - srb) * math.sqrt(4) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    assert M.probabilistic_sharpe_ratio(x, srb) == pytest.approx(PHI(z), rel=1e-10)


def test_psr_edge_cases():
    assert math.isnan(M.probabilistic_sharpe_ratio([0.01, 0.02]))
    assert math.isnan(M.probabilistic_sharpe_ratio([0.01] * 20))
    r = np.random.default_rng(0).standard_normal(500) * 0.01 + 0.001
    p0 = M.probabilistic_sharpe_ratio(r, 0.0)
    assert 0 < M.probabilistic_sharpe_ratio(r, 0.2) < p0 < 1  # decreasing in benchmark


def test_expected_max_sharpe_and_dsr_hand_computed():
    V, N = 0.0004, 10
    g = 0.5772156649015329
    sr0 = math.sqrt(V) * ((1 - g) * PHI_INV(1 - 1 / N) + g * PHI_INV(1 - 1 / (N * math.e)))
    assert M.expected_max_sharpe(N, V) == pytest.approx(sr0, rel=1e-10)
    assert M.expected_max_sharpe(1, V) == 0.0
    r = np.random.default_rng(3).standard_normal(1000) * 0.01 + 0.0008
    assert M.deflated_sharpe_ratio(r, N, V) == pytest.approx(
        M.probabilistic_sharpe_ratio(r, sr0), rel=1e-12)
    assert M.deflated_sharpe_ratio(r, 1, V) == pytest.approx(M.probabilistic_sharpe_ratio(r))
    with pytest.raises(ValueError):
        M.deflated_sharpe_ratio(r, 0, V)
    with pytest.raises(ValueError):
        M.deflated_sharpe_ratio(r, 5, -1.0)


def test_dsr_monotone_in_trials():
    r = np.random.default_rng(4).standard_normal(750) * 0.01 + 0.001
    dsr = [M.deflated_sharpe_ratio(r, n, 0.0005) for n in (1, 2, 5, 10, 100, 1000, 10000)]
    assert all(np.diff(dsr) < 0)
    sr0 = [M.expected_max_sharpe(n, 1.0) for n in (2, 5, 10, 100, 1000)]
    assert all(np.diff(sr0) > 0)


def test_expected_max_sharpe_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for N in (20, 100):
        mc = rng.standard_normal((20000, N)).max(axis=1).mean()
        assert M.expected_max_sharpe(N, 1.0) == pytest.approx(mc, rel=0.03)


# ---------------------------------------------------------------------------- PBO/CSCV


def _brute_force_logits(R, S):
    """Independent loop implementation of CSCV (Bailey et al. 2017)."""
    T, N = R.shape
    size = T // S
    R = R[T - size * S:]
    blocks = [R[i * size:(i + 1) * size] for i in range(S)]
    out = []
    for comb in itertools.combinations(range(S), S // 2):
        J = np.vstack([blocks[i] for i in comb])
        Jb = np.vstack([blocks[i] for i in range(S) if i not in comb])
        is_sr = J.mean(0) / J.std(0, ddof=1)
        oos_sr = Jb.mean(0) / Jb.std(0, ddof=1)
        n_star = int(np.argmax(is_sr))
        rank = 1 + np.sum(oos_sr < oos_sr[n_star])  # no ties with continuous data
        w = rank / (N + 1)
        out.append(math.log(w / (1 - w)))
    return np.array(out)


def test_cscv_matches_brute_force_and_custom_metric():
    R = np.random.default_rng(0).standard_normal((63, 7)) * 0.01  # 63 = 6*10 + 3 dropped
    lam = M.cscv_logits(R, n_splits=6)
    assert lam.shape == (math.comb(6, 3),)
    np.testing.assert_allclose(lam, _brute_force_logits(R, 6), rtol=1e-12)
    lam_custom = M.cscv_logits(pd.DataFrame(R), n_splits=6,
                               metric=lambda A: A.mean(0) / A.std(0, ddof=1))
    np.testing.assert_allclose(lam_custom, lam, rtol=1e-12)
    assert M.pbo_cscv(R, 6) == pytest.approx(np.mean(lam <= 0))


def test_pbo_dominant_config_is_zero():
    rng = np.random.default_rng(1)
    R = 0.01 * rng.standard_normal((1000, 20))
    R[:, 7] += 0.01  # one configuration dominates in every sub-sample
    assert M.pbo_cscv(R, 16) == 0.0


def test_pbo_pure_noise_is_high():
    # iid noise: the IS winner's OOS rank is uniform -> PBO ~ 0.5 (no skill to select)
    vals = [M.pbo_cscv(0.01 * np.random.default_rng(s).standard_normal((800, 20)), 10)
            for s in range(10)]
    assert 0.4 < np.mean(vals) < 0.6
    # column-demeaned noise (zero full-sample mean per config): the IS winner is mechanically
    # the OOS loser -> PBO close to 1 (classic overfitting).
    R = 0.01 * np.random.default_rng(0).standard_normal((800, 20))
    R -= R.mean(axis=0)
    assert M.pbo_cscv(R, 10) > 0.9


def test_pbo_validation():
    R = np.zeros((100, 3)) + np.random.default_rng(0).standard_normal((100, 3))
    with pytest.raises(ValueError):
        M.pbo_cscv(R, 5)  # odd
    with pytest.raises(ValueError):
        M.pbo_cscv(R[:, :1], 4)  # one configuration
    with pytest.raises(ValueError):
        M.pbo_cscv(R[:10], 16)  # too short
    Rn = R.copy()
    Rn[3, 1] = np.nan  # NaN ignored by the default metric
    assert 0.0 <= M.pbo_cscv(Rn, 4) <= 1.0
