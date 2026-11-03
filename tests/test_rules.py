"""Tests for the number-of-factors rules (kfactors.rules).

Strategy: (i) planted truth on a self-contained factor-model generator, (ii) pure noise,
(iii) known-answer tests on panels with an exactly prescribed spectrum, (iv) cross-checks against
independent implementations (statsmodels PCA.ic for Bai-Ng, optht for Gavish-Donoho, direct
residual computations), (v) edge cases (p > n, tiny panels, kmax clipping, DataFrames).
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize

from kfactors.rules import RULES, KResult, select_k
from kfactors.rules.base import RuleWarning, effective_kmax
from kfactors.rules.econometric import ahn_horenstein_stats, bai_ng_criteria
from kfactors.rules.rmt import gd_lambda_star, gd_omega, gd_omega_approx, mp_cdf, mp_median

EXPECTED_RULES = {"variance_share", "kaiser", "mp", "mp_market", "gavish_donoho", "bai_ng",
                  "ahn_horenstein", "onatski", "parallel"}
SHAPES = [(200, 100), (60, 150)]
SEEDS = range(10)


# ------------------------------------------------------------------------------------------------
# generators (self-contained on purpose: no dependency on kfactors.sim)
# ------------------------------------------------------------------------------------------------
def planted(n, p, k=3, seed=0, equal_communality=False):
    """X = F L' + E with k strong pervasive factors (loading sd 1, 0.8, 0.6), iid N(0,1) noise."""
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((p, k)) * np.array([1.0, 0.8, 0.6, 0.5, 0.4][:k])
    if equal_communality:  # every series has the same factor variance 1.44
        L = L / np.linalg.norm(L, axis=1, keepdims=True) * 1.2
    F = rng.standard_normal((n, k))
    return F @ L.T + rng.standard_normal((n, p))


def noise(n, p, seed=0):
    return np.random.default_rng(10_000 + seed).standard_normal((n, p))


def ar1_noise(n, p, phi=0.8, seed=0):
    rng = np.random.default_rng(20_000 + seed)
    u = rng.standard_normal((n, p)) * np.sqrt(1 - phi**2)
    e = np.empty((n, p))
    e[0] = rng.standard_normal(p)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + u[t]
    return e


def with_spectrum(lam, n, p, seed=0):
    """Column-centred n x p matrix Z whose Z'Z/n has exactly the non-zero eigenvalues ``lam``."""
    lam = np.asarray(lam, dtype=float)
    m = lam.size
    assert m <= min(n - 1, p)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, m))
    A -= A.mean(axis=0)                       # orthogonal to the constant -> centred columns
    Q1, _ = np.linalg.qr(A)
    Q2, _ = np.linalg.qr(rng.standard_normal((p, m)))
    return np.sqrt(n) * (Q1 * np.sqrt(lam)) @ Q2.T


def recovery(rule, shape, k=3, standardize=True, gen=planted, **kw):
    n, p = shape
    return [select_k(gen(n, p, k, s), rule, kmax=10, standardize=standardize, random_state=s,
                     **kw).k for s in SEEDS]


# ------------------------------------------------------------------------------------------------
# registry / select_k plumbing
# ------------------------------------------------------------------------------------------------
def test_registry_exact():
    assert set(RULES) == EXPECTED_RULES


def test_unknown_method_lists_rules():
    with pytest.raises(ValueError) as exc:
        select_k(noise(30, 10), "not_a_rule")
    for name in EXPECTED_RULES:
        assert name in str(exc.value)


@pytest.mark.parametrize("n,p,kmax,expected", [
    (200, 100, 20, 20), (200, 100, 200, 98), (60, 150, 100, 57), (10, 50, 20, 7),
    (50, 10, 20, 8), (3, 2, 5, 1), (5, 3, 1, 1), (100, 100, 5, 5),
])
def test_effective_kmax(n, p, kmax, expected):
    assert effective_kmax(n, p, kmax) == expected
    assert effective_kmax(n, p, kmax) == max(1, min(kmax, min(n - 1, p) - 2))


@pytest.mark.parametrize("bad,exc", [(0, ValueError), (-1, ValueError), (2.5, TypeError),
                                     (True, TypeError), ("3", TypeError)])
def test_effective_kmax_rejects_bad(bad, exc):
    with pytest.raises(exc):
        effective_kmax(50, 20, bad)
    with pytest.raises(exc):
        select_k(noise(50, 20), "bai_ng", kmax=bad)


@pytest.mark.parametrize("rule", sorted(EXPECTED_RULES))
def test_kresult_contract(rule):
    X = planted(80, 40, 3, seed=1)
    res = select_k(X, rule, kmax=8, random_state=0)
    assert isinstance(res, KResult)
    assert res.method == rule
    assert 0 <= res.k <= 8
    assert res.info["kmax_effective"] == 8
    assert res.info["n"] == 80 and res.info["p"] == 40 and res.info["kmax_requested"] == 8
    ev = res.eigenvalues
    assert ev.shape == (40,) and np.all(np.diff(ev) <= 0)
    np.testing.assert_allclose(ev.sum(), 40.0)          # trace of a correlation matrix
    if rule != "onatski":
        assert res.criterion is not None and len(res.criterion) == 9   # k = 0..kmax or ranks
    else:
        assert len(res.criterion) == res.info["rmax"]


@pytest.mark.parametrize("rule", sorted(EXPECTED_RULES))
def test_kmax_clipping_and_cap(rule):
    # n = 12, p = 60: only 11 non-trivial eigenvalues -> kmax clipped to 9
    X = planted(12, 60, 3, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuleWarning)
        res = select_k(X, rule, kmax=50, random_state=0)
    assert res.info["kmax_effective"] == 9
    assert 0 <= res.k <= 9
    # with kmax = 1 every rule is capped at 1 (planted k = 3 is strong)
    res1 = select_k(planted(200, 100, 3, seed=2), rule, kmax=1, random_state=0)
    assert res1.k <= 1 and res1.info["kmax_effective"] == 1


def test_threshold_rules_report_uncapped_count():
    X = planted(200, 100, 3, seed=3)
    res = select_k(X, "variance_share", kmax=5)
    assert res.k == 5 and res.info["capped"] and res.info["k_uncapped"] > 5


@pytest.mark.parametrize("rule", sorted(EXPECTED_RULES))
def test_dataframe_equals_array(rule):
    X = planted(70, 30, 3, seed=4)
    df = pd.DataFrame(X, index=pd.date_range("2020-01-01", periods=70),
                      columns=[f"a{i}" for i in range(30)])
    r1 = select_k(X, rule, kmax=6, random_state=7)
    r2 = select_k(df, rule, kmax=6, random_state=7)
    assert r1.k == r2.k
    np.testing.assert_allclose(r1.eigenvalues, r2.eigenvalues)


def test_random_state_forwarded_only_to_stochastic_rules():
    X = planted(60, 30, 3, seed=5)
    # deterministic rules do not accept random_state; select_k must not forward it
    for rule in EXPECTED_RULES - {"parallel"}:
        select_k(X, rule, kmax=5, random_state=123)
    with pytest.raises(TypeError):
        RULES["bai_ng"](X - X.mean(0), kmax=5, random_state=1)


@pytest.mark.parametrize("rule,kw", [("bai_ng", {"criterion": "IC_p9"}),
                                     ("ahn_horenstein", {"stat": "XR"}),
                                     ("mp", {"noise": "max"}),
                                     ("parallel", {"null": "wild"}),
                                     ("parallel", {"quantile": 1.0}),
                                     ("parallel", {"block_size": 0}),
                                     ("variance_share", {"share": 0.0})])
def test_invalid_rule_kwargs(rule, kw):
    with pytest.raises(ValueError):
        select_k(noise(40, 10), rule, kmax=4, **kw)


def test_select_k_input_validation():
    X = noise(30, 10)
    X[3, 3] = np.inf
    with pytest.raises(ValueError):
        select_k(X, "bai_ng")
    with pytest.raises(ValueError):
        select_k(np.ones((30, 4)), "bai_ng")                  # zero variance (standardize)
    with pytest.raises(ValueError, match="zero total variance"):
        select_k(np.ones((30, 4)), "bai_ng", standardize=False)


# ------------------------------------------------------------------------------------------------
# planted truth
# ------------------------------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:parallel\\(null='circular'\\)")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rule,kw", [
    ("mp_market", {}),
    ("mp", {"noise": "mean"}),
    ("bai_ng", {"criterion": "IC_p1"}), ("bai_ng", {"criterion": "IC_p2"}),
    ("bai_ng", {"criterion": "IC_p3"}), ("bai_ng", {"criterion": "PC_p1"}),
    ("bai_ng", {"criterion": "PC_p2"}),
    ("ahn_horenstein", {"stat": "ER"}), ("ahn_horenstein", {"stat": "GR"}),
    ("onatski", {}),
    ("parallel", {"null": "iid"}), ("parallel", {"null": "circular"}),
    ("parallel", {"null": "block"}),
])
def test_planted_three_factors(shape, rule, kw):
    ks = recovery(rule, shape, **kw)
    assert sum(k == 3 for k in ks) >= 9, ks


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rule,kw", [("mp", {}), ("mp", {"noise": "median"}),
                                     ("gavish_donoho", {})])
def test_planted_rmt_rules_on_covariance(shape, rule, kw):
    """MP / GD assume homoscedastic noise; on the covariance of the planted model they recover 3."""
    ks = recovery(rule, shape, standardize=False, **kw)
    assert sum(k == 3 for k in ks) >= 9, ks


@pytest.mark.parametrize("shape", SHAPES)
def test_mp_median_overshoots_on_heteroscedastic_correlation(shape):
    """Documented behaviour: standardising a panel with heterogeneous communalities makes the
    idiosyncratic variance 1/(1 + |l_i|^2) vary across series; the bulk is then wider than the
    Marchenko-Pastur law and the (homoscedastic) median-matched MP edge over-counts. It never
    under-counts the strong factors. With equal communalities (homoscedastic after
    standardising) it is exact. The default noise="mean" is exact on both."""
    ks = recovery("mp", shape, noise="median")
    assert all(k >= 3 for k in ks) and np.mean(ks) > 3.5, ks
    ks_eq = recovery("mp", shape, gen=lambda n, p, k, s: planted(n, p, k, s, True),
                     noise="median")
    assert sum(k == 3 for k in ks_eq) >= 9, ks_eq
    assert sum(k == 3 for k in recovery("mp", shape)) >= 9


def dispersed_market(n, p, seed, idio_sd=0.5):
    """One market factor with betas U(0.5, 1.5) and lognormal(0, idio_sd) idiosyncratic sds."""
    rng = np.random.default_rng(seed)
    return (np.outer(rng.standard_normal(n), rng.uniform(0.5, 1.5, p))
            + rng.standard_normal((n, p)) * np.exp(rng.normal(0.0, idio_sd, p)))


def test_mp_default_is_mean_noise_and_exact_on_return_like_panel():
    """Regression (review R1): the median-matched MP edge over-counted badly (15-20 instead of 1)
    on a return-like correlation matrix; the default is now noise="mean" (classic correlation
    edge (1 + sqrt(gamma))^2), which finds the single market factor."""
    for s in range(5):
        X = dispersed_market(250, 300, s)
        res = select_k(X, "mp")
        assert res.info["noise"] == "mean"
        assert res.k == 1 == select_k(X, "mp", noise="mean").k
        np.testing.assert_allclose(res.threshold, (1 + np.sqrt(300 / 249)) ** 2)
        assert select_k(X, "mp", noise="median").k >= 10      # documented opt-in failure mode


def test_mp_market_overcounts_with_dispersed_residual_variance():
    """Documented behaviour (review R2): Laloux's edge assumes one residual variance for all
    series; with dispersed betas / idiosyncratic sds it over-counts where mp(noise="mean") and
    the econometric rules find the single factor."""
    for s in range(3):
        X = dispersed_market(250, 300, s)
        assert select_k(X, "mp_market").k >= 3
        assert select_k(X, "mp_market").info["market_removed"]
        for rule in ("mp", "onatski", "bai_ng", "ahn_horenstein"):
            assert select_k(X, rule).k == 1, rule


@pytest.mark.parametrize("shape", SHAPES)
def test_gavish_donoho_on_correlation(shape):
    """Documented behaviour: on the correlation matrix (heteroscedastic noise, see above) GD keeps
    the 3 factors and occasionally one extra bulk component (measured 17/20 exact at 200 x 100)."""
    ks = recovery("gavish_donoho", shape)
    assert all(k in (3, 4) for k in ks) and sum(k == 3 for k in ks) >= 7, ks


@pytest.mark.parametrize("shape", SHAPES)
def test_pc_p3_never_underestimates(shape):
    """Documented behaviour: PC_p3 has the weakest penalty (Bai & Ng 2002, Table I shows it
    over-estimates in small samples); it keeps the 3 factors and may add one."""
    ks = recovery("bai_ng", shape, criterion="PC_p3")
    assert all(k in (3, 4) for k in ks), ks


@pytest.mark.parametrize("shape", SHAPES)
def test_textbook_baselines_overestimate(shape):
    """Variance share 90% / Kaiser are not consistent: on the planted model they are capped."""
    for rule in ("variance_share", "kaiser"):
        ks = recovery(rule, shape)
        assert all(k > 3 for k in ks), (rule, ks)


# ------------------------------------------------------------------------------------------------
# pure noise
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rule", ["mp", "gavish_donoho", "parallel", "onatski", "bai_ng",
                                  "ahn_horenstein"])
def test_pure_noise_gives_zero(shape, rule):
    n, p = shape
    ks = [select_k(noise(n, p, s), rule, kmax=10, random_state=s).k for s in SEEDS]
    assert sum(k == 0 for k in ks) >= 9, ks


@pytest.mark.parametrize("shape", SHAPES)
def test_mp_market_on_pure_noise(shape):
    """Documented behaviour: removing lambda_1 from the trace lowers the noise variance by
    ~(1 + sqrt(gamma))^2 / p, so under pure noise the top eigenvalue (which sits at the edge)
    may be counted: mp_market returns at most 1 and never fewer than mp."""
    n, p = shape
    for s in SEEDS:
        X = noise(n, p, s)
        for market in ("auto", "always"):
            k_mkt = select_k(X, "mp_market", kmax=10, market=market).k
            assert k_mkt <= 1
            assert k_mkt >= select_k(X, "mp", kmax=10, noise="mean").k


@pytest.mark.parametrize("n,p", [(1000, 8), (600, 10), (1000, 20)])
def test_mp_market_small_p_noise_regression(n, p):
    """Regression (review R7): always removing lambda_1 biases sigma2 down by ~(1+sqrt(gamma))^2/p,
    which matters for small p (mean k 3.25 at 1000 x 8). market="auto" removes the market mode
    only if lambda_1 is above the plain MP edge, i.e. only if there is one."""
    ks = [select_k(noise(n, p, s), "mp_market").k for s in SEEDS]
    assert max(ks) <= 1 and sum(k == 0 for k in ks) >= 9, ks
    ks_always = [select_k(noise(n, p, s), "mp_market", market="always").k for s in SEEDS]
    assert np.mean(ks_always) > np.mean(ks), ks_always      # the old behaviour, kept as opt-in
    with pytest.raises(ValueError):
        select_k(noise(n, p, 0), "mp_market", market="never")


# ------------------------------------------------------------------------------------------------
# p > n
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rule", sorted(EXPECTED_RULES - {"variance_share", "kaiser", "mp"}))
def test_p_much_larger_than_n(rule):
    n, p = 40, 400
    ks = [select_k(planted(n, p, 3, s), rule, kmax=10, random_state=s).k for s in range(5)]
    assert sum(k == 3 for k in ks) >= 4, ks
    res = select_k(planted(n, p, 3, 0), rule, kmax=10, random_state=0)
    assert res.eigenvalues.shape == (p,)
    np.testing.assert_allclose(res.eigenvalues[n - 1:], 0.0, atol=1e-9)


# ------------------------------------------------------------------------------------------------
# known-answer tests on prescribed spectra (rules called directly on Z)
# ------------------------------------------------------------------------------------------------
def mp_quantiles(beta, size, scale=1.0):
    """Exact MP(beta) quantiles at (i - 0.5) / size."""
    a, b = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2
    qs = [optimize.brentq(lambda x, u=u: mp_cdf(x, beta) - u, a, b)
          for u in (np.arange(size) + 0.5) / size]
    return scale * np.sort(qs)[::-1]


def test_kaiser_and_variance_share_known_answer():
    lam = np.array([10.0, 5.0, 2.0, 1.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5])  # sum 22, mean 2.2
    Z = with_spectrum(lam, 50, 10)
    r = RULES["kaiser"](Z, kmax=8)
    assert r.k == 2 and np.isclose(r.threshold, 2.2)
    cum = np.cumsum(lam) / lam.sum()                    # 0.4545, 0.6818, 0.7727, 0.8409, ...
    for share, k in [(0.4, 1), (0.6818, 2), (0.69, 3), (0.8, 4), (0.85, 5)]:
        assert RULES["variance_share"](Z, kmax=8, share=share).k == k, (share, cum)
    np.testing.assert_allclose(RULES["variance_share"](Z, kmax=8).criterion[1:], cum[:8])


@pytest.mark.parametrize("n,p", [(401, 200), (201, 400)])
def test_mp_recovers_noise_scale_and_spikes(n, p):
    """Bulk = exact MP quantiles with scale s, plus 3 spikes: median matching (iterated) must
    recover s and k = 3; the edge is s (1 + sqrt(beta))^2."""
    m = min(n - 1, p)
    gamma = p / (n - 1)
    beta = min(gamma, 1 / gamma)
    s = 0.7
    lam = np.concatenate([[60.0, 40.0, 20.0], mp_quantiles(beta, m - 3, s)])
    Z = with_spectrum(np.sort(lam)[::-1], n, p)
    r = RULES["mp"](Z, kmax=10, noise="median")
    assert r.k == 3
    assert r.info["converged"] and r.info["iterations"] >= 2
    np.testing.assert_allclose(r.threshold, s * (1 + np.sqrt(beta)) ** 2, rtol=0.01)
    np.testing.assert_allclose(r.info["sigma2"], s / ((n - 1) / n * max(1.0, gamma)), rtol=0.01)


def test_mp_market_known_answer():
    n, p = 101, 50
    lam = np.concatenate([[30.0, 3.0], np.full(48, 0.5)])
    Z = with_spectrum(lam, n, p)
    r = RULES["mp_market"](Z, kmax=10)
    sigma2 = (lam.sum() - 30.0) / p
    edge = sigma2 * (1 + np.sqrt(p / (n - 1))) ** 2
    np.testing.assert_allclose(r.threshold, edge)
    assert r.k == int(np.sum(lam > edge)) == 2 and r.info["market_removed"]
    # no market mode (lambda_1 below the plain MP edge): auto keeps the plain edge
    lam0 = np.concatenate([[1.4], np.full(49, 0.5)])
    Z0 = with_spectrum(lam0, n, p)
    plain = lam0.sum() / p * (1 + np.sqrt(p / (n - 1))) ** 2
    r0 = RULES["mp_market"](Z0, kmax=10)
    assert not r0.info["market_removed"] and np.isclose(r0.threshold, plain) and r0.k == 0
    r1 = RULES["mp_market"](Z0, kmax=10, market="always")
    assert r1.info["market_removed"] and r1.threshold < plain


def test_onatski_known_answer():
    """Bulk lambda_j = 2 - 0.05 (j-1)^{2/3} exactly: the ED regression slope is -0.05, so
    delta = 0.1 and every bulk gap (< 0.05) is below delta; spikes give k = 4."""
    bulk_ranks = np.arange(5, 60)
    bulk = 2.0 - 0.05 * (bulk_ranks - 1.0) ** (2.0 / 3.0)
    lam = np.concatenate([[9.0, 7.0, 5.0, 3.0], bulk])
    Z = with_spectrum(lam, 100, 60)
    r = RULES["onatski"](Z, kmax=10)
    assert r.k == 4
    np.testing.assert_allclose(r.threshold, 0.1, rtol=1e-6)
    assert r.info["converged"] and r.info["iterations"] == len(r.info["path"]) >= 2
    np.testing.assert_allclose(r.criterion, -np.diff(lam)[:10], rtol=1e-8)


def test_onatski_uses_largest_qualifying_gap():
    """k = max{i <= rmax : gap_i >= delta}: a small gap above a large one still counts."""
    bulk = 2.0 - 0.05 * (np.arange(4, 60) - 1.0) ** (2.0 / 3.0)
    lam = np.concatenate([[9.0, 8.95, 4.0], bulk])      # gap_1 = 0.05 < delta, gap_2 = 4.95
    Z = with_spectrum(lam, 100, 60)
    assert RULES["onatski"](Z, kmax=10).k == 3


def test_bai_ng_criteria_match_direct_residuals():
    """V(k) from eigenvalues == mean squared residual of a rank-k truncated SVD."""
    X = planted(50, 30, 3, seed=8)
    Z = X - X.mean(0)
    n, p = Z.shape
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    ev = np.concatenate([s**2 / n, np.zeros(p - s.size)])
    crit = bai_ng_criteria(ev, n, p, kmax=6)
    for k in range(7):
        R = Z - (U[:, :k] * s[:k]) @ Vt[:k]
        np.testing.assert_allclose(crit["V"][k], (R**2).sum() / (n * p), rtol=1e-10)
    c = (n + p) / (n * p)
    ks = np.arange(7)
    np.testing.assert_allclose(crit["IC_p1"], np.log(crit["V"]) + ks * c * np.log(1 / c))
    np.testing.assert_allclose(crit["IC_p2"], np.log(crit["V"]) + ks * c * np.log(min(n, p)))
    np.testing.assert_allclose(crit["IC_p3"],
                               np.log(crit["V"]) + ks * np.log(min(n, p)) / min(n, p))
    np.testing.assert_allclose(crit["PC_p2"],
                               crit["V"] + ks * crit["V"][6] * c * np.log(min(n, p)))


@pytest.mark.parametrize("n,p", [(120, 20), (120, 12), (23, 50), (20, 200)])
def test_bai_ng_short_tail_does_not_select_kmax(n, p):
    """Regression (review R3): with kmax clipped near m = min(n-1, p), V(kmax) is the sum of the
    last eigenvalues and ln V(k) falls faster than the penalty, so IC_p2 returned kmax on pure
    noise. bai_ng now searches k <= m // 2."""
    m = min(n - 1, p)
    ks = [select_k(noise(n, p, s), "bai_ng").k for s in SEEDS]
    assert sum(k == 0 for k in ks) >= 9, ks
    res = select_k(noise(n, p, 0), "bai_ng")
    assert res.info["kmax_bn"] == min(res.info["kmax_effective"], m // 2)
    assert res.criterion.shape == (res.info["kmax_bn"] + 1,)
    rng = np.random.default_rng(1)
    X = np.outer(rng.standard_normal(n), rng.uniform(0.5, 1.5, p)) + noise(n, p, 1)
    assert select_k(X, "bai_ng").k == 1


def test_bai_ng_kmax_bn_does_not_bind_on_large_panels():
    res = select_k(planted(200, 100, 3, seed=0), "bai_ng", kmax=20)
    assert res.info["kmax_bn"] == res.info["kmax_effective"] == 20


def test_ahn_horenstein_stats_from_definitions():
    X = planted(80, 50, 2, seed=9)
    Z = X - X.mean(0)
    n, p = Z.shape
    mu = np.linalg.svd(Z, compute_uv=False) ** 2 / n
    kmax = 6
    er, gr, mock = ahn_horenstein_stats(np.concatenate([mu, np.zeros(p - mu.size)]), n, p, kmax)

    def V(k):  # sum_{j > k} mu_j  (k = -1 uses the mock eigenvalue)
        return mu[k:].sum() if k >= 0 else mu.sum() + mu.sum() / np.log(min(n, p))

    mu0 = mu.sum() / np.log(min(n, p))
    mus = np.concatenate([[mu0], mu])
    for k in range(kmax + 1):
        assert np.isclose(er[k], mus[k] / mus[k + 1])
        assert np.isclose(gr[k], np.log(V(k - 1) / V(k)) / np.log(V(k) / V(k + 1)))
    assert np.isclose(mock[0], mu0)
    assert select_k(X, "ahn_horenstein", kmax=kmax).k == 2


# ------------------------------------------------------------------------------------------------
# Marchenko-Pastur / Gavish-Donoho constants
# ------------------------------------------------------------------------------------------------
def test_mp_cdf_and_median():
    for beta in (0.05, 0.3, 0.7, 1.0):
        a, b = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2
        assert mp_cdf(a, beta) == 0.0 and mp_cdf(b, beta) == 1.0
        np.testing.assert_allclose(mp_cdf(b - 1e-12, beta), 1.0, atol=1e-6)
        assert abs(mp_cdf(mp_median(beta), beta) - 0.5) < 1e-10
        # density integrates to 1 (independent naive integration of the density)
        def dens(x, a=a, b=b, beta=beta):
            return np.sqrt(max((b - x) * (x - a), 0)) / (2 * np.pi * beta * x)

        tot = integrate.quad(dens, a, b, limit=500)[0]
        np.testing.assert_allclose(tot, 1.0, atol=1e-4)
    # Monte-Carlo check of the median (Wishart, unit variance)
    rng = np.random.default_rng(0)
    G = rng.standard_normal((3000, 1500))
    ev = np.linalg.eigvalsh(G.T @ G / 3000)
    np.testing.assert_allclose(np.median(ev), mp_median(0.5), rtol=5e-3)


def test_gd_constants():
    np.testing.assert_allclose(gd_lambda_star(1.0), 4 / np.sqrt(3))
    np.testing.assert_allclose(gd_omega(1.0), 2.858, atol=1e-3)     # GD 2014, Sec. I
    for beta in np.linspace(0.05, 1.0, 12):
        assert abs(gd_omega(beta) - gd_omega_approx(beta)) < 0.02     # Eq. (5) accuracy
    from optht.optht import _median_marcenko_pastur
    for beta in (0.1, 0.39, 0.8, 1.0):
        assert abs(mp_median(beta) - _median_marcenko_pastur(beta)) < 1e-3


# ------------------------------------------------------------------------------------------------
# cross-checks against independent implementations
# ------------------------------------------------------------------------------------------------
def _cross_panels():
    cfg = [(120, 40, 3), (60, 150, 3), (80, 80, 2), (50, 200, 4), (200, 60, 1), (90, 30, 0),
           (40, 100, 2), (150, 150, 5), (70, 20, 3), (30, 90, 1), (100, 300, 3), (250, 50, 2)]
    for i, (n, p, k) in enumerate(cfg):
        X = planted(n, p, k, seed=100 + i) if k else noise(n, p, 100 + i)
        if i % 3 == 1:   # heavy tails / heteroscedasticity for variety
            X = X * np.random.default_rng(i).uniform(0.5, 2.0, size=p)
        yield n, p, X


@pytest.mark.parametrize("standardize", [True, False])
def test_bai_ng_matches_statsmodels(standardize):
    from statsmodels.multivariate.pca import PCA as SMPCA
    count = 0
    for n, p, X in _cross_panels():
        kmax = 8
        res = {c: select_k(X, "bai_ng", kmax=kmax, standardize=standardize, criterion=c)
               for c in ("IC_p1", "IC_p2", "IC_p3")}
        Z = X - X.mean(0)
        if standardize:
            Z = Z / Z.std(0)
        for normalize in (True, False):
            sm = SMPCA(Z, ncomp=kmax, standardize=False, demean=False, normalize=normalize)
            ic = np.asarray(sm.ic)
            assert ic.shape == (kmax + 1, 3)
            for j, c in enumerate(("IC_p1", "IC_p2", "IC_p3")):
                assert int(np.argmin(ic[:, j])) == res[c].k
                # statsmodels uses ln SSR(k) = ln V(k) + ln(n p): identical up to a constant
                np.testing.assert_allclose(ic[:, j] - res[c].criterion, np.log(n * p),
                                           rtol=1e-8, atol=1e-8)
        count += 1
    assert count >= 10


def test_gavish_donoho_matches_optht():
    from optht import optht
    count = 0
    for n, p, X in _cross_panels():
        for standardize in (True, False):
            res = select_k(X, "gavish_donoho", kmax=200, standardize=standardize)
            Z = X - X.mean(0)
            if standardize:
                Z = Z / Z.std(0)
            m = min(n - 1, p)
            sv = np.linalg.svd(Z, compute_uv=False)[:m]
            beta = m / max(n - 1, p)
            try:
                k_ref = int(optht(beta, sv=sv, sigma=None))
            except ValueError:   # optht raises when no singular value survives
                k_ref = 0
            assert res.info["k_uncapped"] == k_ref, (n, p, standardize)
            count += 1
    assert count >= 20


# ------------------------------------------------------------------------------------------------
# parallel analysis
# ------------------------------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:parallel\\(null='circular'\\)")
@pytest.mark.parametrize("null", ["iid", "circular", "block"])
def test_parallel_deterministic(null):
    X = planted(60, 150, 3, seed=11)
    r1 = select_k(X, "parallel", kmax=6, null=null, n_perm=30, random_state=42)
    r2 = select_k(X, "parallel", kmax=6, null=null, n_perm=30, random_state=42)
    r3 = select_k(X, "parallel", kmax=6, null=null, n_perm=30,
                  random_state=np.random.default_rng(42))
    r4 = select_k(X, "parallel", kmax=6, null=null, n_perm=30, random_state=43)
    assert r1.k == r2.k == r3.k
    np.testing.assert_array_equal(r1.threshold, r2.threshold)
    np.testing.assert_array_equal(r1.threshold, r3.threshold)
    assert not np.array_equal(r1.threshold, r4.threshold)
    assert r1.info["null_eigenvalues"].shape == (30, 7)
    assert r1.threshold.shape == (7,) and np.all(np.diff(r1.threshold) <= 0)


def test_parallel_null_preserves_marginals():
    """Every null scheme keeps column means and variances (the null only breaks dependence)."""
    from kfactors.rules.permutation import _null_panel
    Z = ar1_noise(100, 8, seed=1)
    Z = (Z - Z.mean(0)) / Z.std(0)
    rng = np.random.default_rng(0)
    for null in ("iid", "circular", "block"):
        Zb = _null_panel(Z, null, 5, rng, Z.mean(0), Z.std(0))
        np.testing.assert_allclose(Zb.mean(0), 0.0, atol=1e-12)
        np.testing.assert_allclose(Zb.std(0), 1.0)
        if null != "block":   # permutations / rotations keep the exact multiset of values
            np.testing.assert_allclose(np.sort(Zb, 0), np.sort(Z, 0))


def test_parallel_serial_nulls_inflate_quantiles_on_ar1():
    """On AR(1) noise (phi = 0.8) the circular / block nulls keep the autocorrelation, which
    widens the eigenvalue spread, so their null quantiles are larger than the iid null's and
    they select fewer spurious factors."""
    X = ar1_noise(120, 60, phi=0.8, seed=3)
    q = {null: select_k(X, "parallel", kmax=5, null=null, n_perm=60, random_state=0,
                        block_size=10 if null == "block" else None)
         for null in ("iid", "circular", "block")}
    assert q["circular"].threshold[0] > 1.2 * q["iid"].threshold[0]
    assert q["block"].threshold[0] > 1.1 * q["iid"].threshold[0]
    assert np.all(q["circular"].threshold > q["iid"].threshold)
    assert q["circular"].k <= q["iid"].k and q["block"].k <= q["iid"].k
    assert q["iid"].k >= 1          # iid permutation is fooled by serial dependence
    assert q["circular"].k == 0


def test_parallel_block_size_default():
    X = planted(100, 30, 2, seed=1)
    r = select_k(X, "parallel", kmax=4, null="block", n_perm=10, random_state=0)
    assert r.info["block_size"] == int(np.ceil(np.sqrt(100)))


def test_parallel_block_default_on_ar_noise_p_equals_n():
    """Regression (review R6): the old default block length ceil(n^(1/3)) = 7 cut the AR(0.5)
    autocorrelation too often and the block null returned ~kmax on 250 x 250 noise. The default
    ceil(sqrt(n)) = 16 is much better; the circular null is the recommended one for p >= n."""
    def ar(seed, n=250, p=250, phi=0.5):
        rng = np.random.default_rng(seed)
        e = rng.standard_normal((n, p))
        for t in range(1, n):
            e[t] += phi * e[t - 1]
        return e
    new, old, circ = [], [], []
    for s in range(4):
        X = ar(s)
        kw = dict(kmax=10, n_perm=30, random_state=s)
        new.append(select_k(X, "parallel", null="block", **kw).k)
        old.append(select_k(X, "parallel", null="block", block_size=7, **kw).k)
        circ.append(select_k(X, "parallel", null="circular", **kw).k)
    assert np.mean(old) >= 7 and np.mean(new) <= 3 and max(circ) <= 3, (old, new, circ)


def common_vol_panel(n, p, seed, k=0, log_vol_sd=0.5):
    """Independent series (plus an optional market factor) times a shared AR(0.98) volatility."""
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    u = rng.standard_normal(n) * log_vol_sd * np.sqrt(1 - 0.98**2)
    for t in range(1, n):
        h[t] = 0.98 * h[t - 1] + u[t]
    X = rng.standard_normal((n, p))
    if k:
        X = X + np.outer(rng.standard_normal(n), rng.uniform(0.5, 1.5, p))
    return X * np.exp(h)[:, None]


@pytest.mark.parametrize("null", ["iid", "circular", "block"])
def test_parallel_devolatilize_handles_common_volatility(null):
    """Regression (review R4): every null resamples columns independently and desynchronises a
    volatility path shared by all series, so parallel analysis hit kmax on factor-free panels.
    devolatilize=True (row RMS normalisation) restores k = 0 and keeps the market factor."""
    for s in range(3):
        X0 = common_vol_panel(300, 100, s)
        kw = dict(kmax=10, null=null, n_perm=30, random_state=s)
        assert select_k(X0, "parallel", **kw).k == 10           # documented failure (default)
        r = select_k(X0, "parallel", devolatilize=True, **kw)
        assert r.k == 0 and r.info["devolatilize"]
        assert r.info["tested_eigenvalues"].shape == (11,)
        np.testing.assert_allclose(r.eigenvalues, select_k(X0, "kaiser").eigenvalues)
        assert select_k(common_vol_panel(300, 100, s, k=1), "parallel", devolatilize=True,
                        **kw).k == 1


def test_parallel_devolatilize_leaves_iid_panels_alone():
    for s in range(3):
        X = planted(200, 100, 3, seed=s)
        assert select_k(X, "parallel", kmax=8, n_perm=30, random_state=s,
                        devolatilize=True).k == 3


def test_circular_null_transforms_distinct_and_balanced():
    """Regression (review R5): shifts were drawn with replacement, so with p > n many columns
    shared a shift and kept their factor co-movement. Shifts (and time reversals) are now a
    balanced assignment: all distinct for p <= 2n, at most ceil(p / 2n) columns per transform."""
    from kfactors.rules.permutation import _circular_index
    rng = np.random.default_rng(0)
    for n, p in [(50, 30), (50, 100), (20, 130)]:
        idx = _circular_index(n, p, rng)
        assert idx.shape == (n, p)
        assert all(np.array_equal(np.sort(idx[:, j]), np.arange(n)) for j in range(p))
        _, counts = np.unique(idx, axis=1, return_counts=True)
        assert counts.max() == int(np.ceil(p / (2 * n)))
        assert counts.max() - counts.min() <= 1


def test_circular_null_threshold_close_to_iid_on_iid_rows():
    """On i.i.d. rows the circular and iid nulls should agree. The second null quantile was
    ~41% above the iid one at 250 x 500 with a market factor; with distinct transforms it is
    within ~25% (the rest is the periodogram spread of shifted copies of one factor)."""
    rng = np.random.default_rng(0)
    n, p = 250, 500
    X = (np.outer(rng.standard_normal(n), rng.uniform(0.5, 1.5, p))
         + rng.standard_normal((n, 3)) @ (0.3 * rng.standard_normal((p, 3))).T
         + rng.standard_normal((n, p)))
    kw = dict(kmax=6, n_perm=30, random_state=0)
    q_iid = select_k(X, "parallel", **kw).threshold[1]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuleWarning)             # p = 2n: no shared transforms
        r = select_k(X, "parallel", null="circular", **kw)
    assert r.info["max_shared"] == 1
    assert r.threshold[1] < 1.25 * q_iid, (r.threshold[1], q_iid)


def test_circular_null_warns_when_transforms_are_shared():
    with pytest.warns(RuleWarning, match="share each shift"):
        r = select_k(planted(20, 100, 2, seed=0), "parallel", null="circular", kmax=3,
                     n_perm=5, random_state=0)
    assert r.info["max_shared"] == 3


# ------------------------------------------------------------------------------------------------
# tiny / degenerate panels
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rule", sorted(EXPECTED_RULES))
@pytest.mark.parametrize("shape", [(3, 2), (4, 3), (6, 2), (5, 9), (8, 5)])
def test_tiny_panels_do_not_crash(rule, shape):
    X = np.random.default_rng(sum(shape)).standard_normal(shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuleWarning)
        res = select_k(X, rule, kmax=5, random_state=0)
    kmax_eff = max(1, min(5, min(shape[0] - 1, shape[1]) - 2))
    assert res.info["kmax_effective"] == kmax_eff
    assert 0 <= res.k <= kmax_eff


@pytest.mark.parametrize("rule", ["bai_ng", "ahn_horenstein"])
@pytest.mark.parametrize("shape", [(3, 2), (6, 2), (3, 5)])
def test_econometric_rules_degenerate_below_three_eigenvalues(rule, shape):
    """Regression (review R8): with m = min(n-1, p) = 2, kmax_effective is floored at 1 > m - 2
    and both rules returned 1 on noise. They now return 0 with a RuleWarning."""
    for s in range(5):
        with pytest.warns(RuleWarning, match=rule):
            res = select_k(np.random.default_rng(s).standard_normal(shape), rule)
        assert res.k == 0 and res.info["degenerate"] and res.info["kmax_effective"] == 1


def test_rule_warnings_point_at_the_caller():
    """Regression (review U8): warnings raised inside a rule used to point at rules/base.py."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        select_k(noise(6, 5), "onatski")
    assert [w.category for w in caught] == [RuleWarning]
    assert caught[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuleWarning)
        with pytest.raises(RuleWarning):
            select_k(noise(6, 5), "onatski")


def test_onatski_degenerate_warns():
    with pytest.warns(RuleWarning, match="onatski"):
        res = select_k(noise(30, 4), "onatski", kmax=3)
    assert res.k == 0 and res.info["degenerate"]


def test_onatski_rmax_clipped_for_regression_window():
    # n = 12, p = 60: m = 11 non-trivial eigenvalues, kmax_effective 9, rmax = m - 5 = 6
    res = select_k(planted(12, 60, 2, seed=0), "onatski", kmax=20)
    assert res.info["kmax_effective"] == 9 and res.info["rmax"] == 6


def test_exact_low_rank_panel():
    """An exactly rank-2 panel (no noise): IC criteria stay finite and pick 2."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 2)) @ rng.standard_normal((2, 20))
    res = select_k(X, "bai_ng", kmax=5, standardize=False)
    assert res.k == 2 and np.all(np.isfinite(res.criterion))
    assert select_k(X, "ahn_horenstein", kmax=5, standardize=False).k == 2


def test_non_convergence_is_reported():
    X = planted(200, 100, 3, seed=0)
    with pytest.warns(RuleWarning, match="onatski"):
        r = select_k(X, "onatski", kmax=10, max_iter=1)
    assert not r.info["converged"] and r.info["iterations"] == 1 and r.k == r.info["path"][-1]
    with pytest.warns(RuleWarning, match="mp"):
        r = select_k(X, "mp", kmax=10, standardize=True, noise="median", max_iter=1)
    assert not r.info["converged"]


def test_rule_level_input_validation():
    with pytest.raises(ValueError):
        RULES["kaiser"](np.ones(10), kmax=2)
    with pytest.raises(ValueError):
        RULES["kaiser"](np.ones((2, 5)), kmax=2)
    Z = noise(20, 5)
    Z[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        RULES["kaiser"](Z, kmax=2)
    with pytest.raises(ValueError):
        mp_cdf(1.0, 1.5)
    with pytest.raises(ValueError):
        select_k(noise(40, 10), "parallel", kmax=3, n_perm=0)
