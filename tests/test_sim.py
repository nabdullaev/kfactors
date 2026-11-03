"""Tests for kfactors.sim (simulate, design_grid)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kfactors.sim import (
    DEFAULT_STRENGTHS,
    SimData,
    SimSpec,
    design_grid,
    simulate,
)
from kfactors.utils import eigvals_desc, preprocess


def _acf1(A: np.ndarray) -> float:
    """Mean lag-1 autocorrelation over columns."""
    A = A - A.mean(axis=0)
    return float(np.mean((A[1:] * A[:-1]).sum(0) / (A * A).sum(0)))


# ------------------------------------------------------------------------------ shapes / API

@pytest.mark.parametrize("regime", ["pervasive", "spiked"])
@pytest.mark.parametrize("noise", ["gauss", "t", "ar1", "garch", "hetero"])
def test_shapes_and_identity(regime, noise):
    spec = SimSpec(n=60, p=150, k=3, regime=regime, noise=noise)
    d = simulate(spec, random_state=0)
    assert isinstance(d, SimData)
    assert d.X.shape == (60, 150) and d.E.shape == (60, 150)
    assert d.F.shape == (60, 3) and d.L.shape == (150, 3)
    assert d.k == 3 and d.spec is spec
    np.testing.assert_allclose(d.X, d.F @ d.L.T + d.E, rtol=0, atol=1e-12)
    assert d.Sigma.shape == (150, 150)
    np.testing.assert_allclose(d.Sigma, d.Sigma.T)
    np.testing.assert_allclose(d.Sigma, d.L @ d.L.T + np.diag(d.noise_var), atol=1e-12)
    assert np.all(np.isfinite(d.X))


def test_determinism_and_seed_sensitivity():
    spec = SimSpec(n=50, p=80, k=2, noise="garch", factor_ar=0.3)
    a, b = simulate(spec, 7), simulate(spec, 7)
    for f in ("X", "F", "L", "E", "Sigma"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
    c = simulate(spec, 8)
    assert not np.allclose(a.X, c.X)
    # a Generator is consumed, not copied
    rng = np.random.default_rng(3)
    d1, d2 = simulate(spec, rng), simulate(spec, rng)
    assert not np.allclose(d1.X, d2.X)
    np.testing.assert_array_equal(simulate(spec, np.random.default_rng(3)).X, d1.X)


def test_k_zero_is_pure_noise():
    d = simulate(SimSpec(n=40, p=30, k=0), 0)
    assert d.F.shape == (40, 0) and d.L.shape == (30, 0)
    np.testing.assert_array_equal(d.X, d.E)
    np.testing.assert_allclose(d.Sigma, np.eye(30))


def test_sigma_none_for_large_p():
    d = simulate(SimSpec(n=5, p=2001, k=1), 0)
    assert d.Sigma is None and d.X.shape == (5, 2001)
    assert d.noise_var.shape == (2001,)


def test_default_strengths_truncation_and_extension():
    d = simulate(SimSpec(n=20, p=50, k=7, regime="spiked"), 0)
    sv = np.linalg.svd(d.L, compute_uv=False)
    expected = np.array(list(DEFAULT_STRENGTHS) + [1.2, 1.2])
    np.testing.assert_allclose(np.sort(sv)[::-1], np.sort(expected)[::-1], rtol=1e-10)
    # user strengths are truncated to k
    d = simulate(SimSpec(n=20, p=50, k=2, regime="spiked", strengths=(4, 2, 1)), 0)
    np.testing.assert_allclose(np.linalg.svd(d.L, compute_uv=False), [4, 2], rtol=1e-10)


@pytest.mark.parametrize("bad", [
    {"k": 5, "strengths": (1, 2)},             # too few strengths
    {"k": 2, "strengths": (1, -2)},            # non-positive strength
    {"noise": "t", "df": 2.0},                 # infinite variance
    {"noise": "ar1", "phi": 1.0},
    {"noise": "garch", "garch": (0.1, 0.5, 0.5)},
    {"noise": "hetero", "hetero_range": (0.0, 1.0)},
    {"noise": "laplace"},
    {"regime": "weak"},
    {"factor_ar": -1.0},
    {"k": 0, "market": 1.0},
    {"k": 2, "market": -1.0},
    {"k": -1},
    {"k": 31},                              # k > p
    {"n": 1},
])
def test_invalid_specs_raise(bad):
    kw = {"n": 20, "p": 30, "k": 2}
    kw.update(bad)
    with pytest.raises(ValueError):
        simulate(SimSpec(**kw), 0)


# ----------------------------------------------------------------------- population vs sample

def test_spiked_population_spikes_exact():
    spec = SimSpec(n=10, p=120, k=4, regime="spiked")
    d = simulate(spec, 0)
    pop = np.linalg.eigvalsh(d.Sigma)[::-1]
    np.testing.assert_allclose(pop[:4], np.array([5, 3, 2, 1.5]) ** 2 + 1, rtol=1e-10)
    np.testing.assert_allclose(pop[4:], 1.0, atol=1e-10)
    np.testing.assert_allclose(d.L.T @ d.L, np.diag([25, 9, 4, 2.25]), atol=1e-10)


def test_pervasive_spikes_grow_with_p():
    """Bai-Ng loadings: L'L / p -> diag(s_k^2 / 25)."""
    d = simulate(SimSpec(n=10, p=4000, k=3, regime="pervasive"), 1)
    G = d.L.T @ d.L / 4000
    np.testing.assert_allclose(np.diag(G), np.array([25, 9, 4]) / 25, rtol=0.08)
    assert np.max(np.abs(G - np.diag(np.diag(G)))) < 0.05


@pytest.mark.parametrize("regime", ["pervasive", "spiked"])
def test_sample_top_eigenvalues_match_population_for_large_n(regime):
    """With n >> p the sample covariance converges: top-k sample eigenvalues ~ population."""
    spec = SimSpec(n=20000, p=40, k=3, regime=regime)
    d = simulate(spec, 2)
    Z, _, _ = preprocess(d.X, standardize=False)
    sample = eigvals_desc(Z)
    pop = np.linalg.eigvalsh(d.Sigma)[::-1]
    np.testing.assert_allclose(sample[:3], pop[:3], rtol=0.05)
    # bulk is at the noise level 1 (MP with gamma = 0.002 is very narrow)
    assert abs(np.median(sample[3:]) - 1.0) < 0.05
    # sample covariance matches Sigma entrywise
    S = Z.T @ Z / spec.n
    assert np.max(np.abs(S - d.Sigma)) < 0.12 * np.max(np.diag(d.Sigma))


@pytest.mark.parametrize("regime, market, expected", [
    ("pervasive", 1.0, 300.0),     # market * p
    ("spiked", 50.0, 50.0),        # market
])
def test_market_factor_eigenvalue_and_share(regime, market, expected):
    spec = SimSpec(n=10, p=300, k=4, regime=regime, market=market, strengths=(3, 2, 1.5))
    d = simulate(spec, 3)
    b = d.L[:, 0]
    assert np.all(b > 0) or np.mean(b > 0) > 0.99      # positive betas
    np.testing.assert_allclose(b @ b, expected, rtol=1e-10)
    # b is an exact eigenvector of L L' (and of Sigma, noise is homoscedastic)
    LLt_b = d.L @ (d.L.T @ b)
    np.testing.assert_allclose(LLt_b, expected * b, rtol=1e-8, atol=1e-8)
    pop = np.linalg.eigvalsh(d.Sigma)[::-1]
    np.testing.assert_allclose(pop[0], expected + 1.0, rtol=1e-8)
    # dominant: largest population eigenvalue, with a large share of total variance
    assert pop[0] > 2 * pop[1]
    if regime == "pervasive":
        assert pop[0] / pop.sum() > 0.25


def test_market_dominates_sample_spectrum():
    spec = SimSpec(n=500, p=200, k=4, market=1.0, strengths=(3, 2, 1.5))
    d = simulate(spec, 4)
    Z, _, _ = preprocess(d.X)
    ev = eigvals_desc(Z)
    assert ev[0] / ev.sum() > 0.3
    assert ev[0] > 2.5 * ev[1]


# ----------------------------------------------------------------------------- noise moments

def test_gauss_and_t_noise_moments():
    g = simulate(SimSpec(n=2000, p=200, k=0, noise="gauss"), 5).E
    assert abs(g.var() - 1) < 0.01
    assert abs(np.mean(g ** 4) / g.var() ** 2 - 3) < 0.05
    t5 = simulate(SimSpec(n=2000, p=200, k=0, noise="t", df=5), 5).E
    assert abs(t5.var() - 1) < 0.03                    # rescaled to unit variance
    kurt = np.mean(t5 ** 4) / t5.var() ** 2 - 3        # population excess kurtosis = 6
    assert kurt > 2
    t3 = simulate(SimSpec(n=2000, p=200, k=0, noise="t", df=3), 5).E
    assert np.mean(t3 ** 4) / t3.var() ** 2 - 3 > 2
    assert abs(np.median(np.abs(t3)) - np.median(np.abs(g))) > 0.05  # different shape


@pytest.mark.parametrize("phi", [0.5, 0.9, -0.3])
def test_ar1_noise_acf_and_unit_variance(phi):
    E = simulate(SimSpec(n=3000, p=100, k=0, noise="ar1", phi=phi), 6).E
    assert abs(_acf1(E) - phi) < 0.02
    # stationary start: unit variance already in the first rows (no transient)
    assert abs(E[:50].var() - 1) < 0.15
    assert abs(E.var() - 1) < 0.08


def test_garch_noise_volatility_clustering():
    """acf of squares at lag 1 of GARCH(1,1) (Bollerslev, 1988, J. Time Ser. Anal. 9(2)):
    rho_1 = alpha (1 - alpha beta - beta^2) / (1 - 2 alpha beta - beta^2) = 0.179 for
    (alpha, beta) = (0.10, 0.85)."""
    E = simulate(SimSpec(n=3000, p=200, k=0, noise="garch"), 7).E
    assert abs(E.var() - 1) < 0.03
    rho1 = _acf1(E ** 2)
    a, b = 0.10, 0.85
    theory = a * (1 - a * b - b * b) / (1 - 2 * a * b - b * b)
    assert rho1 > 0.05
    assert abs(rho1 - theory) < 0.04
    assert abs(_acf1(E)) < 0.02                        # returns themselves uncorrelated
    g = simulate(SimSpec(n=3000, p=200, k=0, noise="gauss"), 7).E
    assert abs(_acf1(g ** 2)) < 0.01


def test_garch_rescales_nonunit_parameters():
    E = simulate(SimSpec(n=3000, p=100, k=0, noise="garch", garch=(0.2, 0.05, 0.9)), 8).E
    assert abs(E.var() - 1) < 0.05                     # uncond. var 4 rescaled to 1


def test_hetero_noise_range():
    d = simulate(SimSpec(n=3000, p=300, k=0, noise="hetero", hetero_range=(0.5, 3.0)), 9)
    v = d.noise_var
    assert v.min() >= 0.5 and v.max() <= 3.0
    assert v.max() - v.min() > 2.0                     # actually spread over the range
    np.testing.assert_allclose(np.diag(d.Sigma), v)
    np.testing.assert_allclose(d.E.var(axis=0), v, rtol=0.1)


def test_factor_ar():
    d = simulate(SimSpec(n=5000, p=20, k=3, factor_ar=0.7), 10)
    assert abs(_acf1(d.F) - 0.7) < 0.03
    np.testing.assert_allclose(d.F.var(axis=0), 1, atol=0.1)
    C = np.corrcoef(d.F, rowvar=False)
    assert np.max(np.abs(C - np.eye(3))) < 0.1
    d0 = simulate(SimSpec(n=5000, p=20, k=3), 10)
    assert abs(_acf1(d0.F)) < 0.03


# ------------------------------------------------------------------------------ design grids

def test_design_grid_default_contents():
    g = design_grid("default")
    names = [s.name for s in g]
    assert len(g) == 111
    assert 60 <= len(g) <= 120
    assert len(set(names)) == len(names) and all(names)
    assert {s.regime for s in g} == {"pervasive", "spiked"}
    assert {s.noise for s in g} == {"gauss", "t", "ar1", "garch", "hetero"}
    assert {round(s.p / s.n, 3) for s in g} == {0.4, 1.0, 2.5}
    assert any(s.market is not None for s in g)
    assert any(s.k == 0 for s in g)
    # weak factors bracket the BBP threshold s^2 = sqrt(p/n)
    weak = [s for s in g if s.name.startswith("weak_")]
    ratios = sorted({round(s.strengths[3] ** 2 / np.sqrt(s.p / s.n), 6) for s in weak})
    assert ratios == [0.5, 1.0, 1.5, 2.0]
    # fresh objects each call
    g2 = design_grid("default")
    g2[0].n = -1
    assert design_grid("default")[0].n > 0


def test_design_grid_small():
    g = design_grid("small")
    assert 1 <= len(g) <= 8
    assert len({s.name for s in g}) == len(g)
    assert all(s.n <= 100 and s.p <= 250 for s in g)
    # small designs that share a name with a default design are the same design
    default = {s.name: s for s in design_grid("default")}
    shared = [s for s in g if s.name in default]
    assert len(shared) >= 5
    for s in shared:
        assert s == default[s.name]


def test_design_grid_unknown():
    with pytest.raises(ValueError, match="unknown design grid"):
        design_grid("huge")


@pytest.mark.parametrize("grid", ["default", "small"])
def test_all_designs_simulate(grid):
    for spec in design_grid(grid):
        d = simulate(spec, 0)
        assert d.X.shape == (spec.n, spec.p), spec.name
        assert d.F.shape[1] == spec.k == d.k, spec.name
        assert np.all(np.isfinite(d.X)), spec.name
        assert dataclasses.asdict(spec)["name"] == spec.name
