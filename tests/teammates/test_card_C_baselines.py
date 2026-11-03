"""Acceptance tests for Card C: ``kfactors.baselines`` (third-party k-selectors).

Each test skips while the function it exercises raises ``NotImplementedError``.
Run: ``.venv/bin/python -m pytest tests/teammates/test_card_C_baselines.py -v``
"""
from __future__ import annotations

import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from kfactors import baselines
from kfactors.baselines import (
    BaselineUnavailableWarning,
    horns_parallel,
    optht_threshold,
    sklearn_mle,
    sklearn_variance_share,
    statsmodels_bai_ng,
)
from kfactors.rules import select_k
from kfactors.sim import SimSpec, simulate


def call(fn, *args, **kwargs):
    """Call ``fn``; skip the test if that function of the card is not implemented yet."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"Card C function not implemented yet: {exc}")


def panels():
    """Deterministic planted panels: (label, X, true k). Covers p < n, p ~ n and p > n."""
    out = []
    for n, p, k, noise, seed in [(200, 40, 3, "gauss", 0), (150, 150, 2, "hetero", 1),
                                 (80, 200, 3, "gauss", 2), (300, 60, 4, "t", 3),
                                 (120, 90, 1, "hetero", 4)]:
        d = simulate(SimSpec(n=n, p=p, k=k, noise=noise), random_state=seed)
        out.append((f"n{n}_p{p}_k{k}_{noise}", d.X, k))
    rng = np.random.default_rng(5)
    out.append(("pure_noise_n100_p50", rng.normal(size=(100, 50)), 0))
    return out


PANELS = panels()
IDS = [label for label, _, _ in PANELS]


def assert_int(k, lo=0, hi=None):
    assert isinstance(k, (int, np.integer)) and not isinstance(k, bool), type(k)
    assert k >= lo and (hi is None or k <= hi)


# ================================================================ sklearn_variance_share
@pytest.mark.parametrize("share", [0.5, 0.8, 0.9])
@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("label, X, k_true", PANELS, ids=IDS)
def test_variance_share_agrees_with_kfactors(label, X, k_true, share, standardize):
    k = call(sklearn_variance_share, X, share=share, standardize=standardize)
    assert_int(k, lo=1)
    ref = select_k(X, "variance_share", share=share, standardize=standardize)
    assert k == ref.info["k_uncapped"]


def test_variance_share_matches_cumulative_eigenvalues():
    """Known answer from the eigenvalues of the correlation matrix (Jolliffe 2002, 6.1.1)."""
    X = PANELS[0][1]
    Z = (X - X.mean(0)) / X.std(0)
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    for share in (0.3, 0.6, 0.95):
        expected = int(np.argmax(cum > share)) + 1
        assert call(sklearn_variance_share, X, share=share) == expected


def test_variance_share_dataframe_and_validation():
    X = PANELS[1][1]
    df = pd.DataFrame(X, columns=[f"a{i}" for i in range(X.shape[1])])
    assert call(sklearn_variance_share, df) == call(sklearn_variance_share, X)
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            call(sklearn_variance_share, X, share=bad)
    Xn = X.copy()
    Xn[3, 4] = np.nan
    with pytest.raises(ValueError):
        call(sklearn_variance_share, Xn)


# ================================================================ sklearn_mle
def test_mle_returns_none_when_n_less_than_p():
    X = PANELS[2][1]                                  # n=80, p=200
    with pytest.warns(BaselineUnavailableWarning, match="n < p"):
        assert call(sklearn_mle, X) is None


def test_mle_matches_sklearn_directly():
    from sklearn.decomposition import PCA
    for label, X, _ in PANELS:
        n, p = X.shape
        if n < p:
            continue
        for standardize in (True, False):
            Z = X - X.mean(0)
            if standardize:
                Z = Z / Z.std(0)
            expected = PCA(n_components="mle", svd_solver="full").fit(Z).n_components_
            with warnings.catch_warnings():
                warnings.simplefilter("error", BaselineUnavailableWarning)
                k = call(sklearn_mle, X, standardize=standardize)
            assert_int(k)
            assert k == expected, label


def test_mle_recovers_planted_k_with_homoscedastic_noise():
    """Minka's evidence assumes isotropic noise; with gauss noise, n >> p it finds k exactly."""
    for seed, k_true in [(0, 2), (1, 3), (2, 5)]:
        d = simulate(SimSpec(n=600, p=30, k=k_true, noise="gauss"), random_state=seed)
        assert call(sklearn_mle, d.X, standardize=False) == k_true


def test_mle_square_panel_is_supported():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 40))
    with warnings.catch_warnings():
        warnings.simplefilter("error", BaselineUnavailableWarning)
        assert_int(call(sklearn_mle, X))


# ================================================================ statsmodels_bai_ng
@pytest.mark.parametrize("criterion", ["IC_p1", "IC_p2", "IC_p3"])
@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("label, X, k_true", PANELS, ids=IDS)
def test_bai_ng_agrees_with_kfactors(label, X, k_true, criterion, standardize):
    k = call(statsmodels_bai_ng, X, criterion=criterion, kmax=12, standardize=standardize)
    ref = select_k(X, "bai_ng", criterion=criterion, kmax=12, standardize=standardize)
    assert_int(k, hi=ref.info["kmax_effective"])
    assert k == ref.k


def test_bai_ng_recovers_planted_k():
    """IC_p2 finds the planted k on the Gaussian / heteroscedastic panels (with t(3) noise the
    weakest 4th factor is missed by both statsmodels and kfactors, so that panel is excluded)."""
    for label, X, k_true in PANELS:
        if label.endswith("_t"):
            continue
        assert call(statsmodels_bai_ng, X) == k_true, label


def test_bai_ng_kmax_is_clipped_like_kfactors():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(12, 30)) + np.outer(rng.normal(size=12), rng.normal(size=30))
    k = call(statsmodels_bai_ng, X, kmax=50)          # kmax_eff = min(50, 11 - 2) = 9
    assert k == select_k(X, "bai_ng", kmax=50).k
    assert k <= 9


def test_bai_ng_validation():
    X = PANELS[0][1]
    with pytest.raises(ValueError):
        call(statsmodels_bai_ng, X, criterion="PC_p2")
    with pytest.raises(ValueError):
        call(statsmodels_bai_ng, X, kmax=0)
    Xn = X.copy()
    Xn[0, 0] = np.inf
    with pytest.raises(ValueError):
        call(statsmodels_bai_ng, Xn)


# ================================================================ optht_threshold
@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("label, X, k_true", PANELS, ids=IDS)
def test_optht_agrees_with_kfactors_gavish_donoho(label, X, k_true, standardize):
    k = call(optht_threshold, X, standardize=standardize)
    ref = select_k(X, "gavish_donoho", standardize=standardize)
    assert_int(k, hi=min(X.shape[0] - 1, X.shape[1]))
    assert k == ref.info["k_uncapped"]


def test_optht_uses_n_minus_one_aspect_ratio():
    """beta = min(n-1, p) / max(n-1, p): for a square panel beta is (n-1)/n, not 1."""
    calls = []
    import optht as optht_pkg
    real = optht_pkg.optht

    def spy(beta, sv, sigma=None):
        calls.append((beta, np.asarray(sv).copy(), sigma))
        return real(beta, sv=sv, sigma=sigma)

    X = PANELS[1][1]                                   # n = p = 150
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(optht_pkg, "optht", spy)
        if isinstance(getattr(baselines, "optht", None), types.FunctionType):
            mp.setattr(baselines, "optht", spy)        # `from optht import optht` at top level
        call(optht_threshold, X)
    finally:
        mp.undo()
    assert calls, "optht_threshold must call optht.optht(beta, sv=..., sigma=None)"
    beta, sv, sigma = calls[-1]
    assert beta == pytest.approx(149 / 150)
    assert len(sv) == 149 and sigma is None
    Z = (X - X.mean(0)) / X.std(0)
    np.testing.assert_allclose(sv, np.linalg.svd(Z, compute_uv=False)[:149], rtol=1e-10)


def test_optht_returns_zero_when_nothing_survives():
    """A rank-deficient-looking pure-noise panel where no singular value clears the cutoff."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(400, 20))
    k = call(optht_threshold, X)
    assert k == 0


# ================================================================ horns_parallel
def test_horns_missing_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "horns", None)    # makes `import horns` raise ImportError
    with pytest.warns(BaselineUnavailableWarning, match="not installed"):
        assert call(horns_parallel, PANELS[0][1]) is None


def test_horns_wiring_with_fake_package(monkeypatch):
    """A fake ``horns`` module checks the call: raw data, simulations, analysis type, quantile."""
    seen = {}

    def parallel_analysis(data, simulations=10000, randomisation_method="bootstrap",
                          analysis_type="pca", quantile=0.95, full_output=False,
                          verbose=False):
        seen.update(data=np.asarray(data), simulations=simulations,
                    analysis_type=analysis_type, quantile=quantile)
        return np.int64(4)

    fake = types.ModuleType("horns")
    fake.parallel_analysis = parallel_analysis
    monkeypatch.setitem(sys.modules, "horns", fake)
    X = PANELS[0][1]
    k = call(horns_parallel, pd.DataFrame(X), n_sim=123, quantile=0.9)
    assert k == 4 and type(k) is int
    assert seen["simulations"] == 123 and seen["quantile"] == 0.9
    assert seen["analysis_type"] == "pca"
    np.testing.assert_allclose(seen["data"], X)


def test_horns_validation(monkeypatch):
    monkeypatch.setitem(sys.modules, "horns", None)
    for kw in ({"n_sim": 0}, {"quantile": 1.0}, {"quantile": 0.0}):
        with pytest.raises(ValueError):
            call(horns_parallel, PANELS[0][1], **kw)


def test_horns_real_package_if_installed():
    pytest.importorskip("horns")
    d = simulate(SimSpec(n=300, p=30, k=3, noise="gauss"), random_state=0)
    k = call(horns_parallel, d.X, n_sim=200)
    assert abs(k - select_k(d.X, "parallel", random_state=0).k) <= 1


# ================================================================ module hygiene
def test_module_imports_without_optional_packages():
    with open(baselines.__file__, encoding="utf-8") as fh:
        src = fh.read()
    assert "\nimport horns" not in src and "\nfrom horns" not in src
