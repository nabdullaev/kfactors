"""Tests for kfactors.model.FactorModel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kfactors.model import FactorModel
from kfactors.utils import eigvals_desc, preprocess


def _panel(n=120, p=30, k=3, seed=0, scale_spread=True):
    """Planted k-factor panel with heterogeneous volatilities and non-zero means."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, k)) * np.linspace(3.0, 1.5, k)
    B = rng.standard_normal((p, k))
    E = rng.standard_normal((n, p))
    vol = np.exp(rng.uniform(-1.0, 1.0, p)) if scale_spread else np.ones(p)
    return 0.01 * (F @ B.T + E) * vol + 0.001 * rng.standard_normal(p)


SHAPES = [(120, 30), (40, 150)]  # p < n and p > n


@pytest.mark.parametrize("n,p", SHAPES)
@pytest.mark.parametrize("standardize", [True, False])
def test_reconstruction_exact_at_full_rank(n, p, standardize):
    X = _panel(n, p)
    rank = min(n - 1, p)
    m = FactorModel(rank, standardize=standardize).fit(X)
    Xc = X - X.mean(axis=0)
    np.testing.assert_allclose(m.factors_ @ m.loadings_.T, Xc, atol=1e-12)
    assert np.abs(m.residuals_).max() < 1e-12
    np.testing.assert_allclose(m.idio_var_, 1e-12)
    assert m.explained_variance_ratio_.sum() == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("n,p", SHAPES)
@pytest.mark.parametrize("standardize", [True, False])
def test_shapes_and_attribute_identities(n, p, standardize):
    X = _panel(n, p)
    k = 4
    m = FactorModel(k, standardize=standardize).fit(X)
    assert m.mean_.shape == (p,) and m.scale_.shape == (p,)
    assert m.eigenvalues_.shape == (min(n, p),)
    assert m.eigenvectors_.shape == (p, k)
    assert m.factors_.shape == (n, k) and m.loadings_.shape == (p, k)
    assert m.residuals_.shape == (n, p) and m.idio_var_.shape == (p,)
    assert m.explained_variance_ratio_.shape == (k,)
    Z, mean, scale = preprocess(X, standardize)
    np.testing.assert_allclose(m.mean_, mean)
    np.testing.assert_allclose(m.scale_, scale)
    if not standardize:
        np.testing.assert_array_equal(m.scale_, 1.0)
    np.testing.assert_allclose(m.factors_, Z @ m.eigenvectors_, atol=1e-14)
    np.testing.assert_allclose(m.loadings_, m.eigenvectors_ * m.scale_[:, None])
    np.testing.assert_allclose(m.residuals_, X - mean - m.factors_ @ m.loadings_.T, atol=1e-15)
    np.testing.assert_allclose(m.idio_var_, np.maximum(m.residuals_.var(axis=0), 1e-12))
    # eigenvalues: descending, match the shared helper and an independent eigvalsh
    ev = m.eigenvalues_
    assert np.all(np.diff(ev) <= 1e-15)
    np.testing.assert_allclose(ev, eigvals_desc(Z)[: min(n, p)], atol=1e-12)
    np.testing.assert_allclose(
        ev[: min(n, p) - 1], np.sort(np.linalg.eigvalsh(Z.T @ Z / n))[::-1][: min(n, p) - 1],
        rtol=1e-8, atol=1e-13,
    )
    np.testing.assert_allclose(m.explained_variance_ratio_, ev[:k] / np.trace(Z.T @ Z / n))
    # orthonormal eigenvectors
    np.testing.assert_allclose(m.eigenvectors_.T @ m.eigenvectors_, np.eye(k), atol=1e-12)


@pytest.mark.parametrize("n,p", SHAPES)
def test_factors_uncorrelated_with_eigenvalue_variances(n, p):
    X = _panel(n, p)
    m = FactorModel(5).fit(X)
    F = m.factors_
    np.testing.assert_allclose(F.mean(axis=0), 0.0, atol=1e-13)
    np.testing.assert_allclose(F.T @ F / n, np.diag(m.eigenvalues_[:5]), atol=1e-12)


def test_centring_rank_trailing_eigenvalue_zero_when_p_ge_n():
    n, p = 40, 150
    m = FactorModel(2).fit(_panel(n, p))
    assert m.rank_ == n - 1
    assert m.eigenvalues_.shape == (n,)
    assert m.eigenvalues_[-1] < 1e-12 * m.eigenvalues_[0]
    assert m.eigenvalues_[-2] > 1e-6 * m.eigenvalues_[0]


def test_matches_sklearn_pca_up_to_sign():
    from sklearn.decomposition import PCA

    X = _panel(100, 25)
    m = FactorModel(3, standardize=False).fit(X)
    pca = PCA(n_components=3, svd_solver="full").fit(X)
    for j in range(3):
        v, c = m.eigenvectors_[:, j], pca.components_[j]
        assert abs(abs(v @ c) - 1.0) < 1e-10
    # sklearn's explained_variance_ uses ddof=1
    np.testing.assert_allclose(m.eigenvalues_[:3] * 100 / 99, pca.explained_variance_, rtol=1e-10)


@pytest.mark.parametrize("n,p", SHAPES)
@pytest.mark.parametrize("standardize", [True, False])
def test_covariance_psd_and_equals_sample_cov_at_full_rank(n, p, standardize):
    X = _panel(n, p)
    S = np.cov(X, rowvar=False, ddof=0)
    full = FactorModel(min(n - 1, p), standardize=standardize).fit(X)
    np.testing.assert_allclose(full.covariance(), S + 1e-12 * np.eye(p), atol=1e-15, rtol=1e-10)
    for k in (0, 1, 3):
        m = FactorModel(k, standardize=standardize).fit(X)
        for res in ("diag", "ledoit_wolf"):
            C = m.covariance(res)
            assert C.shape == (p, p)
            np.testing.assert_array_equal(C, C.T)
            assert np.linalg.eigvalsh(C).min() > -1e-14 * np.abs(C).max()
        # the "diag" model matches sample variances exactly, for every k
        np.testing.assert_allclose(np.diag(m.covariance("diag")), np.diag(S), rtol=1e-10)


def test_covariance_components_and_ledoit_wolf_cross_check():
    from sklearn.covariance import LedoitWolf

    X = _panel(80, 120)
    m = FactorModel(3).fit(X)
    B = m.loadings_
    common = B @ np.diag(m.eigenvalues_[:3]) @ B.T
    np.testing.assert_allclose(m.covariance("diag"), common + np.diag(m.idio_var_),
                               atol=1e-15, rtol=1e-10)
    lw = LedoitWolf().fit(m.residuals_).covariance_
    np.testing.assert_allclose(m.covariance("ledoit_wolf"), common + lw, atol=1e-15, rtol=1e-10)
    # LW residual covariance is full rank even though p > n
    assert np.linalg.eigvalsh(m.covariance("ledoit_wolf")).min() > 0
    with pytest.raises(ValueError):
        m.covariance("sample")


def test_ledoit_wolf_at_full_rank_has_floor():
    X = _panel(30, 10)
    m = FactorModel(10).fit(X)
    C = m.covariance("ledoit_wolf")
    np.testing.assert_allclose(C, np.cov(X, rowvar=False, ddof=0) + 1e-12 * np.eye(10),
                               atol=1e-15, rtol=1e-9)


def test_k_zero():
    X = _panel(50, 20)
    m = FactorModel(0).fit(X)
    assert m.eigenvectors_.shape == (20, 0) and m.factors_.shape == (50, 0)
    assert m.loadings_.shape == (20, 0) and m.explained_variance_ratio_.shape == (0,)
    np.testing.assert_allclose(m.residuals_, X - X.mean(axis=0))
    np.testing.assert_allclose(m.covariance(), np.diag(X.var(axis=0)))
    assert m.eigenportfolios().shape == (20, 0)
    resid, betas = m.residual_returns()
    assert betas.shape == (20, 0)
    np.testing.assert_allclose(resid, X - X.mean(axis=0), atol=1e-15)
    resid2, _ = m.residual_returns(X[:10])
    np.testing.assert_allclose(resid2, X[:10] - X[:10].mean(axis=0), atol=1e-15)


@pytest.mark.parametrize("n,p", SHAPES)
def test_deterministic_sign_convention(n, p):
    X = _panel(n, p)
    m1 = FactorModel(4).fit(X)
    m2 = FactorModel(4).fit(X.copy())
    np.testing.assert_array_equal(m1.eigenvectors_, m2.eigenvectors_)
    V = m1.eigenvectors_
    assert np.all(V[np.argmax(np.abs(V), axis=0), np.arange(4)] > 0)
    # S is invariant to X -> -X, so the sign-normalised eigenvectors must be identical
    m3 = FactorModel(4).fit(-X)
    np.testing.assert_allclose(m3.eigenvectors_, V, atol=1e-12)
    np.testing.assert_allclose(m3.factors_, -m1.factors_, atol=1e-12)


def test_dataframe_input_equivalent():
    X = _panel(60, 12)
    df = pd.DataFrame(X, columns=[f"a{i}" for i in range(12)])
    m1, m2 = FactorModel(2).fit(X), FactorModel(2).fit(df)
    np.testing.assert_allclose(m1.loadings_, m2.loadings_)
    np.testing.assert_allclose(m1.residual_returns(df)[0], m2.residual_returns(X)[0])


def test_eigenportfolios_avellaneda_lee():
    X = _panel(100, 20)
    m = FactorModel(3).fit(X)
    Q = m.eigenportfolios()
    np.testing.assert_allclose(Q, m.eigenvectors_ / X.std(axis=0)[:, None])
    # eigenportfolio returns equal the factor scores up to a constant
    np.testing.assert_allclose(X @ Q - (X @ Q).mean(axis=0), m.factors_, atol=1e-13)
    m_raw = FactorModel(3, standardize=False).fit(X)
    np.testing.assert_allclose(m_raw.eigenportfolios(), m_raw.eigenvectors_)


@pytest.mark.parametrize("n,p", SHAPES)
@pytest.mark.parametrize("standardize", [True, False])
def test_residual_returns_in_sample(n, p, standardize):
    X = _panel(n, p)
    k = 3
    m = FactorModel(k, standardize=standardize).fit(X)
    resid, betas = m.residual_returns()
    assert resid.shape == (n, p) and betas.shape == (p, k)
    Fq = X @ m.eigenportfolios()
    # OLS normal equations: residuals orthogonal to the regressors and to the intercept
    np.testing.assert_allclose(Fq.T @ resid, 0.0, atol=1e-12)
    np.testing.assert_allclose(resid.mean(axis=0), 0.0, atol=1e-15)
    # in-sample slopes are the loadings, residuals the PCA residuals
    np.testing.assert_allclose(betas, m.loadings_, atol=1e-12)
    np.testing.assert_allclose(resid, m.residuals_, atol=1e-13)


def test_residual_returns_out_of_sample_matches_statsmodels():
    import statsmodels.api as sm

    X = _panel(200, 15, seed=3)
    m = FactorModel(2).fit(X[:120])
    Xo = X[120:]
    resid, betas = m.residual_returns(Xo)
    Fq = Xo @ m.eigenportfolios()
    for i in (0, 7, 14):
        fit = sm.OLS(Xo[:, i], sm.add_constant(Fq)).fit()
        np.testing.assert_allclose(betas[i], fit.params[1:], rtol=1e-9, atol=1e-14)
        np.testing.assert_allclose(resid[:, i], fit.resid, atol=1e-14)
    with pytest.raises(ValueError):
        m.residual_returns(Xo[:, :10])


def test_planted_factors_recovered():
    rng = np.random.default_rng(7)
    n, p, k = 500, 60, 2
    F = rng.standard_normal((n, k)) * [4.0, 2.5]
    B = rng.standard_normal((p, k))
    X = 0.01 * (F @ B.T + rng.standard_normal((n, p)))
    m = FactorModel(k, standardize=False).fit(X)
    # canonical correlations between estimated and true factors close to 1
    Fq = X @ m.eigenportfolios()
    qa, _ = np.linalg.qr(Fq - Fq.mean(0))
    qb, _ = np.linalg.qr(F - F.mean(0))
    cc = np.linalg.svd(qa.T @ qb, compute_uv=False)
    assert cc.min() > 0.98


def test_errors():
    X = _panel(30, 10)
    with pytest.raises(ValueError):
        FactorModel(11).fit(X)  # k > p
    Xw = _panel(10, 50)
    FactorModel(9).fit(Xw)  # n - 1 is allowed
    with pytest.raises(ValueError):
        FactorModel(10).fit(Xw)  # k > n - 1
    with pytest.raises(ValueError):
        FactorModel(-1).fit(X)
    with pytest.raises(TypeError):
        FactorModel(1.5).fit(X)
    with pytest.raises(TypeError):
        FactorModel(True).fit(X)
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        FactorModel(1).fit(Xn)
    m = FactorModel(1)
    for call in (m.covariance, m.eigenportfolios, m.residual_returns):
        with pytest.raises(RuntimeError):
            call()


def test_warns_when_k_exceeds_numerical_rank():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((50, 3))
    X = np.column_stack([base, base @ rng.standard_normal((3, 5))])  # rank 3, p = 8
    with pytest.warns(RuntimeWarning):
        FactorModel(5, standardize=False).fit(X)
