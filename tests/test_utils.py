"""Tests for kfactors.utils: preprocess, eigvals_desc, as_2d_array, check_random_state."""
import numpy as np
import pandas as pd
import pytest

from kfactors.utils import as_2d_array, check_random_state, eigvals_desc, preprocess


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.standard_normal((40, 6)) * np.arange(1, 7) + np.arange(6) * 10.0


def test_preprocess_standardize_matches_numpy(X):
    Z, mean, scale = preprocess(X, standardize=True)
    np.testing.assert_allclose(mean, X.mean(axis=0))
    np.testing.assert_allclose(scale, X.std(axis=0, ddof=0))
    np.testing.assert_allclose(Z, (X - X.mean(axis=0)) / X.std(axis=0))
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1.0)
    # correlation matrix from Z equals numpy's corrcoef
    np.testing.assert_allclose(Z.T @ Z / Z.shape[0], np.corrcoef(X, rowvar=False), atol=1e-12)


def test_preprocess_centre_only(X):
    Z, _mean, scale = preprocess(X, standardize=False)
    np.testing.assert_allclose(scale, 1.0)
    np.testing.assert_allclose(Z, X - X.mean(axis=0))
    np.testing.assert_allclose(Z.T @ Z / Z.shape[0], np.cov(X, rowvar=False, ddof=0), atol=1e-10)


def test_preprocess_accepts_dataframe(X):
    df = pd.DataFrame(X, columns=list("abcdef"))
    Z1, m1, s1 = preprocess(df)
    Z2, m2, s2 = preprocess(X)
    np.testing.assert_allclose(Z1, Z2, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(m1, m2)
    np.testing.assert_allclose(s1, s2)


def test_preprocess_rejects_bad_input(X):
    Xc = X.copy()
    Xc[:, 2] = 3.0
    with pytest.raises(ValueError, match="zero-variance"):
        preprocess(Xc, standardize=True)
    Z, _, _ = preprocess(Xc, standardize=False)  # allowed without standardisation
    assert np.all(Z[:, 2] == 0)
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        preprocess(Xn)
    with pytest.raises(ValueError):
        as_2d_array(np.ones(5))
    with pytest.raises(ValueError):
        as_2d_array(np.ones((2, 5)))
    with pytest.raises(ValueError):
        as_2d_array(np.ones((5, 1)))


@pytest.mark.parametrize("shape", [(50, 10), (10, 50), (20, 20)])
def test_eigvals_desc_matches_numpy(shape):
    rng = np.random.default_rng(1)
    Z, _, _ = preprocess(rng.standard_normal(shape))
    n, p = shape
    ev = eigvals_desc(Z)
    ref = np.sort(np.linalg.eigvalsh(Z.T @ Z / n))[::-1]
    assert ev.shape == (p,)
    assert np.all(np.diff(ev) <= 0)
    np.testing.assert_allclose(ev, np.clip(ref, 0, None), atol=1e-10)
    # trace identity and rank after centring: only min(n - 1, p) non-trivial eigenvalues
    np.testing.assert_allclose(ev.sum(), (Z**2).sum() / n)
    m = min(n - 1, p)
    assert ev[m - 1] > 1e-8
    np.testing.assert_allclose(ev[m:], 0.0, atol=1e-10)
    # singular values of Z agree
    sv = np.linalg.svd(Z, compute_uv=False)
    np.testing.assert_allclose(ev[: sv.size], sv**2 / n, atol=1e-10)


def test_check_random_state():
    g = np.random.default_rng(3)
    assert check_random_state(g) is g
    a = check_random_state(5).standard_normal(3)
    b = check_random_state(5).standard_normal(3)
    np.testing.assert_array_equal(a, b)
    assert isinstance(check_random_state(None), np.random.Generator)
