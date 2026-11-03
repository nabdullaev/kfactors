"""Tests for the scikit-learn estimators (kfactors.sklearn_api)."""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from kfactors.rules import select_k
from kfactors.rules.base import RuleWarning
from kfactors.sklearn_api import FactorCountSelector, SelectedPCA


def planted(n, p, k=3, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((p, k)) * np.array([1.0, 0.8, 0.6, 0.5][:k])
    return rng.standard_normal((n, k)) @ L.T + rng.standard_normal((n, p))


# Full scikit-learn API conformance, no exemptions. Several configurations so that both the k = 0
# path (default onatski on the tiny check datasets) and k > 0 paths are exercised.
ESTIMATORS = [
    FactorCountSelector(),
    FactorCountSelector(method="bai_ng", standardize=False),
    SelectedPCA(),
    SelectedPCA(method="bai_ng"),
    SelectedPCA(method="kaiser", standardize=False),
    SelectedPCA(method="parallel", rule_kwargs={"n_perm": 20}),
    SelectedPCA(method="gavish_donoho"),
]


@parametrize_with_checks(ESTIMATORS)
def test_sklearn_compatible(estimator, check):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuleWarning)
        check(estimator)


def test_k0_and_k_positive_paths_exercised_by_checks():
    rng = np.random.RandomState(0)
    X = 3 * rng.uniform(size=(30, 3))
    with pytest.warns(RuleWarning):
        assert SelectedPCA().fit(X).k_ == 0
    assert SelectedPCA(method="kaiser", standardize=False).fit(X).k_ >= 1


@pytest.mark.parametrize("method", ["onatski", "bai_ng", "ahn_horenstein", "parallel"])
def test_selector_matches_select_k(method):
    X = planted(150, 80, 3, seed=1)
    est = FactorCountSelector(method=method, kmax=10, random_state=3).fit(X)
    ref = select_k(X, method, kmax=10, random_state=3)
    assert est.k_ == ref.k == 3
    assert est.result_.method == method
    assert est.n_features_in_ == 80


def test_rule_kwargs_forwarded_and_not_mutated():
    X = planted(150, 80, 3, seed=2)
    kw = {"criterion": "IC_p3"}
    est = FactorCountSelector(method="bai_ng", rule_kwargs=kw).fit(X)
    assert est.result_.info["criterion"] == "IC_p3"
    assert kw == {"criterion": "IC_p3"}


def test_selected_pca_matches_sklearn_pca():
    X = planted(200, 60, 3, seed=3)
    est = SelectedPCA(method="onatski", kmax=10).fit(X)
    assert est.k_ == est.n_components_ == 3
    Z = (X - X.mean(0)) / X.std(0)
    ref = PCA(n_components=3, svd_solver="full").fit(Z)
    T, Tref = est.transform(X), ref.transform(Z)
    assert T.shape == (200, 3)
    # same subspace and scores up to sign
    for j in range(3):
        sgn = np.sign(T[:, j] @ Tref[:, j])
        np.testing.assert_allclose(T[:, j], sgn * Tref[:, j], atol=1e-8)
    # explained variance uses ddof = 0 (S = Z'Z / n); sklearn uses n - 1
    np.testing.assert_allclose(est.explained_variance_, ref.explained_variance_ * 199 / 200)
    np.testing.assert_allclose(est.explained_variance_ratio_, ref.explained_variance_ratio_)
    np.testing.assert_allclose(est.components_ @ est.components_.T, np.eye(3), atol=1e-12)
    # sign convention: the largest |loading| of every component is positive
    idx = np.argmax(np.abs(est.components_), axis=1)
    assert np.all(est.components_[np.arange(3), idx] > 0)
    np.testing.assert_allclose(est.mean_, X.mean(0))
    np.testing.assert_allclose(est.scale_, X.std(0))


def test_selected_pca_unstandardised():
    X = planted(120, 40, 2, seed=4) * 0.01
    est = SelectedPCA(method="bai_ng", standardize=False).fit(X)
    np.testing.assert_allclose(est.scale_, 1.0)
    Xc = X - X.mean(0)
    np.testing.assert_allclose(est.transform(X), Xc @ est.components_.T)


def test_single_feature_gives_zero_factors():
    X = np.random.default_rng(0).standard_normal((25, 1))
    for est in (FactorCountSelector(), SelectedPCA()):
        est.fit(X)
        assert est.k_ == 0
        assert "single feature" in est.result_.info["degenerate"]
    out = SelectedPCA().fit(X).transform(X)
    assert out.shape == (25, 0)


def test_too_few_samples_raise():
    with pytest.raises(ValueError, match="sample"):
        SelectedPCA().fit(np.ones((2, 5)))


def test_dataframe_and_feature_names():
    X = planted(100, 30, 2, seed=5)
    df = pd.DataFrame(X, columns=[f"s{i}" for i in range(30)])
    est = SelectedPCA(method="bai_ng").fit(df)
    assert list(est.feature_names_in_) == list(df.columns)
    names = est.get_feature_names_out()
    assert list(names) == [f"selectedpca{i}" for i in range(est.k_)]
    out = est.set_output(transform="pandas").transform(df)
    assert isinstance(out, pd.DataFrame) and list(out.columns) == list(names)


def test_random_state_variants_and_clone():
    X = planted(80, 40, 2, seed=6)
    a = FactorCountSelector(method="parallel", random_state=7,
                            rule_kwargs={"n_perm": 20}).fit(X)
    b = clone(a).fit(X)
    np.testing.assert_array_equal(a.result_.threshold, b.result_.threshold)
    c = FactorCountSelector(method="parallel", random_state=np.random.RandomState(1),
                            rule_kwargs={"n_perm": 20}).fit(X)
    assert c.k_ == 2


def test_in_pipeline():
    X = planted(100, 30, 2, seed=8)
    pipe = make_pipeline(SelectedPCA(method="onatski", kmax=8))
    assert pipe.fit_transform(X).shape == (100, 2)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="available rules"):
        FactorCountSelector(method="nope").fit(planted(50, 10))


@pytest.mark.parametrize("key", ["random_state", "kmax", "standardize", "method"])
def test_rule_kwargs_reject_estimator_parameters(key):
    """Regression (review R8): rule_kwargs={'random_state': 1} used to raise an obscure
    'multiple values for keyword argument' TypeError from select_k."""
    X = planted(100, 30, 2, seed=9)
    for cls in (FactorCountSelector, SelectedPCA):
        est = cls(method="parallel", random_state=0, rule_kwargs={key: 1, "n_perm": 5})
        with pytest.raises(ValueError, match=f"must not contain '{key}'"):
            est.fit(X)


def test_rule_warning_points_at_sklearn_api_not_rules():
    """Rule warnings are re-emitted by select_k at its caller (review U8)."""
    X = np.random.default_rng(0).standard_normal((30, 3))
    with pytest.warns(RuleWarning) as rec:
        FactorCountSelector().fit(X)
    assert all("rules" not in w.filename for w in rec)
