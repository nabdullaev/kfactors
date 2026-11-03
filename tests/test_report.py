"""Tests for kfactors.report: diagnose, decision tables, calibration and the report."""
from __future__ import annotations

import json
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from kfactors import RULES, select_k
from kfactors.report import (
    DEFAULT_FEATURES,
    Report,
    calibrate_decision_table,
    diagnose,
    load_decision_table,
    parse_rule_label,
    recommend,
    report,
    rule_label,
    validate_decision_table,
)
from kfactors.sim import SimSpec, simulate

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _quiet_rule_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="kfactors.rules")
        yield


def _sim(seed=0, **kw):
    kw.setdefault("n", 250)
    kw.setdefault("p", 100)
    return simulate(SimSpec(**kw), random_state=seed).X


# ------------------------------------------------------------------------------------------------
# diagnose
# ------------------------------------------------------------------------------------------------
class TestDiagnose:
    def test_keys_types_and_json(self):
        d = diagnose(_sim(k=3))
        for key in ("n", "p", "p_over_n", "spike_share", "top_gap_ratios", "gap_ratio_1",
                    "mean_excess_kurtosis", "mean_abs_acf1", "mean_acf1_sq", "var_dispersion"):
            assert key in d
        assert set(DEFAULT_FEATURES) <= set(d)
        assert (d["n"], d["p"], d["p_over_n"]) == (250, 100, 0.4)
        assert len(d["top_gap_ratios"]) == 5
        json.dumps(d)  # plain python scalars only

    def test_cross_check_independent_implementations(self):
        rng = np.random.default_rng(3)
        X = rng.standard_t(5, size=(80, 30)) * rng.uniform(0.5, 2.0, size=30)
        X[1:] += 0.4 * X[:-1]
        d = diagnose(X)
        ev = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]
        assert d["spike_share"] == pytest.approx(ev[0] / ev.sum(), rel=1e-10)
        assert d["gap_ratio_1"] == pytest.approx(ev[0] / ev[1], rel=1e-10)
        assert d["top_gap_ratios"] == pytest.approx(list(ev[:5] / ev[1:6]), rel=1e-10)
        assert d["mean_excess_kurtosis"] == pytest.approx(
            np.mean(stats.kurtosis(X, axis=0, fisher=True, bias=True)), rel=1e-10)
        from statsmodels.tsa.stattools import acf
        r1 = np.array([acf(X[:, j], nlags=1, fft=False)[1] for j in range(X.shape[1])])
        assert d["mean_acf1"] == pytest.approx(r1.mean(), rel=1e-10)
        assert d["mean_abs_acf1"] == pytest.approx(np.abs(r1).mean(), rel=1e-10)
        sq = (X - X.mean(0)) ** 2
        r1sq = np.array([acf(sq[:, j], nlags=1, fft=False)[1] for j in range(X.shape[1])])
        assert d["mean_acf1_sq"] == pytest.approx(r1sq.mean(), rel=1e-10)
        v = X.var(axis=0)
        assert d["var_dispersion"] == pytest.approx(v.std() / v.mean(), rel=1e-10)

    def test_covariance_variant_and_scale(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 20)) * np.linspace(1, 5, 20)
        d_cor, d_cov = diagnose(X), diagnose(X, standardize=False)
        ev = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
        assert d_cov["spike_share"] == pytest.approx(ev[0] / ev.sum(), rel=1e-10)
        assert d_cov["spike_share"] > d_cor["spike_share"]   # the largest-variance column dominates
        # var_dispersion is always on unstandardised data; other column stats are scale free
        assert d_cov["var_dispersion"] == pytest.approx(d_cor["var_dispersion"])
        d_scaled = diagnose(1e-2 * X)
        for key in ("spike_share", "mean_abs_acf1", "mean_excess_kurtosis", "var_dispersion"):
            assert d_scaled[key] == pytest.approx(d_cor[key], rel=1e-9)

    def test_white_noise_levels(self):
        d = diagnose(_sim(k=0, n=400, p=200))
        assert d["mean_abs_acf1"] == pytest.approx(np.sqrt(2 / (np.pi * 400)), rel=0.15)
        assert abs(d["mean_excess_kurtosis"]) < 0.1
        assert abs(d["mean_acf1_sq"]) < 0.03
        assert d["var_dispersion"] < 0.15

    def test_ar_noise_raises_autocorrelation(self):
        base = diagnose(_sim(k=3, noise="gauss"))
        for phi in (0.5, 0.9):
            ar = diagnose(_sim(k=3, noise="ar1", phi=phi))
            assert ar["mean_abs_acf1"] > base["mean_abs_acf1"] + 0.15
            assert ar["mean_acf1"] > 0.15
        assert diagnose(_sim(k=3, noise="ar1", phi=0.9))["mean_abs_acf1"] > \
            diagnose(_sim(k=3, noise="ar1", phi=0.5))["mean_abs_acf1"]

    def test_market_factor_raises_spike_share(self):
        plain = diagnose(_sim(k=4, strengths=(3, 2, 1.5, 1.2)))
        market = diagnose(_sim(k=4, market=1.0, strengths=(3, 2, 1.5)))
        assert market["spike_share"] > plain["spike_share"] + 0.1
        assert market["gap_ratio_1"] > plain["gap_ratio_1"]
        spiked_market = diagnose(_sim(k=4, regime="spiked", market=100.0, strengths=(3, 2, 1.5)))
        assert spiked_market["gap_ratio_1"] > 4

    def test_heavy_tails_garch_and_hetero(self):
        gauss = diagnose(_sim(k=3))
        t3 = diagnose(_sim(k=3, noise="t", df=3.0))
        garch = diagnose(_sim(k=3, noise="garch"))
        hetero = diagnose(_sim(k=0, noise="hetero"))
        assert t3["mean_excess_kurtosis"] > gauss["mean_excess_kurtosis"] + 1.0
        assert garch["mean_acf1_sq"] > gauss["mean_acf1_sq"] + 0.01
        assert garch["mean_excess_kurtosis"] > gauss["mean_excess_kurtosis"]
        assert hetero["var_dispersion"] > diagnose(_sim(k=0))["var_dispersion"] + 0.2

    def test_tiny_panel_and_errors(self):
        d = diagnose(np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 0.5]]))
        assert len(d["top_gap_ratios"]) == 1 and d["argmax_gap_ratio"] == 1
        X = np.random.default_rng(0).standard_normal((20, 5))
        X[3, 2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            diagnose(X)
        X = np.random.default_rng(0).standard_normal((20, 5))
        X[:, 1] = 3.0
        with pytest.raises(ValueError, match="zero-variance"):
            diagnose(X)
        d = diagnose(X, standardize=False)    # constant column tolerated without standardising
        assert np.isfinite(d["mean_abs_acf1"])

    def test_dataframe_input(self):
        X = _sim(k=2, n=50, p=10)
        df = pd.DataFrame(X, index=pd.date_range("2020", periods=50), columns=list("abcdefghij"))
        a, b = diagnose(df), diagnose(X)
        assert a.keys() == b.keys()
        for key in a:
            assert a[key] == pytest.approx(b[key], rel=1e-12)


# ------------------------------------------------------------------------------------------------
# rule labels
# ------------------------------------------------------------------------------------------------
class TestLabels:
    @pytest.mark.parametrize("name,kw,label", [
        ("onatski", {}, "onatski"),
        ("parallel", {"null": "circular"}, "parallel(null=circular)"),
        ("parallel", {"null": "block", "n_perm": 50, "quantile": 0.9},
         "parallel(n_perm=50, null=block, quantile=0.9)"),
        ("x", {"flag": True, "v": None}, "x(flag=true, v=none)"),
    ])
    def test_round_trip(self, name, kw, label):
        assert rule_label(name, kw) == label
        assert parse_rule_label(label) == (name, kw)

    @pytest.mark.parametrize("bad", ["", "1abc", "parallel(null)", "a(b=1", "p(1x=2)"])
    def test_malformed(self, bad):
        with pytest.raises(ValueError):
            parse_rule_label(bad)

    def test_unencodable_value(self):
        with pytest.raises(ValueError):
            rule_label("parallel", {"null": "a b"})


# ------------------------------------------------------------------------------------------------
# decision tables / recommend
# ------------------------------------------------------------------------------------------------
BASE = {"mean_abs_acf1": 0.05, "gap_ratio_1": 2.0, "spike_share": 0.2, "p_over_n": 0.5}


class TestRecommend:
    def test_default_table_is_valid_and_documented(self):
        t = load_decision_table()
        validate_decision_table(t)
        assert t["kind"] == "heuristic"
        assert "HEURISTIC" in t["meta"]["status"]
        assert len(t["meta"]["branches"]) == 5
        leaves = []

        def walk(node):
            if "feature" in node:
                assert node["feature"] in t["features"]
                walk(node["then"])
                walk(node["else"])
            else:
                leaves.append(node)
        walk(t["tree"])
        assert len(leaves) == 5
        assert all(leaf["rule"] in RULES and leaf.get("why") for leaf in leaves)

    @pytest.mark.parametrize("override,label", [
        ({"mean_abs_acf1": 0.35}, "parallel(devolatilize=true, null=circular)"),
        ({"mean_abs_acf1": 0.35, "gap_ratio_1": 10.0, "p_over_n": 3.0},
         "parallel(devolatilize=true, null=circular)"),
        ({"gap_ratio_1": 8.0}, "onatski"),
        ({"spike_share": 0.45}, "onatski"),
        ({"spike_share": 0.45, "p_over_n": 2.5}, "onatski"),   # market branch before p > n
        ({"p_over_n": 2.5}, "parallel(devolatilize=true)"),
        ({}, "onatski"),
        ({"mean_abs_acf1": 0.2, "gap_ratio_1": 4.0, "spike_share": 0.3, "p_over_n": 1.0},
         "onatski"),   # all thresholds are strict '>'
    ])
    def test_default_paths(self, override, label):
        r = recommend({**BASE, **override})
        assert r["label"] == label
        assert r["rule"] == parse_rule_label(label)[0]
        assert r["kwargs"] == parse_rule_label(label)[1]
        assert r["table_kind"] == "heuristic"
        assert r["reason"].split(" -> ")[1].startswith(label)

    def test_reason_and_path_follow_the_tree(self):
        r = recommend({**BASE, "p_over_n": 2.5})
        assert [s["feature"] for s in r["path"]] == ["mean_abs_acf1", "gap_ratio_1",
                                                     "spike_share", "p_over_n"]
        assert [s["taken"] for s in r["path"]] == ["else", "else", "else", "then"]
        assert "mean_abs_acf1 = 0.05 <= 0.2" in r["reason"]
        assert "p_over_n = 2.5 > 1" in r["reason"]

    def test_nan_goes_to_else_and_missing_raises(self):
        r = recommend({**BASE, "mean_abs_acf1": float("nan")})
        assert r["label"] == "onatski"
        assert "mean_abs_acf1 is NaN" in r["reason"]
        with pytest.raises(KeyError, match="gap_ratio_1"):
            recommend({"mean_abs_acf1": 0.0})

    def test_extra_keys_ignored_and_real_diagnostics(self):
        assert recommend(diagnose(_sim(k=3, noise="ar1", phi=0.9)))["label"] == \
            "parallel(devolatilize=true, null=circular)"
        r = recommend(diagnose(_sim(k=4, regime="spiked", market=100.0, strengths=(3, 2, 1.5))))
        assert r["rule"] == "onatski" and r["path"][1]["taken"] == "then"   # market branch
        assert recommend(diagnose(_sim(k=3, regime="spiked", n=100, p=250)))["label"] == \
            "parallel(devolatilize=true)"
        assert recommend(diagnose(_sim(k=3, regime="spiked")))["label"] == "onatski"

    def test_custom_table_and_copy_semantics(self):
        table = {"tree": {"feature": "x", "op": "<=", "threshold": 1,
                          "then": {"rule": "bai_ng", "kwargs": {"criterion": "IC_p1"}},
                          "else": {"rule": "kaiser"}}}
        r = recommend({"x": 1.0}, table)
        assert r["label"] == "bai_ng(criterion=IC_p1)" and r["table_kind"] == "unknown"
        r["kwargs"]["criterion"] = "changed"
        assert table["tree"]["then"]["kwargs"]["criterion"] == "IC_p1"
        assert recommend({"x": 1.5}, table)["label"] == "kaiser"
        assert recommend({}, {"tree": {"rule": "onatski"}})["reason"] == "onatski"

    @pytest.mark.parametrize("tree,exc", [
        ({"rule": "nope"}, ValueError),
        ({"feature": "x", "op": "==", "threshold": 1, "then": {"rule": "mp"},
          "else": {"rule": "mp"}}, ValueError),
        ({"feature": "x", "op": ">", "threshold": float("nan"), "then": {"rule": "mp"},
          "else": {"rule": "mp"}}, ValueError),
        ({"feature": "x", "op": ">", "threshold": 1, "then": {"rule": "mp"}}, ValueError),
        ({"something": 1}, ValueError),
        ({"rule": "mp", "kwargs": [1]}, ValueError),
        ([1, 2], TypeError),
    ])
    def test_malformed_tables(self, tree, exc):
        with pytest.raises(exc):
            recommend({"x": 0.0}, {"tree": tree})

    def test_table_must_be_mapping_with_tree(self):
        with pytest.raises(TypeError):
            validate_decision_table([1])
        with pytest.raises(ValueError):
            validate_decision_table({"meta": {}})


# ------------------------------------------------------------------------------------------------
# calibration
# ------------------------------------------------------------------------------------------------
def _toy_results(n_panels=40, seed=0):
    """Perfectly separable toy benchmark: best rule = circular PA iff a > 0.5, else onatski."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_panels):
        fam = "famA" if i % 2 == 0 else "famB"
        a, b = rng.uniform(0, 1), rng.uniform(0, 1)
        if 0.45 < a < 0.55:
            a += 0.2 if a > 0.5 else -0.2
        good = "parallel(null=circular)" if a > 0.5 else "onatski"
        for rule in ("onatski", "parallel(null=circular)", "mp_market"):
            k_hat = 3 if rule == good else 3 + 1 + int(rng.integers(0, 3))
            rows.append({"design": f"{fam}_d{i}", "rep": 0, "rule": rule, "k_hat": k_hat,
                         "k_true": 3, "a": a, "b": b})
    return pd.DataFrame(rows)


class TestCalibrate:
    def test_separable_problem_round_trips(self):
        df = _toy_results()
        table = calibrate_decision_table(df, features=["a", "b"], max_depth=3)
        validate_decision_table(table)
        m = table["meta"]
        assert table["kind"] == "calibrated"
        assert m["train_accuracy"] == 1.0
        assert m["leave_one_family_out_accuracy"] == 1.0
        assert set(m["leave_one_family_out_accuracy_by_family"]) == {"famA", "famB"}
        assert m["mae_tree_train"] == 0.0 and m["mae_oracle"] == 0.0
        assert m["n_samples"] == 40 and m["n_designs"] == 40
        # one split on the informative feature, between the classes
        tree = table["tree"]
        assert tree["feature"] == "a" and 0.35 < tree["threshold"] < 0.65
        assert tree["then"]["rule"] == "onatski"
        assert tree["else"] == {**tree["else"], "rule": "parallel", "kwargs": {"null": "circular"}}
        # JSON round trip, then recommend() reproduces the target on every panel
        table2 = json.loads(json.dumps(table))
        panels = df.drop_duplicates(["design", "rep"])
        for _, row in panels.iterrows():
            want = "parallel(null=circular)" if row["a"] > 0.5 else "onatski"
            assert recommend({"a": row["a"], "b": row["b"]}, table2)["label"] == want
        # and generalises to new points
        assert recommend({"a": 0.9, "b": 0.1}, table2)["label"] == "parallel(null=circular)"
        assert recommend({"a": 0.1, "b": 0.9}, table2)["label"] == "onatski"

    def test_best_single_rule_and_mae(self):
        df = _toy_results()
        m = calibrate_decision_table(df, features=["a", "b"])["meta"]
        err = df.assign(e=(df.k_hat - df.k_true).abs()).groupby("rule").e.mean()
        assert m["best_single_rule"] == err.idxmin()
        assert m["mae_best_single_rule"] == pytest.approx(err.min())
        assert m["mae_by_rule"] == pytest.approx(err.to_dict())
        assert sum(m["target_counts"].values()) == 40

    def test_ties_follow_priority(self):
        df = _toy_results()
        df["k_hat"] = 3                               # every rule is exact -> all tied
        t = calibrate_decision_table(df, features=["a", "b"])
        assert t["tree"]["rule"] == "onatski"         # single leaf, default priority
        t = calibrate_decision_table(df, features=["a", "b"], priority=["mp_market"])
        assert t["tree"]["rule"] == "mp_market"
        assert t["meta"]["train_accuracy"] == 1.0

    def test_matches_sklearn_on_noisy_multiclass(self):
        from sklearn.tree import DecisionTreeClassifier
        rng = np.random.default_rng(5)
        rows = []
        rules = ["onatski", "bai_ng", "parallel(null=circular)", "mp_market"]
        for i in range(150):
            f = rng.normal(size=3)
            errs = rng.integers(0, 4, size=len(rules)) + (f[0] > 0) * np.array([2, 0, 2, 2])
            for rule, e in zip(rules, errs, strict=True):
                rows.append({"design": f"g{i % 3}_{i}", "rep": i % 2, "rule": rule,
                             "k_hat": 2 + e, "k_true": 2, "f0": f[0], "f1": f[1], "f2": f[2]})
        df = pd.DataFrame(rows)
        table = calibrate_decision_table(df, features=["f0", "f1", "f2"], max_depth=4)
        assert table["meta"]["leave_one_family_out_accuracy"] is not None
        # independent refit with sklearn on the same targets -> same predictions on new points
        best = (df.assign(e=(df.k_hat - df.k_true).abs(),
                          pr=df.rule.map(lambda r: ["onatski", "bai_ng", "parallel",
                                                    "mp_market"].index(parse_rule_label(r)[0])))
                .sort_values(["e", "pr"], kind="mergesort").groupby(["design", "rep"]).head(1)
                .sort_values(["design", "rep"]))
        clf = DecisionTreeClassifier(max_depth=4, random_state=0)
        clf.fit(best[["f0", "f1", "f2"]].to_numpy(), best.rule.to_numpy())
        new = rng.normal(size=(300, 3))
        ours = [recommend(dict(zip(["f0", "f1", "f2"], x, strict=True)), table)["label"]
                for x in new]
        assert ours == list(clf.predict(new))
        assert table["meta"]["train_accuracy"] == pytest.approx(
            clf.score(best[["f0", "f1", "f2"]].to_numpy(), best.rule.to_numpy()))

    def test_nan_k_hat_and_dropped_panels(self):
        df = _toy_results(10)
        df.loc[df.rule == "onatski", "k_hat"] = np.nan        # onatski fails everywhere
        t = calibrate_decision_table(df, features=["a"])
        assert "onatski" not in t["meta"]["target_counts"]
        df["k_hat"] = np.where(df.design == "famA_d0", np.nan, df["k_hat"])
        with pytest.warns(UserWarning, match="dropped"):
            t = calibrate_decision_table(df, features=["a"])
        assert t["meta"]["n_dropped_panels"] == 1 and t["meta"]["n_samples"] == 9

    def test_default_features_family_column_and_kwargs_column(self):
        df = _toy_results(12).rename(columns={"a": "mean_abs_acf1", "b": "p_over_n"})
        df["family"] = np.where(df.index % 2 == 0, "x", "y")
        df["family"] = df.groupby(["design", "rep"])["family"].transform("first")
        df["kwargs"] = np.where(df.rule == "parallel(null=circular)",
                                json.dumps({"null": "block", "block_size": 3}), None)
        t = calibrate_decision_table(df)
        assert t["features"] == ["p_over_n", "mean_abs_acf1"]
        assert t["meta"]["families"] == sorted(df.family.unique())
        r = recommend({"mean_abs_acf1": 0.95, "p_over_n": 0.2}, t)
        assert r["rule"] == "parallel" and r["kwargs"] == {"null": "block", "block_size": 3}

    def test_single_family_has_no_lofo(self):
        df = _toy_results(10)
        df["design"] = "only_" + df["design"]
        assert calibrate_decision_table(df, features=["a"])["meta"][
            "leave_one_family_out_accuracy"] is None

    def test_input_errors(self):
        df = _toy_results(6)
        with pytest.raises(ValueError, match="lacks columns"):
            calibrate_decision_table(df.drop(columns="k_true"), features=["a"])
        with pytest.raises(ValueError, match="no diagnostics"):
            calibrate_decision_table(df)
        with pytest.raises(ValueError, match="not in sim_results"):
            calibrate_decision_table(df, features=["zzz"])
        bad = df.copy()
        bad.loc[0, "a"] = 99.0
        with pytest.raises(ValueError, match="vary within"):
            calibrate_decision_table(bad, features=["a"])
        bad = df.copy()
        bad["rule"] = bad["rule"].replace("mp_market", "unknown_rule")
        with pytest.raises(ValueError, match="unknown rule"):
            calibrate_decision_table(bad, features=["a"])
        with pytest.raises(ValueError, match="several rows"):
            calibrate_decision_table(pd.concat([df, df.iloc[:1]]), features=["a"])
        bad = df.copy()
        bad["a"] = np.nan
        with pytest.raises(ValueError, match="NaN in feature"):
            calibrate_decision_table(bad, features=["a"])


# ------------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pn_report():
    X = simulate(SimSpec(n=60, p=150, k=3, regime="pervasive"), random_state=11).X
    t0 = time.perf_counter()
    rep = report(X, random_state=0)                    # default n_boot = 200, all rules
    return X, rep, time.perf_counter() - t0


class TestReport:
    def test_runs_fast_on_small_p_gt_n(self, pn_report):
        _, rep, elapsed = pn_report
        assert isinstance(rep, Report)
        assert elapsed < 20.0
        assert rep.info["n_boot"] == 200
        assert rep.diagnostics["p_over_n"] == 2.5

    def test_tables_are_consistent(self, pn_report):
        X, rep, _ = pn_report
        assert set(RULES) <= set(rep.rules.index)
        assert rep.rules["recommended"].sum() == 1
        lab = rep.recommended["label"]
        assert rep.rules.loc[lab, "recommended"]
        assert list(rep.k_distribution.index) == list(rep.rules.index)
        freq = rep.k_distribution.to_numpy()
        np.testing.assert_allclose(freq.sum(axis=1), 1.0)
        assert (rep.rules["n_boot_ok"] + rep.rules["n_boot_failed"] == 200).all()
        # full-sample k equals the public API (non-stochastic rules)
        for name in ("onatski", "bai_ng", "gavish_donoho", "mp_market", "ahn_horenstein"):
            assert rep.rules.loc[name, "k"] == select_k(X, name).k
        # bootstrap summary agrees with the distribution
        for label, row in rep.rules.iterrows():
            dist = rep.k_distribution.loc[label]
            assert row["boot_mode"] == int(dist.idxmax())
            assert row["boot_p_same"] == pytest.approx(dist.get(row["k"], 0.0))
            assert row["boot_q05"] <= row["boot_median"] <= row["boot_q95"]
        assert sum(rep.recommended["k_distribution"].values()) == pytest.approx(1.0)

    def test_planted_truth_recovered(self, pn_report):
        _, rep, _ = pn_report
        assert rep.recommended["k"] == 3
        assert rep.recommended["k_boot_median"] == 3
        assert rep.recommended["k_boot_p_same"] >= 0.8
        assert rep.rules.loc["onatski", "k"] == 3 and rep.rules.loc["bai_ng", "k"] == 3

    def test_markdown_contains_everything(self, pn_report):
        _, rep, _ = pn_report
        md = rep.to_markdown()
        assert md.startswith("# kfactors report")
        for label in rep.rules.index:
            assert f"| {label} |" in md
        for name in RULES:
            assert f"| {name} |" in md
        for key in ("spike_share", "mean_abs_acf1", "mean_acf1_sq", "var_dispersion",
                    "mean_excess_kurtosis", "p_over_n"):
            assert key in md
        assert rep.recommended["reason"] in md
        assert f"k = {rep.recommended['k']}" in md
        assert "distinct rows only" in md
        # every markdown table row has the same number of cells as its header
        for block in md.split("\n\n"):
            lines = [x for x in block.splitlines() if x.startswith("|")]
            if lines:
                assert len({x.count(" | ") for x in lines if not x.startswith("|---")}) == 1

    def test_plot(self, pn_report):
        import matplotlib.pyplot as plt
        _, rep, _ = pn_report
        axes = rep.plot()
        assert len(axes) == 2
        assert [t.get_text() for t in axes[0].get_yticklabels()] == list(rep.rules.index)
        plt.close(axes[0].figure)
        fig, given = plt.subplots(2, 1)
        out = rep.plot(axes=given)
        assert out[0] is given[0] and out[1] is given[1]
        with pytest.raises(ValueError):
            rep.plot(axes=[given[0]])
        plt.close(fig)

    def test_deterministic(self):
        X = _sim(k=2, n=40, p=60, seed=4)
        a = report(X, n_boot=15, random_state=7)
        b = report(X, n_boot=15, random_state=7)
        pd.testing.assert_frame_equal(a.rules, b.rules)
        pd.testing.assert_frame_equal(a.k_distribution, b.k_distribution)
        assert a.recommended == b.recommended

    def test_distinct_rows_keep_null_spectrum(self):
        """Pure noise: duplicated rows fake factors for RMT / permutation rules."""
        X = _sim(k=0, n=200, p=100, seed=1)
        rules = ["onatski", "gavish_donoho", "mp_market", "parallel"]
        uniq = report(X, rules=rules, n_boot=20, random_state=0)
        dup = report(X, rules=rules, n_boot=20, random_state=0, boot_unique_rows=False)
        assert (uniq.rules.loc[rules, "k"] == 0).all()
        assert (uniq.rules.loc[rules, "boot_median"] == 0).all()
        assert 0.55 < uniq.info["boot_mean_rows"] / 200 < 0.75
        assert dup.info["boot_mean_rows"] == 200
        assert dup.rules.loc["gavish_donoho", "boot_median"] >= 2
        assert dup.rules.loc["parallel", "boot_median"] >= 10

    def test_rule_subset_adds_recommendation_and_custom_table(self):
        X = _sim(k=2, n=50, p=30, seed=2)
        table = {"kind": "custom", "tree": {"rule": "bai_ng", "kwargs": {"criterion": "IC_p1"}}}
        rep = report(X, rules=["onatski", ("parallel", {"null": "circular"})], n_boot=10,
                     table=table)
        assert list(rep.rules.index) == ["onatski", "parallel(null=circular)",
                                         "bai_ng(criterion=IC_p1)"]
        assert rep.rules.loc["bai_ng(criterion=IC_p1)", "recommended"]
        assert rep.recommended["k"] == select_k(X, "bai_ng", criterion="IC_p1").k
        assert rep.recommended["table_kind"] == "custom"
        assert json.loads(rep.rules.loc["parallel(null=circular)", "kwargs"]) == {
            "null": "circular"}

    def test_no_bootstrap(self):
        rep = report(_sim(k=1, n=30, p=12), rules=["onatski", "bai_ng"], n_boot=0)
        assert rep.rules["boot_median"].isna().all()
        assert "Bootstrap" in rep.to_markdown()

    def test_bootstrap_failures_are_counted_and_warned(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 8))
        X[:, 0] = 0.0
        X[5, 0] = 1.0                       # a resample without row 5 has a constant column
        with pytest.warns(UserWarning, match="failed"):
            rep = report(X, rules=["bai_ng"], n_boot=30, random_state=0)
        assert rep.rules["n_boot_failed"].sum() > 0
        ok_and_failed = rep.rules["n_boot_ok"] + rep.rules["n_boot_failed"]
        assert (ok_and_failed == 30).all()

    def test_input_validation(self):
        X = _sim(k=1, n=30, p=10)
        with pytest.raises(ValueError, match="n_boot"):
            report(X, n_boot=-1)
        with pytest.raises(ValueError, match="boot_n_perm"):
            report(X, boot_n_perm=0)
        with pytest.raises(ValueError, match="unknown rule"):
            report(X, rules=["nope"], n_boot=0)
        with pytest.raises(TypeError):
            report(X, rules=[3], n_boot=0)
        with pytest.raises(ValueError):
            report(np.full((10, 3), np.nan))


def test_market_branch_does_not_overcount_with_dispersed_idiosyncratic_variance():
    """Regression (2026-11-03): branches 2/3 used mp_market, which returned 3-6 on a one-factor
    market panel with dispersed betas and idiosyncratic volatilities; onatski gives 1."""
    rng = np.random.default_rng(0)
    n, p = 250, 300
    X = (np.outer(rng.standard_normal(n), rng.uniform(0.5, 1.5, p))
         + rng.standard_normal((n, p)) * rng.lognormal(0.0, 0.5, p))
    rec = recommend(diagnose(X))
    assert rec["path"][1]["feature"] == "gap_ratio_1" and rec["path"][1]["taken"] == "then"
    assert rec["label"] == "onatski"
    assert select_k(X, rec["rule"], **rec["kwargs"]).k == 1
    assert select_k(X, "mp_market").k > 1          # the over-count the routing avoids


def test_parallel_branches_devolatilize_common_volatility():
    """Branch 4 (p > n) with a volatility path shared by all series: the plain permutation null
    over-counts pure noise, the routed parallel(devolatilize=true) does not."""
    rng = np.random.default_rng(0)
    n, p = 250, 500
    h = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.98 * h[t - 1] + np.sqrt(1 - 0.98 ** 2) * 0.5 * rng.standard_normal()
    X = rng.standard_normal((n, p)) * np.exp(h)[:, None]
    rec = recommend(diagnose(X))
    assert rec["label"] == "parallel(devolatilize=true)"
    assert select_k(X, rec["rule"], random_state=0, n_perm=20, **rec["kwargs"]).k == 0
    assert select_k(X, "parallel", random_state=0, n_perm=20).k > 5
