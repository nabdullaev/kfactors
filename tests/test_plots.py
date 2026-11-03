"""Tests for kfactors.plots (Agg backend, files written to tmp_path)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from kfactors.plots import (
    mp_median,
    plot_heatmap,
    plot_k_curve,
    plot_spectrum,
    spectrum_edges,
)
from kfactors.sim import SimSpec, simulate
from kfactors.utils import eigvals_desc, preprocess


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _spectrum(spec, seed=0, standardize=False):
    d = simulate(spec, seed)
    Z, _, _ = preprocess(d.X, standardize=standardize)
    return eigvals_desc(Z)


# ------------------------------------------------------------------------- MP helper values

def test_mp_median_closed_form_y1():
    """For y = 1, sqrt(MP) is the quarter-circle law: F(s) = (s sqrt(4-s^2)/2 + 2 asin(s/2))/pi."""
    def F(s):
        return (s * np.sqrt(4 - s * s) / 2 + 2 * np.arcsin(s / 2)) / np.pi - 0.5

    m = brentq(F, 0, 2, xtol=1e-15)
    assert mp_median(1.0) == pytest.approx(m * m, abs=1e-9)
    assert mp_median(1.0) == pytest.approx(0.6528, abs=1e-4)   # Gavish & Donoho (2014)


@pytest.mark.parametrize("y", [0.05, 0.25, 0.5, 0.8, 1.0])
def test_mp_median_matches_optht(y):
    from optht.optht import _median_marcenko_pastur  # independent implementation

    assert mp_median(y) == pytest.approx(_median_marcenko_pastur(y), abs=5e-4)


def test_mp_median_monte_carlo():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 500))
    ev = np.linalg.eigvalsh(X.T @ X / 2000)
    assert np.median(ev) == pytest.approx(mp_median(0.25), rel=0.01)
    with pytest.raises(ValueError):
        mp_median(1.5)


@pytest.mark.parametrize("n, p", [(250, 100), (250, 625), (200, 200)])
def test_spectrum_edges_pure_noise(n, p):
    ev = _spectrum(SimSpec(n=n, p=p, k=0), seed=1)
    e = spectrum_edges(ev, n, p)
    assert e["sigma2_median"] == pytest.approx(1.0, abs=0.06)
    # Laloux removes lambda_1 ~ MP edge from the trace: sigma2 ~ 1 - (1 + sqrt(p/n))^2 / p
    assert e["sigma2_laloux"] == pytest.approx(1 - (1 + np.sqrt(p / n)) ** 2 / p, abs=0.03)
    assert e["edge_mp"] == pytest.approx((1 + np.sqrt(p / n)) ** 2, rel=0.07)
    assert e["n_above_mp"] <= 1 and e["n_above_laloux"] <= 1


def test_spectrum_edges_recover_strong_spikes():
    n, p = 250, 625
    ev = _spectrum(SimSpec(n=n, p=p, k=3, regime="spiked"), seed=2)
    e = spectrum_edges(ev, n, p)
    assert e["n_above_mp"] == 3 and e["n_above_laloux"] == 3


def test_spectrum_edges_laloux_market_adjustment():
    """A dominant market factor inflates the trace-based noise estimate unless removed."""
    n, p = 500, 200
    ev = _spectrum(SimSpec(n=n, p=p, k=4, market=1.0, strengths=(3, 2, 1.5)), seed=3,
                   standardize=True)
    e = spectrum_edges(ev, n, p)
    naive_trace = ev.sum() / p          # = 1 for standardised data
    assert e["sigma2_laloux"] == pytest.approx(naive_trace - ev[0] / p)
    assert e["sigma2_laloux"] < 0.7 * naive_trace


def test_spectrum_edges_validation():
    ev = _spectrum(SimSpec(n=50, p=30, k=1))
    with pytest.raises(ValueError, match="full spectrum"):
        spectrum_edges(ev[:5], 50, 30)
    with pytest.raises(ValueError):
        spectrum_edges(np.r_[ev[:-1], np.nan], 50, 30)
    with pytest.raises(ValueError):
        spectrum_edges(np.zeros(30), 50, 30)
    # unsorted input is accepted
    e1 = spectrum_edges(ev, 50, 30)
    e2 = spectrum_edges(ev[::-1], 50, 30)
    assert e1 == e2


# ----------------------------------------------------------------------------- plot_spectrum

def test_plot_spectrum_writes_file_and_legend_counts(tmp_path):
    n, p = 250, 625
    ev = _spectrum(SimSpec(n=n, p=p, k=3, regime="spiked"), seed=2)
    ax = plot_spectrum(ev, n, p)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.get_yscale() == "log"
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(t.startswith("MP edge") and t.endswith("3 above") for t in texts)
    assert any(t.startswith("Laloux") and t.endswith("3 above") for t in texts)
    assert len(ax.lines[0].get_xdata()) == 50            # default top = 50
    out = tmp_path / "spectrum.png"
    ax.figure.savefig(out)
    assert out.stat().st_size > 1000


def test_plot_spectrum_options(tmp_path):
    n, p = 100, 250                    # p > n: trailing zeros must not break the log plot
    ev = _spectrum(SimSpec(n=n, p=p, k=2), seed=4)
    fig, ax = plt.subplots()
    out_ax = plot_spectrum(ev, n, p, ax=ax, log=False, market_adjusted=False, top=None,
                           title="custom")
    assert out_ax is ax
    assert ax.get_title() == "custom" and ax.get_yscale() == "linear"
    assert len(ax.get_legend().get_texts()) == 2         # eigenvalues + MP edge only
    assert len(ax.lines[0].get_xdata()) == p
    ax2 = plot_spectrum(ev, n, p, top=None)              # log: zeros dropped
    assert len(ax2.lines[0].get_xdata()) == n - 1
    fig.savefig(tmp_path / "s.pdf")
    assert (tmp_path / "s.pdf").exists()
    with pytest.raises(ValueError):
        plot_spectrum(ev, n, p, top=0)


# ------------------------------------------------------------------------------ plot_heatmap

def _long_frame():
    rows = []
    for i, rule in enumerate(["mp", "onatski", "bai_ng"]):
        for j, design in enumerate(["d1", "d2", "d3", "d4"]):
            rows.append({"rule": rule, "design": design, "accuracy": 0.1 * i + 0.2 * j})
    return pd.DataFrame(rows)


def test_plot_heatmap_values_annotations_colorbar(tmp_path):
    df = _long_frame()
    ax = plot_heatmap(df)
    im = ax.get_images()[0]
    M = np.asarray(im.get_array())
    expected = df.pivot(index="rule", columns="design", values="accuracy").loc[
        ["mp", "onatski", "bai_ng"], ["d1", "d2", "d3", "d4"]].to_numpy()
    np.testing.assert_allclose(M, expected)              # order of first appearance kept
    assert [t.get_text() for t in ax.get_yticklabels()] == ["mp", "onatski", "bai_ng"]
    assert len(ax.texts) == 12
    assert {t.get_text() for t in ax.texts} >= {"0.00", "0.80"}
    assert len(ax.figure.axes) == 2                      # heatmap + colorbar
    assert ax.figure.axes[1].get_ylabel() == "accuracy"
    out = tmp_path / "heat.png"
    ax.figure.savefig(out)
    assert out.stat().st_size > 1000


def test_plot_heatmap_missing_duplicates_and_custom_value(tmp_path):
    df = pd.DataFrame({
        "rule": ["a", "a", "a", "b"],
        "design": ["x", "x", "y", "x"],
        "rmse": [1.0, 3.0, 5.0, 4.0],
    })
    ax = plot_heatmap(df, value="rmse", title="RMSE", annotate=True, fmt=".1f")
    M = np.ma.filled(ax.get_images()[0].get_array(), np.nan)
    assert M[0, 0] == pytest.approx(2.0)                 # duplicates averaged
    assert np.isnan(M[1, 1])                             # missing cell masked
    assert len(ax.texts) == 3
    assert ax.get_title() == "RMSE"
    ax.figure.savefig(tmp_path / "heat2.png")
    ax_none = plot_heatmap(df, value="rmse", annotate=False)
    assert len(ax_none.texts) == 0


def test_plot_heatmap_validation():
    with pytest.raises(ValueError, match="missing columns"):
        plot_heatmap(pd.DataFrame({"rule": ["a"], "design": ["b"]}))
    with pytest.raises(ValueError):
        plot_heatmap(pd.DataFrame(columns=["rule", "design", "accuracy"]))
    with pytest.raises(TypeError):
        plot_heatmap([("a", "b", 1.0)])


# ------------------------------------------------------------------------------ plot_k_curve

def _kcurve(**over):
    from kfactors.evaluate import KCurve

    ks = np.arange(1, 11)
    metric = 1.0 - 0.02 * (ks - 5) ** 2
    curve = pd.DataFrame({"metric": metric, "low": metric - 0.2, "high": metric + 0.2},
                         index=pd.Index(ks, name="k"))
    rules = pd.DataFrame({
        "k_mean": [4.2, 9.0, 2.0, np.nan],
        "k_median": [4.0, 9.0, 2.0, np.nan],
        "metric": [0.98, 0.68, 0.82, np.nan],
        "in_set": [True, False, False, False],
    }, index=pd.Index(["onatski", "mp", "bai_ng", "broken"], name="rule"))
    kw = {"curve": curve, "rules": rules, "oracle_k": 5, "oracle_set": [4, 5, 6, 8], "metric": "sharpe_net",
              "results": {}, "meta": {}}
    kw.update(over)
    return KCurve(**kw)


def test_plot_k_curve_elements(tmp_path):
    kc = _kcurve()
    ax = plot_k_curve(kc)
    assert ax.get_ylabel() == "sharpe_net"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "bootstrap CI" in labels and "oracle set" in labels and "oracle k = 5" in labels
    texts = [t.get_text().strip() for t in ax.texts]
    assert texts == ["bai_ng (k=2)", "onatski (k=4)", "mp (k=9)"]   # NaN rule skipped
    vlines = [ln for ln in ax.lines if np.allclose(ln.get_xdata(), ln.get_xdata()[0])
              and len(ln.get_xdata()) == 2]
    assert sorted(float(ln.get_xdata()[0]) for ln in vlines) == [2.0, 4.0, 9.0]
    styles = {float(ln.get_xdata()[0]): ln.get_linestyle() for ln in vlines}
    assert styles[4.0] == "-" and styles[9.0] == "--"
    # two contiguous runs in the oracle set: {4, 5, 6} and {8}
    gold = matplotlib.colors.to_rgb("gold")
    spans = [pt for pt in ax.patches if np.allclose(pt.get_facecolor()[:3], gold)]
    assert len(spans) == 2
    extents = sorted((pt.get_x(), pt.get_x() + pt.get_width()) for pt in spans)
    np.testing.assert_allclose(extents, [(3.5, 6.5), (7.5, 8.5)])
    out = tmp_path / "kcurve.png"
    ax.figure.savefig(out)
    assert out.stat().st_size > 1000


def test_plot_k_curve_groups_rules_with_same_k():
    rules = pd.DataFrame({"k_median": [6.0, 6.0, 3.0], "in_set": [True, True, False]},
                         index=["onatski", "ah", "mp"])
    ax = plot_k_curve(_kcurve(rules=rules))
    assert sorted(t.get_text().strip() for t in ax.texts) == ["mp (k=3)", "onatski, ah (k=6)"]


def test_plot_k_curve_minimal_and_method(tmp_path):
    curve = pd.DataFrame({"metric": [0.1, 0.3, 0.2]}, index=[1, 2, 3])
    kc = _kcurve(curve=curve, rules=pd.DataFrame(), oracle_set=[], oracle_k=2,
                 metric="min_var")
    fig, ax = plt.subplots()
    out = kc.plot(ax=ax)                                  # KCurve.plot delegates
    assert out is ax
    assert len(ax.texts) == 0
    assert ax.get_title() == "min_var vs number of factors k"
    fig.savefig(tmp_path / "kc_min.png")


def test_plot_k_curve_validation():
    with pytest.raises(TypeError):
        plot_k_curve(object())
    with pytest.raises(ValueError):
        plot_k_curve(_kcurve(curve=pd.DataFrame({"x": [1.0]})))
