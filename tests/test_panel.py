"""Tests for kfactors.data.panel.Panel (validation, window, subset, periods_per_year)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kfactors.data.panel import Panel


def make_panel(n=10, p=4, seed=0, freq="D", meta=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D" if freq == "D" else "ME")
    R = pd.DataFrame(rng.normal(0, 0.01, (n, p)), index=idx, columns=[f"a{j}" for j in range(p)])
    R.iloc[:3, 1] = np.nan          # a1 listed at row 3
    R.iloc[6, 2] = np.nan           # a2 has a hole at row 6
    T = R.notna()
    T.iloc[8:, 3] = False           # a3 leaves the universe at row 8 (still has returns)
    return Panel(R, T, name="toy", freq=freq, meta=dict(meta or {}))


# --------------------------------------------------------------------------- validation
def test_valid_panel_roundtrip():
    P = make_panel()
    assert P.shape == (10, 4)
    assert P.returns.dtypes.eq(np.float64).all()
    assert "toy" in repr(P) and "n=10" in repr(P)


def test_integer_returns_are_cast_to_float():
    idx = pd.date_range("2021-01-01", periods=3)
    R = pd.DataFrame({"x": [0, 1, 0], "y": [1, 0, 0]}, index=idx)
    P = Panel(R, R.notna())
    assert P.returns.dtypes.eq(np.float64).all()


@pytest.mark.parametrize("mutate, exc, match", [
    (lambda R, T: (R.reset_index(drop=True), T.reset_index(drop=True)), TypeError,
     "DatetimeIndex"),
    (lambda R, T: (R.iloc[::-1], T.iloc[::-1]), ValueError, "sorted"),
    (lambda R, T: (pd.concat([R, R.iloc[[0]]]).sort_index(),
                   pd.concat([T, T.iloc[[0]]]).sort_index()), ValueError, "duplicated dates"),
    (lambda R, T: (R, T.iloc[:, :3]), ValueError, "shape"),
    (lambda R, T: (R, T.set_axis(["a0", "a1", "a2", "zz"], axis=1)), ValueError, "same index"),
    (lambda R, T: (R, T.astype(int)), TypeError, "boolean"),
    (lambda R, T: (R, T | True), ValueError, "tradable is True where returns is NaN"),
    (lambda R, T: (R.replace(R.iloc[0, 0], np.inf), T), ValueError, "inf"),
    (lambda R, T: (R.set_axis(["a", "a", "b", "c"], axis=1),
                   T.set_axis(["a", "a", "b", "c"], axis=1)), ValueError, "duplicated columns"),
    (lambda R, T: (R.assign(a0="x"), T), TypeError, "numeric"),
    (lambda R, T: (R.values, T), TypeError, "DataFrame"),
])
def test_validation_rejects_bad_inputs(mutate, exc, match):
    P = make_panel()
    R, T = mutate(P.returns.copy(), P.tradable.copy())
    with pytest.raises(exc, match=match):
        Panel(R, T)


def test_bad_freq_and_meta():
    P = make_panel()
    with pytest.raises(ValueError, match="freq"):
        Panel(P.returns, P.tradable, freq="")
    with pytest.raises(TypeError, match="meta"):
        Panel(P.returns, P.tradable, meta=[1])


def test_empty_columns_allowed():
    idx = pd.date_range("2021-01-01", periods=3)
    R = pd.DataFrame(index=idx)
    P = Panel(R, R.astype(bool))
    assert P.shape == (3, 0)


# --------------------------------------------------------------------------- periods per year
def test_periods_per_year_defaults_and_override():
    assert make_panel(freq="D").periods_per_year() == 252
    assert make_panel(freq="M").periods_per_year() == 12
    assert make_panel(meta={"periods_per_year": 365}).periods_per_year() == 365
    P = make_panel()
    P.freq = "W"
    with pytest.raises(ValueError, match="periods_per_year"):
        P.periods_per_year()
    P.meta["periods_per_year"] = 0
    with pytest.raises(ValueError, match="positive"):
        P.periods_per_year()


# --------------------------------------------------------------------------- window
def test_window_rows_and_complete_columns():
    P = make_panel()
    W = P.window(end=5, length=4)                  # rows 2..5
    assert list(W.returns.index) == list(P.returns.index[2:6])
    # a1 has NaN at row 2 -> dropped; a0, a2, a3 complete and tradable at row 5
    assert list(W.returns.columns) == ["a0", "a2", "a3"]
    assert not W.returns.isna().any().any()
    np.testing.assert_array_equal(W.returns.values, P.returns.iloc[2:6][["a0", "a2", "a3"]].values)
    assert W.name == "toy" and W.freq == "D"


def test_window_excludes_untradable_at_end_and_holes():
    P = make_panel()
    W = P.window(end=9, length=3)                  # rows 7..9: a3 not tradable at row 9
    assert list(W.returns.columns) == ["a0", "a1", "a2"]
    W = P.window(end=7, length=3)                  # rows 5..7: a2 has a hole at row 6
    assert list(W.returns.columns) == ["a0", "a1", "a3"]


def test_window_incomplete_keeps_everything():
    P = make_panel()
    W = P.window(end=5, length=6, complete=False)
    assert W.shape == (6, 4)
    assert W.returns.isna().sum().sum() == 3


def test_window_bounds():
    P = make_panel()
    with pytest.raises(IndexError, match="before row 0"):
        P.window(end=2, length=4)
    P.window(end=3, length=4)                      # starts exactly at row 0
    with pytest.raises(IndexError):
        P.window(end=10, length=1)
    with pytest.raises(ValueError):
        P.window(end=5, length=0)
    W = P.window(end=-1, length=2)                 # negative end = from the last row
    assert W.returns.index[-1] == P.returns.index[-1]
    with pytest.raises(TypeError):
        P.window(end=2.5, length=2)


def test_window_has_no_look_ahead():
    """Corrupting every row after `end` must not change the window."""
    P = make_panel(n=30, p=6, seed=3)
    for end in range(5, 29):
        R2 = P.returns.copy()
        R2.iloc[end + 1:] = 99.0
        T2 = P.tradable.copy()
        T2.iloc[end + 1:] = False
        P2 = Panel(R2, T2)
        a, b = P.window(end, 6), P2.window(end, 6)
        pd.testing.assert_frame_equal(a.returns, b.returns)
        pd.testing.assert_frame_equal(a.tradable, b.tradable)


def test_window_meta_is_copied():
    P = make_panel(meta={"source": "x"})
    W = P.window(5, 3)
    W.meta["source"] = "changed"
    assert P.meta["source"] == "x"


# --------------------------------------------------------------------------- subset
def test_subset_dates_inclusive_and_assets():
    P = make_panel()
    S = P.subset("2020-01-03", "2020-01-05", assets=["a3", "a0"])
    assert list(S.returns.index) == list(pd.date_range("2020-01-03", "2020-01-05"))
    assert list(S.returns.columns) == ["a3", "a0"]
    pd.testing.assert_frame_equal(S.tradable, P.tradable.loc["2020-01-03":"2020-01-05",
                                                              ["a3", "a0"]])
    assert P.subset().shape == P.shape
    assert P.subset(start="2020-01-09").shape == (2, 4)
    assert P.subset(end="2019-01-01").shape == (0, 4)


def test_subset_errors():
    P = make_panel()
    with pytest.raises(KeyError, match="zz"):
        P.subset(assets=["a0", "zz"])
    with pytest.raises(ValueError, match="after"):
        P.subset("2020-02-01", "2020-01-01")


def test_subset_tz_aware_index():
    P = make_panel()
    R = P.returns.tz_localize("UTC")
    T = P.tradable.tz_localize("UTC")
    Q = Panel(R, T)
    assert Q.subset("2020-01-02", "2020-01-03").shape == (2, 4)
