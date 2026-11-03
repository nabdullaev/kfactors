"""Tests for the benchmark suite (kfactors.bench): fully offline (synthetic panels only)."""
from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from kfactors.bench import (
    Cache,
    derive_seed,
    load_config,
    parse_k_grid,
    resolve_rules,
    stable_hash,
)
from kfactors.bench.__main__ import main
from kfactors.bench.leaderboard import build_leaderboard, read_csv_str, sim_scores
from kfactors.bench.real_bench import (
    LabeledRule,
    latest_window,
    rolling_k,
    synthetic_panel,
    window_universe,
)
from kfactors.bench.sim_bench import accuracy_table, build_designs
from kfactors.engine import WalkForward, WalkForwardConfig
from kfactors.rules import select_k
from kfactors.strategies import TangencyPCA

REPO = Path(__file__).resolve().parents[1]
CI = REPO / "configs" / "ci.yaml"
DEFAULT = REPO / "configs" / "default.yaml"


def _ci_dict() -> dict:
    with open(CI, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _write_cfg(tmp_path: Path, cfg: dict, name: str = "cfg.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _raw_no_time(out: Path) -> pd.DataFrame:
    return read_csv_str(out / "sim" / "raw.csv").drop(columns=["seconds"])


# ------------------------------------------------------------------------------------------
# end-to-end CLI run on configs/ci.yaml (module-scoped: run once)
# ------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ci_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("ci_out")
    code = main(["all", "--config", str(CI), "--out", str(out)])
    return code, out


def test_ci_exit_code_and_files(ci_run):
    code, out = ci_run
    assert code == 0
    expected = [
        "leaderboard.csv", "leaderboard.md", "bench.log",
        "sim/raw.csv", "sim/accuracy.csv", "sim/heatmap.png", "sim/heatmap_mae.png",
        "sim/timings.csv", "sim/designs.csv", "sim/run_info.json",
        "real/status.csv", "real/run_info.json",
    ]
    pdir = "real/synthetic_monthly/"
    expected += [pdir + f for f in (
        "panel.json", "spectrum.png", "spectrum.json", "rolling_k.csv", "rolling_k.png",
        "rolling_k_summary.csv", "rolling_k_info.json", "report.md", "report.json",
        "report.png", "report_rules.csv", "report_k_distribution.csv")]
    edir = pdir + "experiments/tangency_w60/"
    expected += [edir + f for f in (
        "kcurve.csv", "curve.csv", "rules.csv", "k_used.csv", "returns.csv",
        "specs_summary.csv", "summary.json", "kcurve.png")]
    expected.append(pdir + "experiments/minvar_w60/summary.json")
    missing = [f for f in expected if not (out / f).is_file() or (out / f).stat().st_size == 0]
    assert not missing, missing
    # figures are real PNGs
    for f in expected:
        if f.endswith(".png"):
            assert (out / f).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_ci_sim_raw_and_accuracy(ci_run):
    _, out = ci_run
    cfg = load_config(CI)
    raw = read_csv_str(out / "sim" / "raw.csv")
    labels = [r[0] for r in resolve_rules(cfg["sim"]["rules"])]
    n_designs, reps = len(cfg["sim"]["grid"]), cfg["sim"]["reps"]
    assert len(raw) == n_designs * reps * len(labels)
    assert raw.groupby(["design", "rep", "rule"]).size().max() == 1
    assert set(raw["rule"]) == set(labels)
    assert (raw["error"].isna() | (raw["error"] == "")).all()
    assert raw["k_hat"].between(0, cfg["sim"]["kmax"]).all()
    # diagnostics are constant within a panel and k_true matches the design
    diag = raw.groupby(["design", "rep"])[["spike_share", "p_over_n"]].nunique()
    assert (diag == 1).all().all()
    assert (raw.loc[raw["design"].str.startswith("null"), "k_true"] == 0).all()
    np.testing.assert_allclose(raw["p_over_n"], raw["p"] / raw["n"])
    acc = read_csv_str(out / "sim" / "accuracy.csv")
    assert len(acc) == n_designs * len(labels)
    # recompute exact % and MAE for one cell independently
    g = raw[(raw["rule"] == "bai_ng") & (raw["design"] == "spiked_t5_n60_p120")]
    row = acc[(acc["rule"] == "bai_ng") & (acc["design"] == "spiked_t5_n60_p120")].iloc[0]
    assert row["exact_pct"] == pytest.approx(100 * np.mean(g["k_hat"] == g["k_true"]))
    assert row["mae"] == pytest.approx(np.mean(np.abs(g["k_hat"] - g["k_true"])))
    # planted truth: a strong pervasive 3-factor design is recovered by onatski
    on = acc[(acc["rule"] == "onatski") & (acc["design"] == "pervasive_gauss_n80_p40")]
    assert on["exact_pct"].iloc[0] == 100.0


def test_ci_real_outputs_consistent(ci_run):
    _, out = ci_run
    pdir = out / "real" / "synthetic_monthly"
    rk = pd.read_csv(pdir / "rolling_k.csv", index_col=0, parse_dates=True)
    assert list(rk.columns[:1]) == ["n_assets"]
    # planted k=3: onatski finds it in (almost) every window
    assert (rk["onatski"] == 3).mean() >= 0.9
    # late-listed assets join the universe once they have a complete window
    assert rk["n_assets"].iloc[0] == 26 and rk["n_assets"].iloc[-1] == 30
    s = json.loads((pdir / "experiments" / "tangency_w60" / "summary.json").read_text())
    assert s["status"] == "ok"
    assert s["oracle_k"] in s["k_grid"] and s["oracle_k"] in s["oracle_set"]
    assert 3 in s["oracle_set"]
    rules = read_csv_str(pdir / "experiments" / "tangency_w60" / "rules.csv").set_index("rule")
    np.testing.assert_allclose(rules["dist_to_oracle"], (rules["k_median"] - s["oracle_k"]).abs())
    # k_used of a rule in the k-curve run equals the rule's rolling k_t on the same dates
    ku = pd.read_csv(pdir / "experiments" / "tangency_w60" / "k_used.csv", index_col=0,
                     parse_dates=True)
    assert rules.loc["onatski", "k_median"] == 3
    assert set(ku.columns) == set(rules.index)
    sm = json.loads((pdir / "experiments" / "minvar_w60" / "summary.json").read_text())
    assert sm["status"] in ("skipped", "ok")
    rep = json.loads((pdir / "report.json").read_text())
    assert rep["n"] == 60 and rep["p"] == 30
    assert "# kfactors report" in (pdir / "report.md").read_text()


def test_ci_leaderboard_rows_sane(ci_run):
    _, out = ci_run
    board = read_csv_str(out / "leaderboard.csv")
    assert set(board["section"]) == {"sim", "real", "real_overall"}
    cfg = load_config(CI)
    sim_labels = [r[0] for r in resolve_rules(cfg["sim"]["rules"])]
    ov = board[(board["section"] == "sim") & (board["group"] == "overall")]
    assert sorted(ov["rule"]) == sorted(sim_labels)
    assert ov["exact_pct"].between(0, 100).all() and (ov["mae"] >= 0).all()
    assert sorted(ov["rank"].astype(int)) == list(range(1, len(sim_labels) + 1))
    n_panels = len(cfg["sim"]["grid"]) * cfg["sim"]["reps"]
    assert (ov["n"] == n_panels).all()
    for grp in ("family=null", "family=pervasive", "family=spiked", "noise=t5"):
        assert (board["group"] == grp).any(), grp
    real = board[(board["section"] == "real") & (board["experiment"] == "tangency_w60")]
    real_labels = [r[0] for r in resolve_rules(cfg["real"]["rules"])]
    assert sorted(real["rule"]) == sorted(real_labels)
    assert (real["status"] == "ok").all()
    assert real["oracle_k"].nunique() == 1
    oset = [int(k) for k in str(real["oracle_set"].iloc[0]).split(",")]
    for _, r in real.iterrows():
        assert r["dist_to_oracle"] == pytest.approx(abs(r["k_median"] - r["oracle_k"]))
        assert bool(r["in_set"]) == (int(np.floor(r["k_median"] + 0.5)) in oset)
    mv = board[board["experiment"] == "minvar_w60"]
    assert len(mv) >= 1 and set(mv["status"]) <= {"skipped", "ok"}
    md = (out / "leaderboard.md").read_text()
    assert "## Simulations" in md and "## Real data" in md and "tangency_w60" in md


# ------------------------------------------------------------------------------------------
# determinism, cache, parallelism
# ------------------------------------------------------------------------------------------
def test_sim_deterministic_cache_and_n_jobs(tmp_path, ci_run):
    _, out1 = ci_run
    ref = _raw_no_time(out1)
    # same out dir: everything comes from the cache, outputs identical
    assert main(["sim", "--config", str(CI), "--out", str(out1)]) == 0
    pd.testing.assert_frame_equal(_raw_no_time(out1), ref)
    timings = pd.read_csv(out1 / "sim" / "timings.csv")
    assert (timings["n_cached"] == timings["reps"]).all()
    # fresh dir, no cache, two processes: identical numbers
    out2 = tmp_path / "par"
    assert main(["sim", "--config", str(CI), "--out", str(out2), "--no-cache",
                 "--n-jobs", "2"]) == 0
    pd.testing.assert_frame_equal(_raw_no_time(out2), ref)
    assert not (out2 / "cache").exists()


def test_sim_reps_and_designs_override(tmp_path):
    out = tmp_path / "o"
    assert main(["sim", "--config", str(CI), "--out", str(out), "--reps", "1",
                 "--designs", "null_*"]) == 0
    raw = read_csv_str(out / "sim" / "raw.csv")
    assert set(raw["design"]) == {"null_gauss_n80_p40"} and set(raw["rep"]) == {0}
    # replication seeds do not depend on the filter / number of reps
    assert raw["seed"].iloc[0] == derive_seed(0, "sim", "null_gauss_n80_p40", 0)


def test_real_deterministic(tmp_path, ci_run):
    _, out1 = ci_run
    out2 = tmp_path / "r"
    assert main(["real", "--config", str(CI), "--out", str(out2), "--no-cache"]) == 0
    sub = Path("real/synthetic_monthly")
    for f in ("rolling_k.csv", "experiments/tangency_w60/kcurve.csv",
              "experiments/tangency_w60/returns.csv", "report_rules.csv"):
        a = (out1 / sub / f).read_text()
        b = (out2 / sub / f).read_text()
        assert a == b, f


# ------------------------------------------------------------------------------------------
# calibration never touches the packaged table
# ------------------------------------------------------------------------------------------
def test_calibrate_writes_table_not_resource(tmp_path, capsys):
    from kfactors.report import recommend, validate_decision_table

    res = resources.files("kfactors") / "resources" / "decision_table.json"
    before = hashlib.sha256(res.read_bytes()).hexdigest()
    out = tmp_path / "cal"
    assert main(["sim", "--config", str(CI), "--out", str(out), "--calibrate"]) == 0
    assert hashlib.sha256(res.read_bytes()).hexdigest() == before
    table = json.loads((out / "sim" / "decision_table.json").read_text())
    validate_decision_table(table)
    assert table["kind"] == "calibrated"
    raw = read_csv_str(out / "sim" / "raw.csv")
    diag = raw.iloc[0][list(table["features"])].to_dict()
    assert recommend(diag, table)["rule"] in {r[1] for r in resolve_rules(
        load_config(CI)["sim"]["rules"])}
    printed = capsys.readouterr().out
    assert "NOT changed" in printed and "cp " in printed


# ------------------------------------------------------------------------------------------
# skipped / failed strategies and config errors
# ------------------------------------------------------------------------------------------
def _real_only(cfg: dict, experiments: list[dict]) -> dict:
    cfg = dict(cfg)
    cfg.pop("sim")
    cfg["real"] = dict(cfg["real"])
    panel = dict(cfg["real"]["panels"][0])
    panel["experiments"] = experiments
    panel["rolling"] = {"enabled": False}
    panel["report"] = {"enabled": False}
    panel["spectrum"] = {"enabled": False}
    cfg["real"]["panels"] = [panel]
    return cfg


def test_not_implemented_strategy_is_skipped(tmp_path):
    cfg = _real_only(_ci_dict(), [{"name": "stub", "strategy": "kfactors.strategy:Strategy",
                                   "k_grid": "0..2"}])
    out = tmp_path / "o"
    assert main(["real", "--config", str(_write_cfg(tmp_path, cfg)), "--out", str(out)]) == 0
    s = json.loads((out / "real/synthetic_monthly/experiments/stub/summary.json").read_text())
    assert s["status"] == "skipped" and "NotImplementedError" in s["message"]
    board = read_csv_str(out / "leaderboard.csv")
    assert board.loc[board["experiment"] == "stub", "status"].tolist() == ["skipped"]
    st = read_csv_str(out / "real" / "status.csv")
    assert st.loc[st["step"] == "kcurve:stub", "status"].item() == "skipped"


def test_failing_experiment_recorded_and_exit_1(tmp_path):
    cfg = _real_only(_ci_dict(), [
        {"name": "bad", "strategy": "kfactors.strategies.sdf:TangencyPCA",
         "strategy_kwargs": {"gamma": -1.0}, "k_grid": "0..2"},
        {"name": "good", "strategy": "kfactors.strategies.sdf:TangencyPCA", "k_grid": "1..3",
         "n_boot": 50}])
    out = tmp_path / "o"
    assert main(["real", "--config", str(_write_cfg(tmp_path, cfg)), "--out", str(out)]) == 1
    exp = out / "real/synthetic_monthly/experiments"
    bad = json.loads((exp / "bad" / "summary.json").read_text())
    assert bad["status"] == "failed" and "gamma" in bad["message"]
    assert json.loads((exp / "good" / "summary.json").read_text())["status"] == "ok"
    with pytest.raises(ValueError, match="gamma"):
        main(["real", "--config", str(_write_cfg(tmp_path, cfg)), "--out", str(out),
              "--fail-fast"])


@pytest.mark.parametrize("mutate, match", [
    (lambda c: c.update(typo=1), "unknown key"),
    (lambda c: c["sim"].update(rules=["no_such_rule"]), "unknown rule"),
    (lambda c: c["sim"].update(reps=0), "reps"),
    (lambda c: c["real"]["panels"][0].update(source="ftp"), "source"),
    (lambda c: c["real"]["panels"][0]["experiments"][0].update(k_grid="5..1"), "k_grid"),
    (lambda c: c["real"]["panels"][0]["experiments"][0].update(strategy="no_colon"),
     "module:Class"),
    (lambda c: c["real"]["panels"][0]["experiments"][0].update(windw=3), "unknown key"),
])
def test_config_validation(tmp_path, mutate, match):
    cfg = _ci_dict()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        load_config(cfg)
    out = tmp_path / "o"
    assert main(["all", "--config", str(_write_cfg(tmp_path, cfg)), "--out", str(out)]) == 2


def test_unknown_walkforward_key_fails_step(tmp_path):
    cfg = _real_only(_ci_dict(), [{"name": "t", "strategy": "kfactors.strategies.sdf:TangencyPCA",
                                   "k_grid": "1..2", "walkforward": {"windoww": 10}}])
    out = tmp_path / "o"
    assert main(["real", "--config", str(_write_cfg(tmp_path, cfg)), "--out", str(out)]) == 1
    s = json.loads((out / "real/synthetic_monthly/experiments/t/summary.json").read_text())
    assert "windoww" in s["message"]


def test_default_config_is_valid():
    cfg = load_config(DEFAULT)
    assert cfg["sim"]["grid"] == "default" and cfg["real"]["rules"] == "all"
    panels = {p["name"]: p for p in cfg["real"]["panels"]}
    assert set(panels) == {"jkp_usa_monthly", "binance_top100", "french_100_daily"}
    jkp = panels["jkp_usa_monthly"]
    assert jkp["walkforward"]["window"] == 240
    assert {e["name"] for e in jkp["experiments"]} == {
        "tangency_w240", "tangency_w120", "tangency_w240_cost10", "tangency_w120_cost10"}
    # rules may reach the top of the k grid on JKP (kmax 40), and the cost variants charge 10 bps
    assert jkp["walkforward"]["kmax"] == 40
    costs = {e["name"]: e.get("walkforward", {}).get("cost_bps") for e in jkp["experiments"]}
    assert costs["tangency_w240_cost10"] == costs["tangency_w120_cost10"] == 10.0
    assert all(parse_k_grid(e["k_grid"]) == list(range(1, 41)) for e in jkp["experiments"])
    bn = panels["binance_top100"]
    assert parse_k_grid(bn["experiments"][0]["k_grid"]) == list(range(16))
    assert panels["french_100_daily"]["walkforward"]["window"] == 126
    assert len(build_designs(cfg["sim"]["grid"])) == 111


# ------------------------------------------------------------------------------------------
# helpers: unit tests
# ------------------------------------------------------------------------------------------
def test_parse_k_grid_and_rules():
    assert parse_k_grid("0..3") == [0, 1, 2, 3]
    assert parse_k_grid("2..10:4") == [2, 6, 10]
    assert parse_k_grid([5, 1]) == [1, 5]
    for bad in ("3", "a..b", "4..1", [1, 1], [-1], []):
        with pytest.raises(ValueError):
            parse_k_grid(bad)
    from kfactors.rules import RULES

    assert [r[0] for r in resolve_rules("all")] == sorted(RULES)
    assert resolve_rules(["parallel( null = circular )"]) == [
        ("parallel(null=circular)", "parallel", {"null": "circular"})]
    with pytest.raises(ValueError, match="duplicated"):
        resolve_rules(["onatski", "onatski"])


def test_seeds_and_hash():
    a = derive_seed(0, "sim", "d", 1)
    assert a == derive_seed(0, "sim", "d", 1)
    assert len({a, derive_seed(0, "sim", "d", 2), derive_seed(1, "sim", "d", 1),
                derive_seed(0, "sim", "e", 1)}) == 4
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_cache_roundtrip(tmp_path):
    c = Cache(tmp_path, enabled=True)
    assert c.get("x", "k") is None
    c.put("x", "k", {"v": np.arange(3)})
    np.testing.assert_array_equal(c.get("x", "k")["v"], np.arange(3))
    (tmp_path / "x" / "bad.pkl").write_bytes(b"not a pickle")
    assert c.get("x", "bad") is None              # corrupt entry -> recompute
    off = Cache(tmp_path, enabled=False)
    off.put("y", "k", 1)
    assert off.get("x", "k") is None and not (tmp_path / "y").exists()


def test_accuracy_table_known_answer():
    raw = pd.DataFrame({
        "design": ["d"] * 4, "family": "f", "regime": "pervasive", "noise": "gauss",
        "market": False, "n": 50, "p": 20, "k_true": 2, "rep": [0, 1, 2, 3],
        "rule": "r", "k_hat": [2.0, 3.0, 0.0, np.nan]})
    acc = accuracy_table(raw).iloc[0]
    assert acc["n_ok"] == 3 and acc["n_failed"] == 1
    assert acc["exact_pct"] == pytest.approx(100 / 3)
    assert acc["over_pct"] == pytest.approx(100 / 3)
    assert acc["under_pct"] == pytest.approx(100 / 3)
    assert acc["mae"] == pytest.approx(1.0) and acc["bias"] == pytest.approx(-1 / 3)
    assert acc["mean_k"] == pytest.approx(5 / 3) and acc["sd_k"] == pytest.approx(1.5275252)


def test_sim_scores_pooling_known_answer():
    acc = pd.DataFrame({
        "rule": ["a", "a", "b", "b"], "design": ["d1", "d2", "d1", "d2"],
        "family": ["x", "y", "x", "y"], "regime": "pervasive", "noise": "gauss",
        "n_ok": [10, 30, 10, 30], "n_failed": [0, 1, 0, 0],
        "exact_pct": [100.0, 0.0, 50.0, 50.0], "mae": [0.0, 2.0, 1.0, 1.0],
        "bias": [0.0, 2.0, 1.0, -1.0], "mean_k": [3.0, 5.0, 4.0, 2.0]})
    s = sim_scores(acc)
    ov = s[s["group"] == "overall"].set_index("rule")
    assert ov.loc["a", "exact_pct"] == pytest.approx(25.0)   # (10*100 + 30*0) / 40
    assert ov.loc["a", "mae"] == pytest.approx(1.5)
    assert ov.loc["b", "mae"] == pytest.approx(1.0)
    assert ov.loc["b", "rank"] == 1 and ov.loc["a", "rank"] == 2
    assert ov.loc["a", "n"] == 40 and ov.loc["a", "n_failed"] == 1
    fx = s[s["group"] == "family=x"].set_index("rule")
    assert fx.loc["a", "rank"] == 1 and fx.loc["a", "exact_pct"] == 100.0


def test_leaderboard_requires_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_leaderboard(tmp_path)


# ------------------------------------------------------------------------------------------
# real-data helpers: synthetic panel, universe, rolling k (leakage, cross-checks with engine)
# ------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def panel():
    return synthetic_panel({"n": 160, "p": 25, "k": 2, "premia": [0.3, 0.2], "freq": "D",
                            "late_assets": 3, "late_rows": 50}, seed=7, name="syn")


def test_synthetic_panel(panel):
    again = synthetic_panel({"n": 160, "p": 25, "k": 2, "premia": [0.3, 0.2], "freq": "D",
                             "late_assets": 3, "late_rows": 50}, seed=7, name="syn")
    pd.testing.assert_frame_equal(panel.returns, again.returns)
    assert panel.returns.iloc[:50, :3].isna().all().all()
    assert panel.returns.iloc[50:].notna().all().all()
    assert (panel.tradable == panel.returns.notna()).all().all()
    assert panel.meta["k_true"] == 2 and panel.periods_per_year() == 252
    # premia shift the population mean by scale * premia @ L'
    from kfactors.sim import SimSpec, simulate

    d = simulate(SimSpec(n=160, p=25, k=2, name="syn"), 7)
    full = 0.01 * (d.X + np.array([0.3, 0.2]) @ d.L.T)
    np.testing.assert_allclose(panel.returns.iloc[50:].to_numpy(), full[50:])
    with pytest.raises(ValueError, match="premia"):
        synthetic_panel({"n": 50, "p": 10, "k": 2, "premia": [1, 2, 3]}, seed=0)
    with pytest.raises(ValueError, match="freq"):
        synthetic_panel({"n": 50, "p": 10, "k": 1, "freq": "W"}, seed=0)


def test_window_universe_matches_panel_window(panel):
    R = panel.returns.to_numpy()
    T = panel.tradable.to_numpy()
    cum = np.vstack([np.zeros((1, R.shape[1]), int), np.cumsum(np.isnan(R), axis=0)])
    for t in (59, 70, 100, 159):
        cols = window_universe(R, T, cum, t, 60, standardize=True)
        ref = panel.window(t, 60).returns.columns     # independent implementation
        assert list(panel.returns.columns[cols]) == list(ref)
    t, cols = latest_window(panel, 60, 10, True)
    assert t == 159 and cols.size == 25
    t, cols = latest_window(panel, 60, 10, True, end=panel.returns.index[80])
    assert t == 80 and cols.size == 22
    with pytest.raises(ValueError, match="no window"):
        latest_window(panel, 60, 26, True)


def test_rolling_k_crosscheck_and_no_lookahead(panel):
    rules = resolve_rules(["onatski", "bai_ng", "parallel(n_perm=10)"])
    rk, info = rolling_k(panel, rules, window=60, step=5, kmax=6, random_state=3)
    assert info["errors"] == {}
    assert rk.index[0] == panel.returns.index[59] and len(rk) == len(range(59, 160, 5))
    # independent recomputation with Panel.window + select_k
    for date in rk.index[::4]:
        t = panel.returns.index.get_loc(date)
        X = panel.window(t, 60).returns
        for lab, name, kw in rules:
            assert rk.loc[date, lab] == select_k(X, name, kmax=6, random_state=3, **kw).k
    # corrupting rows after a decision row leaves every k_t up to it unchanged: random values
    # (any leak changes k) and NaN (any leaked row makes select_k fail -> NaN k)
    from kfactors.data.panel import Panel

    cut = 109                                   # a decision row (59 + 10 * 5)
    early = rk.index <= panel.returns.index[cut]
    rng = np.random.default_rng(0)
    for fill in ("random", "nan"):
        bad = panel.returns.copy()
        bad.iloc[cut + 1:] = (rng.standard_normal(bad.iloc[cut + 1:].shape) if fill == "random"
                              else np.nan)
        p2 = Panel(bad, bad.notna(), name="bad", freq="D")
        rk2, _ = rolling_k(p2, rules, window=60, step=5, kmax=6, random_state=3)
        pd.testing.assert_frame_equal(rk[early], rk2[early])
        assert rk2.loc[early, "onatski"].notna().all()
    # n_jobs does not change the result
    rk3, _ = rolling_k(panel, rules, window=60, step=5, kmax=6, random_state=3, n_jobs=2)
    pd.testing.assert_frame_equal(rk, rk3)


def test_rolling_k_equals_engine_k_used(panel):
    """The rolling k_t on the engine's decision grid equals the engine's own rule choice."""
    cfg = WalkForwardConfig(window=60, rebalance=5, min_assets=10, kmax=6, random_state=3)
    eng = WalkForward(panel, TangencyPCA(), cfg)
    rk, _ = rolling_k(panel, resolve_rules(["onatski"]), window=60, step=5, kmax=6,
                      min_assets=10, random_state=3)
    ku = eng.run("onatski").k_used
    common = ku.index.intersection(rk.index)
    assert len(common) == len(ku) >= 15
    np.testing.assert_array_equal(rk.loc[common, "onatski"].to_numpy(), ku.to_numpy())
    # a LabeledRule with the rule's default options reproduces the engine's rule path
    lr = LabeledRule("bai_ng(criterion=IC_p2)", "bai_ng", {"criterion": "IC_p2"}, 6, True, 3)
    pd.testing.assert_series_equal(eng.run(lr).k_used, eng.run("bai_ng").k_used)


def test_run_experiment_rule_k_equals_engine(panel):
    """Precomputed rule choices reproduce the engine's own rule path exactly."""
    from kfactors.bench.real_bench import run_experiment

    rules = resolve_rules(["onatski", "bai_ng", "parallel(n_perm=10)"])
    wf = {"window": 60, "rebalance": 5, "min_assets": 10, "kmax": 6, "random_state": 3}
    exp = {"name": "t", "strategy": "kfactors.strategies.sdf:TangencyPCA", "k_grid": "0..3"}
    res = run_experiment(panel, exp, rules=rules, wf=wf, seed=0, n_boot=50, alpha=0.1,
                         mean_block=None)
    eng = WalkForward(panel, TangencyPCA(), WalkForwardConfig(**wf))
    for lab, name in (("onatski", "onatski"), ("bai_ng", "bai_ng")):
        ref = eng.run(name)
        np.testing.assert_array_equal(res["k_used"][lab].to_numpy(), ref.k_used.to_numpy())
        np.testing.assert_allclose(res["returns"][lab].to_numpy(), ref.net.to_numpy())
    lr = LabeledRule("parallel(n_perm=10)", "parallel", {"n_perm": 10}, 6, True, 3)
    np.testing.assert_array_equal(res["k_used"]["parallel(n_perm=10)"].to_numpy(),
                                  eng.run(lr).k_used.to_numpy())
    # fixed-k part of the curve equals direct engine runs
    for k in range(4):
        np.testing.assert_allclose(res["returns"][f"k={k}"].to_numpy(),
                                   eng.run(k).net.to_numpy())
    assert res["oracle_k"] in res["oracle_set"]
    assert res["n_table_misses"] == 0                # every window was found in the table


def test_code_change_invalidates_cache(tmp_path, monkeypatch):
    from kfactors.bench import code_fingerprint, file_fingerprint, sim_bench

    assert code_fingerprint() == code_fingerprint() and len(code_fingerprint()) == 20
    assert file_fingerprint("kfactors.strategies.sdf") != ""
    assert file_fingerprint("no_such_module_xyz") == ""
    out = tmp_path / "o"
    args = ["sim", "--config", str(CI), "--out", str(out), "--reps", "1", "--designs", "null_*"]
    assert main(args) == 0
    assert main(args) == 0
    assert pd.read_csv(out / "sim" / "timings.csv")["n_cached"].tolist() == [1]
    monkeypatch.setattr(sim_bench, "code_fingerprint", lambda: "edited-code")
    assert main(args) == 0
    assert pd.read_csv(out / "sim" / "timings.csv")["n_cached"].tolist() == [0]


# ===================================================== repository packaging / contributor docs
# Cheap consistency checks on files the benchmark docs and the contributor cards point to.

def _pyproject() -> dict:
    import tomllib

    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def test_pytest_addopts_keep_summary_line():
    """`pytest -q` (documented in README/CONTRIBUTING) must still print "N passed".

    A ``-q`` in addopts plus the documented ``-q`` gives ``-qq``, which drops the summary line.
    """
    import shlex

    opts = shlex.split(_pyproject()["tool"]["pytest"]["ini_options"]["addopts"])
    assert not any(o.startswith("-q") or o == "--quiet" for o in opts), opts
    assert "not network and not slow" in opts


def test_license_is_spdx_expression():
    """The TOML-table licence form is deprecated by setuptools (removal 2027-02-18)."""
    cfg = _pyproject()
    assert cfg["project"]["license"] == "MIT"
    assert cfg["project"]["license-files"] == ["LICENSE"]
    assert (REPO / "LICENSE").is_file()
    assert any(r.startswith("setuptools>=") and int(r.split(">=")[1].split(".")[0]) >= 77
               for r in cfg["build-system"]["requires"])


def test_card_d_instructions_are_buildable():
    card = (REPO / "docs" / "tasks" / "card_D_docs_examples.md").read_text(encoding="utf-8")
    # nav must not reference a file outside docs/ (mkdocs --strict aborts on it)
    assert "Contributing: contributing.md" in card
    assert "- Contributing: ../CONTRIBUTING.md" not in card
    assert '--8<-- "CONTRIBUTING.md"' in card
    # example output directory is created before saving (results/ is not in git)
    assert 'out.mkdir(parents=True, exist_ok=True)' in card
    assert 'to_csv("results/' not in card
    # bench is shipped: no stale "wave C / fallback" wording
    assert "wave C" not in card and "Fallback" not in card
    yml = card.split("```yaml\n", 1)[1].split("```", 1)[0]
    cfg = yaml.safe_load("\n".join(line[3:] for line in yml.splitlines()))
    mk = next(p["mkdocstrings"] for p in cfg["plugins"] if isinstance(p, dict))
    assert mk["handlers"]["python"]["options"]["docstring_style"] == "numpy"


def test_card_d_may_not_edit_core_configs():
    """Card D owns only *new* robustness configs; ci.yaml / default.yaml back these tests."""
    contributing = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    card = (REPO / "docs" / "tasks" / "card_D_docs_examples.md").read_text(encoding="utf-8")
    for text in (contributing, card):
        text = " ".join(text.split())  # ignore line wrapping
        assert "`configs/**`" not in text
        assert "not `ci.yaml` / `default.yaml`" in text


def test_design_warns_jkp_returns_already_signed():
    """meta["direction"] must not be applied again to JKP returns (double sign flip)."""
    design = " ".join((REPO / "docs" / "design.md").read_text(encoding="utf-8").split())
    assert "**already signed** by the JKP `direction`" in design
