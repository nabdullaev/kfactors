"""Derived simulation tables for RESULTS.md (reads results/sim/*, uses only kfactors APIs).

Usage: python sim_tables.py <results_dir> <outdir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kfactors.report import calibrate_decision_table, load_decision_table, recommend
from kfactors.bench.sim_bench import DIAG_KEYS

res, outdir = Path(sys.argv[1]), Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(res / "sim" / "raw.csv", keep_default_na=False, na_values=[""])
table = json.load(open(res / "sim" / "decision_table.json"))
opts = {"max_depth": table["meta"]["max_depth"],
        "min_samples_leaf": table["meta"]["min_samples_leaf"]}
feats = table["meta"]["features"]
raw["abs_err"] = (raw["k_hat"] - raw["k_true"]).abs()
raw["exact"] = raw["abs_err"] == 0
keys = ["design", "rep"]

# ---------- best rule per group (by exact %, ties by MAE) and by MAE
def best(group_col):
    g = raw.groupby([group_col, "rule"]).agg(exact_pct=("exact", "mean"), mae=("abs_err", "mean"),
                                             n=("exact", "size")).reset_index()
    g["exact_pct"] *= 100
    rows = []
    for val, sub in g.groupby(group_col):
        be = sub.sort_values(["exact_pct", "mae"], ascending=[False, True]).iloc[0]
        bm = sub.sort_values(["mae", "exact_pct"], ascending=[True, False]).iloc[0]
        on = sub.set_index("rule").loc["onatski"]
        tie_e = sub.loc[np.isclose(sub["exact_pct"], be["exact_pct"]), "rule"].tolist()
        tie_m = sub.loc[np.isclose(sub["mae"], bm["mae"]), "rule"].tolist()
        rows.append({group_col: val, "n_panels": int(be["n"]),
                     "best_by_exact": " | ".join(tie_e) if len(tie_e) <= 3 else
                     f"{len(tie_e)} rules tie", "best_exact_pct": round(be["exact_pct"], 1),
                     "best_by_mae": " | ".join(tie_m) if len(tie_m) <= 3 else
                     f"{len(tie_m)} rules tie", "best_mae": round(bm["mae"], 3),
                     "onatski_exact_pct": round(on["exact_pct"], 1),
                     "onatski_mae": round(on["mae"], 3)})
    return pd.DataFrame(rows)

raw["family_noise"] = raw["family"] + "/" + raw["noise"]
raw["family_regime"] = raw["family"] + "/" + raw["regime"]
for col in ["family", "noise", "regime", "family_regime", "family_noise"]:
    best(col).to_csv(outdir / f"sim_best_by_{col}.csv", index=False)

# ---------- decision tables evaluated as k selectors on the sim panels
panels = raw.groupby(keys, sort=True).agg({**{c: "first" for c in DIAG_KEYS},
                                           "family": "first", "k_true": "first"})
khat = raw.pivot_table(index=keys, columns="rule", values="k_hat", aggfunc="first")
khat = khat.reindex(panels.index)

def score(labels):
    labels = pd.Series(labels, index=panels.index)
    k = np.array([khat.at[i, lab] for i, lab in zip(panels.index, labels)], dtype=float)
    err = np.abs(k - panels["k_true"].to_numpy(dtype=float))
    return pd.DataFrame({"label": labels, "k": k, "abs_err": err, "exact": err == 0,
                         "family": panels["family"]})

def pick(tab):
    return [recommend({f: row[f] for f in DIAG_KEYS}, tab)["label"]
            for _, row in panels.iterrows()]

out = {}
calib_in = score(pick(table))
packaged = load_decision_table()
pack = score(pick(packaged))
# leave-one-family-out: refit on the other families with the same options, predict held-out
cols = ["design", "rep", "rule", "k_hat", "k_true", "family"] + feats
lofo_labels = pd.Series(index=panels.index, dtype=object)
import warnings
for fam in sorted(panels["family"].unique()):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = calibrate_decision_table(raw.loc[raw["family"] != fam, cols].copy(), **opts)
    idx = panels.index[panels["family"] == fam]
    for i in idx:
        lofo_labels.at[i] = recommend({f: panels.at[i, f] for f in DIAG_KEYS}, t)["label"]
calib_lofo = score(lofo_labels.tolist())
single = {r: score([r] * len(panels)) for r in khat.columns}
oracle_err = (khat.sub(panels["k_true"], axis=0)).abs().min(axis=1)

def summ(df):
    d = {"exact_pct": 100 * df["exact"].mean(), "mae": df["abs_err"].mean()}
    for fam, g in df.groupby("family"):
        d[f"exact_pct[{fam}]"] = 100 * g["exact"].mean()
        d[f"mae[{fam}]"] = g["abs_err"].mean()
    return d

rows = [{"selector": "calibrated table (in-sample)", **summ(calib_in)},
        {"selector": "calibrated table (leave-one-family-out)", **summ(calib_lofo)},
        {"selector": "packaged heuristic table (report() default)", **summ(pack)}]
srank = sorted(single, key=lambda r: (single[r]["abs_err"].mean(), -single[r]["exact"].mean()))
for r in srank[:5]:
    rows.append({"selector": f"single rule: {r}", **summ(single[r])})
best_exact = max(single, key=lambda r: single[r]["exact"].mean())
if best_exact not in srank[:5]:
    rows.append({"selector": f"single rule: {best_exact}", **summ(single[best_exact])})
rows.append({"selector": "per-panel oracle (best rule in hindsight)",
             "exact_pct": 100 * (oracle_err == 0).mean(), "mae": oracle_err.mean()})
dt = pd.DataFrame(rows)
dt.to_csv(outdir / "sim_decision_table_eval.csv", index=False)
pd.set_option("display.width", 250)
print(dt.round(3).to_string())
print("calibrated picks (in-sample):", calib_in["label"].value_counts().to_dict())
print("packaged picks:", pack["label"].value_counts().to_dict())
print("lofo picks:", calib_lofo["label"].value_counts().to_dict())
json.dump({"meta": {k: table["meta"][k] for k in [
    "train_accuracy", "leave_one_family_out_accuracy", "leave_one_family_out_accuracy_by_family",
    "mae_tree_train", "mae_tree_lofo", "best_single_rule", "mae_best_single_rule", "mae_oracle",
    "target_counts", "max_depth", "min_samples_leaf", "features"]}},
    open(outdir / "sim_decision_table_meta.json", "w"), indent=2)

w = raw[raw["family"] == "weak"].copy()
w["bbp"] = w["design"].str.extract(r"bbp([0-9.]+)_")[0].astype(float)
w["k_eq_3"] = w["k_hat"] == 3
w["k_ge_4"] = w["k_hat"] >= 4
wt = (w.groupby(["bbp", "rule"]).agg(exact4_pct=("exact", "mean"), k3_pct=("k_eq_3", "mean"),
                                      mean_k=("k_hat", "mean")).reset_index())
wt["exact4_pct"] *= 100
wt["k3_pct"] *= 100
wt.to_csv(outdir / "sim_weak_by_bbp.csv", index=False)
sel = ["onatski", "parallel(null=circular)", "parallel(devolatilize=true, null=circular)",
       "mp_market", "mp", "mp(noise=median)"]
print(wt[wt["rule"].isin(sel)].pivot(index="rule", columns="bbp",
                                      values=["exact4_pct", "k3_pct", "mean_k"]).round(1).to_string())
