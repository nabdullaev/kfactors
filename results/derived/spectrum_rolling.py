"""Eigenvalues above the naive (median-matched) MP edge and the Laloux edge in every rolling
window of the real-data benchmark (same decision grid/universe as rolling_k.csv).

Usage: python spectrum_rolling.py <repo>/configs/default.yaml <outdir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kfactors.bench import load_config, _merge
from kfactors.bench.real_bench import (_nan_cumsum, _row_range, _walkforward_config, load_panel,
                                       window_universe)
from kfactors.plots import spectrum_edges
from kfactors.utils import eigvals_desc, preprocess
import dataclasses

cfg = load_config(sys.argv[1])
outdir = Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
rc, seed = cfg["real"], cfg["seed"]
rows_all = []
for pc in rc["panels"]:
    panel = load_panel(pc, seed)
    wf = _merge(dataclasses.asdict(_walkforward_config({}, seed)), rc["walkforward"],
                pc.get("walkforward"))
    ro = _merge({"window": wf["window"], "step": wf["rebalance"], "start": wf["start"],
                 "end": wf["end"], "min_assets": wf["min_assets"],
                 "standardize": wf["standardize"]}, rc["rolling"], pc.get("rolling"))
    R = panel.returns.to_numpy(dtype=float)
    T = panel.tradable.to_numpy(dtype=bool)
    cum = _nan_cumsum(R)
    first, last = _row_range(panel.returns.index, ro["window"], ro["start"], ro["end"])
    win = ro["window"]
    for t in range(first, last + 1, ro["step"]):
        cols = window_universe(R, T, cum, t, win, ro["standardize"])
        if cols.size < ro["min_assets"]:
            continue
        ev = eigvals_desc(preprocess(R[t - win + 1:t + 1, cols], standardize=ro["standardize"])[0])
        e = spectrum_edges(ev, win, int(cols.size))
        rows_all.append({"panel": pc["name"], "date": panel.returns.index[t], "n": win,
                         "p": int(cols.size), "lambda1": ev[0], "lambda1_share": ev[0] / ev.sum(),
                         "edge_mp": e["edge_mp"], "n_above_mp": e["n_above_mp"],
                         "edge_laloux": e["edge_laloux"], "n_above_laloux": e["n_above_laloux"]})
df = pd.DataFrame(rows_all)
df.to_csv(outdir / "spectrum_rolling.csv", index=False)
agg = df.groupby("panel", sort=False).agg(
    n_windows=("date", "size"), first=("date", "min"), last=("date", "max"),
    mp_median=("n_above_mp", "median"), mp_q25=("n_above_mp", lambda s: s.quantile(.25)),
    mp_q75=("n_above_mp", lambda s: s.quantile(.75)), mp_min=("n_above_mp", "min"),
    mp_max=("n_above_mp", "max"),
    lal_median=("n_above_laloux", "median"), lal_q25=("n_above_laloux", lambda s: s.quantile(.25)),
    lal_q75=("n_above_laloux", lambda s: s.quantile(.75)), lal_min=("n_above_laloux", "min"),
    lal_max=("n_above_laloux", "max"), lambda1_share_median=("lambda1_share", "median"))
agg.to_csv(outdir / "spectrum_rolling_summary.csv")
pd.set_option("display.width", 250)
print(agg.to_string())
