"""Derived real-data tables for RESULTS.md (reads results/real/*).

Usage: python real_tables.py <results_dir> <outdir>
"""
import json
import sys
from pathlib import Path

import pandas as pd

res, outdir = Path(sys.argv[1]) / "real", Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250)

# (c) rolling k_t per rule: median and IQR (rolling_k.csv + monthly k_used of the JKP k-curves)
rows = []
def add(name, df, source):
    for lab in df.columns:
        k = df[lab].dropna()
        rows.append({"panel": name, "source": source, "rule": lab, "n_windows": int(k.size),
                     "median": k.median(), "q25": k.quantile(.25), "q75": k.quantile(.75),
                     "iqr": k.quantile(.75) - k.quantile(.25), "min": k.min(), "max": k.max(),
                     "mean": k.mean()})
for p in ["jkp_usa_monthly", "french_100_daily", "binance_top100"]:
    rk = pd.read_csv(res / p / "rolling_k.csv", index_col=0, parse_dates=True)
    info = json.load(open(res / p / "rolling_k_info.json"))
    add(p, rk.drop(columns="n_assets"),
        f"rolling_k.csv (window {info['window']}, step {info['step']})")
for e in ["tangency_w240", "tangency_w120"]:
    ku = pd.read_csv(res / "jkp_usa_monthly" / "experiments" / e / "k_used.csv", index_col=0)
    add("jkp_usa_monthly", ku, f"experiments/{e}/k_used.csv (monthly decisions)")
roll = pd.DataFrame(rows)
roll.to_csv(outdir / "real_rolling_k_iqr.csv", index=False)
print(roll.round(2).to_string())

# (d) JKP k-curves
for e in ["tangency_w240", "tangency_w120"]:
    d = res / "jkp_usa_monthly" / "experiments" / e
    s = json.load(open(d / "summary.json"))
    c = pd.read_csv(d / "curve.csv", index_col=0)
    r = pd.read_csv(d / "rules.csv", index_col=0)
    ok = s["oracle_k"]
    print(e, "oracle", ok, s["oracle_set"], "sharpe", c.loc[ok, ["metric", "low", "high"]].round(3).tolist(),
          "n_common", s["n_common"], s["common_start"], s["common_end"])
    print(r[["k_median", "k_mean", "metric", "low", "high", "in_set", "regret", "regret_low",
             "regret_high"]].round(3).to_string())
