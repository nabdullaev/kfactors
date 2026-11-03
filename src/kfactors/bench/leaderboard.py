"""Leaderboard: combine simulation accuracy and real-data k-curve scores of every rule.

:func:`build_leaderboard` reads what the ``sim`` and ``real`` commands left in a results
directory and writes ``leaderboard.csv`` (long format, one row per (section, group, rule)) and
``leaderboard.md``.

Sections of ``leaderboard.csv`` (column ``section``):

* ``sim`` -- statistical accuracy pooled over replications, for ``group`` = ``overall``,
  ``regime=<...>``, ``noise=<...>`` and ``family=<...>``: ``exact_pct`` (share of panels with
  ``k_hat == k_true``), ``mae``, ``bias``, ``mean_k``, ``n`` (successful rule calls),
  ``n_failed`` and ``rank`` (within the group, by MAE then exact %).
* ``real`` -- one row per (panel, experiment, rule) with the k-curve scores of
  :func:`kfactors.evaluate.curve_from_results`: ``oracle_k``, ``oracle_set``, the rule's
  ``k_median``/``k_mean``, ``metric`` [``low``, ``high``], ``dist_to_oracle``, ``in_set`` and
  ``regret``. Skipped/failed experiments appear as one row with ``rule`` empty and their
  ``status``/``message``.
* ``real_overall`` -- per rule across the successful experiments: ``n`` experiments,
  ``in_set_pct``, mean ``dist_to_oracle`` and mean ``regret`` (regret is in the units of each
  experiment's metric, so it is only comparable within one metric).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("kfactors.bench")

COLUMNS: tuple[str, ...] = (
    "section", "group", "panel", "experiment", "strategy", "metric_name", "status", "rule",
    "exact_pct", "mae", "bias", "mean_k", "n", "n_failed", "rank", "oracle_k", "oracle_set",
    "k_median", "k_mean", "metric", "low", "high", "dist_to_oracle", "in_set", "regret",
    "in_set_pct", "message",
)


def read_csv_str(path) -> pd.DataFrame:
    """``read_csv`` where only empty fields are missing (labels like ``"null"`` stay strings)."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


# ------------------------------------------------------------------------------------------
# simulation part
# ------------------------------------------------------------------------------------------
def sim_scores(acc: pd.DataFrame) -> pd.DataFrame:
    """Pooled rule accuracy per group from ``accuracy.csv`` (see module docstring).

    Pooling weights each design by its number of successful replications, so the result
    equals the statistic over all (design, rep) panels of the group.
    """
    acc = acc.copy()
    w = acc["n_ok"].astype(float)
    acc["_w_exact"] = acc["exact_pct"] * w
    acc["_w_mae"] = acc["mae"] * w
    acc["_w_bias"] = acc["bias"] * w
    acc["_w_mean_k"] = acc["mean_k"] * w
    acc["_w"] = w
    groups = [("overall", None)] + [(g, g) for g in ("regime", "noise", "family")
                                    if g in acc.columns]
    out = []
    rule_order = list(pd.unique(acc["rule"]))
    for gname, col in groups:
        keys = ["rule"] if col is None else [col, "rule"]
        agg = acc.groupby(keys, sort=False)[["_w_exact", "_w_mae", "_w_bias", "_w_mean_k", "_w",
                                             "n_failed"]].sum().reset_index()
        with np.errstate(invalid="ignore", divide="ignore"):
            agg["exact_pct"] = agg["_w_exact"] / agg["_w"]
            agg["mae"] = agg["_w_mae"] / agg["_w"]
            agg["bias"] = agg["_w_bias"] / agg["_w"]
            agg["mean_k"] = agg["_w_mean_k"] / agg["_w"]
        agg["n"] = agg["_w"].astype(int)
        agg["group"] = "overall" if col is None else gname + "=" + agg[col].astype(str)
        agg["_ord"] = agg["rule"].map({r: i for i, r in enumerate(rule_order)})
        agg = agg.sort_values(["group", "mae", "exact_pct", "_ord"],
                              ascending=[True, True, False, True], kind="mergesort")
        agg["rank"] = agg.groupby("group", sort=False).cumcount() + 1
        out.append(agg[["group", "rule", "exact_pct", "mae", "bias", "mean_k", "n", "n_failed",
                        "rank"]])
    res = pd.concat(out, ignore_index=True)
    res.insert(0, "section", "sim")
    return res


# ------------------------------------------------------------------------------------------
# real part
# ------------------------------------------------------------------------------------------
def _fmt_set(s) -> str:
    return ",".join(str(int(k)) for k in s) if s else ""


def real_scores(real_dir: Path) -> pd.DataFrame:
    """One row per (panel, experiment, rule) from ``real/*/experiments/*/summary.json``."""
    rows = []
    for sp in sorted(Path(real_dir).glob("*/experiments/*/summary.json")):
        with open(sp, encoding="utf-8") as fh:
            s = json.load(fh)
        base = {"section": "real", "group": f"{s['panel']}/{s['experiment']}",
                "panel": s["panel"], "experiment": s["experiment"], "strategy": s["strategy"],
                "metric_name": s.get("metric_name", s.get("metric")),
                "status": s.get("status", "unknown")}
        if s.get("status") != "ok":
            rows.append({**base, "message": s.get("message", "")})
            continue
        rpath = sp.parent / "rules.csv"
        if not rpath.exists():
            rows.append({**base, "status": "failed", "message": "rules.csv missing"})
            continue
        rt = read_csv_str(rpath).set_index("rule")
        for rule, r in rt.iterrows():
            rows.append({**base, "rule": str(rule), "oracle_k": s["oracle_k"],
                         "oracle_set": _fmt_set(s["oracle_set"]), "k_median": r["k_median"],
                         "k_mean": r["k_mean"], "metric": r["metric"], "low": r["low"],
                         "high": r["high"], "dist_to_oracle": r["dist_to_oracle"],
                         "in_set": bool(r["in_set"]), "regret": r["regret"]})
    return pd.DataFrame(rows, columns=list(COLUMNS))


def real_overall(real: pd.DataFrame) -> pd.DataFrame:
    """Per rule across successful experiments: n, in-set %, mean distance and mean regret."""
    ok = real[(real["status"] == "ok") & real["rule"].notna()]
    if ok.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    g = ok.groupby("rule", sort=False)
    res = pd.DataFrame({
        "n": g.size(),
        "in_set_pct": 100.0 * g["in_set"].mean(),
        "dist_to_oracle": g["dist_to_oracle"].mean(),
        "regret": g["regret"].mean(),
    }).reset_index()
    res = res.sort_values(["in_set_pct", "dist_to_oracle"], ascending=[False, True],
                          kind="mergesort")
    res["rank"] = np.arange(1, len(res) + 1)
    res["section"] = "real_overall"
    res["group"] = "all experiments"
    return res


# ------------------------------------------------------------------------------------------
# markdown
# ------------------------------------------------------------------------------------------
def _cell(v, fmt: str = ".2f") -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(float(v)):
            return ""
        return format(float(v), fmt)
    return str(v).replace("|", "\\|").replace("\n", " ")


def md_table(df: pd.DataFrame, fmts: dict | None = None) -> str:
    """GitHub-flavoured markdown table (no dependency on ``tabulate``)."""
    fmts = fmts or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_cell(r[c], fmts.get(c, ".2f")) for c in cols) + " |")
    return "\n".join(lines)


def _sim_md(sim: pd.DataFrame) -> list[str]:
    out = ["## Simulations: does the rule find the true k?", ""]
    ov = sim[sim["group"] == "overall"].set_index("rule")
    order = list(ov.sort_values("rank").index)
    tab = pd.DataFrame({"rank": ov.loc[order, "rank"], "rule": order,
                        "exact %": ov.loc[order, "exact_pct"], "MAE": ov.loc[order, "mae"],
                        "bias": ov.loc[order, "bias"], "panels": ov.loc[order, "n"],
                        "failed": ov.loc[order, "n_failed"]})
    out += ["Pooled over all designs and replications (rank by MAE, then exact %).", "",
            md_table(tab, {"exact %": ".1f", "MAE": ".3f", "bias": "+.3f"}), ""]
    for dim, title in (("family", "exact % by design family"),
                       ("regime", "exact % by regime"), ("noise", "exact % by noise")):
        sub = sim[sim["group"].str.startswith(dim + "=")]
        if sub.empty:
            continue
        wide = sub.pivot_table(index="rule", columns="group", values="exact_pct",
                               aggfunc="first").reindex(order)
        wide.columns = [c.split("=", 1)[1] for c in wide.columns]
        wide = wide.reset_index()
        out += [f"### {title.capitalize()}", "",
                md_table(wide, dict.fromkeys(wide.columns, ".1f")), ""]
    return out


def _real_md(real: pd.DataFrame, overall: pd.DataFrame) -> list[str]:
    out = ["## Real data: what does the choice of k cost?", "",
           ("Oracle k = best fixed k in hindsight; oracle set = smallest set of k covering the "
            "bootstrap argmax draws at the configured level. A rule is *in set* when its median "
            "k lies in the oracle set; regret = metric(oracle k) - metric(rule)."), ""]
    if not overall.empty:
        tab = overall[["rank", "rule", "n", "in_set_pct", "dist_to_oracle", "regret"]].rename(
            columns={"n": "experiments", "in_set_pct": "in set %",
                     "dist_to_oracle": "mean |k - oracle|", "regret": "mean regret"})
        out += ["### All experiments", "", md_table(tab, {"in set %": ".0f",
                                                          "mean |k - oracle|": ".2f",
                                                          "mean regret": ".3f"}), ""]
    for (panel, exp), g in real.groupby(["panel", "experiment"], sort=False):
        s = g.iloc[0]
        out.append(f"### {panel} / {exp}")
        out.append("")
        if s["status"] != "ok":
            out += [f"**{s['status']}** ({s['strategy']}): {s['message']}", ""]
            continue
        out += [(f"Strategy `{s['strategy']}`, metric `{s['metric_name']}`: oracle k = "
                 f"**{int(s['oracle_k'])}**, oracle set = {{{s['oracle_set']}}}."), ""]
        tab = g[["rule", "k_median", "k_mean", "metric", "low", "high", "dist_to_oracle",
                 "in_set", "regret"]].sort_values(["dist_to_oracle", "rule"])
        out += [md_table(tab, {"k_median": ".1f", "k_mean": ".2f", "metric": ".3f",
                               "low": ".3f", "high": ".3f", "dist_to_oracle": ".1f",
                               "regret": ".3f"}), ""]
    return out


# ------------------------------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------------------------------
def build_leaderboard(out) -> tuple[pd.DataFrame, str]:
    """Write ``<out>/leaderboard.csv`` and ``<out>/leaderboard.md`` from existing results.

    Parameters
    ----------
    out : path
        Results root (containing ``sim/accuracy.csv`` and/or ``real/<panel>/experiments``).

    Returns
    -------
    (DataFrame, str)
        The long leaderboard table and the markdown text. Missing sections are noted in the
        markdown; with neither section present a ``FileNotFoundError`` is raised.
    """
    out = Path(out)
    parts, md = [], ["# kfactors benchmark leaderboard", ""]
    acc_path = out / "sim" / "accuracy.csv"
    sim = None
    if acc_path.exists():
        acc = read_csv_str(acc_path)
        sim = sim_scores(acc)
        parts.append(sim)
        info_path = out / "sim" / "run_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            md += [(f"Simulation config `{info.get('config')}`: {info.get('n_designs')} "
                    f"designs x {info.get('reps')} replications, kmax = {info.get('kmax')}, "
                    f"seed {info.get('seed')}."), ""]
        md += _sim_md(sim)
    else:
        md += ["## Simulations", "", "_No simulation results (run `sim`)._", ""]
    real_dir = out / "real"
    real = real_scores(real_dir) if real_dir.exists() else pd.DataFrame(columns=list(COLUMNS))
    if not real.empty:
        overall = real_overall(real)
        parts += [real, overall]
        md += _real_md(real, overall)
    else:
        md += ["## Real data", "", "_No real-data results (run `real`)._", ""]
    if sim is None and real.empty:
        raise FileNotFoundError(f"no benchmark results under {out} (expected sim/accuracy.csv "
                                f"or real/*/experiments/*/summary.json)")
    board = pd.concat([p.reindex(columns=list(COLUMNS)) for p in parts], ignore_index=True)
    board.to_csv(out / "leaderboard.csv", index=False)
    text = "\n".join(md).rstrip() + "\n"
    (out / "leaderboard.md").write_text(text, encoding="utf-8")
    LOG.info("leaderboard: %d rows -> %s", len(board), out / "leaderboard.csv")
    return board, text
