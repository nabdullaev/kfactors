"""Simulation benchmark: accuracy of every number-of-factors rule on designs with a known k.

For every design of a :func:`kfactors.sim.design_grid` (or an inline list of
:class:`~kfactors.sim.SimSpec`) and every replication, a panel ``X = F L' + E`` is drawn with
:func:`kfactors.sim.simulate`, every rule configuration is applied with
:func:`kfactors.rules.select_k` and the panel diagnostics of :func:`kfactors.report.diagnose`
are recorded. This is the Monte-Carlo protocol of Bai & Ng (2002, Econometrica 70(1), Sec. 6)
and Onatski (2010, REStat 92(4), Sec. V): report, per design, how often each estimator hits the
true number of factors and its mean absolute error.

Outputs (in ``<out>/sim/``): ``raw.csv`` (+ ``raw.parquet`` if pyarrow is available),
``accuracy.csv``, ``heatmap.png``, ``heatmap_mae.png``, ``timings.csv``, ``designs.csv`` and,
with calibration, ``decision_table.json``.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from kfactors import __version__ as KF_VERSION
from kfactors.bench import (
    CACHE_SCHEMA,
    Cache,
    code_fingerprint,
    derive_seed,
    new_figure,
    resolve_rules,
    select_designs,
    stable_hash,
    timed,
    to_json,
)

LOG = logging.getLogger("kfactors.bench")

#: Scalar keys of :func:`kfactors.report.diagnose` stored as ``diag_<key>``-free columns.
DIAG_KEYS: tuple[str, ...] = (
    "p_over_n", "spike_share", "gap_ratio_1", "max_gap_ratio", "argmax_gap_ratio",
    "mean_excess_kurtosis", "median_excess_kurtosis", "mean_acf1", "mean_abs_acf1",
    "mean_acf1_sq", "var_dispersion",
)

RAW_COLUMNS: tuple[str, ...] = (
    "design", "family", "regime", "noise", "market", "n", "p", "k_true", "rep", "seed", "rule",
    "rule_name", "k_hat", "error", "n_warnings", "warning", "seconds",
) + DIAG_KEYS


# ------------------------------------------------------------------------------------------
# designs
# ------------------------------------------------------------------------------------------
def noise_label(spec) -> str:
    """Short noise tag of a spec: ``gauss``, ``t3``, ``ar1-0.5``, ``garch``, ``hetero``."""
    if spec.noise == "t":
        return f"t{spec.df:g}"
    if spec.noise == "ar1":
        return f"ar1-{spec.phi:g}"
    return str(spec.noise)


def design_info(spec) -> dict:
    """Descriptive columns of a design: name, family (name prefix), regime, noise, shape, k."""
    return {"design": spec.name, "family": spec.name.split("_", 1)[0], "regime": spec.regime,
            "noise": noise_label(spec), "market": spec.market is not None, "n": int(spec.n),
            "p": int(spec.p), "k_true": int(spec.k)}


def build_designs(grid, patterns=None) -> list:
    """SimSpecs of a named grid or of an inline list of SimSpec dicts, filtered by patterns."""
    from kfactors.sim import SimSpec, design_grid

    if isinstance(grid, str):
        specs = design_grid(grid)
    else:
        specs = []
        for i, d in enumerate(grid):
            d = dict(d)
            for key in ("strengths", "garch", "hetero_range"):
                if key in d and d[key] is not None:
                    d[key] = tuple(d[key])
            try:
                s = SimSpec(**d)
            except TypeError as exc:
                raise ValueError(f"sim.grid[{i}]: {exc}") from None
            if not s.name:
                s.name = (f"custom{i}_{s.regime}_{noise_label(s)}_n{s.n}_p{s.p}_k{s.k}")
            specs.append(s)
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicated design names in the grid: {names}")
    keep = set(select_designs(names, patterns))
    specs = [s for s in specs if s.name in keep]
    if not specs:
        raise ValueError(f"no design matches {patterns!r}")
    return specs


# ------------------------------------------------------------------------------------------
# one replication
# ------------------------------------------------------------------------------------------
def _rep_rows(spec, rep: int, rules, kmax: int, standardize: bool, seed: int) -> list[dict]:
    from kfactors.report import diagnose
    from kfactors.rules import select_k
    from kfactors.sim import simulate

    s = derive_seed(seed, "sim", spec.name, rep)
    data = simulate(spec, s)
    X = data.X
    diag = diagnose(X, standardize=standardize)
    base = design_info(spec)
    base["k_true"] = int(data.k)
    dcols = {k: float(diag[k]) if diag.get(k) is not None else math.nan for k in DIAG_KEYS}
    rows = []
    for label, name, kw in rules:
        rs = derive_seed(seed, "rule", spec.name, rep, label)
        t0 = time.perf_counter()
        err, msg = "", ""
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            try:
                k_hat = float(select_k(X, name, kmax=kmax, standardize=standardize,
                                       random_state=rs, **kw).k)
            except Exception as exc:
                k_hat = math.nan
                err = f"{type(exc).__name__}: {exc}"[:300]
        if wrec:
            msg = str(wrec[0].message).splitlines()[0][:200]
        rows.append({**base, "rep": int(rep), "seed": int(s), "rule": label, "rule_name": name,
                     "k_hat": k_hat, "error": err, "n_warnings": len(wrec), "warning": msg,
                     "seconds": round(time.perf_counter() - t0, 6), **dcols})
    return rows


def _design_task(spec, reps: list[int], rules, kmax: int, standardize: bool, seed: int,
                 cache_root: str | None, cache_on: bool) -> tuple[list[dict], dict]:
    """All replications of one design (worker). Returns (rows, timing record)."""
    cache = Cache(cache_root or ".", enabled=bool(cache_root) and cache_on)
    code = code_fingerprint()
    t0 = time.perf_counter()
    rows: list[dict] = []
    n_cached = 0
    for rep in reps:
        key = stable_hash({"schema": CACHE_SCHEMA, "v": KF_VERSION, "code": code,
                           "spec": dataclasses.asdict(spec), "rep": rep, "rules": rules,
                           "kmax": kmax, "standardize": standardize, "seed": seed})
        got = cache.get("sim", key)
        if got is None:
            got = _rep_rows(spec, rep, rules, kmax, standardize, seed)
            cache.put("sim", key, got)
        else:
            n_cached += 1
        rows.extend(got)
    return rows, {"design": spec.name, "n": spec.n, "p": spec.p, "reps": len(reps),
                  "n_cached": n_cached, "seconds": round(time.perf_counter() - t0, 4)}


# ------------------------------------------------------------------------------------------
# aggregation
# ------------------------------------------------------------------------------------------
def accuracy_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Rule x design accuracy from the raw table.

    Columns: ``rule, design, family, regime, noise, market, n, p, k_true, n_ok, n_failed,
    exact_pct`` (share of successful replications with ``k_hat == k_true``, in percent),
    ``under_pct``, ``over_pct``, ``mae`` (mean ``|k_hat - k_true|``), ``bias``, ``mean_k``,
    ``sd_k`` (ddof=1; NaN for one replication). Failed rule calls (NaN ``k_hat``) are excluded
    from the statistics and counted in ``n_failed``. Order: rules and designs by first
    appearance in ``raw``.
    """
    if raw.empty:
        raise ValueError("empty raw table")
    rules = list(pd.unique(raw["rule"]))
    designs = list(pd.unique(raw["design"]))
    keys = ["design", "family", "regime", "noise", "market", "n", "p", "k_true"]
    out = []
    for (rule, _design), g in raw.groupby(["rule", "design"], sort=False):
        ok = g["k_hat"].notna()
        k = g.loc[ok, "k_hat"].to_numpy(dtype=float)
        kt = g.loc[ok, "k_true"].to_numpy(dtype=float)
        e = k - kt
        nk = k.size
        row = {"rule": rule, **{c: g[c].iloc[0] for c in keys},
               "n_ok": int(nk), "n_failed": int((~ok).sum())}
        row.update({
            "exact_pct": 100.0 * float(np.mean(e == 0)) if nk else math.nan,
            "under_pct": 100.0 * float(np.mean(e < 0)) if nk else math.nan,
            "over_pct": 100.0 * float(np.mean(e > 0)) if nk else math.nan,
            "mae": float(np.mean(np.abs(e))) if nk else math.nan,
            "bias": float(np.mean(e)) if nk else math.nan,
            "mean_k": float(np.mean(k)) if nk else math.nan,
            "sd_k": float(np.std(k, ddof=1)) if nk > 1 else math.nan,
        })
        out.append(row)
    acc = pd.DataFrame(out)
    acc["rule"] = pd.Categorical(acc["rule"], categories=rules, ordered=True)
    acc["design"] = pd.Categorical(acc["design"], categories=designs, ordered=True)
    acc = acc.sort_values(["rule", "design"]).reset_index(drop=True)
    acc["rule"] = acc["rule"].astype(str)
    acc["design"] = acc["design"].astype(str)
    return acc


def _save_heatmap(acc: pd.DataFrame, value: str, path: Path, title: str, cmap: str) -> None:
    from kfactors.plots import plot_heatmap

    nr = acc["rule"].nunique()
    nc = acc["design"].nunique()
    fig = new_figure((min(5 + 0.22 * nc, 60), min(2.5 + 0.4 * nr, 30)))
    ax = fig.subplots()
    plot_heatmap(acc[["rule", "design", value]], ax=ax, value=value, title=title,
                 fmt=".0f" if value.endswith("pct") else ".2f", cmap=cmap)
    fig.savefig(path, dpi=110)


def _write_raw(raw: pd.DataFrame, outdir: Path) -> list[Path]:
    paths = [outdir / "raw.csv"]
    raw.to_csv(paths[0], index=False)
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        LOG.info("pyarrow not installed: raw results written as CSV only")
    else:
        raw.to_parquet(outdir / "raw.parquet", index=False)
        paths.append(outdir / "raw.parquet")
    return paths


# ------------------------------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------------------------------
def run_sim(cfg: dict, out, *, n_jobs: int | None = None, reps: int | None = None,
            calibrate: bool | None = None, designs=None, use_cache: bool = True) -> dict:
    """Run the simulation benchmark of a loaded config (see :mod:`kfactors.bench`).

    Parameters
    ----------
    cfg : dict
        Output of :func:`kfactors.bench.load_config` (needs a ``sim`` section).
    out : path
        Results root; files go to ``out/sim/``, the cache to ``out/cache/sim/``.
    n_jobs : int, optional
        joblib processes over designs (default ``cfg["n_jobs"]``). Results do not depend on it.
    reps : int, optional
        Overrides ``sim.reps``.
    calibrate : bool, optional
        Overrides ``sim.calibrate``: learn a decision table from the raw results.
    designs : list of str, optional
        fnmatch patterns overriding ``sim.designs``.
    use_cache : bool
        Reuse/store per-(design, replication) results in ``out/cache``.

    Returns
    -------
    dict
        ``{"raw": DataFrame, "accuracy": DataFrame, "files": [Path, ...],
        "decision_table": dict | None}``.
    """
    if "sim" not in cfg:
        raise ValueError("the config has no 'sim' section")
    sc = cfg["sim"]
    seed = int(cfg["seed"])
    n_jobs = int(cfg["n_jobs"] if n_jobs is None else n_jobs)
    n_reps = int(sc["reps"] if reps is None else reps)
    if n_reps < 1:
        raise ValueError(f"reps must be >= 1, got {n_reps}")
    calibrate = bool(sc["calibrate"] if calibrate is None else calibrate)
    rules = resolve_rules(sc["rules"])
    kmax, standardize = int(sc["kmax"]), bool(sc["standardize"])
    specs = build_designs(sc["grid"], designs if designs is not None else sc["designs"])
    outdir = Path(out) / "sim"
    outdir.mkdir(parents=True, exist_ok=True)
    cache_root = str(Path(out) / "cache")
    rep_ids = list(range(n_reps))
    LOG.info("sim: %d designs x %d reps x %d rules (kmax=%d, standardize=%s, n_jobs=%d)",
             len(specs), n_reps, len(rules), kmax, standardize, n_jobs)

    steps: list = []
    t0 = time.perf_counter()
    with timed(f"sim: {len(specs)} designs", steps):
        args = [(s, rep_ids, rules, kmax, standardize, seed, cache_root, use_cache)
                for s in specs]
        if n_jobs == 1:
            done = []
            for i, a in enumerate(args, 1):
                done.append(_design_task(*a))
                LOG.info("sim: [%d/%d] %s (%.2f s, %d/%d reps cached)", i, len(args),
                         a[0].name, done[-1][1]["seconds"], done[-1][1]["n_cached"], n_reps)
        else:
            from joblib import Parallel, delayed

            done = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_design_task)(*a) for a in args)
            for i, (_, tm) in enumerate(done, 1):
                LOG.info("sim: [%d/%d] %s (%.2f s, %d/%d reps cached)", i, len(done),
                         tm["design"], tm["seconds"], tm["n_cached"], n_reps)
    raw = pd.DataFrame([r for rows, _ in done for r in rows], columns=list(RAW_COLUMNS))
    timings = pd.DataFrame([tm for _, tm in done])

    n_err = int(raw["error"].astype(bool).sum())
    if n_err:
        first = raw.loc[raw["error"].astype(bool), ["design", "rep", "rule", "error"]].iloc[0]
        LOG.warning("sim: %d rule calls failed (k_hat=NaN), e.g. %s rep %d %s: %s", n_err,
                    first["design"], first["rep"], first["rule"], first["error"])
    n_warn = int((raw["n_warnings"] > 0).sum())
    if n_warn:
        LOG.info("sim: %d rule calls emitted warnings (see raw columns n_warnings/warning)",
                 n_warn)

    files = _write_raw(raw, outdir)
    acc = accuracy_table(raw)
    acc.to_csv(outdir / "accuracy.csv", index=False)
    files.append(outdir / "accuracy.csv")
    pd.DataFrame([{**design_info(s), "spec": json.dumps(dataclasses.asdict(s))}
                  for s in specs]).to_csv(outdir / "designs.csv", index=False)
    files.append(outdir / "designs.csv")
    with timed("sim: heatmaps", steps):
        _save_heatmap(acc, "exact_pct", outdir / "heatmap.png",
                      f"exact k (% of {n_reps} reps) by rule and design", "viridis")
        _save_heatmap(acc, "mae", outdir / "heatmap_mae.png",
                      "mean |k_hat - k_true| by rule and design", "viridis_r")
    files += [outdir / "heatmap.png", outdir / "heatmap_mae.png"]
    timings.to_csv(outdir / "timings.csv", index=False)
    files.append(outdir / "timings.csv")

    table = None
    if calibrate:
        table = calibrate_from_raw(raw, sc.get("calibrate_options") or {})
        path = outdir / "decision_table.json"
        to_json(table, path)
        files.append(path)
        meta = table.get("meta", {})
        LOG.info("sim: calibrated decision table: train acc %.3f, LOFO acc %s, MAE tree %.3f "
                 "vs best single rule %s %.3f", meta.get("train_accuracy", math.nan),
                 meta.get("leave_one_family_out_accuracy"), meta.get("mae_tree_train", math.nan),
                 meta.get("best_single_rule"), meta.get("mae_best_single_rule", math.nan))
        print("Calibrated decision table written to", path)
        print("The packaged table was NOT changed. To install this table, run:")
        print(f'  cp "{path}" "{_packaged_table_path()}"')

    to_json({"config": cfg["name"], "seed": seed, "reps": n_reps, "n_designs": len(specs),
             "rules": [r[0] for r in rules], "kmax": kmax, "standardize": standardize,
             "n_rows": len(raw), "n_failed": n_err, "kfactors_version": KF_VERSION,
             "seconds": round(time.perf_counter() - t0, 3), "steps": steps},
            outdir / "run_info.json")
    files.append(outdir / "run_info.json")
    return {"raw": raw, "accuracy": acc, "files": files, "decision_table": table}


def calibrate_from_raw(raw: pd.DataFrame, options: dict | None = None) -> dict:
    """Decision table from raw sim results (:func:`kfactors.report.calibrate_decision_table`).

    Uses the columns ``design, rep, rule, k_hat, k_true, family`` and the diagnostics in
    :data:`kfactors.report.DEFAULT_FEATURES` (unless ``options["features"]`` is given).
    """
    from kfactors.report import DEFAULT_FEATURES, calibrate_decision_table

    options = dict(options or {})
    feats = options.get("features") or [f for f in DEFAULT_FEATURES if f in raw.columns]
    cols = ["design", "rep", "rule", "k_hat", "k_true", "family"] + list(feats)
    return calibrate_decision_table(raw[cols].copy(), **options)


def _packaged_table_path() -> str:
    from importlib import resources

    return str(resources.files("kfactors") / "resources" / "decision_table.json")
