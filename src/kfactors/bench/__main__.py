"""CLI: ``python -m kfactors.bench {sim,real,all,leaderboard} --config CONFIG --out DIR``.

See :mod:`kfactors.bench` for the configuration keys and the outputs. Exit status: 0 on
success (skipped not-yet-implemented strategies included), 1 if any step failed, 2 on usage or
configuration errors.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

LOG = logging.getLogger("kfactors.bench")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m kfactors.bench",
        description="kfactors benchmark suite: simulations (rule accuracy against a known k), "
                    "real-data k-curves (what the choice of k costs), and a leaderboard.",
        epilog="Config keys and outputs are documented in the kfactors.bench module docstring "
               "(python -c 'import kfactors.bench as b; help(b)').")
    p.add_argument("command", choices=["sim", "real", "all", "leaderboard"],
                   help="sim: simulation grid; real: real-data panels; all: both; "
                        "leaderboard: only rebuild leaderboard.csv/.md from existing results")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    p.add_argument("--out", default="results", help="results directory (cache in <out>/cache)")
    p.add_argument("--reps", type=int, default=None, help="override sim.reps")
    p.add_argument("--n-jobs", type=int, default=None, help="joblib processes (-1 = all cores)")
    p.add_argument("--calibrate", action="store_true",
                   help="learn a decision table from the sim results "
                        "(written to <out>/sim/decision_table.json)")
    p.add_argument("--panels", nargs="+", default=None, metavar="NAME",
                   help="only these real-data panels")
    p.add_argument("--designs", nargs="+", default=None, metavar="PATTERN",
                   help="only sim designs matching these fnmatch patterns")
    p.add_argument("--seed", type=int, default=None, help="override the config seed")
    p.add_argument("--no-cache", action="store_true", help="recompute everything")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first failure")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv=None) -> int:
    """Run the benchmark CLI; returns the process exit status (see module docstring)."""
    args = _parser().parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")  # figures are only written to files

    from kfactors import __version__
    from kfactors.bench import load_config, to_json

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    handlers = [logging.StreamHandler(sys.stderr),
                logging.FileHandler(out / "bench.log", encoding="utf-8")]
    for h in handlers:
        h.setFormatter(fmt)
        LOG.addHandler(h)
    old_level, old_prop = LOG.level, LOG.propagate
    LOG.setLevel(args.log_level)
    LOG.propagate = False
    t0 = time.perf_counter()
    try:
        try:
            cfg = load_config(args.config)
        except (OSError, ValueError, TypeError) as exc:
            LOG.error("invalid config %s: %s", args.config, exc)
            return 2
        if args.seed is not None:
            cfg["seed"] = int(args.seed)
        if args.n_jobs is not None:
            if args.n_jobs == 0:
                LOG.error("--n-jobs must be non-zero")
                return 2
            cfg["n_jobs"] = int(args.n_jobs)
        if args.reps is not None and args.reps < 1:
            LOG.error("--reps must be >= 1")
            return 2
        if args.command in ("sim", "all") and "sim" not in cfg:
            LOG.error("config %s has no 'sim' section", args.config)
            return 2
        if args.command in ("real", "all") and "real" not in cfg:
            LOG.error("config %s has no 'real' section", args.config)
            return 2
        LOG.info("kfactors %s bench %s: config %s (%s), out %s, seed %d, n_jobs %d",
                 __version__, args.command, args.config, cfg["name"], out, cfg["seed"],
                 cfg["n_jobs"])
        resolved = dict(cfg)
        if args.reps is not None and "sim" in resolved:
            resolved["sim"] = {**resolved["sim"], "reps": args.reps}
        with open(out / f"config_resolved_{args.command}.json", "w", encoding="utf-8") as fh:
            json.dump(resolved, fh, indent=2, default=str)

        n_failed = 0
        if args.command in ("sim", "all"):
            from kfactors.bench.sim_bench import run_sim

            run_sim(cfg, out, reps=args.reps, calibrate=True if args.calibrate else None,
                    designs=args.designs, use_cache=not args.no_cache)
        if args.command in ("real", "all"):
            from kfactors.bench.real_bench import run_real

            res = run_real(cfg, out, panels=args.panels, use_cache=not args.no_cache,
                           fail_fast=args.fail_fast)
            n_failed += res["n_failed"]
            if res["n_skipped"]:
                LOG.warning("%d experiment(s) skipped: strategy not implemented yet",
                            res["n_skipped"])

        from kfactors.bench.leaderboard import build_leaderboard

        try:
            build_leaderboard(out)
        except FileNotFoundError as exc:
            LOG.error("%s", exc)
            return 1
        dt = time.perf_counter() - t0
        to_json({"command": args.command, "config": str(args.config), "seconds": round(dt, 3),
                 "n_failed": n_failed, "kfactors_version": __version__},
                out / f"run_info_{args.command}.json")
        if n_failed:
            LOG.error("finished in %.1f s with %d failed step(s); see %s", dt, n_failed,
                      out / "real" / "status.csv")
            return 1
        LOG.info("finished in %.1f s; leaderboard: %s", dt, out / "leaderboard.md")
        return 0
    finally:
        for h in handlers:
            LOG.removeHandler(h)
            h.close()
        LOG.setLevel(old_level)
        LOG.propagate = old_prop


if __name__ == "__main__":
    sys.exit(main())
