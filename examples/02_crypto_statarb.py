"""Crypto residual statistical-arbitrage example.

Run from the repository root. The first full run downloads Binance Vision archives and can take
several minutes; later runs reuse the local cache.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kfactors.data import load_binance
from kfactors.engine import WalkForward, WalkForwardConfig
from kfactors.evaluate import curve_from_results, k_curve
from kfactors.strategies.statarb import ResidualReversion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="use a smaller 2024 one-year panel for a quick smoke test")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="number of stationary-bootstrap draws (default: 1000)")
    return parser.parse_args()


def save_curve(result, path: Path) -> None:
    """Save a k-curve table and its plot with a shared filename prefix."""
    result.to_frame().to_csv(path.with_suffix(".csv"), index=False)
    ax = result.plot()
    ax.figure.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    out = Path("results/examples")
    out.mkdir(parents=True, exist_ok=True)

    # The quick mode is a smoke test. The default is the documented four-year experiment.
    load_kwargs = dict(top=50, start="2022-01-01", end="2025-12-31", n_jobs=32)
    if args.quick:
        load_kwargs.update(top=30, start="2024-01-01", end="2024-12-31", max_symbols=60)
    panel = load_binance(**load_kwargs)

    config = WalkForwardConfig(
        window=120,
        rebalance=1,
        lag=1,
        cost_bps=10,
        min_assets=20,
    )
    strategy = ResidualReversion()
    engine = WalkForward(panel, strategy, config)
    grid = range(0, 16)

    # Run all backtests once. Re-score the same results gross of costs, avoiding a second run.
    net = k_curve(engine, k_grid=grid, metric="sharpe_net", n_boot=args.n_boot)
    gross = curve_from_results(net.results, k_grid=grid,
                               rules=net.rules.index, metric="sharpe_gross",
                               n_boot=args.n_boot)
    save_curve(net, out / "crypto_statarb_net_curve")
    save_curve(gross, out / "crypto_statarb_gross_curve")

    fixed_turnover = pd.Series(
        {f"k={k}": result.summary()["mean_turnover"]
         for k, result in net.results.items() if isinstance(k, int)},
        name="mean_turnover",
    )
    fixed_turnover.to_csv(out / "crypto_statarb_turnover.csv")

    print("Net oracle k:", net.oracle_k)
    print("Net oracle set:", net.oracle_set)
    print("Gross oracle k:", gross.oracle_k)
    print("Gross oracle set:", gross.oracle_set)
    print("\nNet rule comparison:")
    print(net.rules)
    print("\nMean turnover by fixed k:")
    print(fixed_turnover)
    print("\nResults saved to:", out.resolve())


if __name__ == "__main__":
    main()
