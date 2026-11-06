
"""Summarize Ken French robustness experiments and plot a heatmap."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from kfactors.plots import plot_heatmap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "french"


def collect_results() -> pd.DataFrame:
    """Collect rule results from all four experiments."""
    records = []

    experiments = [
        ("french_100_daily", "tangency_gamma_0", 0.0),
        ("french_100_daily", "tangency_gamma_01", 0.1),
        ("french_49_monthly", "tangency_gamma_0", 0.0),
        ("french_49_monthly", "tangency_gamma_01", 0.1),
    ]

    for panel, experiment, gamma in experiments:
        folder = OUT / "real" / panel / "experiments" / experiment

        with (folder / "summary.json").open(encoding="utf-8") as file:
            summary = json.load(file)

        rules = pd.read_csv(folder / "rules.csv")
        design = f"{panel} / gamma={gamma}"

        for _, row in rules.iterrows():
            records.append(
                {
                    "panel": panel,
                    "experiment": experiment,
                    "design": design,
                    "gamma": gamma,
                    "rule": row["rule"],
                    "k_median": row["k_median"],
                    "k_mean": row["k_mean"],
                    "metric": row["metric"],
                    "in_set": row["in_set"],
                    "regret": row["regret"],
                    "oracle_k": summary["oracle_k"],
                    "oracle_set": summary["oracle_set"],
                }
            )

    return pd.DataFrame(records)


def main_example() -> None:
    """Generate summary tables and a heatmap from existing results."""
    OUT.mkdir(parents=True, exist_ok=True)
    results = collect_results()

    results.to_csv(
        OUT / "french_robustness_summary.csv",
        index=False,
    )

    heatmap_data = results[["rule", "design", "k_median"]].rename(
        columns={"k_median": "value"}
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_heatmap(
        heatmap_data,
        ax=ax,
        value="value",
        title="Median number of factors by rule and design",
        fmt=".1f",
    )
    fig.tight_layout()
    fig.savefig(OUT / "french_robustness_heatmap.png", dpi=200)
    plt.close(fig)

    print("\nFrench robustness summary:")
    print(results.to_string(index=False))
    print(f"\nSummary saved to: {OUT / 'french_robustness_summary.csv'}")
    print(f"Heatmap saved to: {OUT / 'french_robustness_heatmap.png'}")


if __name__ == "__main__":
    main_example()
