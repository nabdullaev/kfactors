
from pathlib import Path

from kfactors.data import load_jkp
from kfactors.engine import WalkForward, WalkForwardConfig
from kfactors.evaluate import k_curve
from kfactors.plots import plot_spectrum
from kfactors.report import report
from kfactors.strategies.sdf import TangencyPCA
from kfactors.utils import preprocess, eigvals_desc


def main():
    out = Path("results/examples")
    out.mkdir(parents=True, exist_ok=True)

    # Load the monthly US Factor Zoo panel.
    panel = load_jkp("usa", "monthly")

    # Configure the walk-forward experiment.
    config = WalkForwardConfig(
        window=120,
        rebalance=1,
        min_assets=50,
        start="1985-01-01",
    )

    strategy = TangencyPCA(gamma=0.0)
    backtest = WalkForward(panel, strategy, config)

    # Evaluate performance for different numbers of principal components.
    result = k_curve(
        backtest,
        k_grid=range(0, 21),
        metric="sharpe_net",
        n_boot=1000,
    )

    # Save the performance curve and factor-selection rules.
    result.curve.to_csv(out / "factor_zoo_curve.csv")
    result.rules.to_csv(out / "factor_zoo_rules.csv")

    ax = result.plot()
    ax.figure.savefig(
        out / "factor_zoo_k_curve.png",
        dpi=150,
        bbox_inches="tight",
    )

    # Run diagnostics on the last 120 complete observations.
    diagnostic_panel = panel.window(
        end=-1,
        length=120,
        complete=True,
    )

    diagnostic = report(
        diagnostic_panel.returns,
        kmax=20,
        n_boot=200,
        standardize=True,
        random_state=0,
    )

    # Save the diagnostic report and its plots.
    (out / "factor_zoo_report.md").write_text(
        diagnostic.to_markdown(),
        encoding="utf-8",
    )

    report_axes = diagnostic.plot()
    report_axes[0].figure.savefig(
        out / "factor_zoo_diagnostics.png",
        dpi=150,
        bbox_inches="tight",
    )

    # Plot the eigenvalue spectrum of the same window.
    Z, _, _ = preprocess(
        diagnostic_panel.returns,
        standardize=True,
    )
    eigenvalues = eigvals_desc(Z)

    spectrum_ax = plot_spectrum(
        eigenvalues,
        n=Z.shape[0],
        p=Z.shape[1],
    )
    spectrum_ax.figure.savefig(
        out / "factor_zoo_spectrum.png",
        dpi=150,
        bbox_inches="tight",
    )

    print("Oracle k:", result.oracle_k)
    print("Oracle set:", result.oracle_set)
    print("\nFactor-selection rules:")
    print(result.rules)
    print("\nDiagnostic recommendations:")
    print(diagnostic.recommended)
    print("\nResults saved to:", out.resolve())


if __name__ == "__main__":
    main()
