# Factor Zoo: Portfolio Performance and Factor Selection

This example demonstrates how to use `kfactors` to estimate the number of principal components that maximizes portfolio performance on the US Factor Zoo dataset.

The experiment uses a walk-forward evaluation procedure to avoid look-ahead bias. It also compares factor-selection rules and performs additional diagnostics on the last complete observation window.

## 1. Loading the Data

The example uses the monthly US Factor Zoo dataset:

```python
from kfactors.data import load_jkp

panel = load_jkp("usa", "monthly")
```

## 2. Configuring the Backtest

The backtest uses a rolling window of 120 months, monthly rebalancing, and a minimum of 50 assets.

```python
from kfactors.engine import WalkForwardConfig

config = WalkForwardConfig(
    window=120,
    rebalance=1,
    min_assets=50,
    start="1985-01-01",
)
```

## 3. Running the Tangency PCA Strategy

Create a Tangency PCA strategy and run the walk-forward backtest:

```python
from kfactors.engine import WalkForward
from kfactors.strategies.sdf import TangencyPCA

strategy = TangencyPCA(gamma=0.0)
backtest = WalkForward(panel, strategy, config)
```

Evaluate portfolio performance for different numbers of principal components:

```python
from kfactors.evaluate import k_curve

result = k_curve(
    backtest,
    k_grid=range(0, 21),
    metric="sharpe_net",
    n_boot=1000,
)
```

## 4. Visualizing Portfolio Performance

Plot the performance curve:

```python
result.plot()
```

The curve shows how portfolio performance changes with the number of principal components. Bootstrap confidence bands help assess uncertainty in the estimated performance.

The result also provides the oracle number of factors, its confidence set, and the performance of different factor-selection rules.

## 5. Factor Diagnostics

The example also examines the last 120 complete observations:

```python
from kfactors.report import report

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
```

The report includes factor-selection results, bootstrap information, and diagnostic plots.

The eigenvalue spectrum can also be examined:

```python
from kfactors.plots import plot_spectrum
from kfactors.utils import preprocess, eigvals_desc

Z, _, _ = preprocess(
    diagnostic_panel.returns,
    standardize=True,
)

eigenvalues = eigvals_desc(Z)

plot_spectrum(
    eigenvalues,
    n=Z.shape[0],
    p=Z.shape[1],
)
```

## 6. Running the Complete Example

The complete, reproducible script is available at `examples/01_factor_zoo.py`.

Run it from the repository root:

```bash
python examples/01_factor_zoo.py
```

The script saves the results to `results/examples/`:

| File | Description |
|---|---|
| `factor_zoo_curve.csv` | Performance curve and bootstrap intervals |
| `factor_zoo_rules.csv` | Factor-selection rule results |
| `factor_zoo_k_curve.png` | Portfolio performance versus k |
| `factor_zoo_report.md` | Diagnostic report |
| `factor_zoo_diagnostics.png` | Diagnostic plots |
| `factor_zoo_spectrum.png` | Eigenvalue spectrum |

## 7. Experimental Results

In the observed run, the walk-forward experiment produced:

- Oracle k: **18**.
- Oracle set: **[15, 17, 18, 19, 20]**.
- Gavish–Donoho selected k = 20, which belongs to the oracle set.
- Bai–Ng selected k = 16, close to the oracle range but outside the set.

The diagnostic report for the last 120-month window recommended the Onatski rule with k = 2. Its bootstrap probability of selecting the same value was 0.865.

These results are not contradictory. The walk-forward k-curve evaluates portfolio performance across multiple evaluation windows, while the diagnostic report examines the factor structure of one particular window.

## 8. Interpretation

The experiment illustrates that the number of factors maximizing portfolio performance does not necessarily coincide with the number suggested by statistical factor-selection rules.

The oracle value is an empirical benchmark, not a factor-selection rule that can be used without knowledge of the evaluation results. Comparing the rules with the oracle set helps assess how much portfolio performance is lost when using an estimated number of factors.

Results may vary with the dataset version, preprocessing choices, and bootstrap settings.