
# Factor Zoo: Portfolio Performance and Factor Selection

This example demonstrates how to use `kfactors` to estimate the number of principal components that maximizes portfolio performance on the US Factor Zoo dataset.

The experiment uses a walk-forward evaluation procedure to avoid look-ahead bias.

## 1. Loading the Data

Load the monthly US Factor Zoo dataset:

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

## 4. Visualizing the Results

Plot the performance curve:

```python
result.plot()
```

The curve shows how portfolio performance changes with the number of principal components. Bootstrap confidence bands help assess uncertainty in the estimated performance.

The result also provides information about the oracle number of factors, its confidence set, and the performance of different factor-selection rules.

## 5. Saving the Results

Save the performance curve, factor-selection results, and plot:

```python
from pathlib import Path

out = Path("results/examples")
out.mkdir(parents=True, exist_ok=True)

result.curve.to_csv(out / "factor_zoo_curve.csv")
result.rules.to_csv(out / "factor_zoo_rules.csv")

result.plot().figure.savefig(
    out / "factor_zoo_k_curve.png",
    dpi=150,
)

print("Results saved to", out.resolve())
```

## 6. Interpretation

This experiment helps investigate whether a relatively small number of principal components is sufficient to achieve good portfolio performance.

Compare the estimated number of factors across selection rules and examine whether their choices belong to the oracle confidence set.

The conclusions should be based on the actual experimental results rather than assumed in advance.
