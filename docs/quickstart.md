# Quick start

This guide shows the basic workflow of `kfactors`: loading financial data, configuring a backtest, and evaluating a portfolio strategy.

## 1. Import the required modules

```python
from kfactors.data import load_jkp
from kfactors.engine import WalkForward, WalkForwardConfig
from kfactors.strategies.sdf import TangencyPCA
from kfactors.evaluate import k_curve
```

## 2. Load the data

Load the monthly US factor zoo dataset:

```python
panel = load_jkp("usa", "monthly")
```

## 3. Configure the backtest

Set the rolling window, rebalance frequency, and minimum number of assets:

```python
config = WalkForwardConfig(
    window=120,
    rebalance=1,
    min_assets=50,
    start="1985-01-01",
)
```

## 4. Run the strategy

Create a walk-forward backtest using the Tangency PCA strategy:

```python
strategy = TangencyPCA(gamma=0.0)
backtest = WalkForward(panel, strategy, config)
```

## 5. Evaluate the results

Estimate the performance for different numbers of principal components:

```python
result = k_curve(
    backtest,
    k_grid=range(0, 21),
    metric="sharpe_net",
    n_boot=1000,
)
```

Display the resulting performance curve:

```python
result.plot()
```

The result contains the estimated performance curve and information about the selected number of factors.

For more detailed examples, see the [Factor Zoo](examples/factor_zoo.md) and [Ken French robustness](examples/french_robustness.md) guides.