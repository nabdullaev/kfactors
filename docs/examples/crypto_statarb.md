
# Crypto Statistical Arbitrage

This example demonstrates how to use `kfactors` for a statistical arbitrage strategy on cryptocurrency market data.

## 1. Loading the Data

Load cryptocurrency data using the Binance data loader:

```python
from kfactors.data import load_binance

panel = load_binance(
    top=50,
    start="2022-01-01",
    end="2025-12-31",
    n_jobs=32,
)
```

## 2. Configuring the Backtest

Configure a rolling-window backtest with monthly rebalancing:

```python
from kfactors.engine import WalkForwardConfig

config = WalkForwardConfig(
    window=120,
    rebalance=1,
    min_assets=10,
)
```

## 3. Running a Portfolio Strategy

The following example uses the Minimum Variance strategy:

```python
from kfactors.engine import WalkForward
from kfactors.strategies.minvar import MinVariance

strategy = MinVariance()
backtest = WalkForward(panel, strategy, config)
```

## 4. Evaluating the Results

Use the performance evaluation tools to examine the strategy:

```python
from kfactors.evaluate import k_curve

result = k_curve(
    backtest,
    k_grid=range(0, 21),
    metric="sharpe_net",
    n_boot=1000,
)

result.plot()
```

## 5. Interpretation

The experiment can be used to study how factor selection affects portfolio performance in cryptocurrency markets.

Cryptocurrency returns may exhibit high volatility, changing correlations, and different market conditions over time. Results should therefore be interpreted with care.
