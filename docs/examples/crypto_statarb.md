
# Crypto Statistical Arbitrage

This example evaluates Avellaneda--Lee residual statistical arbitrage on cryptocurrency market
data. PCA first removes the common movement of the crypto market. The strategy then trades only
the residual movement of each coin when it is far from its estimated equilibrium.

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

Configure a daily walk-forward backtest. A 120-day estimation window is used, positions are held
for one day, and `lag=1` is the conservative assumption that a signal at today's close can first
be traded at the next close.

```python
from kfactors.engine import WalkForwardConfig

config = WalkForwardConfig(
    window=120,
    rebalance=1,
    lag=1,
    cost_bps=10,
    min_assets=20,
)
```

## 3. Running Residual Statistical Arbitrage

`ResidualReversion` removes `k` PCA factors, fits a mean-reverting model to the cumulative
residual of each eligible coin, and takes a long or short position when its s-score crosses an
entry threshold. Every position is hedged with the PCA eigenportfolios, so the resulting book is
factor-neutral on the estimation window.

```python
from kfactors.engine import WalkForward
from kfactors.strategies.statarb import ResidualReversion

strategy = ResidualReversion()
backtest = WalkForward(panel, strategy, config)
```

## 4. Evaluating the Results

Use the performance evaluation tools to examine the strategy:

```python
from kfactors.evaluate import k_curve

result = k_curve(
    backtest,
    k_grid=range(0, 16),
    metric="sharpe_net",
    n_boot=1000,
)

result.plot()
```

The complete example saves both a net-of-cost and a gross k-curve. Comparing them shows the drag
from turnover and transaction costs. It also saves the mean turnover for each fixed value of `k`.

## 5. Running the Complete Example

From the repository root:

```bash
python examples/02_crypto_statarb.py
```

The first run downloads Binance Vision daily archives and can take several minutes. Data are
cached outside the repository, so later runs are much faster. For a small smoke test first, use:

```bash
python examples/02_crypto_statarb.py --quick --n-boot 100
```

The script writes the following regenerated outputs to `results/examples/`:

| File | Description |
|---|---|
| `crypto_statarb_net_curve.csv` / `.png` | Net-of-cost Sharpe k-curve with bootstrap intervals |
| `crypto_statarb_gross_curve.csv` / `.png` | Gross Sharpe k-curve using the same backtests |
| `crypto_statarb_turnover.csv` | Mean turnover for each fixed k |

## 6. Interpretation

The key question is whether a statistical rule such as Onatski or Bai--Ng chooses a median `k`
inside the oracle confidence set. The oracle is only a hindsight benchmark: it identifies the
best realised `k` in this experiment and cannot be known in advance.

Compare `k = 0` against positive values of `k`: with `k = 0` the residuals still contain common
market movement, while positive `k` values progressively remove it. Conversely, overly large
`k` can absorb idiosyncratic movements that the strategy needs to trade. Cryptocurrency returns
have high volatility and changing correlations, so conclusions should be based on the full curve
and its bootstrap bands rather than on one visually best point.

The strategy follows Avellaneda and Lee (2010), *Statistical Arbitrage in the U.S. Equities
Market*, adapted here to the Binance crypto panel.
