# Ken French Robustness Experiments

This example evaluates factor-selection rules on two datasets from the Ken French Data Library and examines whether the results are sensitive to the regularization parameter of the Tangency PCA strategy.

## Experimental setup

Two datasets are considered:

- **100 Size-BM portfolios (daily):** daily value-weighted returns, using a 252-day estimation window and rebalancing every 21 trading days.
- **49 Industry portfolios (monthly):** monthly value-weighted returns, using a 120-month estimation window and monthly rebalancing.

The experiment compares nine factor-selection rules:

- Variance Share
- Kaiser
- Bai–Ng
- Ahn–Horenstein
- Onatski
- Parallel Analysis
- Marchenko–Pastur (MP)
- Market-adjusted MP
- Gavish–Donoho

For each dataset, the Tangency PCA strategy is evaluated with two regularization settings: `gamma = 0.0` and `gamma = 0.1`. The candidate number of factors ranges from 0 to 20.

The benchmark uses 200 bootstrap replications, a significance level of 0.10, and random seed 42.

## Running the benchmark

Run the benchmark from the repository root:

```bash
python -m kfactors.bench real --config configs/french_robustness.yaml --out results/french/
```

Generate the summary table and heatmap from the benchmark results:

```bash
python examples/03_french_robustness.py
```

The script saves the following files:

- `results/french/french_robustness_summary.csv`
- `results/french/french_robustness_heatmap.png`

## Results

The benchmark completed successfully for all four experimental configurations.

| Dataset | Gamma | Observations | Oracle k | Oracle set |
|---|---:|---:|---:|---|
| 100 Size-BM, daily | 0.0 | 14,307 | 20 | {17, 20} |
| 100 Size-BM, daily | 0.1 | 14,307 | 20 | {17, 20} |
| 49 Industries, monthly | 0.0 | 548 | 1 | {1, 2, 8, 10, 11, 14, 15} |
| 49 Industries, monthly | 0.1 | 548 | 1 | {1, 2, 8, 10, 11, 14, 15} |

The median number of factors selected by each rule is shown below.

| Rule | Daily, γ=0.0 | Daily, γ=0.1 | Monthly, γ=0.0 | Monthly, γ=0.1 |
|---|---:|---:|---:|---:|
| Variance Share | 20 | 20 | 17 | 17 |
| Kaiser | 5 | 5 | 5 | 5 |
| Bai–Ng | 3 | 3 | 3 | 3 |
| Ahn–Horenstein | 1 | 1 | 1 | 1 |
| Onatski | 3 | 3 | 2 | 2 |
| Parallel Analysis | 2 | 2 | 2 | 2 |
| MP | 2 | 2 | 2 | 2 |
| MP Market | 6 | 6 | 5 | 5 |
| Gavish–Donoho | 7 | 7 | 6 | 6 |

The heatmap in `french_robustness_heatmap.png` visualizes these values across all four configurations.

## Interpretation

The results show that the selected median number of factors is unchanged when the Tangency PCA regularization parameter increases from 0.0 to 0.1. This holds for all nine rules and both datasets.

However, the daily and monthly datasets produce different estimates for some rules. For example, Variance Share selects a median of 20 factors for the daily portfolios and 17 for the monthly portfolios. Onatski selects 3 and 2 factors, respectively.

The oracle results also differ substantially between the datasets. The daily experiment has an oracle value of 20, while the monthly experiment has an oracle value of 1. This indicates that the preferred factor dimension depends on the dataset and sampling frequency.

The oracle set contains values whose performance is statistically close to the best fixed number of factors under the benchmark's criterion. Therefore, a rule can select a value outside this set even if its performance is relatively competitive.

## Notes

The `100_size_bm` dataset contains portfolios with incomplete historical coverage. The benchmark therefore uses complete rolling windows, and the number of available assets may vary over time.

All returns are handled in decimal units. The benchmark uses 252 periods per year for daily data and 12 periods per year for monthly data when annualizing performance metrics.