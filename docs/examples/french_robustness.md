
# Ken French Data: Robustness Analysis

This example describes how to evaluate factor-selection methods on datasets from the Kenneth French Data Library.

The goal is to compare the estimated number of factors across different datasets, frequencies, and rolling-window configurations.

## 1. Datasets

The experiment uses two datasets:

- **100 Size-BM portfolios** — portfolios formed on size and book-to-market characteristics.
- **49 Industry portfolios** — portfolios representing different US industries.

Both monthly and daily data can be considered where available.

## 2. Experimental Configurations

The robustness analysis considers different rolling-window lengths:

- Monthly data: 60 and 120 months.
- Daily data: 252 and 504 trading days.

The analysis compares factor-selection rules under different configurations.

## 3. Running the Experiment

The experiment can be launched using the benchmark command:

```bash
python -m kfactors.bench real --config configs/french_robustness.yaml --out results/french/
```

The configuration file specifies the datasets, frequencies, rolling windows, strategies, and factor-selection rules.

## 4. Comparing the Results

The results can be used to compare:

- The oracle number of factors and its confidence set.
- The number of factors selected by each rule.
- Whether each rule's estimate belongs to the oracle confidence set.
- The stability of the estimates across datasets and configurations.

## 5. Interpretation

The results help assess whether factor-selection methods produce consistent estimates across different portfolio datasets and sampling frequencies.

Differences between configurations may reflect changes in the underlying return structure, sample size, and estimation uncertainty.

The conclusions should be based on the generated benchmark results.
