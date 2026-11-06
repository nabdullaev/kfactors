# Core concepts

## Factor models

Factor models describe the returns of multiple assets using a smaller number of common factors.

A typical factor model can be written as:

\[
R_{it} = \alpha_i + \beta_i^T f_t + \varepsilon_{it}
\]

where:

- \(R_{it}\) is the return of asset \(i\) at time \(t\).
- \(\alpha_i\) is the asset-specific intercept.
- \(\beta_i\) represents factor loadings.
- \(f_t\) is the vector of common factors.
- \(\varepsilon_{it}\) is the idiosyncratic component.

## Number of factors

An important problem in factor modeling is determining how many factors should be included.

Using too few factors may fail to capture important common variation. Using too many factors may introduce noise and reduce the stability of estimates.

The `kfactors` library provides different methods for estimating the appropriate number of factors.

## Principal Component Analysis

Principal Component Analysis (PCA) transforms correlated variables into a set of orthogonal components.

The components are ordered by the amount of variance they explain. The first components capture the largest share of variation in the data.

PCA can be used to construct latent factors from a financial panel.

## Factor selection rules

Factor selection rules determine how many components or factors should be retained.

Different rules use different criteria, such as:

- Eigenvalue thresholds.
- Information criteria.
- Statistical tests.
- Portfolio performance.

The estimated number of factors may differ depending on the selected rule and dataset.

## Walk-forward evaluation

Walk-forward evaluation tests a strategy over time using a rolling or expanding historical window.

At each evaluation step:

1. Historical data are used to estimate the model.
2. The strategy constructs portfolio weights.
3. The portfolio is evaluated on subsequent observations.
4. The process moves forward through time.

This approach helps avoid look-ahead bias because future observations are not used to make past investment decisions.

## Portfolio strategies

The library includes portfolio strategies such as Minimum Variance and Tangency PCA.

These strategies use estimated information from financial data to construct portfolio weights.

Their performance can be compared using metrics such as the Sharpe ratio.