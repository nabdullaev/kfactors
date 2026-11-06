# Choosing the number of factors

The `kfactors` library provides several methods for estimating the number of latent factors in a data matrix.

Different methods use different statistical assumptions and selection criteria. The choice of method may affect the estimated number of factors.

## Available methods

### 1. Baseline methods

Baseline methods provide simple reference estimates:

- **Variance Share (`variance_share`)** — selects the number of components needed to explain a specified proportion of total variance.
- **Kaiser (`kaiser`)** — uses the Kaiser eigenvalue criterion.

### 2. Econometric methods

These methods are based on established factor-number estimation procedures:

- **Bai–Ng (`bai_ng`)** — uses information criteria to estimate the number of factors.
- **Ahn–Horenstein (`ahn_horenstein`)** — uses eigenvalue-based statistics, including ER.
- **Onatski (`onatski`)** — estimates the number of factors using the behavior of the eigenvalue spectrum.

### 3. Random Matrix Theory methods

These methods use random matrix theory to distinguish informative eigenvalues from noise:

- **Marchenko–Pastur (`mp`)** — compares the observed eigenvalue spectrum with a Marchenko–Pastur noise distribution.
- **Market-adjusted Marchenko–Pastur (`mp_market`)** — provides a market-adjusted version of the Marchenko–Pastur approach.
- **Gavish–Donoho (`gavish_donoho`)** — uses an optimal singular-value threshold to separate signal from noise.

### 4. Permutation methods

- **Parallel Analysis (`parallel`)** — compares observed eigenvalues with eigenvalues obtained from permuted data to estimate how many factors should be retained.

## Selecting a method

The main selection interface is `select_k`.

Example:

```python
from kfactors.rules import select_k

result = select_k(
    X,
    method="onatski",
    kmax=20,
    standardize=True,
)

print(result)
```

Here:

- `X` is the input data matrix.
- `method` specifies the factor-number selection method.
- `kmax` sets the maximum number of factors to consider.
- `standardize` controls whether the input data are standardized.

The selection function returns a `KResult` object containing the estimation result and related information.

## How to choose a method

There is no universally best method for every dataset.

Consider the following guidelines:

| Situation | Methods to consider |
|---|---|
| Simple baseline comparison | Variance Share, Kaiser |
| Econometric factor estimation | Bai–Ng, Ahn–Horenstein, Onatski |
| Noise and eigenvalue spectrum analysis | Marchenko–Pastur, Gavish–Donoho |
| Permutation-based comparison | Parallel Analysis |

For robust analysis, compare several methods rather than relying on a single estimate.

## Further information

See the [API reference](api/rules.md) for the available functions and their parameters.