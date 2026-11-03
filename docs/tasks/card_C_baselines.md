# Card C: baseline selectors from established libraries

| | |
|---|---|
| **File you edit** | `src/kfactors/baselines.py` (only this file) |
| **Tests you must make pass** | `tests/teammates/test_card_C_baselines.py` (do not edit it) |
| **Size** | 5 short functions, about 60 lines in total |
| **Difficulty** | easy-medium (mostly reading library documentation) |
| **Prerequisites** | numpy, calling scikit-learn / statsmodels |

## 1. Goal

Wrap off-the-shelf implementations of "how many components?" rules so that kfactors can
(a) cross-check its own implementations against independent code and (b) show in the
benchmark what a practitioner gets from standard tools:

| Function | Library call | Must agree with |
|---|---|---|
| `sklearn_variance_share(X, share=0.9)` | `sklearn.decomposition.PCA(n_components=share)` | `select_k(X, "variance_share", share=share).info["k_uncapped"]` |
| `sklearn_mle(X)` | `PCA(n_components="mle")` (Minka 2000) | sklearn itself; returns `None` when `n < p` |
| `statsmodels_bai_ng(X, criterion="IC_p2", kmax=20)` | `statsmodels.multivariate.pca.PCA(...).ic` | `select_k(X, "bai_ng", criterion=..., kmax=...).k` |
| `horns_parallel(X, n_sim, quantile)` | `horns.parallel_analysis` (optional package) | `None` when horns is not installed |
| `optht_threshold(X)` | `optht.optht(beta, sv=sv, sigma=None)` | `select_k(X, "gavish_donoho").info["k_uncapped"]` |

Every function returns a Python `int`, or `None` when the method cannot be applied. In the
`None` case it emits a `BaselineUnavailableWarning` explaining why; that class is already
defined for you.

## 2. Background

**What these rules do.** A return panel `X` has `n` rows (dates) and `p` columns (assets). PCA
finds orthogonal directions ordered by explained variance, and every rule decides how many of
them are real "factors" rather than noise:

* The **cumulative variance share** keeps components until 90% of the variance is explained
  (Jolliffe 2002). It is simple, but has no statistical justification.
* **Minka's MLE** chooses the dimension with the highest Bayesian evidence under a
  probabilistic-PCA model (Minka 2000). scikit-learn only supports it when `n >= p`.
* The **Bai & Ng (2002) information criteria** minimise
  `log(residual variance) + k * penalty(n, p)`. This is the econometrics standard.
* **Horn's parallel analysis** keeps components whose eigenvalues exceed those of random data
  of the same shape (Horn 1965).
* The **Gavish-Donoho optimal hard threshold** keeps singular values above
  `omega(beta) * median(singular values)`, the optimal cutoff for matrix denoising
  (Gavish & Donoho 2014).

**Why we need baselines.** kfactors re-implements these rules (in `kfactors.rules`) so that they
handle `p > n`, heavy tails and a `kmax` cap consistently. Agreement with independent library
code is strong evidence that our implementations are right. If your wrapper disagrees with the
core, **do not "fix" the core**: report it in your PR. One of the two has a bug, and we want to
know which.

**Conventions you must follow** (they are what makes the numbers comparable):
* Pre-process exactly like kfactors: `Z, _, _ = kfactors.utils.preprocess(X, standardize)`.
  This centres each column and, if `standardize=True`, divides it by its ddof=0 standard
  deviation. The result is PCA on the correlation matrix.
* After centring, only `m = min(n - 1, p)` singular values are non-zero, so `optht` must use
  `beta = m / max(n - 1, p)` and the first `m` singular values.
* `kmax` for Bai-Ng is clipped with `kfactors.rules.base.effective_kmax(n, p, kmax)`.

**References**
* I. T. Jolliffe (2002), *Principal Component Analysis*, 2nd ed., Springer, Sec. 6.1.1.
* T. P. Minka (2000), "Automatic choice of dimensionality for PCA", *NIPS 13*.
* J. Bai, S. Ng (2002), "Determining the number of factors in approximate factor models",
  *Econometrica* 70(1), 191-221.
* J. L. Horn (1965), "A rationale and test for the number of factors in factor analysis",
  *Psychometrika* 30(2), 179-185.
* M. Gavish, D. L. Donoho (2014), "The optimal hard threshold for singular values is
  4/sqrt(3)", *IEEE Trans. Information Theory* 60(8), 5040-5053.
* Library documentation: scikit-learn `PCA`, statsmodels `multivariate.pca.PCA`, the `optht`
  README.

## 3. Step-by-step plan

0. `git switch -c card-c-baselines`. Read the module docstring and each function's docstring in
   `src/kfactors/baselines.py`.
1. **`sklearn_variance_share`**:
   1. Validate `0 < share < 1`, otherwise raise `ValueError`.
   2. Compute `Z = preprocess(X, standardize)[0]`.
   3. Return `int(PCA(n_components=share, svd_solver="full").fit(Z).n_components_)`.
2. **`sklearn_mle`**:
   1. Compute `Z` as above, then `n, p = Z.shape`.
   2. If `n < p`, call `warnings.warn(f"... n < p (n={n}, p={p}) ...", BaselineUnavailableWarning,
      stacklevel=2)` and return `None`.
   3. Otherwise return the int `n_components_` of `PCA(n_components="mle", svd_solver="full")`.
3. **`statsmodels_bai_ng`**:
   1. Validate `criterion in ("IC_p1", "IC_p2", "IC_p3")`.
   2. Compute `A = as_2d_array(X)` and `kmax_eff = effective_kmax(n, p, kmax)`. It raises
      `ValueError` for `kmax < 1`.
   3. Run `pc = PCA(pd.DataFrame(A), ncomp=kmax_eff, standardize=standardize, demean=True)`.
   4. Return `int(pc.ic[criterion].to_numpy().argmin())`.
4. **`optht_threshold`**:
   1. Compute `Z`, then `m = min(n - 1, p)`.
   2. Compute `sv = np.linalg.svd(Z, compute_uv=False)[:m]` and `beta = m / max(n - 1, p)`.
   3. Wrap the call in `try: return int(optht.optht(beta, sv=sv, sigma=None))`, with
      `except ValueError: return 0`. Import as `import optht` and call `optht.optht(...)`.
5. **`horns_parallel`**:
   1. Validate `n_sim >= 1` and `0 < quantile < 1` (raise `ValueError`).
   2. Compute `A = as_2d_array(X)`.
   3. Write `try: import horns` with `except ImportError:`, then warn with "horns is not
      installed" and return `None`.
   4. Otherwise return
      `int(horns.parallel_analysis(A, simulations=n_sim, analysis_type="pca", quantile=quantile))`.
6. Run the tests and `ruff`, commit, open the PR.

**About `horns`.** The latest horns (0.6.0) requires `numpy<2` and `numba`. Our environment uses
numpy 2.x, so horns **cannot be installed** here. Do not try to downgrade numpy. The tests check
the "not installed" path and use a *fake* horns module to check that you call it correctly.

## 4. How to run your tests

```bash
.venv/bin/python -m pytest tests/teammates/test_card_C_baselines.py -v
.venv/bin/python -m pytest tests/teammates/test_card_C_baselines.py -v -k bai_ng   # one function
.venv/bin/ruff check src/kfactors/baselines.py
```

Expected at the end: everything passes, and exactly **one** test is skipped:
`test_horns_real_package_if_installed`, because horns is not installed.

## 5. What "done" means

* Every test passes except that single horns skip, and `ruff` is clean.
* Only `src/kfactors/baselines.py` changed.
* The tests check the following:
  * Exact agreement with kfactors on 6 panels: p < n, p = n, p > n, Gaussian, heavy-tailed and
    heteroscedastic noise, pure noise. This covers both pre-processing modes and all three IC
    criteria.
  * A known-answer check of the variance share computed from eigenvalues.
  * `sklearn_mle` returns `None` with a warning when `n < p` and finds the planted k on
    homoscedastic data.
  * `optht` is called with `beta = (n-1)/n` on a square panel.
  * Correct horns wiring.
* Optional for the PR description: a small table of `k` from every baseline next to the
  matching `select_k` rule on 2-3 of the simulated panels from the test file.

## 6. Suggested prompts for ChatGPT / Claude

> Here is the docstring of `statsmodels_bai_ng` from my project (pasted). Implement the body.
> Use `statsmodels.multivariate.pca.PCA` exactly as described; do not re-implement the
> criterion by hand. Keep the signature.

> In scikit-learn, what does `PCA(n_components=0.9)` select exactly: the smallest k with
> cumulative explained variance ratio `> 0.9` or `>= 0.9`? Show me the relevant line of the
> scikit-learn source.

> My `optht_threshold` returns 3 but kfactors' `gavish_donoho` returns 4 (output pasted).
> What aspect ratio `beta` and which singular values should I pass, given that the columns
> were centred?

Do not let the AI "fix" disagreements by editing `kfactors/rules/*` or the tests. Report them.

## 7. Pull request checklist

- [ ] Branch `card-c-...`; only `src/kfactors/baselines.py` changed.
- [ ] `pytest tests/teammates/test_card_C_baselines.py -v`: all passed, 1 skipped (horns);
      paste the summary line.
- [ ] `ruff check src/kfactors/baselines.py` is clean.
- [ ] Optional packages (`horns`) are imported **inside** the function, never at module level.
- [ ] No new hard dependencies in `pyproject.toml`. statsmodels and optht are already dev
      dependencies.

## 8. Common pitfalls

* **Forgetting to standardise.** sklearn's `PCA` only centres. Pass `Z` from `preprocess`. For
  statsmodels, pass the raw data and `standardize=standardize`; statsmodels standardises itself
  with ddof=0, like kfactors.
* **Off-by-one in k.** `pc.ic` row 0 is k = 0 (no factors). sklearn's `n_components_` is
  already the count.
* **`n` vs `n - 1`.** After centring there are only `n - 1` informative rows. optht must get
  `beta = min(n-1, p) / max(n-1, p)`.
* **Letting library errors escape.** `sklearn_mle` with `n < p` must return `None` with a
  warning, not raise. When `optht` finds nothing, return `0`.
* **NaN.** Validate the input (`preprocess` / `as_2d_array` do it); NaN must raise `ValueError`.
* **numpy integers.** Return `int(...)`, not `np.int64`. Some tests check `type(k) is int`.
* **Look-ahead** is not an issue here: these functions see only the matrix they are given. When
  the engine uses them as callables (`WalkForward.run(callable)`), it passes past windows only.
