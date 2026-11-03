# Card B: Avellaneda-Lee residual stat-arb (the "trading evaluator")

| | |
|---|---|
| **File you edit** | `src/kfactors/strategies/statarb.py` (only this file) |
| **Tests you must make pass** | `tests/teammates/test_card_B_statarb.py` (do not edit it) |
| **Size** | about 80-120 lines of code, in 5 small functions |
| **Difficulty** | medium; do it step by step, each step has its own tests |
| **Prerequisites** | numpy, pandas, linear regression; Card A is a good warm-up |

## 1. Goal

Implement the statistical-arbitrage strategy of Avellaneda & Lee (2010) as a kfactors
`Strategy`. It must:

1. Compute each asset's residual return after removing k PCA factors.
2. Model the cumulative residual as a mean-reverting (Ornstein-Uhlenbeck) process.
3. Compute the **s-score**: how far the asset is from its equilibrium.
4. Buy assets that are "too cheap" and sell assets that are "too rich", using the paper's
   thresholds.
5. Hedge every position with the factor portfolios, so that the book is factor-neutral.

You implement five functions in this order. Each one has its own tests, so you can finish and
check them one at a time:

| Step | Function | What it does |
|---|---|---|
| 1 | `fit_ar1(x)` | least-squares AR(1) fit `x[t+1] = a + b x[t] + zeta` |
| 2 | `s_scores(residuals, max_tau=..., center=...)` | OU parameters and s-score per asset |
| 3 | `update_positions(s, previous, ...)` | the open/close rule, positions in {-1, 0, +1} |
| 4 | `hedge_weights(q, Q, betas)` | hedge with eigenportfolios, gross exposure 1 |
| 5 | `ResidualReversion.weights(window, model, context)` | glue: steps 1-4 plus state |

## 2. Background

**Why this strategy exists.** kfactors asks how many factors k a PCA factor model should have,
and what the choice costs. This strategy measures the **trading** cost of a wrong k. Residuals
are "returns minus the part explained by k factors". If k is too small, the residuals still
contain market moves, which do not mean-revert, and the strategy loses money. If k is too large,
the factors swallow the idiosyncratic moves we want to trade. The realised Sharpe ratio as a
function of k is the k-curve. In our synthetic tests the reference solution reaches an
annualised Sharpe of about 10.9 with the true k = 1, and only 0.7 with k = 0.

**The model (Avellaneda & Lee 2010, Sec. 2-4).** For each asset `i`, regress its last `L = 60`
daily returns on the k eigenportfolio returns:
`R_i = beta_0i + sum_j beta_ij F_j + eps_i`. The core already does this in
`model.residual_returns(...)`. The cumulative residual `X_i(t) = eps_i(1) + ... + eps_i(t)`
behaves like a spring around an equilibrium level (an Ornstein-Uhlenbeck process). In discrete
time this is an AR(1): `X(t+1) = a + b X(t) + zeta`. From `a`, `b` and the variance of `zeta`:

* the speed of mean reversion is `kappa = -ln(b)`, and the mean-reversion time is
  `tau = 1/kappa` days;
* the equilibrium mean is `m = a / (1 - b)`;
* the equilibrium standard deviation is `sigma_eq = sqrt(var(zeta) / (1 - b^2))`;
* the **s-score** is `s = (X(L) - m) / sigma_eq`.

Only assets that revert fast enough are traded: `tau < L/2`, i.e. 30 days. AL use a "modified"
s-score in which `m` is centred across assets.

**The trading rule (AL Sec. 4).**
* Open a long if `s < -1.25` ("cheap").
* Open a short if `s > +1.25` ("rich").
* Close a long once `s > -0.50`.
* Close a short once `s < +0.75`.

A position stays open between those thresholds. You therefore need to remember yesterday's
positions; the engine gives you `context.state`, a dict that persists during one backtest. A
long of 1 dollar in asset `i` is hedged by shorting `beta_ij` dollars of eigenportfolio `j`. The
hedged weights are `w = q - Q (betas' q)`, which are then scaled so that `sum |w| = 1`.

**References**
* M. Avellaneda, J.-H. Lee (2010), "Statistical arbitrage in the US equities market",
  *Quantitative Finance* 10(7), 761-782. Read Sec. 2 (PCA, eigenportfolios), Sec. 4 (s-score,
  thresholds) and the appendix (AR(1) estimation).
* G. E. Uhlenbeck, L. S. Ornstein (1930), "On the theory of the Brownian motion", *Physical
  Review* 36, 823. Background on OU processes only; you do not need to read it.
* `docs/design.md` covers the `Strategy`, `Context` and `FactorModel` interfaces and the engine
  timing.
* Core helpers you call: `model.residual_returns(X)`, `model.eigenportfolios()`.

## 3. Step-by-step plan

0. `git switch -c card-b-statarb`. Read the module docstring of
   `src/kfactors/strategies/statarb.py`. Every function's docstring is its exact specification,
   including the edge cases.
1. **`fit_ar1`**: use the closed-form simple regression given in the docstring. Validate the
   input (`ValueError`). Return `(nan, nan, nan)` when `x[:-1]` is constant.
   Test: `pytest tests/teammates/test_card_B_statarb.py -k fit_ar1 -v`.
2. **`s_scores`**:
   * `X = np.cumsum(residuals, axis=0)`, then call `fit_ar1` on every column.
   * Compute `kappa`, `tau`, `m`, `sigma_eq` only where `0 < b < 1`; elsewhere set NaN.
   * `eligible = (0 < b < 1) & (var_zeta > 0) & (tau < max_tau)`.
   * If `center`, subtract the mean of `m` over eligible assets.
   * `s = (X[-1] - m) / sigma_eq` for eligible assets, NaN otherwise.
   * Return a DataFrame with columns exactly `S_SCORE_COLUMNS`.
   Test with `-k s_scores`.
3. **`update_positions`**: implement the table in the docstring. All comparisons are strict
   (`<`, `>`). Align `previous` to `s.index`: assets that are missing start at 0, extra assets
   are dropped. Return integers. Test with `-k update_positions`.
4. **`hedge_weights`**: compute `h = q - Q @ (betas.T @ q)`, divide by `abs(h).sum()`, and
   return zeros when nothing is held. Test with `-k hedge`.
5. **`ResidualReversion.weights`**: follow the 8 steps in its docstring. Store `q` in
   `context.state["positions"]` and the s-score table in `context.state["s_scores"]`. Then run
   the whole file.
6. Run `ruff`, commit, push and open the PR.

## 4. How to run your tests

```bash
.venv/bin/python -m pytest tests/teammates/test_card_B_statarb.py -v            # all steps
.venv/bin/python -m pytest tests/teammates/test_card_B_statarb.py -v -k fit_ar1 # one step
.venv/bin/ruff check src/kfactors/strategies/statarb.py
```

Tests for unimplemented steps show `SKIPPED (Card B step not implemented yet ...)`.

## 5. What "done" means

* All tests in `tests/teammates/test_card_B_statarb.py` pass with 0 skipped, and `ruff` is clean.
* Only `src/kfactors/strategies/statarb.py` changed.
* The tests check the following:
  * `fit_ar1` agrees with `numpy.linalg.lstsq`.
  * The s-scores agree with an independent re-derivation; the `tau` filter and the centring
    work.
  * All 18 threshold transitions match the AL rule.
  * The hedged book has exactly zero covariance with every factor over the estimation window.
  * On a planted panel, the strategy goes long the cheap asset and short the rich one.
  * Only the last `L` rows are used, the state is carried between dates, and the output is
    deterministic.
  * In the walk-forward engine, gross exposure is always 0 or 1, and the true k = 1 earns a
    gross Sharpe above 5, versus k = 0.

## 6. Suggested prompts for ChatGPT / Claude

Work one function at a time and paste only that function's docstring.

> Here is the docstring of `fit_ar1` from my project (pasted below). Implement the body with
> numpy only, following the docstring exactly, including the ValueError cases and the
> `(nan, nan, nan)` case. Do not change the signature.

> Implement `update_positions` from this docstring (pasted). Write it as a clear loop first,
> with a comment for each branch of the rule. All comparisons must be strict.

> My test `test_book_is_factor_neutral_on_estimation_window` fails (output pasted). Explain
> what "factor-neutral" means for `w = q - Q (B' q)` and what my code might be doing wrong.

> Explain the Avellaneda-Lee s-score with a 5-number numerical example of a cumulative
> residual series.

Never let the AI edit `tests/` or core files (`model.py`, `engine.py`, ...). If it says a core
function is wrong, ask the core team in the PR instead.

## 7. Pull request checklist

- [ ] Branch `card-b-...`; the PR targets `main`; only `statarb.py` changed.
- [ ] `pytest tests/teammates/test_card_B_statarb.py -v`: all passed, 0 skipped. Paste the
      summary line in the PR.
- [ ] `ruff check src/kfactors/strategies/statarb.py`: clean.
- [ ] No `print` statements left; no new dependencies.
- [ ] Docstrings unchanged, or updated if you clarified something.

## 8. Common pitfalls

* **Look-ahead.** Use only `window.returns` (past rows) and the last `L` rows of it. Never use
  `panel` or future data. The engine's leakage test corrupts future rows and checks that your
  weights do not change.
* **Wrong rows for the residuals.** Call `model.residual_returns(R[-L_eff:])`, not
  `residual_returns()` with no argument (that uses the whole window).
* **Sign of the trade.** A *low* s-score (`s < -1.25`) means *cheap*, so you **buy** (`+1`).
* **Swapped exit thresholds.** Longs close at `-0.50`, shorts close at `+0.75`. They are not
  symmetric.
* **Forgetting the state.** Positions must persist between dates via `context.state`. A new
  `Context()` means "start flat".
* **`log(b)` for `b <= 0` or division by `1 - b` for `b >= 1`.** These assets are simply not
  eligible. Produce NaN, never `inf`, and suppress numpy warnings with `np.errstate` if needed.
* **NaN weights.** The engine rejects non-finite weights. Ineligible assets get position 0, not
  NaN weights.
* **Units.** Residuals are in return units (decimals). The s-score is unit-free. `kappa` is
  per period (per row); AL multiply it by 252 to annualise, but we do not.
