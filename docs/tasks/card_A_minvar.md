# Card A: minimum-variance strategy (the "risk evaluator")

| | |
|---|---|
| **File you edit** | `src/kfactors/strategies/minvar.py` (only this file) |
| **Tests you must make pass** | `tests/teammates/test_card_A_minvar.py` (do not edit it) |
| **Size** | about 10 lines of code |
| **Difficulty** | easy, a good first PR |
| **Prerequisites** | Python basics, numpy arrays, `git` (see `CONTRIBUTING.md`) |

## 1. Goal

Implement `MinVariance.weights(window, model, context)`. It must return the **global
minimum-variance (GMV) portfolio** computed from the factor-model covariance matrix:

```
S = model.covariance(residual=self.residual)          # p x p numpy array
w = S^{-1} 1 / (1' S^{-1} 1)                          # weights sum to 1
```

Return `w` as a `pandas.Series` indexed by `window.returns.columns`.

## 2. Background (read this once)

**Why this strategy exists.** kfactors studies one question: *how many factors k should a
statistical (PCA) factor model have?* Statistics gives several answers (the "rules" in
`kfactors.rules`). We also want to know what the choice **costs in money**. So we plug each
choice of k into a strategy, run it out of sample with the walk-forward engine, and compare
results. Your strategy measures the **risk** side. A covariance matrix with too few factors
misses correlations. One with too many treats noise as structure. Either way the GMV portfolio
built from it has a higher realised volatility out of sample. The k with the lowest realised
volatility is the "economically best" k for risk.

**The formula.** The GMV portfolio solves `min_w w'Sw` subject to `sum(w) = 1`. Short positions
are allowed. Solving with a Lagrange multiplier gives the closed form above (Merton 1972). With
more assets than observations (`p > n`) the plain sample covariance is singular and cannot be
inverted. The factor-model covariance `S = B Cov(F) B' + D` is always invertible, because the
diagonal residual matrix `D` is strictly positive. This is the reason factor models are used for
large portfolios (Fan, Liao & Mincheva 2013). The core team already implemented `S`: you only
call `model.covariance(...)`.

**Where your code is called.** At each decision date the engine (`kfactors.engine.WalkForward`)
does five things:
1. It cuts a past-only window of returns.
2. It fits `FactorModel(k)` on that window.
3. It calls your `weights(window, model, context)`.
4. It holds the returned weights over the next period(s).
5. It records the realised returns.

`kfactors.evaluate.k_curve(..., metric="neg_vol")` then repeats this for k = 0, 1, 2, ...
and draws the k-curve.

**References**
* R. C. Merton (1972), "An analytic derivation of the efficient portfolio frontier", *Journal of
  Financial and Quantitative Analysis* 7(4), 1851-1872. This is the GMV closed form.
* J. Fan, Y. Liao, M. Mincheva (2013), "Large covariance estimation by thresholding principal
  orthogonal complements", *JRSS-B* 75(4). It covers factor-model covariances when p > n.
* O. Ledoit, M. Wolf (2004), "A well-conditioned estimator for large-dimensional covariance
  matrices", *J. Multivariate Analysis* 88(2). This is the `residual="ledoit_wolf"` option.
* `docs/design.md` has the `Strategy` and `FactorModel` interfaces and the engine timing.

## 3. Step-by-step plan

1. Create your branch: `git switch -c card-a-minvar`.
2. Open `src/kfactors/strategies/minvar.py` and read the docstring of `weights`. It is the
   specification.
3. Replace `raise NotImplementedError(...)` with:
   1. `S = model.covariance(residual=self.residual)`
   2. `x = np.linalg.solve(S, np.ones(S.shape[0]))`. Use `solve`, **not** `np.linalg.inv`.
   3. Check that `x.sum() > 0` and that everything is finite; otherwise `raise ValueError(...)`.
   4. `w = x / x.sum()`
   5. `return pd.Series(w, index=window.returns.columns, name=self.name)`
4. Add `import numpy as np` at the top of the file.
5. Run the tests (section 4). Fix any failure, then run them again.
6. Run `ruff` (section 4).
7. Optional extra: write a short example in the class docstring.
8. Commit, push and open a pull request (see `CONTRIBUTING.md`).

## 4. How to run your tests

From the repository root:

```bash
.venv/bin/python -m pytest tests/teammates/test_card_A_minvar.py -v
.venv/bin/ruff check src/kfactors/strategies/minvar.py
```

Before you start, every test shows `SKIPPED (Card A not implemented yet ...)`. That is expected.
After your change they must show `PASSED`.

## 5. What "done" means

* All tests in `tests/teammates/test_card_A_minvar.py` pass, none is skipped, and `ruff` is
  clean.
* You changed only `src/kfactors/strategies/minvar.py`.
* What the tests check:
  * The weights equal the closed form, computed independently with an explicit inverse, for
    k = 0, 1, 3 and both residual types.
  * They sum to 1, work for p > n and do not modify their inputs.
  * They pass a 2-asset hand-computed example: `[0.25, 0.75]`.
  * On a simulated 3-factor market, k = 3 gives a much lower true risk than k = 0 or k = 50.
  * The strategy runs through the walk-forward engine, with a fixed k and with a rule-selected
    k.
* Optional demo for your PR description. On a simulated 4-factor panel, the reference solution
  gives a k-curve whose best k is 4: realised volatility is lowest at k = 4 and the oracle
  confidence set is {3, 4, 6}.

```python
import pandas as pd
from kfactors.data.panel import Panel
from kfactors.engine import WalkForward, WalkForwardConfig
from kfactors.evaluate import k_curve
from kfactors.sim import SimSpec, simulate
from kfactors.strategies.minvar import MinVariance

d = simulate(SimSpec(n=600, p=80, k=4, noise="t"), random_state=1)
dates = pd.bdate_range("2020-01-01", periods=600)
cols = [f"S{i}" for i in range(80)]
panel = Panel(pd.DataFrame(d.X * 0.01, index=dates, columns=cols),
              pd.DataFrame(True, index=dates, columns=cols))
kc = k_curve(WalkForward(panel, MinVariance(), WalkForwardConfig(window=150, rebalance=21)),
             k_grid=[0, 1, 2, 3, 4, 6, 8, 12, 20], rules=["onatski", "bai_ng"],
             metric="neg_vol", n_boot=200)
print(kc.curve, kc.oracle_k, kc.oracle_set)
```

## 6. Suggested prompts for ChatGPT / Claude

Paste the whole docstring of the file together with the prompt.

> I am implementing a method in a Python package. Here is the file
> `src/kfactors/strategies/minvar.py` (pasted below). Implement only the body of
> `MinVariance.weights` following its docstring exactly. Use `np.linalg.solve`, not `inv`.
> Do not change any signature, do not add dependencies, do not refit the model.
> Explain each line in one sentence.

> Here is a failing pytest output (pasted below) and my implementation. Explain what the test
> checks and why my code fails it. Do not change the test.

> Explain the global minimum-variance portfolio formula w = S^-1 1 / (1' S^-1 1) with a
> 2-asset numerical example, as if to a first-year student.

**Do not** ask the AI to "make the tests pass" by editing the tests, and do not paste the test
expectations into your code as constants.

## 7. Pull request checklist

- [ ] Branch named `card-a-...`; the PR targets `main`.
- [ ] Only `src/kfactors/strategies/minvar.py` changed (`git diff --stat main` shows 1 file).
- [ ] `pytest tests/teammates/test_card_A_minvar.py -v`: all passed, 0 skipped.
- [ ] `ruff check src/kfactors/strategies/minvar.py`: "All checks passed!".
- [ ] The docstring still describes what the code does.
- [ ] The PR description pastes the pytest summary line.

## 8. Common pitfalls

* **Look-ahead.** Use only `window` and `model`. Never load the full panel or future data.
  The engine already gives you exactly the past window.
* **Refitting the model.** Do not create a new `FactorModel`. The engine fitted `model` with the
  k being evaluated. Refitting would silently evaluate a different k.
* **`np.linalg.inv`.** It works on small examples but is slower and less accurate. Use `solve`.
* **Normalising by the wrong thing.** Weights must sum to 1 (`x / x.sum()`), not have gross
  exposure 1 (`x / abs(x).sum()`). The tests catch this.
* **Index mismatch.** Return a `Series` whose index is exactly `window.returns.columns`. A
  plain numpy array or a default `0..p-1` index breaks the engine's alignment.
* **NaN.** The window never contains NaN. If your result has NaN, something is wrong:
  raise `ValueError` instead of returning it.
* **Units.** Returns are decimals (0.01 = 1%). The covariance is in squared decimal returns per
  period, but the GMV weights do not depend on the units.
