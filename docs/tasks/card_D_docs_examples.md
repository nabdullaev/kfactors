# Card D: documentation site, worked examples, Ken French robustness runs

| | |
|---|---|
| **Files you create/edit** | `mkdocs.yml` (repo root), `docs/` (except `docs/design.md` and `docs/tasks/`), `examples/`, new robustness configs in `configs/` |
| **Tests** | none to pass. "Done" = the site builds without warnings and every example runs end to end |
| **Size** | several small pages + 3 scripts/notebooks |
| **Difficulty** | easy to start, open-ended to polish |
| **Prerequisites** | Markdown, running Python scripts, basic plotting |

## 1. Goal

Make kfactors usable by someone who has never seen it:

1. A **documentation site** built with [MkDocs](https://www.mkdocs.org/) and the
   [Material theme](https://squidfunk.github.io/mkdocs-material/). Its pages: home, install,
   quick start, concepts, "which rule should I use?", API reference generated from the
   docstrings, and examples.
2. **Two worked examples** (scripts in `examples/`, optionally also as notebooks):
   * *Factor zoo*: how many PCs of 153 JKP anomaly factors does a mean-variance investor need?
   * *Crypto stat-arb*: the Avellaneda-Lee residual strategy on Binance coins, Sharpe as a
     function of k.
3. **Ken French robustness runs**: the same k-curve analysis on the classic Fama-French
   portfolio sets (25 / 100 size-B/M, 49 industries; daily and monthly), using the benchmark
   CLI `kfactors.bench` (already shipped and tested by the core team).

## 2. Background

**What the package does (one paragraph for the home page).** With many assets and relatively
few observations (`p >= n`), PCA on returns is the standard way to build a factor model. Its key
parameter is k, the number of principal components treated as real structure. kfactors offers
two things:
* Statistical rules to choose k (`kfactors.select_k`, `kfactors.report`).
* A leakage-safe walk-forward engine (`kfactors.engine.WalkForward`) that measures what each
  choice of k costs in pricing, trading or risk. For every k, `kfactors.evaluate.k_curve`
  reports an out-of-sample metric with bootstrap confidence intervals. It also gives the best k
  in hindsight (the "oracle") and checks whether each rule lands in the oracle's confidence set.

**Core API you will document and use** (all implemented by the core team; see
`docs/design.md`):

| What | Where |
|---|---|
| Choose k | `kfactors.select_k(X, method, kmax=20)`; rules in `kfactors.RULES` |
| Diagnostics and recommendation | `kfactors.report.report(X)` (`.to_markdown()`, `.plot()`) |
| Data | `kfactors.data.load_jkp`, `load_french`, `load_binance` (cached, return a `Panel`) |
| Factor model | `kfactors.model.FactorModel` |
| Strategies | `kfactors.strategies.sdf.TangencyPCA` (core), `minvar.MinVariance` (Card A), `statarb.ResidualReversion` (Card B) |
| Backtest | `kfactors.engine.WalkForward(panel, strategy, WalkForwardConfig(...)).run(k)` |
| k-curve | `kfactors.evaluate.k_curve(engine, k_grid, rules, metric)`; `metric` in `"sharpe_net"`, `"sharpe_gross"`, `"neg_vol"` |
| Plots | `kfactors.plots` (`plot_spectrum`, `plot_k_curve`, `plot_heatmap`) |
| Benchmark | `python -m kfactors.bench {sim,real,all,leaderboard} --config configs/<name>.yaml --out results/`; options in `python -m kfactors.bench --help`, config keys in the `kfactors.bench` module docstring |

**References to cite in the docs**
* J. Bai, S. Ng (2002), *Econometrica* 70(1), on information criteria for k.
* A. Onatski (2010), "Determining the number of factors from empirical distribution of
  eigenvalues", *Review of Economics and Statistics* 92(4), 1004-1016.
* S. C. Ahn, A. R. Horenstein (2013), "Eigenvalue ratio test for the number of factors",
  *Econometrica* 81(3), 1203-1227.
* M. Gavish, D. L. Donoho (2014), *IEEE Trans. Inf. Theory* 60(8).
* S. Kozak, S. Nagel, S. Santosh (2020), "Shrinking the cross-section", *J. Financial
  Economics* 135(2). This is the factor-zoo SDF motivation.
* T. I. Jensen, B. Kelly, L. H. Pedersen (2023), "Is there a replication crisis in finance?",
  *Journal of Finance* 78(5). This is the source of the JKP factors.
* M. Avellaneda, J.-H. Lee (2010), *Quantitative Finance* 10(7). This is the stat-arb
  strategy.
* E. Fama, K. French (1993), *J. Financial Economics* 33(1), and the
  [Ken French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

## 3. Step-by-step plan

### 3.1 Docs site (do this first)

1. `git switch -c card-d-docs`.
2. Install the docs tools into the project venv, **not** into the package dependencies:
   `.venv/bin/python -m pip install mkdocs-material "mkdocstrings[python]"`. Record them in
   `docs/requirements.txt` so others can install them the same way.
3. Create `mkdocs.yml` at the repo root. This minimal file builds with `--strict` (checked
   with mkdocs 1.6.1, mkdocs-material 9.8.0, mkdocstrings-python 2.0.10); copy it and then
   add your pages to `nav:`. The nesting under `mkdocstrings:` matters: the numpy option sits
   under `handlers: python: options:`.

   ```yaml
   site_name: kfactors
   theme:
     name: material
   plugins:
     - search
     - mkdocstrings:
         handlers:
           python:
             paths: [src]              # import kfactors from src/ (works without installing)
             options:
               docstring_style: numpy
   markdown_extensions:
     - pymdownx.snippets:            # ships with mkdocs-material; used by docs/contributing.md
         check_paths: true           # a missing included file is an error, not silence
   not_in_nav: |
     tasks/*
   nav:
     - Home: index.md
     - Install: install.md
     - Quick start: quickstart.md
     - Concepts: concepts.md
     - Choosing a rule: rules.md
     - Examples:
         - Factor zoo: examples/factor_zoo.md
         - Crypto stat-arb: examples/crypto_statarb.md
         - Ken French robustness: examples/french_robustness.md
     - API:
         - Rules: api/rules.md
         # one entry per page in docs/api/
     - Design: design.md
     - Contributing: contributing.md
   ```

   * `not_in_nav: tasks/*` keeps the task cards out of the site navigation without an
     "is not included in the nav" message.
   * **Contributing**: do not point `nav` at `../CONTRIBUTING.md`. That file is outside
     `docs/`, so `mkdocs build --strict` aborts with "A reference to '../CONTRIBUTING.md' is
     included in the 'nav' configuration, which is not found in the documentation files".
     `CONTRIBUTING.md` is a core file, so do not move it either. Instead create
     `docs/contributing.md` containing only the line `--8<-- "CONTRIBUTING.md"`. The
     `pymdownx.snippets` extension above pastes the root file into the page at build time.
     Run `mkdocs` from the repo root, because snippet paths are relative to the working
     directory.
4. Write the pages in `docs/`: `index.md`, `install.md`, `quickstart.md`, `concepts.md`,
   `rules.md`, `contributing.md` (the one-line include above), `examples/*.md`, `api/*.md`.
   Each API page is a single line such as `::: kfactors.rules` and is generated from the
   docstrings.
5. Build: `.venv/bin/mkdocs build --strict`. "Strict" turns warnings into errors. Preview with
   `.venv/bin/mkdocs serve` and open http://127.0.0.1:8000.
6. Add a line to `.gitignore` for `site/`. That edit is allowed for this card.

### 3.2 Example 1: factor zoo (`examples/01_factor_zoo.py`)

```python
from pathlib import Path

from kfactors.data import load_jkp
from kfactors.engine import WalkForward, WalkForwardConfig
from kfactors.evaluate import k_curve
from kfactors.strategies.sdf import TangencyPCA

out = Path("results/examples")           # results/ is not in git: create it before saving
out.mkdir(parents=True, exist_ok=True)

panel = load_jkp("usa", "monthly")       # 1200 x 153, complete from 1971-11 (downloads ~2.7 MB once)
cfg = WalkForwardConfig(window=120, rebalance=1, min_assets=50, start="1985-01-01")
kc = k_curve(WalkForward(panel, TangencyPCA(gamma=0.0), cfg), k_grid=range(0, 21),
             metric="sharpe_net", n_boot=1000)
kc.curve.to_csv(out / "factor_zoo_curve.csv")
kc.rules.to_csv(out / "factor_zoo_rules.csv")
kc.plot().figure.savefig(out / "factor_zoo_k_curve.png", dpi=150)
print("saved to", out.resolve())
```

Create the output directory **before** the long `k_curve` run: pandas raises
`OSError: Cannot save file into a non-existent directory` otherwise, and a fresh clone has no
`results/` folder. Do the same in the other examples (`results/examples/`, `results/french/`).

Also show `kfactors.report.report(panel.window(len(panel.returns) - 1, 120).returns)`, a
diagnostic of the last window, and a spectrum plot (`kfactors.plots.plot_spectrum`).

**Expected output (describe it, do not invent it).**
* A k-curve, i.e. the out-of-sample Sharpe for k = 0..20, with 90% bootstrap bands.
* The oracle k and its confidence set.
* For each rule: its median k, whether it lands in the set, and the regret.
* Kozak, Nagel & Santosh (2020) lead us to expect that a handful of PCs captures most of the
  achievable Sharpe ratio, with flat or declining gains after that.
* Report what you actually get. We have no reference numbers yet, so your run *is* the
  reference.
* Try `gamma > 0` as a robustness check.

### 3.3 Example 2: crypto stat-arb (`examples/02_crypto_statarb.py`)

```python
from kfactors.data import load_binance
from kfactors.engine import WalkForward, WalkForwardConfig
from kfactors.evaluate import k_curve
from kfactors.strategies.statarb import ResidualReversion   # Card B

panel = load_binance(top=50, start="2022-01-01", end="2025-12-31", n_jobs=32)
cfg = WalkForwardConfig(window=120, rebalance=1, cost_bps=10, min_assets=20)
kc = k_curve(WalkForward(panel, ResidualReversion(), cfg), k_grid=range(0, 16),
             metric="sharpe_net", n_boot=1000)
```

Notes:
* The **first** `load_binance` call downloads about 19,000 small monthly files. Expect about
  7 minutes with `n_jobs=32`; later calls take under a second because of the cache. For a quick
  demo use `max_symbols=60, top=30` and a one-year range.
* Crypto uses 365 periods per year (`panel.periods_per_year()`).
* Until Card B is merged, develop the example with `TangencyPCA` or `MinVariance` and switch at
  the end.

**Expected output.**
* A net-of-cost k-curve, plus a gross curve (`metric="sharpe_gross"`) to show the cost drag.
* Turnover from `result.summary()["mean_turnover"]`.
* The question to answer: does a k chosen by a rule (e.g. `onatski`, `bai_ng`) land in the
  oracle set?
* Synthetic tests show the k = 0 versus k = 1 contrast (no market factor removed means no
  edge). Check whether the real data show it too.

### 3.4 Ken French robustness (`examples/03_french_robustness.py` + `configs/`)

The question: do the conclusions hold on classic, long, public data? We run the grid
"dataset x frequency x window x strategy":
* datasets `25_size_bm`, `100_size_bm`, `49_industries`;
* frequencies daily / monthly;
* windows 60/120 months, or 252/504 days;
* strategies `MinVariance` (`metric="neg_vol"`) and `TangencyPCA` (`"sharpe_net"`).

* Write a **new** YAML config in `configs/` (e.g. `configs/french_robustness.yaml`; do not edit
  `configs/ci.yaml` or `configs/default.yaml`, which the core tests use) and run
  `python -m kfactors.bench real --config configs/french_robustness.yaml --out results/french/`.
  The French panels in `configs/default.yaml` are a working template. Config keys are in the
  `kfactors.bench` module docstring; CLI options in `python -m kfactors.bench --help`.
* `examples/03_french_robustness.py` runs the same thing from Python
  (`from kfactors.bench.__main__ import main; main(["real", "--config", ..., "--out", ...])`)
  and then builds the summary table and the heatmap below from the files the bench wrote.

Data facts (from the loader builder) that you must mention in the docs:
* French files are in percent and the loader converts them to decimals.
* `100_size_bm` has whole formation years missing for some portfolios, e.g. ME10-BM8 is
  missing from 1975-07 to 2000-06. The engine therefore drops incomplete assets window by
  window, so `p` varies over time; that is expected, not a bug.
* `49_industries` is complete after 1969-06 (monthly) or 1971-03 (daily).

**Expected outputs.**
* One k-curve CSV and PNG per configuration.
* A summary table (rows = configurations; columns = oracle k, oracle set, the k of each rule,
  and whether it is in the set).
* A heatmap made with `kfactors.plots.plot_heatmap` from a long table with columns
  `rule, design, value`.
* A short "robustness" page in the docs that interprets the table: for which datasets and
  frequencies is the oracle set small, and which rules land in it consistently.

## 4. How to check your work

```bash
.venv/bin/mkdocs build --strict                         # site builds, no warnings
.venv/bin/python examples/01_factor_zoo.py              # runs end to end (network on first run)
.venv/bin/python examples/02_crypto_statarb.py
.venv/bin/python examples/03_french_robustness.py
.venv/bin/ruff check examples/
.venv/bin/python -m pytest -q                           # nothing else broke
```

## 5. What "done" means

* `mkdocs build --strict` passes. The API pages render the numpy docstrings of `kfactors.rules`,
  `model`, `engine`, `evaluate`, `data`, `strategies` and `report`.
* The examples run from a clean clone with a fresh cache (`KFACTORS_DATA=/tmp/kf_cache`), save
  their CSV/PNG outputs under `results/examples/` or `results/french/`, and print where they
  saved them.
* Each example page shows the figure, the rules table and 3-5 sentences of interpretation
  written by you. Report your numbers as you got them; never round them to look nicer.
* Every claim in the docs about a paper cites the paper.

## 6. Suggested prompts for ChatGPT / Claude

> Write an `mkdocs.yml` for a Python package `kfactors` using the Material theme and
> mkdocstrings with numpy-style docstrings. Navigation: (list). Keep it minimal and explain
> every key.

> Here is the docstring of `kfactors.evaluate.k_curve` and of `KCurve` (pasted). Write a
> quick-start page that explains, for a finance student, what the k-curve, the oracle k and
> the oracle confidence set are, using this output table (pasted).

> Review this example script (pasted) for look-ahead bias: does any line use data after the
> decision date? (The engine is leakage-safe; look for things done *outside* the engine, such
> as selecting assets with full-sample statistics.)

Do not let the AI invent results, numbers or citations. Every number on the site must come
from a run you did, and every citation must be checked against the paper.

## 7. Pull request checklist

- [ ] Branch `card-d-...`. Changed files are only `mkdocs.yml`, `docs/**` (not `design.md`,
      not `tasks/`), `examples/**`, new robustness config files in `configs/` (not `ci.yaml` /
      `default.yaml`), `.gitignore` (adding `site/`).
- [ ] `mkdocs build --strict` passes; paste the last line in the PR.
- [ ] Every example runs; the PR lists the runtime and the output files of each.
- [ ] No large data or result files are committed. `results/` outputs are regenerated; commit
      only small CSV/PNG files you reference from the docs (under 1 MB each).
- [ ] `ruff check examples/` is clean.

## 8. Common pitfalls

* **Look-ahead in examples.** Choosing the asset universe or k with full-sample statistics
  (e.g. "the 50 most volatile coins over 2022-2025") leaks the future. Use the loaders'
  point-in-time `tradable` mask and the engine.
* **Units.** Returns are decimals (French files are in percent and the loader converts them).
  Sharpe ratios from the engine are annualised with the panel's `periods_per_year()`: 12
  monthly, 252 daily, 365 crypto.
* **NaN.** Real panels have gaps. Do not `fillna(0)` returns before the engine; it handles
  missing data per window.
* **Cherry-picking.** Show the full k-curve with confidence bands, not only the best k.
* **Committing the cache.** Data lives in `~/.cache/kfactors` (or `$KFACTORS_DATA`), never in
  the repo.
* **Editing core files to make an example work.** Open an issue or ask in the PR instead.
