# kfactors

![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![licence](https://img.shields.io/badge/licence-MIT-green)

**Choose the number of factors *k* in high-dimensional financial return panels, and measure what
the choice costs.**

kfactors targets panels where the textbook assumptions fail: more series than observations
(`p >= n`), heavy tails, serial dependence and one dominant market factor. It provides

1. **Nine selection rules** behind one entry point, `select_k(X, method)`. Each returns the
   chosen `k` together with the eigenvalues, the criterion path, the threshold and diagnostics.
2. **`report(X)`**: runs every rule and bootstraps the sampling distribution of each rule's `k`.
   It also measures the panel properties that break particular rules and recommends one rule,
   with a written reason.
3. **A leakage-safe walk-forward engine** (`WalkForward`) and **economic k-curves**
   (`k_curve`). For a strategy (SDF / tangency, minimum variance, stat-arb) they report the
   out-of-sample Sharpe ratio or volatility as a function of `k`, with marginal percentile
   intervals per configuration plus paired (shared-index) block-bootstrap intervals for the
   regret against the oracle. They also give the best `k` in hindsight (the *oracle*), its
   confidence set, and whether each rule's `k` lands in that set.
4. **Reproducible public data** (JKP factor zoo, Ken French portfolios, Binance spot klines) with
   a hashed download cache and point-in-time tradability masks.

HDDA course project (industrial track), 2026:
Nursultan Abdullaev, Kamilla Farzalieva, Adel Bariev.

---

## Why

PCA is the standard way to build a statistical factor model of returns. Its key parameter is the
number of components treated as real structure. The tools people actually use either cannot
choose it in the `p >= n` regime or hide the choice:

| Library | What happens | Evidence |
|---|---|---|
| scikit-learn 1.9.1 | The only automatic choices are a variance share and `PCA(n_components="mle")`. `"mle"` raises `ValueError: n_components='mle' is only supported if n_samples >= n_features`, which rules out exactly the `p > n` regime. | Reproduced in this repo's venv on a 50 x 100 matrix (2026-11-03). |
| statsmodels 0.15.0 | `PCA.ic` (the Bai-Ng criteria) sits under the source comment `# TODO: This needs careful testing`. No file in `statsmodels/multivariate/tests/` references `.ic`, and nothing returns the chosen `k`. | Checked on the installed package (2026-11-03). |
| skfolio, arbitragelab | skfolio's `DenoiseCovariance` computes a Marchenko-Pastur count internally but never exposes it. arbitragelab's `PCAStrategy` hard-codes `n_components=15`. | From our proposal's source review (Sept. 2026). Neither package is installed here, so this was not re-checked for this README. |

The choice also matters economically. On anomaly factors, pricing typically needs more PCs than
the statistical rules detect, while trading residuals may need fewer. kfactors makes that gap
measurable instead of anecdotal ("detectable vs useful").

## Install

Requires Python >= 3.11.

```bash
git clone <repository-url> kfactors && cd kfactors
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"     # dev = pytest, pytest-cov, ruff, statsmodels, optht
.venv/bin/python -m pytest -q                    # offline suite; network/slow tests are excluded
```

Runtime dependencies: numpy, pandas, scipy, scikit-learn, joblib, matplotlib, pyyaml, requests.

## 30-second quickstart

```python
from kfactors import select_k
from kfactors.sim import SimSpec, simulate

# A p > n panel (250 x 400) with 4 planted factors and Student-t(3) noise.
X = simulate(SimSpec(n=250, p=400, k=4, noise="t", df=3.0), random_state=0).X

res = select_k(X, "onatski", kmax=20)       # default method is "onatski"
res.k                                       # 4
res.eigenvalues[:6], res.info["kmax_effective"]   # spectrum of S; kmax after clipping (20)
# res.threshold is rule-specific: for onatski it is the calibrated delta (an eigenvalue-gap
# size, 0.43 here), not an eigenvalue cut-off; mp/mp_market/kaiser/... return eigenvalue units.

for rule in ["bai_ng", "ahn_horenstein", "mp", "mp_market", "gavish_donoho", "parallel"]:
    print(rule, select_k(X, rule, random_state=0).k)     # all 4 on this panel
```

**Which rule should I trust on *my* data?** `report` diagnoses the panel, runs every rule,
bootstraps `k` and recommends a rule:

```python
from kfactors.report import report

rep = report(X, n_boot=50, random_state=0)
rep.recommended["label"], rep.recommended["k"]   # ('onatski', 4): spike_share > 0.3 (market branch)
rep.rules[["k", "boot_median", "boot_q05", "boot_q95", "boot_p_same"]]
# On this panel variance_share and kaiser hit kmax = 20; the other seven rules return 4 in
# every bootstrap draw (mp(noise="median"), not run by report, would give 18).
print(rep.to_markdown())                         # recommendation + reason + diagnostics
rep.plot()                                       # spectrum and per-rule k distributions
```

**Factor model for a given k** (thin SVD; it never forms the `p x p` Gram matrix):

```python
from kfactors.model import FactorModel

fm = FactorModel(k=res.k).fit(X)
fm.loadings_, fm.factors_, fm.idio_var_
Sigma = fm.covariance("ledoit_wolf")             # p x p, full rank even when p > n
W = fm.eigenportfolios()                         # Avellaneda-Lee eigenportfolios v_j / sigma_i
```

**What does k cost?** Run a walk-forward backtest for every `k` and every rule, then read off
the oracle and its confidence set:

```python
import pandas as pd
from kfactors.data import Panel
from kfactors.engine import WalkForward, WalkForwardConfig
from kfactors.evaluate import k_curve
from kfactors.strategies import TangencyPCA

sim = simulate(SimSpec(n=1500, p=60, k=3), random_state=1)
Xr = (sim.F + 0.1) @ sim.L.T + sim.E                     # the 3 factors earn a premium
dates = pd.bdate_range("2019-01-01", periods=len(Xr))
R = pd.DataFrame(0.01 * Xr, index=dates, columns=[f"asset{j}" for j in range(Xr.shape[1])])
panel = Panel(returns=R, tradable=R.notna(), name="toy", freq="D")

wf = WalkForward(panel, TangencyPCA(),
                 WalkForwardConfig(window=250, rebalance=21, cost_bps=5))
kc = k_curve(wf, k_grid=range(1, 11), rules=["onatski", "bai_ng", "mp_market"], n_boot=200)
kc.oracle_k, kc.oracle_set                    # (3, [1, 2, 3])
kc.curve[["metric", "low", "high"]]           # net Sharpe by k, 90% bootstrap interval
kc.rules[["k_median", "metric", "in_set", "regret", "regret_low", "regret_high"]]
kc.plot()
```

The numbers in the comments come from running these snippets with the pinned seeds (whole
script about 5 s on a laptop; re-checked 2026-11-03). On real data, load a panel instead of simulating one:
`load_jkp()`, `load_french("49_industries", "monthly")` or `load_binance(top=100)`.

There is also a scikit-learn interface: `kfactors.sklearn_api.FactorCountSelector` and
`SelectedPCA` (a PCA whose `n_components` is chosen by a rule; it works inside `Pipeline`). Both
pass `sklearn.utils.estimator_checks.check_estimator`.

## Selection rules

`select_k(X, method, *, kmax=20, standardize=True, random_state=None, **options)`. `X` is
centred and, by default, standardised (correlation matrix). Eigenvalues are those of
`S = Z'Z / n`. Because of centring only `m = min(n - 1, p)` of them are non-trivial. `kmax` is
clipped to `max(1, min(kmax, m - 2))`, and every rule returns `0 <= k <= kmax_effective`.
`res.threshold` is in eigenvalue units of `S` for `kaiser`, `mp`, `mp_market`, `gavish_donoho`
and `parallel` (de-volatilised units with `devolatilize=True`), a share for `variance_share`,
Onatski's delta (an eigenvalue-gap size) for `onatski`, and `None` for `bai_ng`,
`ahn_horenstein` and degenerate panels.

| `method` | Rule | Options | Reference |
|---|---|---|---|
| `variance_share` | Smallest k explaining `share` of the variance (what `PCA(n_components=0.9)` does) | `share=0.9` | Jolliffe (2002), *Principal Component Analysis*, Sec. 6.1.1 |
| `kaiser` | Eigenvalues above their mean ("eigenvalue > 1" on a correlation matrix) | none | Kaiser (1960), *Educ. Psychol. Meas.* 20, 141-151 |
| `mp` | Eigenvalues above the Marchenko-Pastur bulk edge `sigma^2 (1 + sqrt(p/(n-1)))^2` | `noise="mean"` (default; average variance, the classic correlation edge) or `"median"` (iterated median matching, opt-in) | Marchenko & Pastur (1967), *Math. USSR-Sbornik* 1(4) |
| `mp_market` | MP edge with the market eigenvalue removed from the noise variance | `market="auto"` (default: remove `lambda_1` only if it is above the plain MP edge) or `"always"` (Laloux et al. as written) | Laloux, Cizeau, Bouchaud & Potters (1999), *PRL* 83(7) |
| `gavish_donoho` | Optimal hard threshold for singular values, unknown noise level | none | Gavish & Donoho (2014), *IEEE Trans. Inf. Theory* 60(8) |
| `bai_ng` | Information criteria `IC_p1..3`, `PC_p1..3`, searched over `k <= kmax_bn = min(kmax_effective, max(1, m // 2))` (`info["kmax_bn"]`) | `criterion="IC_p2"` | Bai & Ng (2002), *Econometrica* 70(1) |
| `ahn_horenstein` | Eigenvalue ratio / growth ratio | `stat="ER"` or `"GR"` | Ahn & Horenstein (2013), *Econometrica* 81(3) |
| `onatski` (default) | Edge distribution: eigenvalue-difference test with calibrated delta | `max_iter=20` | Onatski (2010), *Rev. Econ. Stat.* 92(4) |
| `parallel` | Parallel analysis against a permutation / circular-shift (random shift plus random time reversal) / moving-block null | `null="iid"`, `"circular"` or `"block"`; `n_perm=100`, `quantile=0.95`, `block_size=None` (= `ceil(sqrt(n))`), `devolatilize=False` (divide rows by their cross-sectional RMS first, against market-wide volatility clustering) | Horn (1965), *Psychometrika* 30(2); Buja & Eyuboglu (1992); Kunsch (1989) |

Known behaviour, documented and tested:

* `mp(noise="median")` is opt-in and exact only for homoskedastic, light-tailed, serially
  independent noise. It over-counts badly on return-like panels (heterogeneous idiosyncratic
  variance, heavy tails, AR noise). The default `noise="mean"` is the classic correlation edge.
* `mp_market(market="auto")` (default) removes `lambda_1` only when it exceeds the plain MP
  edge; on pure noise it then returns 0 in >= 90% of cases even for small `p`.
  `market="always"` (Laloux as written) can return 2-4 for `p` below about 30.
* `mp_market` over-counts with dispersed market betas or idiosyncratic volatilities (one market
  factor at 250 x 300: 3-6 instead of 1), so `report()` no longer recommends it.
  `gavish_donoho` over-counts with heterogeneous idiosyncratic variance: at 250 x 300 with
  lognormal(0, 0.5) idiosyncratic sd it hit `kmax` on the covariance (`standardize=False`) and
  gave 7-12 instead of 3 on the correlation matrix.
* `onatski` needs `m >= 6`; `bai_ng` and `ahn_horenstein` need `m >= 3`. On smaller panels they
  return `k = 0` with a `RuleWarning` and set `info["degenerate"]`.
* `parallel` is the only stochastic rule; it is deterministic for a fixed `random_state`.
  `null="circular"` emits a `RuleWarning` (`info["max_shared"]`) when `p > 2n`, because columns
  then share shifts. `null="block"` is unreliable for `p >= n`; prefer `"circular"` there.
  None of the nulls preserves a volatility path shared by all series; use `devolatilize=True`.

`kfactors.baselines` wraps the library baselines for comparison: the sklearn variance share and
`'mle'`, statsmodels `PCA.ic`, `horns` and `optht`. It is contributor Card C.

## Strategies and evaluation

| Strategy | Module | Loss it measures |
|---|---|---|
| `TangencyPCA(gamma=0)`: tangency portfolio of the first k PC portfolios (Kozak, Nagel & Santosh 2018, 2020) | `kfactors.strategies.sdf` | pricing: net Sharpe of the SDF |
| `MinVariance(residual="diag")`: global minimum variance with the k-factor covariance | `kfactors.strategies.minvar` (Card A) | risk: `metric="neg_vol"` |
| `ResidualReversion(...)`: Avellaneda & Lee (2010) s-score stat-arb on PCA residuals | `kfactors.strategies.statarb` (Card B) | trading: net Sharpe |
| `EqualWeight`: minimal plug-in template | `kfactors.strategies._template` | none (template) |

Engine timing (the leakage contract): at decision row `t` the universe is the set of assets
tradable at `t` with a complete window. The strategy sees only rows `t-window+1 .. t`. The
weights earn rows `t+1+lag .. t+rebalance+lag`, and costs are linear in turnover. By default
(`drift=False`) weights are held constant, so intra-period re-balancing is not charged and net
results with `rebalance > 1` are optimistic; `drift=True` gives buy-and-hold inside the period.
A held asset with a NaN return is exited at its last price (counted in `info["n_nan_exits"]`;
no delisting haircut). The strategy window's `meta` carries only static descriptors
(`engine.WINDOW_META_KEYS`). A test corrupts every row after `t` and checks that `w_t` does not
change. `kfactors.metrics` adds the Sharpe
ratio, the stationary bootstrap, Sharpe-difference CIs, PSR/DSR (Bailey & Lopez de Prado) and
PBO via CSCV.

## Data sources

All loaders return a `Panel` (returns in decimals plus a point-in-time `tradable` mask). Raw
files are cached under `$KFACTORS_DATA` (default `~/.cache/kfactors`) with a `manifest.json` of
URL, SHA-256, size and download time. Provenance and licence are copied into `Panel.meta`.
Nothing is redistributed in this repository except tiny synthetic test fixtures.

| Loader | Source | Licence / terms |
|---|---|---|
| `load_jkp(region="usa", freq="monthly", weighting="vw_cap")` | JKP Global Factor Data, 153 anomaly factors (Jensen, Kelly & Pedersen 2023, *J. Finance* 78(5)), [jkpfactors.com](https://jkpfactors.com) | **CC BY-NC 4.0** (stated on jkpfactors.com) |
| `load_french(dataset="100_size_bm" \| "25_size_bm" \| "49_industries", freq="daily" \| "monthly", weighting="vw" \| "ew")` | Kenneth R. French Data Library | No formal licence; free for research use, cite the source |
| `load_binance(top=100, start, end, ...)` | Binance Vision daily spot klines (USDT pairs), [data.binance.vision](https://data.binance.vision); top-`top` universe by trailing median quote volume. Stablecoins, leveraged tokens and (by default) wrapped/staked duplicates (WBTC, WBETH, BETH, BNSOL; `DERIVATIVE_BASES`) are excluded. A reused ticker becomes `<BASE>_2` after a gap > `max_gap` days or a gap with a price jump > `max_jump` (LUNA -> LUNA_2 from 2022-05-31) | **CC BY-NC-SA 4.0** (Binance Vision Dataset Terms v1.0, Aug. 2026). Commercial use needs a separate licence from Binance. |

Data quirks worth knowing:

* JKP is complete only from 1971-11 (use `start="1972-01-01"`).
* Some French 100 size/B-M portfolios are empty for whole formation years, so walk-forward runs
  on that panel rely on the window completeness filter.
* Use `start >= 1963-07` for French daily `100_size_bm` / `49_industries` (thin-portfolio
  artefacts before).
* A cold `load_binance(top=100)` download is about 19k monthly files; pass `n_jobs=32`.
* Binance ticker renames/migrations (MATIC->POL, RNDR->RENDER, FTM->S, EOS->A, MKR->SKY,
  BTT->BTTC, TOMO->VIC) are not chained; the successor is a new asset that enters the universe
  after `volume_window` days and the walk-forward after `window` days.
* Binance bars shorter than 24 h (delisting / migration last bars) keep their return but are
  not tradable (`meta["partial_bars"]`).
* For Binance, `lag=1` is the conservative execution assumption (`lag=0` trades at the signal
  close); `configs/default.yaml` uses `lag: 1` for the Binance panel.

## Benchmark

One command runs the simulation study, the real-data k-curves and the leaderboard:

```bash
python -m kfactors.bench all --config configs/default.yaml --out results/ --n-jobs -1
# tiny offline run (synthetic panels only, no downloads), as used by the tests
python -m kfactors.bench all --config configs/ci.yaml --out /tmp/kfactors_ci
# the console script is equivalent
kfactors-bench {sim,real,all,leaderboard} --config ... --out ...
```

* `sim` applies every rule to every design of `kfactors.sim.design_grid("default")` and every
  replication. The grid has 111 designs:
  * pervasive or spiked factors;
  * Gaussian, t(3), AR(1), GARCH or heteroskedastic noise;
  * a market factor;
  * weak factors around the BBP threshold;
  * null panels.

  Outputs go to `<out>/sim/` (`raw.csv`, `accuracy.csv`, heatmaps, timings). With
  `--calibrate` it also learns a decision table for `report()`, written to
  `<out>/sim/decision_table.json`; the packaged table is never overwritten.
* `real` handles each configured panel (JKP, French, Binance, a CSV or synthetic data). It
  writes the spectrum, the rolling `k_t` of every rule, a `report()` on the latest window and
  the k-curve of every configured strategy.
* `all` = `sim` then `real`. Every command rebuilds `<out>/leaderboard.csv` and
  `leaderboard.md`.

Runs are deterministic given `seed` (per-design seeds do not depend on `--n-jobs` or on design
filters), and intermediate results are cached under `<out>/cache/`. The configs are not part of
the installed package: `kfactors-bench` needs a repository checkout (or your own YAML file), and
the default `--config configs/default.yaml` is resolved relative to the working directory. The config keys are
documented in the `kfactors.bench` module docstring; see also `python -m kfactors.bench --help`.

## Project structure

```
src/kfactors/
  rules/          base.py (KResult, RULES, select_k), baseline.py, rmt.py, econometric.py, permutation.py
  sklearn_api.py  FactorCountSelector, SelectedPCA
  sim.py          planted-factor simulator and benchmark design grids
  data/           panel.py (Panel), cache.py (hashed download cache), loaders.py (JKP, French, Binance)
  model.py        FactorModel (PCA factor model, covariance, eigenportfolios, residuals)
  metrics.py      Sharpe, bootstrap CIs, PSR/DSR, PBO (CSCV)
  strategy.py     Strategy plug-in base + Context
  engine.py       WalkForward, WalkForwardConfig, BacktestResult
  evaluate.py     k_curve, KCurve, oracle confidence set, metric registry
  strategies/     sdf.py (TangencyPCA), minvar.py (Card A), statarb.py (Card B), _template.py
  baselines.py    library baselines (Card C)
  report.py       diagnose / recommend / report / calibrate_decision_table
  resources/      decision_table.json (heuristic draft, see its meta.status)
  plots.py        spectrum, heatmap, k-curve plots
  bench/          benchmark CLI
tests/            one test file per module; tests/teammates/ = card acceptance tests; tests/fixtures/
docs/             design.md (interfaces + ownership), tasks/ (contributor cards)
configs/          benchmark configurations
.github/workflows/ci.yml   ruff + pytest with coverage on Python 3.11 / 3.12 / 3.13; tests against the built wheel
```

### Ownership

| Area | Owner | Files |
|---|---|---|
| Rules, sklearn API, simulator, data, model, metrics, engine, evaluation, SDF strategy, report, benchmark, plots | core team | everything in `src/kfactors/` not listed below |
| **Card A**: minimum-variance strategy | teammate | `strategies/minvar.py` |
| **Card B**: Avellaneda-Lee stat-arb | teammate | `strategies/statarb.py` |
| **Card C**: library baselines | teammate | `baselines.py` |
| **Card D**: docs site, worked examples, Ken French robustness runs | teammate | `mkdocs.yml`, `docs/**` (not `design.md`, not `tasks/`), `examples/**`, new robustness configs in `configs/` (not `ci.yaml` / `default.yaml`) |

The full table, the interfaces and the leakage contract are in
[`docs/design.md`](docs/design.md).

## Contributing

Start with [`CONTRIBUTING.md`](CONTRIBUTING.md) (setup, branch/PR workflow, code standards, rules
for using AI assistants). Each contributor card is self-contained:

**New team members: read [docs/tasks/START_HERE.md](docs/tasks/START_HERE.md) first.**

* [Card A: minimum-variance strategy](docs/tasks/card_A_minvar.md)
* [Card B: Avellaneda-Lee residual stat-arb](docs/tasks/card_B_statarb.md)
* [Card C: library baselines](docs/tasks/card_C_baselines.md)
* [Card D: docs site, examples, Ken French robustness](docs/tasks/card_D_docs_examples.md)

Card acceptance tests live in `tests/teammates/`. They skip while a card is unimplemented.

## Citation

```bibtex
@software{kfactors2026,
  title   = {kfactors: choosing the number of factors in high-dimensional financial panels},
  author  = {Abdullaev, Nursultan and Farzalieva, Kamilla and Bariev, Adel},
  year    = {2026},
  version = {0.1.0a1},
  note    = {HDDA course project (industrial track)}
}
```

When you publish results computed on the data sources above, also cite them: Jensen, Kelly &
Pedersen (2023) for JKP, the Kenneth R. French Data Library, and Binance Vision.

## Licence

The code is released under the [MIT License](LICENSE), © 2026 Nursultan Abdullaev, Kamilla
Farzalieva, Adel Bariev. The data sources keep their own licences (see the table above); the JKP
and Binance data are non-commercial.
