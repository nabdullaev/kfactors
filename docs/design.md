# kfactors: design and interfaces (v0.1)

This document is the contract between modules and between the core team and contributors. It
describes the behaviour that is **actually implemented** (version `0.1.0a1`, updated
2026-11-03 from the code and the builders' notes). Implementations must not change public
signatures without updating this file. Additions are allowed only as backwards-compatible
keyword arguments or trailing dataclass fields with defaults, and they must be listed here.

## Ownership

| Area | Owner | Files |
|---|---|---|
| Selection rules + sklearn API | core | `rules/`, `sklearn_api.py` |
| Shared helpers (`check_random_state`, `as_2d_array`, `preprocess`, `eigvals_desc`) | core | `utils.py` |
| Simulator | core | `sim.py` |
| Data loaders + `Panel` + download cache | core | `data/` |
| Factor model | core | `model.py` |
| Metrics (Sharpe, bootstrap, PSR/DSR, PBO) | core | `metrics.py` |
| Walk-forward engine + `Strategy` base | core | `engine.py`, `strategy.py` |
| k-curve evaluation (economic optimum, confidence set) | core | `evaluate.py` |
| Reference strategy: factor-zoo SDF (PCA tangency) + plug-in template | core | `strategies/sdf.py`, `strategies/_template.py`, `strategies/__init__.py` |
| `report()` diagnostic | core | `report.py`, `resources/decision_table.json` |
| Benchmark suite | core | `bench/`, benchmark configs in `configs/` |
| Plots | core | `plots.py` |
| CI, README, licence, this document, contributor guide and cards | core | `.github/`, `README.md`, `LICENSE`, `docs/design.md`, `CONTRIBUTING.md`, `docs/tasks/`, `pyproject.toml` |
| **Card A**: minimum-variance strategy (risk evaluator) | teammate | `strategies/minvar.py` |
| **Card B**: stat-arb trading rule | teammate | `strategies/statarb.py` |
| **Card C**: baseline selectors (sklearn, statsmodels, horns, optht) | teammate | `baselines.py` |
| **Card D**: docs site, worked examples, Ken French robustness runs | teammate | `mkdocs.yml`, `docs/**` (except `design.md` and `tasks/`), `examples/**`, new robustness configs in `configs/` (e.g. `configs/french_robustness.yaml`), the `site/` line in `.gitignore` |

Teammate files must only import public core APIs. Acceptance tests for cards live in
`tests/teammates/`. They are skipped while the card is unimplemented:
`tests/teammates/conftest.py` turns a `NotImplementedError` whose message contains "Card" into
a skip. `kfactors.strategies` re-exports only `TangencyPCA` for now. Import the others from
their modules: `kfactors.strategies.minvar.MinVariance`,
`kfactors.strategies.statarb.ResidualReversion` and `kfactors.strategies._template.EqualWeight`.

## Conventions

* Data matrices are **n × p**: rows are time observations, columns are assets/series.
* Returns are **simple returns in decimals** (0.01 = 1%).
* Pre-processing (`kfactors.utils.preprocess`): `Z = X - mean`, and if `standardize=True`
  also `/ sd` (ddof=0). A zero-variance column with `standardize=True` raises `ValueError`.
  `as_2d_array` requires `n >= 3`, `p >= 2` and finite values; DataFrames are accepted.
* Eigenvalues are those of `S = Z'Z / n`, sorted descending, length `p`. Column-centring removes
  one degree of freedom, so the rank of `Z` is at most **`m = min(n - 1, p)`**. Only the first
  `m` eigenvalues are non-trivial; the rest are (numerically) zero. Medians, ratios and
  regressions in the rules use only the non-trivial part.
* `kmax` is clipped inside every rule by
  `rules.base.effective_kmax(n, p, kmax) = max(1, min(kmax, min(n - 1, p) - 2))`. The `- 2`
  keeps `lambda_{kmax+1} > 0` and the Ahn-Horenstein / Bai-Ng tails finite. A non-integer
  `kmax` (bools included) raises `TypeError`; `kmax < 1` raises `ValueError`.
  `bai_ng` searches `k <= kmax_bn = min(kmax_effective, max(1, m // 2))` (`info["kmax_bn"]`);
  `bai_ng` and `ahn_horenstein` return `k = 0` with a `RuleWarning` and `info["degenerate"]`
  when `m < 3`.
* RMT conventions: the noise Wishart has `n - 1` degrees of freedom, so `gamma = p / (n - 1)`
  and the MP edge of `S` is `c sigma^2 (1 + sqrt(gamma))^2` with `c = (n - 1) / n`.
  Gavish-Donoho uses `beta = m / max(n - 1, p)` and the `m` non-trivial singular values.
* Randomness: every stochastic function takes `random_state` (int | np.random.Generator |
  None), normalised with `kfactors.utils.check_random_state`. A passed Generator is consumed.
  Results are deterministic for an int seed.
* Errors: invalid input raises `ValueError` / `TypeError` (`KeyError` for unknown asset ids).
  A rule that cannot be applied as designed emits `kfactors.rules.base.RuleWarning` (a
  `UserWarning`) and records it in `info`. `select_k` re-issues warnings raised inside rules
  with `stacklevel=2`, so they point at the caller of `select_k`.
* Citations in docstrings are by paper and section. Where an equation number could not be
  checked against the PDF, the builders' reports flag it as unverified.

## Core types

```python
@dataclass(frozen=True)
class KResult:                      # kfactors.rules.base
    k: int                          # selected number of factors, 0 <= k <= info["kmax_effective"]
    method: str                     # registry name, e.g. "onatski"
    eigenvalues: np.ndarray         # all p eigenvalues of S = Z'Z/n, descending
    criterion: np.ndarray | None    # rule-specific path over k = 0..kmax (or per eigenvalue rank)
    threshold: float | np.ndarray | None   # see below
    info: dict                      # always: kmax_effective, n, p, kmax_requested, standardize
```

`threshold`: eigenvalue units of `S` for `kaiser` / `mp` / `mp_market` / `gavish_donoho` /
`parallel` (`parallel`: de-volatilised units if `devolatilize=True`); a share for
`variance_share`; Onatski's delta (eigenvalue-gap units) for `onatski`; `None` for `bai_ng`,
`ahn_horenstein` and degenerate panels.

Threshold rules also report `info["k_uncapped"]` and `info["capped"]`. Rules that iterate (mp
median matching, onatski) set `info["converged"]` and emit a `RuleWarning` when they do not
converge. Degenerate panels set `info["degenerate"]`.

### Rule registry (`kfactors.RULES`, entry point `kfactors.select_k`)

```python
select_k(X, method="onatski", *, kmax=20, standardize=True, random_state=None, **kwargs) -> KResult
```

`random_state` is forwarded only to rules whose signature accepts it (currently `parallel`).
An unknown `method` raises `ValueError`, listing the rules. An unknown keyword raises
`TypeError`. Every rule has the signature `rule(Z, *, kmax, **options) -> KResult` on the
already pre-processed `Z`.

| name | module | options (defaults) | notes |
|---|---|---|---|
| `variance_share` | `baseline` | `share=0.9` | Jolliffe (2002) |
| `kaiser` | `baseline` | none | eigenvalue > mean eigenvalue (Kaiser 1960) |
| `mp` | `rmt` | `noise="mean"` (default) or `"median"`, `max_iter=100` | `mean` = `max(1, gamma) * trace(S)/p`, no iteration; `median` = iterated MP-median matching (opt-in; default until 2026-11-03) |
| `mp_market` | `rmt` | `market="auto"` (default) or `"always"` | Laloux et al. (1999): `sigma2 = (trace(S) - lambda_1)/p`. `"auto"` removes `lambda_1` only if it exceeds the plain MP edge (else the edge of `mp(noise="mean")`); `"always"` always removes it. `info["market_removed"]` |
| `gavish_donoho` | `rmt` | none | unknown-noise optimal hard threshold; `info["sv_cutoff"]`; returns 0 when nothing survives |
| `bai_ng` | `econometric` | `criterion="IC_p2"` (`IC_p1..3`, `PC_p1..3`) | `V(k)` floored at `1e-12 V(0)`; searches `k <= kmax_bn = min(kmax_effective, max(1, m // 2))`; `m < 3`: `k = 0`, `RuleWarning`, `info["degenerate"]` |
| `ahn_horenstein` | `econometric` | `stat="ER"` or `"GR"` | mock eigenvalue uses `log(min(n, p))` as in the paper; `m < 3`: `k = 0`, `RuleWarning`, `info["degenerate"]` |
| `onatski` | `econometric` | `max_iter=20` | `rmax = min(kmax_effective, m - 5)`. For `m < 6` it returns `k = 0` with a `RuleWarning` and `info["degenerate"] = True` |
| `parallel` | `permutation` | `null="iid"`, `"circular"` or `"block"`; `n_perm=100`, `quantile=0.95`, `block_size=None` (= `ceil(sqrt(n))`; `ceil(n^(1/3))` until 2026-11-03), `devolatilize=False`, `random_state` | the block null restores each column's mean and sd after the moving-block bootstrap. The circular null is a random shift plus random time reversal per column, with distinct transforms for `p <= 2n`; for `p > 2n` columns share transforms and a `RuleWarning` is emitted (`info["max_shared"]`). `block` is unreliable for `p >= n`; prefer `circular` there. `devolatilize=True` divides every row by its cross-sectional RMS (restoring column means and sds) before the observed and null spectra, against market-wide volatility clustering |

Public helpers: `rules.base.effective_kmax`, `RuleWarning`, `register`;
`rules.rmt.mp_cdf`, `mp_median`, `gd_lambda_star`, `gd_omega`, `gd_omega_approx`;
`rules.econometric.bai_ng_criteria`, `ahn_horenstein_stats`, `BAI_NG_CRITERIA`, `RANK_TOL`.

Known, tested behaviour:
* `mp(noise="median")` is exact only for homoskedastic, light-tailed, serially independent
  noise and over-counts badly on return-like panels (heterogeneous idiosyncratic variance, heavy
  tails, AR noise); the default `noise="mean"` is the classic correlation edge.
* `mp_market(market="auto", default)` removes `lambda_1` only when it exceeds the plain MP
  edge; on pure noise it then returns 0 in >= 90% of cases even for small `p`.
  `market="always"` (Laloux as written) can return 2-4 for `p < ~30`.
* `mp_market` over-counts with dispersed betas or idiosyncratic volatilities (250 x 300, one
  market factor, betas U(0.5, 1.5), idiosyncratic sd lognormal(0, 0.5): 3-6 instead of 1).
* `gavish_donoho` over-counts with heterogeneous idiosyncratic variance: up to `kmax` on
  covariances (`standardize=False`); on correlation matrices it may give `k + 1` or more.
* `bai_ng(PC_p3)` may give `k + 1`.

### scikit-learn API (`kfactors.sklearn_api`)

`FactorCountSelector(method="onatski", kmax=20, standardize=True, random_state=None,
rule_kwargs=None)` sets `k_`, `result_` and `n_features_in_`. `SelectedPCA` (same parameters)
also sets the following:
* `n_components_ = k_`;
* `components_` (`k x p`, orthonormal, in the pre-processed space, largest-|loading| entry
  positive);
* `mean_`, `scale_`;
* `explained_variance_` (ddof=0, i.e. `(n-1)/n` times sklearn's) and
  `explained_variance_ratio_`.

`transform(X) = ((X - mean_) / scale_) @ components_.T`, which may have 0 columns. The class
supports `get_feature_names_out` (prefix `selectedpca`) and `set_output`. For `p = 1` the
estimators set `k_ = 0` with `result_.info["degenerate"]`; `n < 3` raises `ValueError`. Both pass
`check_estimator` with no expected failures.

### Panel and data (`kfactors.data`)

```python
@dataclass(eq=False)
class Panel:                        # kfactors.data.panel
    returns: pd.DataFrame           # DatetimeIndex (sorted, unique) x unique asset ids; NaN = no data
    tradable: pd.DataFrame          # bool, same index/columns; False wherever returns is NaN
    name: str = "panel"
    freq: str = "D"                 # "D" or "M"; other values need meta["periods_per_year"]
    meta: dict                      # url, sha256, bytes, downloaded_at, licence, citation, ...
    shape -> tuple[int, int]
    def window(self, end: int, length: int, *, complete: bool = True) -> Panel
    def subset(self, start=None, end=None, assets=None) -> Panel
    def periods_per_year(self) -> int   # meta override, else 252 ("D") / 12 ("M")
```

* **Validation** happens in `__post_init__` and raises, never repairs. It covers unsorted or
  duplicated dates or columns, label or shape mismatches, a non-bool `tradable`, `±inf`, and
  `tradable` True on NaN. Integer returns are cast to float64.
* **`window(end, length, complete=True)`** returns rows `end-length+1 .. end`, where a negative
  `end` counts from the last row. With `complete=True` it keeps only the columns tradable at row
  `end` that have no NaN in the window. It never looks past `end`, and the result is a new
  validated Panel with a copied `meta`. That copy is the **full** sample-level meta (e.g.
  `end`, `ticker_splits`, `symbol`), which describes the whole sample; the engine does not use
  `Panel.window` for the strategy window and passes only `engine.WINDOW_META_KEYS`.
* **`subset`** is label based and inclusive. Unknown assets raise `KeyError`.

Loaders (all cached; `refresh=True` re-downloads):

* `load_jkp(region="usa", freq="monthly", weighting="vw_cap", *, start, end, refresh)`
  * 153 factors, month-end dates, decimal returns.
  * `tradable = returns.notna()`.
  * `meta["first_complete_date"]` is 1971-11-30 for usa/monthly/vw_cap. Licence CC BY-NC 4.0.
  * The returns (JKP `ret`) are **already signed** by the JKP `direction`: each factor is long
    the leg the characteristic predicts to earn more. `meta["direction"]` (+1 high-minus-low,
    -1 low-minus-high) is informational only; do **not** multiply returns by it again. Check on
    the 2026 snapshot, usa/monthly/vw_cap from 1972: 91.2% of the 80 `direction = +1` factors
    and 86.3% of the 73 `direction = -1` factors have a positive mean return.
* `load_french(dataset="100_size_bm" | "25_size_bm" | "49_industries", freq="daily" | "monthly",
  weighting="vw" | "ew", *, start, end, refresh)`
  * The files are in percent; the loader divides by 100. The sentinels -99.99 and -999 become
    NaN.
  * Some 100 size/B-M portfolios are empty for whole July-June formation years, so walk-forward
    runs rely on the window completeness filter.
  * No outlier filtering. Use `start >= 1963-07` for daily `100_size_bm` / `49_industries`
    (thin-portfolio artefacts before, e.g. -100% / +196% days).
* `load_binance(top=100, start="2022-01-01", end="2025-12-31", *, quote="USDT",
  volume_window=30, history_start="2021-06-01", max_symbols=None, max_gap=30, refresh=False,
  n_jobs=8, symbols=None, max_jump=5.0, exclude_derivatives=True)`
  * Daily spot klines, UTC.
  * `r_t = close_t / close_{t-1} - 1`, only when both days have bars.
  * `tradable[t]` needs a bar on each of the last `volume_window` days, a non-NaN return at `t`,
    a full 24 h bar at `t`, `t >= start`, and a rank within the top `top` by trailing median
    quote volume (ties go alphabetically). Partial bars (delisting / migration last bars) keep
    their return but are not tradable; they are listed in `meta["partial_bars"]`.
  * Stablecoins, leveraged tokens and (by default) wrapped/staked duplicates (WBTC, WBETH, BETH,
    BNSOL; `DERIVATIVE_BASES`) are excluded. A reused ticker becomes `<BASE>_2` after a gap
    > `max_gap` days or a gap with a price jump > `max_jump` (LUNA -> LUNA_2 from 2022-05-31);
    `meta["ticker_splits"]` records each split.
  * Ticker renames / migrations (MATIC->POL, RNDR->RENDER, FTM->S, EOS->A, MKR->SKY, BTT->BTTC,
    TOMO->VIC) are not chained: the successor is a new asset that enters the universe after
    `volume_window` days and the walk-forward after `window` days.
  * Execution: `lag=1` is the conservative assumption (`lag=0` trades at the signal close).
    Strategies may robustify (standardise / winsorise) their estimation windows; realised
    returns are never filtered.
  * Rows between `history_start` and `start` are estimation history with `tradable = False`.
  * `meta["periods_per_year"] = 365`. Licence CC BY-NC-SA 4.0 (Binance Vision Dataset Terms).

Cache (`kfactors.data.cache`):
* The root is `$KFACTORS_DATA`, or `~/.cache/kfactors` if unset; it is read on every call.
* `manifest.json` maps each relpath to `{url, relpath, sha256, bytes, downloaded_at}`.
* Public functions: `fetch`, `fetch_many`, `http_get`, `cache_root`, `manifest`,
  `manifest_entry`.
* A 404 raises `DownloadNotFound` (a `FileNotFoundError`); other failures raise
  `DownloadError`.
* Offline tests monkeypatch `cache._http_get` and `cache._sleep`.

### Simulator (`kfactors.sim`)

`simulate(SimSpec, random_state) -> SimData(X, F, L, E, k, Sigma, spec, noise_var)` draws
`X = F L' + E` with unit noise variance (multiply by e.g. 0.01 for return-like scale).

* **Regimes**: `pervasive` (Bai-Ng loadings, spikes about `p s^2 / 25 + 1`) and `spiked` (exact
  spikes `s^2 + 1`).
* **Noise**: `gauss`, `t`, `ar1`, `garch`, `hetero`.
* **Market and strengths**: with `market` set, `strengths` refers to the `k - 1` non-market
  factors.
* **`Sigma`** is `None` for `p > 2000` (`SIGMA_MAX_P`).
* **Grids**: `design_grid("default")` returns 111 designs (base 72, market 18, weak/BBP 12,
  null 9). `design_grid("small")` returns 8 designs.

### Factor model (`kfactors.model`)

```python
class FactorModel:
    def __init__(self, k: int, standardize: bool = True)
    def fit(self, X) -> FactorModel
    # fitted: mean_, scale_ (ones if standardize=False), eigenvalues_ (min(n,p),), eigenvectors_ (p x k),
    #         factors_ (n x k) = Z @ eigenvectors_ (mean exactly 0), loadings_ = scale_[:,None]*eigenvectors_,
    #         residuals_ (n x p), idio_var_ (p,, ddof=0, floored at 1e-12), explained_variance_ratio_ (k,),
    #         n_samples_, n_features_, rank_ = min(n-1, p), k_
    def covariance(self, residual: str = "diag") -> np.ndarray   # "diag" | "ledoit_wolf"; p x p, ddof=0
    def eigenportfolios(self) -> np.ndarray                      # p x k = eigenvectors_ / scale_[:,None]
    def residual_returns(self, X=None) -> tuple[np.ndarray, np.ndarray]  # (residuals n x p, betas p x k)
```

* **Fitting** uses the thin SVD of `Z`; the `p x p` Gram matrix is never formed.
* **`k`** must be an int (bool rejected) with `0 <= k <= rank_`, otherwise
  `TypeError` / `ValueError`. When `k` exceeds the numerical rank, `fit` emits a
  `RuntimeWarning`.
* **Sign convention**: the largest-|v| entry of each eigenvector is positive.
* **`covariance("diag")`** has the sample variances on its diagonal for every `k`.
  **`"ledoit_wolf"`** is `B Cov(F) B'` plus a Ledoit-Wolf shrunk residual covariance, and is
  full rank even when `p > n`.
* **`residual_returns(X=None)`** uses the fitted window by default. It regresses `X` on the
  eigenportfolio returns by OLS with an intercept; `X` needs the same `p` and at least `k + 1`
  rows.

### Strategy plug-in (`kfactors.strategy`)

```python
class Strategy:
    name: str = "strategy"
    def weights(self, window: Panel, model: FactorModel, context: Context) -> pd.Series

@dataclass
class Context:
    date: pd.Timestamp              # decision date (last row of the window)
    k: int                          # number of factors used to fit `model` (after clipping)
    k_source: str = "fixed"         # "fixed", the rule name, or "callable:<__name__>"
    previous_weights: pd.Series | None = None   # None at the first executed decision; then the
                                                # non-zero positions held (empty after a flat period)
    state: dict                     # fresh dict per run, shared across that run's decisions
```

Strategy output may be:
* a `pd.Series` over a subset of the window columns (missing assets = 0);
* a `dict`;
* an ndarray of length `n_assets`.

Duplicate ids, unknown ids or non-finite values raise `ValueError`. A wrongly shaped array
raises `TypeError`. Weights are **not** rescaled by the engine: `sum |w|` is the gross exposure.
Per-run state must live in `context.state`, not on `self` (with `n_jobs=1` state on `self`
would leak between runs).

Implemented strategies:
* `strategies.sdf.TangencyPCA(gamma=0.0)`, name `sdf_pca_tangency`.
  * `b = (Cov(F) + gamma I)^-1 mean(F)` on the **uncentred** eigenportfolio returns
    `F = X @ model.eigenportfolios()`. `model.factors_` has mean exactly 0, so it cannot be used.
  * `w = (eigenvectors_ @ b) / scale_`, normalised to `sum |w| = 1`.
  * Zero weights when `k = 0` or when every factor mean is zero up to round-off.
* `strategies._template.EqualWeight(gross=1.0)`: the contributor template.
* Card stubs:
  * `strategies.minvar.MinVariance(residual="diag")`: GMV on `model.covariance(residual)`.
  * `strategies.statarb.ResidualReversion(estimation_window=60, s_open=1.25,
    s_close_long=0.50, s_close_short=0.75, max_tau=None, center=True)`, plus the step
    functions `fit_ar1`, `s_scores`, `update_positions` and `hedge_weights`. Its `__init__` is
    core code and validates the parameters (`TypeError` for a non-integer
    `estimation_window`; `ValueError` for `estimation_window < MIN_ESTIMATION_ROWS`, invalid
    thresholds or `max_tau` not `None` and not `> 0`). `max_tau=None` means `L_eff / 2`.
  * `baselines` (`sklearn_variance_share`, `sklearn_mle`, `statsmodels_bai_ng`,
    `horns_parallel`, `optht_threshold`, `BaselineUnavailableWarning`).
  * Unimplemented entry points raise `NotImplementedError("Card X: ...")`.

## Engine timing (leakage contract)

`WalkForward(panel, strategy, config=WalkForwardConfig())`. The config fields and their
defaults:

* `window=120`, `rebalance=1`, `lag=0`, `cost_bps=0.0`, `min_assets=10`;
* `standardize=True`, `kmax=20`;
* `start=None`, `end=None`;
* `rule_kwargs={}` (`{rule_name: {...}}`), `random_state=0`;
* `drift=False` (added).

The config is validated on use: integers with lower bounds, and `cost_bps` finite and `>= 0`.
`rule_kwargs` keys must be `RULES` names with dict values; `kmax`, `standardize` and
`random_state` are rejected inside them (`ValueError`). `random_state` must be `None` or an int
(a numpy `Generator` is still accepted for backward compatibility); `drift` must be a bool.
`start` / `end`: a naive date on a tz-aware index is localised to the index time zone.

Row range:
* Rows after `config.end` are ignored entirely.
* The first decision is the first row with date `>= start`, but never before row `window - 1`.
* Decisions are at `t0, t0 + rebalance, ...`. The last one is the last such `t` with
  `t + 1 + lag <= last row`.
* The final holding period may be shorter than `rebalance`.

At decision index `t` (row position in `panel.returns`):

1. Universe = assets with `tradable[t] == True` **and** no NaN in rows `t-window+1 .. t`. When
   `standardize=True`, assets with numerically zero variance in the window are also dropped
   (same tolerance as `utils.preprocess`; window rows only).
2. `window` = those rows/columns only. The strategy never sees rows `> t`. Its `meta` holds only
   `engine.WINDOW_META_KEYS` (static descriptors); sample-level keys (`end`, `ticker_splits`,
   `symbol`, `first_complete_date`, ...) are dropped. Strategies must not read `window.meta`
   for signals.
3. `k` is one of:
   * a fixed int `>= 0` (bool rejected);
   * a rule name in `RULES`, evaluated as
     `select_k(window, rule, kmax=config.kmax, standardize=config.standardize,
     random_state=config.random_state, **config.rule_kwargs.get(rule, {}))`;
   * a callable `f(window ndarray) -> int`.

   A fixed or callable `k` is clipped to the window rank `min(window - 1, n_assets)`; the
   clipped value goes into `k_used` and is counted in `info["n_clipped"]`. Rule results are
   cached per (rule settings, `t`, universe) on the engine instance. Build a new engine if the
   panel changes.
4. `model = FactorModel(k, standardize=config.standardize).fit(window.returns.values)`.
5. `w_t = strategy.weights(window, model, context)`.
6. `w_t` earns the returns of rows `t+1+lag .. t+rebalance+lag`. With the default `drift=False`
   the weights are held constant: intra-period re-balancing is **not** charged, so net results
   with `rebalance > 1` are optimistic. `drift=True` gives buy-and-hold inside the period
   (`w <- w (1 + r) / (1 + w'r)`, cash earns 0), with turnover measured against the drifted
   book. A held asset with a NaN return is exited at its last price: it contributes 0 from its
   first NaN row to the end of that holding period (no re-entry until the next decision);
   counted in `info["n_nan_exits"]`; no delisting haircut.
7. Turnover at execution = `sum |w_t - w_prev|` (assets missing on one side count as 0; `w_prev`
   is the drifted book when `drift=True`). Net return = gross − `cost_bps/1e4 * turnover` on the
   first holding row.

**Skips.** A decision with fewer than `min_assets` eligible assets is skipped.
* Before the first executed decision, a skip produces no output (warm-up).
* After it, a skip makes the book flat: an all-zero weights row and zero returns for that
  period, plus the cost of closing the previous book. `k_used` has no entry for that date.
* If no decision ever executes, `run` raises `ValueError`.

The realised series is therefore contiguous from the first executed holding row.

The default rule `onatski` needs `min(window - 1, n_assets) >= 6`. Use `window >= 7` and
`min_assets >= 7`.

**Leakage test** (`tests/test_engine.py`): corrupting all rows `> t` must leave `w_t` unchanged
for every `t`. A mutation check confirmed that injecting a one-row look-ahead into the window or
into the rule window makes this test fail.

```python
@dataclass
class BacktestResult:
    gross: pd.Series; net: pd.Series       # by holding date, contiguous
    weights: pd.DataFrame                  # decision date x every asset ever in an executed universe
    turnover: pd.Series                    # by first holding date
    k_used: pd.Series                      # int64, by executed decision date
    k_spec: object; strategy: str; config: WalkForwardConfig; periods_per_year: int
    skipped: pd.DatetimeIndex              # (added) skipped decision dates, warm-up included
    universe_size: pd.Series               # (added) eligible assets at every decision date
    info: dict                             # (added) n_decisions, n_executed, n_skipped, n_clipped, k_source,
                                           # n_nan_exits
    def summary(self) -> dict              # sharpe_gross, sharpe_net, ann_return (geometric), ann_vol,
                                           # max_drawdown (all on net), mean_turnover, k_mean, k_median,
                                           # n_periods, n_decisions, n_skipped
```

Other engine methods:
* `run(k)`: a single specification.
* `run_many(ks, n_jobs=1)`: deduplicates the specs and returns a dict in first-appearance order.
  `n_jobs != 1` uses joblib (loky) and gives results identical to a sequential run.
* `schedule()`: one row per decision with `t`, `n_assets`, `executed`, `hold_start` and
  `hold_end`.

## Evaluation

```python
k_curve(engine, k_grid=range(1, 21),
        rules=("mp", "mp_market", "bai_ng", "ahn_horenstein", "onatski", "gavish_donoho", "parallel"),
        metric="sharpe_net", n_boot=1000, alpha=0.10, mean_block=None, random_state=0, n_jobs=1) -> KCurve
```

`k_curve` runs the engine for every fixed `k` and every rule, then calls
`curve_from_results(results, k_grid, rules, ...)`. That function can also be called directly on
stored `BacktestResult`s.

* **Common dates and bootstrap.** All series are restricted to the dates common to every
  result. One stationary-bootstrap index matrix (`metrics.stationary_bootstrap_indices`, mean
  block `max(1, n^(1/3))` by default) is drawn and shared by every configuration, so draws are
  paired across `k`.
* **Metrics** (registry `evaluate.METRICS`, higher is better):
  * `sharpe_net` (default) and `sharpe_gross`: annualised Sharpe ratio.
  * `neg_vol`: minus the annualised vol of net returns; use it for the minimum-variance risk
    evaluator. It is NaN for flat / constant series and only meaningful for always-invested
    strategies.
  * Add your own with
    `register_metric(name, fn, *, series="net"|"gross", description="", vectorized=True,
    overwrite=False)`.
* **Oracle k** = argmax of the point metric over `k_grid`. Ties go to the smallest `k`, and NaN
  is ignored: `k = 0` with `TangencyPCA` has zero returns and a NaN Sharpe. If every `k` is NaN,
  it raises `ValueError`.
* **Oracle set** (`oracle_confidence_set`).
  * Order `k` by how often it is the bootstrap argmax (ties: closer to the oracle, then smaller
    `k`).
  * Keep the smallest prefix that covers `>= 1 - alpha` of the draws, returned sorted.
  * This is a simple percentile construction. It is related to, but not the same as, the model
    confidence set of Hansen-Lunde-Nason or the rank confidence sets of Mogstad et al.
* **Intervals** `low` / `high` are the marginal `alpha/2` and `1 - alpha/2` bootstrap
  percentiles of each configuration (90% by default). `regret_low` / `regret_high` are paired:
  percentiles of the per-draw differences `draws[oracle_k] - draws[spec]` on the shared
  indices (conditional on the chosen `oracle_k`).

```python
@dataclass
class KCurve:
    curve: pd.DataFrame   # index k: metric, low, high, se, argmax_freq, k_used_mean, regret_low, regret_high
    rules: pd.DataFrame   # index rule: k_mean, k_median, metric, low, high, se, dist_to_oracle, in_set, regret,
                          #             regret_low, regret_high
    oracle_k: int
    oracle_set: list[int]
    metric: str
    results: dict         # k_spec -> BacktestResult
    meta: dict            # series, n_boot, alpha, mean_block, random_state, periods_per_year, strategy,
                          # common_start, common_end, n_common, n_valid_draws, draws (spec x n_boot), n_dropped
    def plot(self, ax=None)
    def to_frame(self) -> pd.DataFrame   # long: kind, spec, k, k_mean, metric, low, high, se,
                                         # dist_to_oracle, in_set, is_oracle, regret, regret_low,
                                         # regret_high, metric_name
```

Each rule is scored by `dist_to_oracle = |k_median - oracle_k|`, by `in_set` (its median `k`,
rounded half up, lies in the oracle set) and by `regret = metric(oracle_k) - metric(rule)`.
Stochastic rules use the same `config.random_state` in every window. This is deterministic and
leakage-free, but the draws are correlated across dates.

`kfactors.metrics` provides:
* `sharpe`, `annual_return` (geometric), `annual_vol`, `max_drawdown`;
* `stationary_bootstrap_indices`, `bootstrap_metric`, `sharpe_diff_ci` (paired; adds `n` and
  `p_value`);
* `probabilistic_sharpe_ratio`, `expected_max_sharpe`, `deflated_sharpe_ratio` (per-period
  units);
* `cscv_logits`, `pbo_cscv`.

## Report (`kfactors.report`)

* `diagnose(X, standardize=True) -> dict` (JSON-serialisable). Keys: `n`, `p`, `p_over_n`,
  `standardize`, `spike_share`, `top_gap_ratios`, `gap_ratio_1`, `max_gap_ratio`,
  `argmax_gap_ratio`, `mean_excess_kurtosis`, `median_excess_kurtosis`, `mean_acf1`,
  `mean_abs_acf1`, `mean_acf1_sq`, `var_dispersion`.
* `recommend(diagnostics, table=None) -> {rule, kwargs, label, reason, path, table_kind}`
  evaluates a JSON decision tree (schema `kfactors.decision_table/1`).
  * The packaged `resources/decision_table.json` is a **heuristic draft**, as its
    `meta.status` says:
    * `mean_abs_acf1 > 0.2` → `parallel(devolatilize=true, null=circular)`;
    * `gap_ratio_1 > 4` or `spike_share > 0.3` → `onatski` (`mp_market` until 2026-11-03; it
      over-counts with dispersed betas or idiosyncratic volatilities);
    * `p_over_n > 1` → `parallel(devolatilize=true)`;
    * otherwise → `onatski`.
    * The evidence for the 2026-11-03 changes is in the table's `meta.revisions`.
  * Replace it with `calibrate_decision_table(sim_results, features=None, max_depth=3,
    random_state=0, *, priority=None, min_samples_leaf=1)` output once the benchmark exists.
* `report(X, *, kmax=20, rules=None, n_boot=200, mean_block=None, standardize=True,
  random_state=0, table=None, boot_n_perm=20, boot_unique_rows=True) -> Report`
  * Runs every rule plus the recommended configuration.
  * The bootstrap is a shared stationary bootstrap that keeps **distinct rows only** by
    default, because duplicated rows fake factors.
  * `Report.rules` (index = rule label), `Report.k_distribution`, `Report.recommended`,
    `Report.diagnostics`, `Report.info`.
  * `Report.to_markdown()` and `Report.plot()`.
* Rule labels are `name` or `name(k1=v1, k2=v2)` with keys sorted (`rule_label`,
  `parse_rule_label`). The same labels are used in benchmark results.

## Plots (`kfactors.plots`)

`plot_spectrum(eigenvalues, n, p, ax=None, *, log=True, title=None, market_adjusted=True,
top=50)` needs the full spectrum. `plot_heatmap(df, ax=None, *, value="accuracy", ...)` takes
long-format columns `rule`, `design`, `<value>`. `plot_k_curve(curve, ax=None)` also exists.
Every function returns an Axes and never calls `plt.show`. `spectrum_edges` and `mp_median` are
independent cross-checks of the RMT edges.

## Benchmark (`kfactors.bench`)

CLI: `python -m kfactors.bench {sim,real,all,leaderboard} --config configs/<name>.yaml --out
results/` (console script `kfactors-bench`). Options: `--reps`, `--n-jobs`, `--calibrate`,
`--panels`, `--designs`, `--seed`, `--no-cache`, `--fail-fast`, `--log-level`. Modules:
`bench/__init__.py` (config, seeds, cache, orchestration), `sim_bench.py`, `real_bench.py`,
`leaderboard.py`, `__main__.py`. Configs: `configs/default.yaml` (full study) and
`configs/ci.yaml` (tiny, offline, synthetic panels only).

* `sim` writes `<out>/sim/raw.csv`, with one row per design x rep x rule and columns `design`,
  `rep`, `rule` (rule label), `k_hat`, `k_true` and the diagnostics. This is the input format
  of `calibrate_decision_table`. It also writes `accuracy.csv`, the heatmaps and
  `timings.csv`. `--calibrate` writes `<out>/sim/decision_table.json` and never overwrites the
  packaged resource.
* `real` writes, per panel, the spectrum, the rolling `k_t`, the report markdown and the
  k-curves of the configured experiments (`strategy: "module:Class"`, `metric`, `k_grid`).
  Strategies that raise `NotImplementedError` (unfinished cards) are skipped with a message.
  Other failures are recorded and make the exit code 1.
* Seeds are `derive_seed(seed, "sim", design, rep)`, independent of grid order, filters and
  `--n-jobs`. The cache lives under `<out>/cache/`, keyed by a hash of every input, the data
  fingerprint and the kfactors version.
* The full key reference is the `kfactors.bench` module docstring.

## Quality gates

* CI (`.github/workflows/ci.yml`) runs on Python 3.11, 3.12 and 3.13 (ubuntu-latest):
  `pip install -e ".[dev]"`, `ruff check src tests`, then
  `pytest --cov=kfactors --cov-report=term-missing --cov-fail-under=85`.
* A second CI job builds the wheel, installs it non-editably into a clean venv and runs the
  suite from a directory outside the checkout (with `tests/`, `configs/`, `docs/` and the
  top-level files the tests read copied next to it).
* Tests marked `network` (real downloads) or `slow` are excluded by default through
  `pyproject.toml` addopts. Run them with `-m network` or `-m slow`.
* `configs/` is not packaged: the benchmark CLI needs a repository checkout or a user config.
