# kfactors milestone results (PRELIMINARY)

> **Status: preliminary milestone evidence (2026-11-03).** These numbers come from one run of
> the package on the configurations below. They are not final and support **no alpha claims**.
> An independent audit re-computed the headline numbers (Section 3(f)), but the methodology has
> not been reviewed. The caveats are listed in Section 4. Teammate strategies
> (Card A `MinVariance`, Card B `ResidualReversion`) are not implemented yet. Their experiments
> were skipped, so the only economic k-curves in this run are the JKP factor-zoo SDF curves.

## 1. What was run

### Code and environment

| item | value |
|---|---|
| package | `kfactors 0.1.0a1` (editable install from this checkout) |
| git HEAD | `b85d4f770c44cd5b8239e3f93d26bdc1463919ed` ("Core backbone: ...", 2026-11-03 16:02 +0300) |
| working tree | **dirty**: 36 tracked files modified relative to HEAD (`git status --porcelain`); `git diff HEAD --binary \| shasum -a 256` = `2dc7a17cf275f440950199c9e7f184552da3ccdfd498b26f4a914b99ce5ea2d9`. The results belong to this working tree, not to HEAD. |
| code fingerprint | `kfactors.bench.code_fingerprint()` = `7a4b1d0ce740c2b99d6c` (SHA-256 over every `.py` file of the package; it is part of every bench cache key). No file under `src/` or `configs/` was modified after 16:45 local time, and the runs started at 16:47. |
| committed as | `41a474d` ("Review fixes: ..."). Verified after the run: `git diff b85d4f7 41a474d -- . ':!results' \| shasum -a 256` = `2dc7a17c...a2d9` and `code_fingerprint()` = `7a4b1d0ce740c2b99d6c`, identical to the run, so these results reproduce from commit `41a474d`. |
| JKP re-run | 2026-11-03 17:37, `bench real --panels jkp_usa_monthly`, same code (`code_fingerprint` 7a4b1d0ce740c2b99d6c), config sha256 `4b52c794eb551c054a4be7ea0a038492adaea276b6acfa38c1eb0b9cfe8e318f` (JKP `kmax: 40` + two 10 bps experiments). Everything except the JKP panel outputs and the leaderboard is from the first run. |
| package fixes in this run | none (no run failed) |
| config | `configs/default.yaml`, sha256 `72e8aca1f94c79d1f21a87ddfe6aa59c764db9ce3873982b4588a76b339e242c`; resolved copies in `results/config_resolved_{sim,real,leaderboard}.json` |
| seed | 0 |
| Python / libs | Python 3.11.14, numpy 2.4.6, pandas 3.1.0, scipy 1.17.1, scikit-learn 1.9.1, joblib 1.6.0 (`.venv`) |
| machine | Apple M4 Pro, 14 cores, 48 GB RAM, macOS 26.2 |
| tests at run time | `pytest -q`: 926 passed, 167 skipped, 3 deselected; `ruff check src tests`: clean |

### Commands (run from the repository root, in this order)

```bash
# 1) data (default cache ~/.cache/kfactors; KFACTORS_DATA unset)
.venv/bin/python -c "
from kfactors.data import load_jkp, load_french, load_binance
load_jkp(region='usa', freq='monthly', weighting='vw_cap', start='1971-11-30')
load_french('100_size_bm', freq='daily', weighting='vw', start='1963-07-01')
load_binance(top=100, start='2022-01-01', end='2025-12-31', history_start='2021-06-01', n_jobs=32)"
# 2) simulation benchmark + calibration
.venv/bin/python -m kfactors.bench sim --config configs/default.yaml --out results/ --reps 20 --n-jobs 8 --calibrate
# 3) real-data benchmark
.venv/bin/python -m kfactors.bench real --config configs/default.yaml --out results/ --n-jobs 8
# 4) leaderboard
.venv/bin/python -m kfactors.bench leaderboard --config configs/default.yaml --out results/
# 5) derived tables quoted below (read the outputs above; deterministic)
.venv/bin/python results/derived/sim_tables.py results results/derived
.venv/bin/python results/derived/real_tables.py results results/derived
.venv/bin/python results/derived/spectrum_rolling.py configs/default.yaml results/derived
```

In step 1 the loaders were called from an equivalent scratch script. `sim` used the configured
20 replications (`--reps 20`); it was not reduced. The bench log is `results/bench.log`.

### Runtime (wall clock)

| step | time | notes |
|---|---|---|
| JKP download + parse | 2.1 s | 1 file, 2.7 MB |
| French 100 daily download + parse | 4.5 s | 1 file, 15.8 MB |
| Binance cold download + build | 398 s (6 min 38 s) | 19,021 monthly kline zips (32.9 MB), `n_jobs=32`, 0 HTTP 404s |
| `bench sim` | 305.9 s (grid 303.5 s) | 111 designs x 20 reps x 17 rule configs = 37,740 rule calls, 0 failures; 2,264 s CPU; slowest design 45.6 s (p = 625). The `parallel` variants take about 0.2 s per call, every other rule about 1.5 ms. |
| `bench real` | 66.1 s | 3 panels; 0 failed steps, 3 skipped experiments (cards A and B); cache hits 0 |
| `bench leaderboard` | 0.4 s | |
| derived tables | about 7 s | |

### Data snapshot (from `~/.cache/kfactors/manifest.json`)

| panel | file / source | sha256 | bytes | downloaded (UTC) |
|---|---|---|---|---|
| JKP usa monthly vw_cap | `jkp/usa_all_factors_monthly_vw_cap.zip` from `https://jkpfactors-data.s3.amazonaws.com/public/[usa]_[all_factors]_[monthly]_[vw_cap].zip` | `f766aa2c7db1f97594f79f9063275161b7038e936db9ea960eac6eb7ec27ca80` | 2,674,337 | 2026-11-03T13:47:49 |
| French 100 size x B/M daily | `french/100_Portfolios_10x10_Daily_CSV.zip` from `mba.tuck.dartmouth.edu/.../ftp/` | `8fad24e04196408125ac739f7a065c4c799d7094f53cc48f9ee34df5405a0775` | 15,805,142 | 2026-11-03T13:47:53 |
| Binance spot 1d klines | 19,021 entries `binance/data/spot/monthly/klines/<SYM>/1d/*.zip` (552 symbols) | combined digest `52c461ed2b33abe19fbbc1edbca95d2d279f245c307dc4bdf0d232de39d95aeb` (1) | 32,857,813 | 13:48:09 .. 13:54:33 |

(1) The combined digest is the SHA-256 of the sorted lines `relpath<TAB>sha256\n` over all
`binance/` manifest entries. The symbol listing cache
(`~/.cache/kfactors/binance/listings/`) is not in the manifest.

The panels as built (fingerprints from `results/real/<panel>/panel.json`, which hash the
returns, the tradable mask, the dates, the ids and the frequency):

| panel | rows x assets | dates | missing share | fingerprint |
|---|---|---|---|---|
| `jkp_usa_monthly` | 650 x 153 | 1971-11-30 .. 2025-12-31 (M) | 0.0 | `b8bf7111882b458def72` |
| `french_100_daily` | 15,918 x 100 | 1963-07-01 .. 2026-09-30 (D, 252/yr) | 0.0023 | `a431e5887d7112a1484e` |
| `binance_top100` | 1,675 x 560 | 2021-06-01 .. 2025-12-31 (D, 365/yr) | 0.39 | `af0dabbb46884f18f0e2` |

On Binance, 694 candidate symbols remain after the stablecoin, leveraged-token and derivative
exclusions. The panel has 560 asset ids, with 8 ticker splits (LUNA_2, FTT_2, BNX_2, CVC_2,
QUICK_2, STRAX_2, SUN_2, VIDT_2) and 140 partial (delisting or migration) last bars that keep
their return but are not tradable.

## 2. Figures

| figure | path |
|---|---|
| sim exact-% heatmap (rule x design) | `results/sim/heatmap.png` |
| sim MAE heatmap | `results/sim/heatmap_mae.png` |
| spectra (latest window, naive MP and Laloux edges) | `results/real/{jkp_usa_monthly,french_100_daily,binance_top100}/spectrum.png` |
| rolling k_t by rule | `results/real/{jkp_usa_monthly,french_100_daily,binance_top100}/rolling_k.png` |
| report() spectrum + bootstrap k distributions | `results/real/{jkp_usa_monthly,french_100_daily,binance_top100}/report.png` |
| JKP k-curves (gross and 10 bps) | `results/real/jkp_usa_monthly/experiments/tangency_w{240,120}{,_cost10}/kcurve.png` |

## 3. Results

### (a) Simulation: does the rule find the true k?

Design: `sim.design_grid("default")` with 111 designs and 20 reps each, which gives 2,220
panels. `kmax = 10`; correlation matrix. The families are:

* pervasive: 36 designs = 6 noise types x 6 shapes;
* spiked: 36 designs, same layout;
* market: 18 designs;
* weak: 12 designs (near the BBP threshold);
* null: 9 designs, k = 0.

Every non-null design has k = 4. Shapes are (n, p) in {100, 250} x p/n in {0.4, 1, 2.5}.
Accuracy is "exact %" (`k_hat == k_true`) and MAE. Sources: `results/leaderboard.md`,
`results/sim/accuracy.csv`, `results/derived/sim_best_by_*.csv`.

**Overall** (2,220 panels per rule, 0 failed calls; ranked by MAE):

| rule | exact % | MAE | bias |
|---|---|---|---|
| onatski | 55.3 | **0.776** | -0.743 |
| parallel(null=circular) | 56.8 | 0.801 | -0.073 |
| parallel(devolatilize=true, null=circular) | 59.2 | 0.833 | +0.034 |
| parallel(null=block) | 57.1 | 0.871 | +0.238 |
| mp_market | **59.6** | 1.046 | +0.601 |
| parallel | 56.0 | 1.101 | +0.489 |
| mp | 43.8 | 1.125 | +0.088 |
| parallel(devolatilize=true) | 57.4 | 1.140 | +0.622 |
| bai_ng(criterion=PC_p1) | 40.0 | 1.656 | -0.051 |
| gavish_donoho | 42.2 | 1.670 | +0.572 |
| bai_ng(criterion=IC_p1) | 39.1 | 1.770 | -0.473 |
| bai_ng (IC_p2) | 36.4 | 1.903 | -0.606 |
| ahn_horenstein(stat=GR) | 35.3 | 1.937 | -1.917 |
| ahn_horenstein (ER) | 32.0 | 2.110 | -2.104 |
| mp(noise=median) | 26.4 | 3.049 | +2.865 |
| kaiser | 0.0 | 6.156 | +6.156 |
| variance_share | 0.0 | 6.324 | +6.324 |

**Exact % by family** (selected rules; the full table is in `results/leaderboard.md`):

| rule | market | null | pervasive | spiked | weak |
|---|---|---|---|---|---|
| onatski | 71.9 | 98.3 | **76.4** | 28.2 | 16.2 |
| parallel(null=circular) | 50.3 | 94.4 | 72.6 | 44.4 | 27.9 |
| parallel(devolatilize=true, null=circular) | 55.3 | 98.3 | 74.9 | **45.7** | 29.2 |
| mp_market | **85.0** | 92.8 | 70.7 | 38.8 | 25.8 |
| mp | 49.7 | 92.8 | 68.8 | 15.8 | 7.1 |
| gavish_donoho | 58.3 | 100.0 | 64.6 | 11.4 | 0.0 |
| bai_ng | 47.8 | 100.0 | 63.2 | 0.3 | 0.0 |
| ahn_horenstein | 40.6 | 100.0 | 53.1 | 0.3 | 0.0 |
| mp(noise=median) | 17.5 | 91.1 | 0.6 | 33.3 | **48.3** |

**Exact % by noise type** (selected rules):

| rule | gauss | t3 | garch | hetero | ar1-0.5 | ar1-0.9 |
|---|---|---|---|---|---|---|
| onatski | 54.2 | 79.8 | 77.3 | 51.7 | 45.8 | 2.9 |
| parallel(null=circular) | 53.9 | 71.9 | 80.7 | 57.2 | 60.0 | 4.6 |
| parallel(devolatilize=true, null=circular) | 57.4 | 77.9 | 82.0 | 58.1 | **62.9** | 0.8 |
| mp_market | **62.1** | **88.6** | **83.3** | **61.7** | 28.8 | 0.0 |
| mp | 37.7 | 58.3 | 57.7 | 41.7 | 53.3 | **11.2** |

**Best rule per regime.** The best rule is chosen by exact %, with MAE in brackets; the
`onatski` figures are given for reference. Source: `results/derived/sim_best_by_family_noise.csv`,
`..._by_family.csv`, `..._by_noise.csv`.

| regime | best rule (exact %, MAE) | onatski (exact %, MAE) |
|---|---|---|
| pervasive, gauss | gavish_donoho (98.3, 0.017) | 95.0, 0.058 |
| pervasive, t3 | gavish_donoho (97.5, 0.025) | 91.7, 0.092 |
| pervasive, garch | mp_market (100, 0) | 96.7, 0.042 |
| pervasive, hetero | mp_market (90.0, 0.108) | 85.0, 0.267 |
| pervasive, ar1-0.5 | bai_ng(criterion=IC_p1) (90.0, 0.100) | 84.2, 0.275 |
| pervasive, ar1-0.9 | mp (13.3); lowest MAE onatski (1.95) | 5.8, 1.95 |
| spiked, gauss | parallel(devolatilize=true) (66.7, 0.342) | 45.0, 0.758 |
| spiked, t3 | mp_market (71.7, 0.283) | 55.8, 0.558 |
| spiked, garch | parallel(devolatilize=true) and parallel(null=block) tie (65.8; MAE 0.342 for devolatilize) | 46.7, 0.717 |
| spiked, hetero | mp(noise=median) (41.7, 0.658) | 14.2, 1.425 |
| spiked, ar1-0.5 | parallel(devolatilize=true, null=circular) (39.2, 1.025) | 7.5, 1.833 |
| spiked, ar1-0.9 | mp (9.2); lowest MAE ahn_horenstein(stat=GR) (3.31) | 0.0, 3.50 |
| market, pervasive | 6 rules tie at 100% (mp_market and all five parallel variants) | 97.8, 0.039 |
| market, spiked | mp_market (70.0, 0.317) | 46.1, 0.694 |
| weak (spiked, gauss) | mp(noise=median) (48.3, 0.558), see note | 16.2, 0.867 |
| null (k = 0) | 6 rules tie at 100% (bai_ng x3, ahn_horenstein x2, gavish_donoho) | 98.3, 0.033 |
| *by noise, pooled over families* | gauss, t3, hetero: mp_market (62.1 / 88.6 / 61.7); garch: mp_market = parallel(devolatilize=true) (83.3); ar1-0.5: parallel(devolatilize=true, null=circular) (62.9); ar1-0.9: no rule above 11.2% | |

Notes on the simulation results:

* **Weak family.** The truth is k = 4, but the fourth spike is `r sqrt(p/n)` with r in
  {0.5, 1, 1.5, 2}. For r <= 1 it is asymptotically invisible in the spectrum. There `onatski`
  returns 3 in 98.3% of the panels, which is right about detectability but scored as wrong.
  `mp(noise=median)` returns 4 in 18.3 / 36.7 / 65.0 / 73.3% of panels for r = 0.5 / 1 / 1.5 / 2
  (`results/derived/sim_weak_by_bbp.csv`). For r <= 1 those hits are lucky over-counts (it also
  returns 5-7 there). The family score is only partly a scoring artefact, though: above the
  threshold (r = 1.5 / 2) `onatski` still returns 3 in 81.7% / 45.0% of panels, so it is
  under-powered just above BBP.
* **AR(0.9) noise.** No rule exceeds 13.3% exact in any family x noise cell (pooled over the six
  shapes). Single designs vary: `mp` gets 80% on `pervasive_ar1-0.9_n100_p40` and 55% on
  `spiked_ar1-0.9_n100_p40`; every other rule x AR(0.9) design cell is at most 20%.
* **Kaiser and variance share** reach kmax = 10 on almost every panel (exact 0%).
* **Warnings.** 1,480 calls of the two circular-null `parallel` variants (740 each) warned that
  `p > 2n` (columns share shifts), as documented. One `onatski` call did not converge.

**Calibrated decision table** (`--calibrate`). Source: `results/sim/decision_table.json`. It is
a CART tree with `max_depth=3, min_samples_leaf=5`, fitted on 7 diagnostics. The target is the
rule with the smallest |error| per panel. Its splits:

* `spike_share <= 0.247`:
  * `gap_ratio_1 <= 1.088` and `spike_share <= 0.074` -> onatski;
  * `gap_ratio_1 <= 1.088` and `spike_share > 0.074` -> parallel(null=block) (a 5-panel leaf);
  * `gap_ratio_1 > 1.088` -> onatski.
* `spike_share > 0.247`:
  * `p_over_n <= 0.70` and `mean_abs_acf1 <= 0.275` -> onatski;
  * `p_over_n <= 0.70` and `mean_abs_acf1 > 0.275` -> mp;
  * `p_over_n > 0.70` -> onatski.

In-sample, the tree picks onatski for 2,152 of 2,220 panels. The packaged table was **not**
replaced.

The table below scores each selector as a k estimator on the 2,220 sim panels: the rule it
picks per panel, compared with k_true. Source: `results/derived/sim_decision_table_eval.csv`.

| selector | exact % | MAE | exact %: market / null / pervasive / spiked / weak |
|---|---|---|---|
| calibrated table, in-sample | 56.4 | 0.746 | 71.9 / 98.3 / 78.2 / 29.7 / 16.2 |
| calibrated table, leave-one-family-out | 55.0 | 0.890 | 71.9 / 98.3 / 75.6 / 28.2 / 16.2 |
| best single rule by MAE: onatski | 55.3 | 0.776 | 71.9 / 98.3 / 76.4 / 28.2 / 16.2 |
| best single rule by exact %: mp_market | 59.6 | 1.046 | 85.0 / 92.8 / 70.7 / 38.8 / 25.8 |
| packaged heuristic table (current `report()` default) | 60.6 | 0.860 | 71.9 / 98.3 / 75.6 / 42.2 / 25.8 |
| per-panel oracle (best rule in hindsight) | 77.9 | 0.330 | |

The bench's own label metrics count a pick as correct when it is the target rule (the
per-panel best rule, with ties broken by priority):

* train accuracy 0.718;
* leave-one-family-out accuracy 0.639 (market 0.769, null 0.983, pervasive 0.661, spiked 0.479,
  weak 0.600);
* MAE of the tree's picks: 0.746 in-sample and 0.890 leave-one-family-out, against 0.776 for
  onatski.

**Reading.** The calibrated table beats the best single rule (onatski) in-sample only slightly,
by 0.03 MAE and 1.1 points of exact %. Leave-one-family-out it is worse: MAE 0.890 vs 0.776 and
exact 55.0% vs 55.3%. In this run the calibration does not generalise across design families.
Applied to the three real panels' diagnostics, the calibrated table also recommends `onatski`.

### (b) Spectra: eigenvalues above the naive MP edge and the Laloux edge

The edges come from `kfactors.plots.spectrum_edges` on the correlation matrix:

* The *naive* MP edge is `sigma2 (1 + sqrt(p/n))^2`, with `sigma2` fitted by matching the bulk
  median to the MP median.
* The *Laloux* edge uses `sigma2 = (trace - lambda_1) / p`.

The naive edge is not the edge of the `mp` rule. That rule uses mean noise (`sigma2 = 1` for a
correlation matrix) and `gamma = p / (n - 1)`; see its k in (c).

Latest window (`results/real/<panel>/spectrum.json`, `spectrum.png`):

| panel | window | n x p | lambda_1 | naive MP edge (sigma2) | # above naive | Laloux edge (sigma2) | # above Laloux |
|---|---|---|---|---|---|---|---|
| JKP | 2006-01-31 .. 2025-12-31 | 240 x 153 | 43.06 (next 34.49) | 0.174 (0.054) | **50** | 2.324 (0.719) | **10** |
| French 100 | 2026-04-01 .. 2026-09-30 | 126 x 99 | 45.74 | 1.022 (0.287) | **13** | 1.914 (0.538) | **6** |
| Binance | 2025-10-03 .. 2025-12-31 | 90 x 92 | 57.80 | 0.757 (0.187) | **17** | 1.504 (0.372) | **7** |

All rolling windows use the same decision grid and universe as (c). Source:
`results/derived/spectrum_rolling_summary.csv`, per window in `spectrum_rolling.csv`.

| panel | windows | # above naive: median [IQR] (min..max) | # above Laloux: median [IQR] (min..max) | median lambda_1 / p |
|---|---|---|---|---|
| JKP (240 m, step 12) | 34 | 49 [47.25, 50] (45..51) | 11.5 [11, 12] (9..13) | 0.43 |
| French 100 (126 d, step 21) | 753 | 9 [7, 12] (2..20) | 5 [4, 5] (2..7) | 0.67 |
| Binance (90 d, step 7) | 209 | 11 [10, 13] (5..18) | 6 [5, 6] (4..7) | 0.58 |

The Laloux counts have the same medians as the rolling `mp_market` k_t: 11.5 / 5 / 6. Window by
window they agree in 100% / 97% / 89% of the windows (JKP / French / Binance). Where they
differ, the Laloux count is one higher, because it uses `gamma = p/n` and not `p/(n-1)`.

The September pilot's "MP edge detects 6-9 factors on JKP" used `sigma2 = 1` and
`gamma = p/n`. That is the `mp` rule, not the median-matched "naive" count above. Recomputed
on the same 34 JKP windows, it gives a median of 7 (range 6..9) and equals the `mp` rule's k
in every window (see (c)).

### (c) Rolling k_t per rule

The values are median [q25, q75] of the k chosen in each walk-forward window. The universe is
the engine's (tradable at t and complete in the window); `kmax = 20`. Source:
`results/real/<panel>/rolling_k.csv` and `rolling_k_summary.csv`, with IQRs in
`results/derived/real_rolling_k_iqr.csv`.

| rule | JKP w240 (34 windows, yearly) | JKP w240, monthly k_used (407) | JKP w120, monthly k_used (407) | French 100 w126 (753) | Binance w90 (209) |
|---|---|---|---|---|---|
| ahn_horenstein | 1 [1, 1] | 1 [1, 1] | 1 [1, 2] | 1 [1, 1] | 1 [1, 1] |
| bai_ng | 20 [20, 20] | 20 [20, 20] | 17 [14, 20] | 3 [2, 3] | 1 [1, 1] |
| gavish_donoho | 20 [20, 20] | 20 [20, 20] | 20 [20, 20] | 5 [3, 8] | 7 [6, 9] |
| kaiser | 18 [17, 20] | 18 [17, 20] | 18 [16, 19] | 6 [4, 10] | 14 [12, 17] |
| mp | 7 [7, 7.75] | 7 [7, 7] | 6 [5, 6] | 2 [1, 2] | 1 [1, 1] |
| mp_market | 11.5 [11, 12] | 11 [11, 12] | 8 [8, 9] | 5 [4, 5] | 6 [5, 6] |
| onatski | 2 [1, 4] | 2 [1, 4] | 2 [1, 4] | 3 [2, 3] | 1 [1, 1] |
| parallel | 8 [8, 9] | 8 [8, 9] | 7 [6, 7] | 2 [1, 2] | 1 [1, 1] |
| variance_share | 18 [18, 20] | 18 [18, 20] | 18 [13, 19] | 20 [14, 20] | 20 [20, 20] |

The rolling grids are:

* JKP: 1992-01-31 .. 2025-01-31, 153 assets in every window.
* French: 1963-12-31 .. 2026-09-30, 96 to 100 assets.
* Binance: 2022-01-01 .. 2025-12-27, 80 to 99 assets (median 93).

No rule failed or warned in any rolling window (`rolling_k_info.json`). On JKP, `bai_ng`,
`gavish_donoho`, `kaiser` and `variance_share` sit at or near the cap of 20.

**JKP re-run with `kmax = 40` (2026-11-03 17:37):** the JKP yearly rolling file now gives `bai_ng` 36
(range 20-40), `gavish_donoho` 40 in every window (the new cap), `variance_share` 18 (17-24) and
`kaiser` 18 (16-20); all other rules are unchanged. The JKP columns of the table above are from the
first run (`kmax = 20`).

### (d) JKP factor zoo: what does k cost?

> **Re-run 2026-11-03 17:37** (`bench real --panels jkp_usa_monthly`, config with `kmax: 40` on the
> JKP panel and two new 10 bps experiments). The first run capped the rules at `kmax = 20` while the
> k grid went to 40, so no rule could reach the oracle. This section, the JKP rolling k_t files and
> the JKP `report()` now use `kmax = 40`. The fixed-k curves are unchanged (they do not depend on kmax).

The strategy is `TangencyPCA`: the tangency portfolio of the first k PC portfolios,
`sum |w| = 1`. Settings:

* panel: 153 JKP usa monthly vw_cap factors;
* walk-forward: rebalance 1, lag 0, `min_assets=100`, standardised; costs 0 bps (gross) and 10 bps
  per unit of turnover across factor weights (a crude proxy: the factors are themselves long-short
  portfolios with their own internal turnover, which is not charged);
* out-of-sample period: 407 months, 1992-02-29 .. 2025-12-31, the same for both windows;
* metric: annualised net Sharpe ratio;
* bootstrap: 1,000 stationary-bootstrap draws shared across configurations; the oracle set
  covers at least 90% of the argmax draws;
* `k_grid = 1..40`, rules with `kmax = 40`.

Sources: `results/real/jkp_usa_monthly/experiments/tangency_w{240,120}{,_cost10}/{summary.json,curve.csv,rules.csv,kcurve.png}`.
Rule rows: median k (mean k), Sharpe [90% marginal], in the oracle set, regret against the oracle
[90% paired].

**gross (0 bps)**

| | window 240 | window 120 |
|---|---|---|
| oracle k | **38** | **33** |
| oracle confidence set | {23, 27, 29, 30, 31, 36, 38, 40} | {18, 19, 26, 32, 33, 36, 40} |
| Sharpe at oracle [90%] | **1.861 [1.542, 2.228]** | **1.869 [1.505, 2.253]** |
| fixed-k Sharpe at k = 1 / 2 / 5 / 10 / 20 / 40 | 0.087 / 0.181 / 1.044 / 1.151 / 1.601 / 1.823 | 0.100 / 0.050 / 0.754 / 1.181 / 1.700 / 1.812 |

| rule | w240: k (mean) | w240: Sharpe [90%] | in set | regret [90% paired] | w120: k (mean) | w120: Sharpe [90%] | in set | regret [90% paired] |
|---|---|---|---|---|---|---|---|---|
| ahn_horenstein | 1 (1.16) | 0.067 [-0.283, 0.446] | no | 1.794 [1.303, 2.219] | 1 (1.45) | 0.156 [-0.079, 0.409] | no | 1.713 [1.258, 2.137] |
| onatski | 2 (2.85) | 0.096 [-0.289, 0.583] | no | 1.765 [1.206, 2.222] | 2 (2.61) | 0.242 [0.002, 0.496] | no | 1.627 [1.179, 2.042] |
| mp | 7 (7.29) | 1.164 [0.830, 1.525] | no | 0.698 [0.420, 0.960] | 6 (5.68) | 0.917 [0.645, 1.226] | no | 0.952 [0.583, 1.297] |
| parallel | 8 (8.52) | 1.149 [0.812, 1.508] | no | 0.713 [0.419, 0.974] | 7 (6.65) | 0.998 [0.678, 1.321] | no | 0.871 [0.547, 1.211] |
| mp_market | 11 (11.16) | 1.127 [0.769, 1.535] | no | 0.734 [0.460, 0.974] | 8 (8.27) | 0.946 [0.604, 1.286] | no | 0.923 [0.628, 1.243] |
| kaiser | 18 (18.25) | 1.594 [1.214, 1.990] | no | 0.267 [0.031, 0.492] | 18 (17.90) | 1.685 [1.359, 2.032] | **yes** | 0.185 [-0.015, 0.405] |
| variance_share | 18 (19.77) | 1.592 [1.216, 1.960] | no | 0.269 [0.051, 0.479] | 18 (17.09) | 1.684 [1.359, 2.030] | **yes** | 0.186 [-0.030, 0.421] |
| bai_ng | 36 (33.38) | 1.762 [1.445, 2.120] | **yes** | 0.099 [-0.007, 0.199] | 17 (19.36) | 1.592 [1.280, 1.922] | no | 0.278 [0.094, 0.474] |
| gavish_donoho | 40 (40.00) | 1.823 [1.478, 2.190] | **yes** | 0.038 [-0.030, 0.095] | 30 (30.05) | 1.816 [1.427, 2.201] | no | 0.053 [-0.055, 0.171] |

**net of 10 bps per unit turnover**

| | window 240 | window 120 |
|---|---|---|
| oracle k | **38** | **18** |
| oracle confidence set | {17, 23, 28, 29, 30, 31, 36, 38, 40} | {13, 14, 15, 17, 18, 19, 26, 32, 33, 36} |
| Sharpe at oracle [90%] | **1.712 [1.388, 2.084]** | **1.595 [1.249, 1.963]** |
| fixed-k Sharpe at k = 1 / 2 / 5 / 10 / 20 / 40 | 0.082 / 0.167 / 1.017 / 1.072 / 1.505 / 1.663 | 0.075 / 0.020 / 0.694 / 1.068 / 1.534 / 1.454 |

| rule | w240: k (mean) | w240: Sharpe [90%] | in set | regret [90% paired] | w120: k (mean) | w120: Sharpe [90%] | in set | regret [90% paired] |
|---|---|---|---|---|---|---|---|---|
| ahn_horenstein | 1 (1.16) | 0.061 [-0.290, 0.442] | no | 1.651 [1.145, 2.078] | 1 (1.45) | 0.132 [-0.103, 0.383] | no | 1.463 [1.044, 1.867] |
| onatski | 2 (2.85) | 0.072 [-0.306, 0.559] | no | 1.640 [1.077, 2.098] | 2 (2.61) | 0.188 [-0.056, 0.434] | no | 1.407 [1.002, 1.787] |
| mp | 7 (7.29) | 1.124 [0.794, 1.483] | no | 0.588 [0.306, 0.844] | 6 (5.68) | 0.868 [0.591, 1.175] | no | 0.728 [0.412, 1.014] |
| parallel | 8 (8.52) | 1.094 [0.758, 1.445] | no | 0.618 [0.317, 0.882] | 7 (6.65) | 0.909 [0.589, 1.239] | no | 0.686 [0.417, 0.943] |
| mp_market | 11 (11.16) | 1.051 [0.694, 1.452] | no | 0.661 [0.386, 0.900] | 8 (8.27) | 0.863 [0.530, 1.202] | no | 0.732 [0.477, 1.001] |
| kaiser | 18 (18.25) | 1.501 [1.117, 1.898] | no | 0.211 [-0.027, 0.435] | 18 (17.90) | 1.525 [1.197, 1.874] | **yes** | 0.070 [0.010, 0.136] |
| variance_share | 18 (19.77) | 1.480 [1.101, 1.853] | no | 0.231 [0.006, 0.439] | 18 (17.09) | 1.518 [1.190, 1.858] | **yes** | 0.077 [-0.012, 0.175] |
| bai_ng | 36 (33.38) | 1.573 [1.253, 1.931] | **yes** | 0.138 [0.035, 0.243] | 17 (19.36) | 1.395 [1.074, 1.720] | **yes** | 0.200 [0.075, 0.342] |
| gavish_donoho | 40 (40.00) | 1.662 [1.319, 2.034] | **yes** | 0.049 [-0.017, 0.109] | 30 (30.05) | 1.498 [1.102, 1.888] | no | 0.097 [-0.112, 0.310] |

**Reading** (preliminary, a "detectable vs useful" pattern):

* The rules that *detect* structure pick few factors on JKP: onatski 2, ahn_horenstein 1, mp 6-7,
  parallel 7-8, mp_market 8-11. Their paired regret intervals exclude 0 in all four experiments
  (gross regret 0.70-1.79 Sharpe units).
* The out-of-sample Sharpe climbs steeply to about k = 14-17 (window 240) / 12 (window 120), then
  rises slowly into a broad plateau; the point argmax is 38 / 33 gross.
* Costs move the optimum down but do not change the pattern: at 10 bps the window-240 oracle stays
  at 38 (Sharpe 1.71) while the window-120 oracle drops to 18 (Sharpe 1.60), and its confidence set
  widens to 13-36. Detection rules still sit well below the set.
* `bai_ng` and `gavish_donoho` now land in (or near) the oracle set at window 240, **but not because
  they detect the structure**: with the larger kmax they slide to it. `gavish_donoho` sits exactly at
  the cap (k = 40 in every month at window 240) and `bai_ng` moves from 20 (kmax 20) to 33-36
  (kmax 40) on the same data; in the `report()` bootstrap `bai_ng` falls to 13-19 (P(k* = k) = 0).
  This is the known kmax sensitivity of Bai-Ng criteria (penalties are weak when n and p are of the
  same order) and of hard thresholding on a slowly decaying spectrum. Treat their in-set status as
  an artefact of the cap, not as evidence that they choose k well.
* `kaiser` and `variance_share` (k = 18) fall inside the window-120 oracle set, gross and net.

The real-data leaderboard (`results/leaderboard.md`, "All experiments") now covers these four
experiments.

### (e) `report()` recommendation per panel (latest window, packaged heuristic table)

Sources: `results/real/<panel>/report.md`, `report.json`, `report_rules.csv`.
`n_boot = 200` (stationary bootstrap, distinct rows only), `kmax = 20`.

| panel | window (n x p) | recommendation | k | bootstrap median, P(k* = k) | decision path |
|---|---|---|---|---|---|
| JKP | 2006-01 .. 2025-12 (240 x 153) | onatski | **2** | 2, 0.75 (draws: 2 75%, 6 22.5%) | mean_abs_acf1 0.093 <= 0.2; gap_ratio_1 1.25 <= 4; spike_share 0.281 <= 0.3; p/n 0.64 <= 1 -> default onatski |
| French 100 | 2026-04-01 .. 2026-09-30 (126 x 99) | onatski | **1** | 2, 0.47 (1: 47%, 2: 37%) | gap_ratio_1 5.90 > 4 (dominant market mode) -> onatski |
| Binance | 2025-10-03 .. 2025-12-31 (90 x 92) | onatski | **3** | 2, 0.255 (1: 42.5%, 2: 30.5%, 3: 25.5%) | gap_ratio_1 18.43 > 4 (dominant market mode) -> onatski |

How to read the bootstrap columns:

* The `report()` bootstrap keeps distinct rows only. On average a draw has 152.5 / 81.9 / 57.5
  of the 240 / 126 / 90 rows (JKP / French / Binance), so the draws have a smaller effective n.
* Threshold rules therefore shift down in the draws: Binance `mp_market` is 7 on the full
  window but 4-6 in the draws (P = 0), and JKP `mp_market` is 10 but 8-9 in the draws.
* P(k* = k) measures how stable k is under this resampling. It is not a sampling distribution
  centred on the point estimate.
* Binance's latest-window onatski k = 3 is not typical: the rolling median is 1 [1, 1] (see (c)).

JKP `report()` was regenerated with `kmax = 40`: the recommendation is unchanged (onatski, k = 2, P = 0.75); `bai_ng` is now 33 (bootstrap 13-19), `gavish_donoho` 40, `kaiser` 20, `variance_share` 23. The table below is from the first run (`kmax = 20`).

Every rule's k on the same latest windows:

| rule | JKP | French 100 | Binance |
|---|---|---|---|
| ahn_horenstein | 2 | 1 | 1 |
| bai_ng | 20 | 4 | 1 |
| gavish_donoho | 20 | 10 | 13 |
| kaiser | 20 | 13 | 13 |
| mp | 7 | 3 | 1 |
| mp_market | 10 | 6 | 7 |
| onatski | 2 | 1 | 3 |
| parallel | 9 | 4 | 1 |
| variance_share | 20 | 20 | 20 |

The diagnostics of the three panels are:

| panel | mean excess kurtosis | spike_share | var_dispersion |
|---|---|---|---|
| JKP | 2.31 | 0.28 | 0.78 |
| French | 1.06 | 0.46 | 0.64 |
| Binance | 6.81 | 0.63 | 1.65 |

### (f) Independent checks and comparison with the September pilot

The independent audit (2026-11-03) used scratch scripts, not bench outputs.

**Reproduced exactly:**

* The sim tables in (a), from `sim/raw.csv`.
* The decision-table scores, by re-running `kfactors.report.recommend` per panel.
* The latest-window spectra and rule k in (b) and (e), with plain numpy and `select_k`.
* The JKP k-curve returns, with a plain-numpy walk-forward that does not use the engine:
  * the window is rows t-239..t (or t-119..t), the PCs come from the correlation matrix, and
    w is scaled to `sum |w| = 1`;
  * the weights earn row t+1;
  * the result matches `returns.csv` to <= 1e-15 for k in {1, 2, 4, 10, 20, 30, 33, 38, 40} in
    both windows.

  This rules out timing look-ahead in this experiment. It does not address the data look-ahead
  in Section 4.

**Comparison with the September pilot** (`pilot/frontier/factor_zoo/pilot4.py` and
`pilot/frontier/stat_arb/`):

* **JKP MP count: agrees.** The pilot's 6-9 (median 7) is the `mp` rule here: 7 [7, 7.75],
  range 6..9 at window 240; 6 [5, 6] at window 120. See (b).
* **JKP k-curve: same shape, different peak location.** The pilot reported a hump that peaked
  at 15-30 PCs. Here the point argmax is 33-38.
  * The pilot's conventions differ: covariance PCA, unit in-sample-volatility scaling, annual
    rebalancing, out-of-sample from 1995-01, and the grid {.., 20, 30, 40, 60, 80}.
  * Re-run with those conventions, it reproduces the pilot: 0.76 / 1.18 / 1.55 / 1.64 at
    k = 5 / 10 / 20 / 30. On a finer grid it peaks at k = 35 (1.74), then gives 1.55 at k = 40
    and 1.32 at k = 60.
  * The package's conventions extended to k = 100 keep the point argmax at 38 / 33. Window 240
    gives 1.71 at k = 50, 1.63 at k = 60 and 1.52 at k = 100. Window 120 gives 1.72, 1.63 and
    1.12 at the same k.
  * So both versions are hump-shaped with a broad top around k = 30-40. The pilot's "15-30"
    combined two things: the PCA peak at k = 30 on a coarse grid, and RP-PCA peaks at
    k = 15-20 (RP-PCA is not run here).
  * Levels at k = 10 / 20 agree within 0.05: the pilot gives 1.18 / 1.55 against 1.15 / 1.60
    here.
* **Binance spectrum and rules: agree.**
  * lambda_1 / p has a rolling median of 0.58 and is 0.63 in the latest window; the pilot
    reported about 0.6.
  * The econometric rules and `mp` pick k = 1 [1, 1].
  * The Laloux / `mp_market` count is 6 [5, 6]. The pilot's market-adjusted edge at window 90
    had a median of 6 (range 4-8).
  * The pilot's economic optimum (k = 5-10 for stat-arb) cannot be checked yet, because Card B
    is not implemented.

## 4. Caveats (read before quoting any number)

* **Preliminary.** This is a single run with seed 0 on a dirty working tree (Section 1). Nothing
  here is final. The audit in 3(f) re-checked the numbers, not the methodology.
* **No alpha claims.** The JKP returns are *already signed* by the JKP `direction`, so each
  factor is long the leg that the literature found to earn more. That sign is fixed ex post
  relative to our 1992-2025 test period. The tangency Sharpe ratios (about 1.6-1.9) are
  therefore upward biased and are not investable performance. They use 0 costs (factor returns
  are already portfolio returns, but trading the factors is not free), and the weights are
  `sum |w| = 1` with no leverage or capacity constraints. Only the *shape* of the k-curve and
  the relative regrets are of interest.
* **Factor selection is also ex post.** The 153-factor zoo includes factors published after the
  start of the test period, and they enter from 1992. In the September pilot, restricting to
  factors already published lowered the window-240 PCA k = 20 Sharpe from 1.55 to 1.24 (pilot
  numbers, not re-run here).
* **Costs and turnover.** Turnover rises with k: at window 240 the mean monthly turnover is 0.03
  at k = 1, 0.055 for mp's k, 0.09 at k = 20 and 0.12 at k = 38; at window 120 it is 0.24 at
  k = 33. The high-k portfolios also have very low volatility (0.8-0.9% a year). A rough
  approximation shows what 10 bps per unit of turnover would do:
  * the Sharpe reduction is about 12 x cost x turnover / annual volatility;
  * the oracle Sharpe would fall by about 0.15 at window 240 and 0.35 at window 120, against
    0.04-0.05 for `mp`;
  * at window 120 the oracle would then be about level with `kaiser`.

  The k-curves have not been re-run with costs, so the location of the optimum is not
  cost-robust.
* **Oracle optimism.** The oracle Sharpe is the maximum of 40 noisy estimates, so it is biased
  upward (winner's curse). Regrets are measured against this maximum.
* **Rule cap vs grid.** In the k-curves the rules are capped at `kmax = 20` while the grid goes
  to 40. The oracle k (38, 33) is therefore unreachable by construction, which drives most
  `in_set = no`. The oracle is near the top of the grid, but extending the grid does not move
  it: in the audit re-implementation ((f)), with k up to 100 the point argmax stays at 38 / 33,
  and the curve declines slowly beyond about k = 45.
* **Oracle set.** The oracle set is a simple bootstrap-percentile construction (it is not the
  Hansen-Lunde-Nason MCS). It is scattered (non-contiguous) in both windows, and the argmax
  share of the oracle is only 0.49 / 0.27, so the location of the optimum is weakly identified.
* **Crypto universe.** The Binance universe is built point in time from all Binance Vision
  symbols, including delisted ones (140 partial delisting or migration bars; LUNA_2, FTT_2 and
  others), so it is free of survivorship bias with respect to Binance listings. Limits:
  * it is not free of exchange-selection bias (only Binance USDT spot pairs);
  * held assets that disappear exit at their last price with no delisting haircut;
  * renamed tickers are not chained (MATIC->POL and others become new assets);
  * stablecoins, leveraged tokens and wrapped/staked duplicates are excluded;
  * the latest Binance window has p > n (92 x 90).
* **Spectra.** The "naive" MP edge of `spectrum_edges` is median-matched with `gamma = p/n`, a
  plotting cross-check that is known to over-count on return-like panels. Its counts (JKP 50)
  are not a factor estimate. The Laloux counts equal the `mp_market` rule in the latest windows
  and in 89-100% of the rolling windows; otherwise they are one higher. Spectra and
  `report()` are reported for the latest window only; the rolling counts are in
  `results/derived/`.
* **Simulation scoring.**
  * Weak family: k_true = 4 includes a factor that is below the BBP threshold for r <= 1, which
    rewards over-counting rules.
  * Regime grouping: null designs carry `regime = "pervasive"` (the SimSpec default), so the
    leaderboard's "Exact % by regime / pervasive" column pools 180 null panels. The per-family
    tables do not have this problem.
  * Packaged heuristic table: its sim score (60.6% exact) is not an out-of-sample result. Its
    `meta.revisions` say that it was tuned partly on 3-seed runs of the same design grid.
* **Calibration.** The calibrated decision table is written to
  `results/sim/decision_table.json` only; `report()` still uses the packaged heuristic draft.
  Its leave-one-family-out performance is worse than the best single rule.
* **Skipped experiments.** `binance_top100/statarb_w90`, `statarb_w60` (Card B) and
  `french_100_daily/minvar_w126` (Card A) were skipped because the strategies raise
  `NotImplementedError`. There is no economic k-curve for Binance or French yet.
* **French data.** French data run to 2026-09-30; the latest French window drops one portfolio
  (p = 99).
* **Licences.** JKP data are CC BY-NC 4.0 and Binance Vision data are CC BY-NC-SA 4.0
  (non-commercial). Cite JKP (Jensen, Kelly & Pedersen 2023), the Kenneth R. French Data Library
  and Binance Vision.

## 5. File index

* `results/sim/`:
  * `raw.csv` (37,740 rows), `accuracy.csv`, `designs.csv`, `timings.csv`, `run_info.json`.
    Read the CSVs with `pd.read_csv(path, keep_default_na=False, na_values=[""])`. With the
    pandas defaults, the family label `null` becomes NaN and the 180 null panels drop out of
    group-bys. The bench and `results/derived/sim_tables.py` already read them this way;
  * `decision_table.json` (calibrated, not installed);
  * `heatmap.png`, `heatmap_mae.png`.
* `results/real/<panel>/`:
  * `panel.json`, `spectrum.{json,png}`;
  * `rolling_k.{csv,png}`, `rolling_k_summary.csv`, `rolling_k_info.json`;
  * `report.{md,json,png}`, `report_rules.csv`, `report_k_distribution.csv`;
  * `experiments/<exp>/summary.json`.
* `results/real/jkp_usa_monthly/experiments/tangency_w{240,120}/`: `curve.csv`, `rules.csv`,
  `kcurve.{csv,png}`, `k_used.csv`, `returns.csv`, `specs_summary.csv`.
* `results/real/status.csv`, `results/real/run_info.json`.
* `results/leaderboard.{csv,md}`, `results/bench.log`, `results/run_info_*.json`,
  `results/config_resolved_*.json`.
* `results/derived/`: the scripts `sim_tables.py`, `real_tables.py` and `spectrum_rolling.py`
  plus their outputs:
  * `sim_best_by_{family,noise,regime,family_regime,family_noise}.csv`;
  * `sim_decision_table_eval.csv`, `sim_decision_table_meta.json`, `sim_weak_by_bbp.csv`;
  * `real_rolling_k_iqr.csv`, `spectrum_rolling{,_summary}.csv`.
* `results/cache/`: the bench pickle cache (git-ignored).
