<!-- panel french_100_daily, window 2026-04-01 .. 2026-09-30 (n=126, p=99) -->

# kfactors report

Panel: n = 126 observations x p = 99 series (p/n = 0.786); correlation matrix; kmax = 20 (effective).

## Recommendation

**onatski** -> **k = 1** (bootstrap median 2, P(k* = k) = 0.47).

Reason (heuristic decision table): mean_abs_acf1 = 0.07451 <= 0.2; gap_ratio_1 = 5.899 > 4 [dominant first eigenvalue (market mode)] -> onatski: dominant market mode: Onatski (2010) needs no noise-variance estimate, so the market eigenvalue does not bias it; mp_market over-counts when betas or idiosyncratic volatilities are dispersed (branch 2 in meta)

## Diagnostics

| diagnostic | value |
|---|---|
| n | 126 |
| p | 99 |
| p_over_n | 0.7857 |
| spike_share | 0.4621 |
| gap_ratio_1 | 5.899 |
| max_gap_ratio | 5.899 |
| argmax_gap_ratio | 1 |
| mean_excess_kurtosis | 1.064 |
| median_excess_kurtosis | 0.5723 |
| mean_acf1 | -0.04775 |
| mean_abs_acf1 | 0.07451 |
| mean_acf1_sq | 0.02547 |
| var_dispersion | 0.6367 |

## Rules

| rule | k | boot_median | boot_q05 | boot_q95 | boot_p_same | n_boot_failed | n_boot_warnings | recommended |
|---|---|---|---|---|---|---|---|---|
| ahn_horenstein | 1 | 1 | 1 | 1 | 1 | 0 | 0 |  |
| bai_ng | 4 | 4 | 2 | 4 | 0.5 | 0 | 0 |  |
| gavish_donoho | 10 | 8 | 7 | 9 | 0.005 | 0 | 0 |  |
| kaiser | 13 | 15 | 13 | 17 | 0.055 | 0 | 0 |  |
| mp | 3 | 3 | 2 | 4 | 0.675 | 0 | 0 |  |
| mp_market | 6 | 6 | 5 | 6 | 0.83 | 0 | 0 |  |
| onatski | 1 | 2 | 1 | 6 | 0.47 | 0 | 0 | yes |
| parallel | 4 | 4 | 3 | 4 | 0.55 | 0 | 0 |  |
| variance_share | 20 | 20 | 20 | 20 | 1 | 0 | 0 |  |

## Bootstrap distribution of k

Stationary bootstrap of rows (Politis & Romano 1994): 200 draws, mean block 5.013, distinct rows only (on average 81.89 of 126 rows per draw).

| rule | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ahn_horenstein | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| bai_ng | 0.00 | 0.00 | 0.08 | 0.41 | 0.50 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gavish_donoho | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.24 | 0.53 | 0.18 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| kaiser | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.06 | 0.30 | 0.35 | 0.23 | 0.04 | 0.01 | 0.00 | 0.00 |
| mp | 0.00 | 0.00 | 0.26 | 0.68 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| mp_market | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | 0.83 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| onatski | 0.00 | 0.47 | 0.37 | 0.08 | 0.01 | 0.01 | 0.04 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| parallel | 0.00 | 0.00 | 0.02 | 0.43 | 0.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| variance_share | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
