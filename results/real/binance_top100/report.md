<!-- panel binance_top100, window 2025-10-03 .. 2025-12-31 (n=90, p=92) -->

# kfactors report

Panel: n = 90 observations x p = 92 series (p/n = 1.02); correlation matrix; kmax = 20 (effective).

## Recommendation

**onatski** -> **k = 3** (bootstrap median 2, P(k* = k) = 0.255).

Reason (heuristic decision table): mean_abs_acf1 = 0.06728 <= 0.2; gap_ratio_1 = 18.43 > 4 [dominant first eigenvalue (market mode)] -> onatski: dominant market mode: Onatski (2010) needs no noise-variance estimate, so the market eigenvalue does not bias it; mp_market over-counts when betas or idiosyncratic volatilities are dispersed (branch 2 in meta)

## Diagnostics

| diagnostic | value |
|---|---|
| n | 90 |
| p | 92 |
| p_over_n | 1.022 |
| spike_share | 0.6282 |
| gap_ratio_1 | 18.43 |
| max_gap_ratio | 18.43 |
| argmax_gap_ratio | 1 |
| mean_excess_kurtosis | 6.805 |
| median_excess_kurtosis | 5.13 |
| mean_acf1 | 0.02756 |
| mean_abs_acf1 | 0.06728 |
| mean_acf1_sq | 0.04007 |
| var_dispersion | 1.653 |

## Rules

| rule | k | boot_median | boot_q05 | boot_q95 | boot_p_same | n_boot_failed | n_boot_warnings | recommended |
|---|---|---|---|---|---|---|---|---|
| ahn_horenstein | 1 | 1 | 1 | 1 | 1 | 0 | 0 |  |
| bai_ng | 1 | 1 | 1 | 2 | 0.68 | 0 | 0 |  |
| gavish_donoho | 13 | 6 | 3 | 8 | 0 | 0 | 0 |  |
| kaiser | 13 | 12 | 11 | 16 | 0.115 | 0 | 0 |  |
| mp | 1 | 1 | 1 | 1 | 1 | 0 | 0 |  |
| mp_market | 7 | 5 | 4 | 6 | 0 | 0 | 0 |  |
| onatski | 3 | 2 | 1 | 3 | 0.255 | 0 | 0 | yes |
| parallel | 1 | 1 | 1 | 1 | 0.995 | 0 | 0 |  |
| variance_share | 20 | 17 | 14 | 20 | 0.125 | 0 | 0 |  |

## Bootstrap distribution of k

Stationary bootstrap of rows (Politis & Romano 1994): 200 draws, mean block 4.481, distinct rows only (on average 57.49 of 90 rows per draw).

| rule | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ahn_horenstein | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| bai_ng | 0.00 | 0.68 | 0.30 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gavish_donoho | 0.00 | 0.00 | 0.01 | 0.05 | 0.08 | 0.15 | 0.30 | 0.18 | 0.20 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| kaiser | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.03 | 0.15 | 0.36 | 0.12 | 0.10 | 0.12 | 0.10 | 0.03 | 0.00 | 0.00 | 0.00 |
| mp | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| mp_market | 0.00 | 0.00 | 0.00 | 0.04 | 0.27 | 0.56 | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| onatski | 0.00 | 0.42 | 0.30 | 0.26 | 0.00 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| parallel | 0.00 | 0.99 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| variance_share | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.05 | 0.17 | 0.21 | 0.20 | 0.14 | 0.07 | 0.12 |
