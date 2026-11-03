# kfactors benchmark leaderboard

Simulation config `default`: 111 designs x 20 replications, kmax = 10, seed 0.

## Simulations: does the rule find the true k?

Pooled over all designs and replications (rank by MAE, then exact %).

| rank | rule | exact % | MAE | bias | panels | failed |
|---|---|---|---|---|---|---|
| 1 | onatski | 55.3 | 0.776 | -0.743 | 2220 | 0 |
| 2 | parallel(null=circular) | 56.8 | 0.801 | -0.073 | 2220 | 0 |
| 3 | parallel(devolatilize=true, null=circular) | 59.2 | 0.833 | +0.034 | 2220 | 0 |
| 4 | parallel(null=block) | 57.1 | 0.871 | +0.238 | 2220 | 0 |
| 5 | mp_market | 59.6 | 1.046 | +0.601 | 2220 | 0 |
| 6 | parallel | 56.0 | 1.101 | +0.489 | 2220 | 0 |
| 7 | mp | 43.8 | 1.125 | +0.088 | 2220 | 0 |
| 8 | parallel(devolatilize=true) | 57.4 | 1.140 | +0.622 | 2220 | 0 |
| 9 | bai_ng(criterion=PC_p1) | 40.0 | 1.656 | -0.051 | 2220 | 0 |
| 10 | gavish_donoho | 42.2 | 1.670 | +0.572 | 2220 | 0 |
| 11 | bai_ng(criterion=IC_p1) | 39.1 | 1.770 | -0.473 | 2220 | 0 |
| 12 | bai_ng | 36.4 | 1.903 | -0.606 | 2220 | 0 |
| 13 | ahn_horenstein(stat=GR) | 35.3 | 1.937 | -1.917 | 2220 | 0 |
| 14 | ahn_horenstein | 32.0 | 2.110 | -2.104 | 2220 | 0 |
| 15 | mp(noise=median) | 26.4 | 3.049 | +2.865 | 2220 | 0 |
| 16 | kaiser | 0.0 | 6.156 | +6.156 | 2220 | 0 |
| 17 | variance_share | 0.0 | 6.324 | +6.324 | 2220 | 0 |

### Exact % by design family

| rule | market | null | pervasive | spiked | weak |
|---|---|---|---|---|---|
| onatski | 71.9 | 98.3 | 76.4 | 28.2 | 16.2 |
| parallel(null=circular) | 50.3 | 94.4 | 72.6 | 44.4 | 27.9 |
| parallel(devolatilize=true, null=circular) | 55.3 | 98.3 | 74.9 | 45.7 | 29.2 |
| parallel(null=block) | 50.3 | 94.4 | 73.2 | 44.3 | 29.2 |
| mp_market | 85.0 | 92.8 | 70.7 | 38.8 | 25.8 |
| parallel | 50.3 | 95.6 | 73.6 | 40.7 | 27.9 |
| mp | 49.7 | 92.8 | 68.8 | 15.8 | 7.1 |
| parallel(devolatilize=true) | 58.3 | 98.3 | 70.6 | 42.8 | 30.0 |
| bai_ng(criterion=PC_p1) | 49.7 | 100.0 | 66.0 | 7.4 | 0.0 |
| gavish_donoho | 58.3 | 100.0 | 64.6 | 11.4 | 0.0 |
| bai_ng(criterion=IC_p1) | 48.6 | 100.0 | 70.6 | 0.8 | 0.0 |
| bai_ng | 47.8 | 100.0 | 63.2 | 0.3 | 0.0 |
| ahn_horenstein(stat=GR) | 46.1 | 100.0 | 60.0 | 0.8 | 0.0 |
| ahn_horenstein | 40.6 | 100.0 | 53.1 | 0.3 | 0.0 |
| mp(noise=median) | 17.5 | 91.1 | 0.6 | 33.3 | 48.3 |
| kaiser | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| variance_share | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Exact % by regime

| rule | pervasive | spiked |
|---|---|---|
| onatski | 83.6 | 28.5 |
| parallel(null=circular) | 80.8 | 34.0 |
| parallel(devolatilize=true, null=circular) | 83.0 | 36.7 |
| parallel(null=block) | 81.2 | 34.2 |
| mp_market | 79.3 | 41.0 |
| parallel | 81.7 | 31.7 |
| mp | 77.9 | 11.5 |
| parallel(devolatilize=true) | 80.1 | 36.0 |
| bai_ng(criterion=PC_p1) | 77.2 | 4.6 |
| gavish_donoho | 76.3 | 9.9 |
| bai_ng(criterion=IC_p1) | 79.9 | 0.5 |
| bai_ng | 74.7 | 0.2 |
| ahn_horenstein(stat=GR) | 72.0 | 0.5 |
| ahn_horenstein | 65.6 | 0.2 |
| mp(noise=median) | 15.6 | 36.8 |
| kaiser | 0.0 | 0.0 |
| variance_share | 0.0 | 0.0 |

### Exact % by noise

| rule | ar1-0.5 | ar1-0.9 | garch | gauss | hetero | t3 |
|---|---|---|---|---|---|---|
| onatski | 45.8 | 2.9 | 77.3 | 54.2 | 51.7 | 79.8 |
| parallel(null=circular) | 60.0 | 4.6 | 80.7 | 53.9 | 57.2 | 71.9 |
| parallel(devolatilize=true, null=circular) | 62.9 | 0.8 | 82.0 | 57.4 | 58.1 | 77.9 |
| parallel(null=block) | 58.3 | 2.9 | 82.7 | 54.5 | 58.1 | 72.1 |
| mp_market | 28.8 | 0.0 | 83.3 | 62.1 | 61.7 | 88.6 |
| parallel | 51.2 | 0.0 | 82.3 | 54.1 | 57.8 | 73.3 |
| mp | 53.3 | 11.2 | 57.7 | 37.7 | 41.7 | 58.3 |
| parallel(devolatilize=true) | 36.2 | 0.0 | 83.3 | 59.2 | 59.2 | 79.5 |
| bai_ng(criterion=PC_p1) | 32.5 | 0.0 | 57.0 | 34.7 | 49.7 | 54.8 |
| gavish_donoho | 12.9 | 0.0 | 65.0 | 40.2 | 45.3 | 67.4 |
| bai_ng(criterion=IC_p1) | 46.2 | 0.0 | 56.3 | 34.7 | 35.3 | 55.5 |
| bai_ng | 39.6 | 0.0 | 53.3 | 33.0 | 31.7 | 52.9 |
| ahn_horenstein(stat=GR) | 30.0 | 0.8 | 52.7 | 33.0 | 32.8 | 51.4 |
| ahn_horenstein | 24.2 | 0.4 | 47.3 | 30.9 | 28.1 | 48.6 |
| mp(noise=median) | 0.0 | 0.0 | 41.3 | 38.6 | 18.6 | 33.6 |
| kaiser | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| variance_share | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Real data: what does the choice of k cost?

Oracle k = best fixed k in hindsight; oracle set = smallest set of k covering the bootstrap argmax draws at the configured level. A rule is *in set* when its median k lies in the oracle set; regret = metric(oracle k) - metric(rule).

### All experiments

| rank | rule | experiments | in set % | mean |k - oracle| | mean regret |
|---|---|---|---|---|---|
| 1 | bai_ng | 4 | 75 | 5.25 | 0.179 |
| 2 | gavish_donoho | 4 | 50 | 4.75 | 0.059 |
| 3 | kaiser | 4 | 50 | 13.75 | 0.183 |
| 4 | variance_share | 4 | 50 | 13.75 | 0.191 |
| 5 | mp_market | 4 | 0 | 22.25 | 0.763 |
| 6 | parallel | 4 | 0 | 24.25 | 0.722 |
| 7 | mp | 4 | 0 | 25.25 | 0.741 |
| 8 | onatski | 4 | 0 | 29.75 | 1.610 |
| 9 | ahn_horenstein | 4 | 0 | 30.75 | 1.655 |

### binance_top100 / statarb_w60

**skipped** (kfactors.strategies.statarb:ResidualReversion): strategy kfactors.strategies.statarb:ResidualReversion is not implemented yet (NotImplementedError: Card B: ResidualReversion.weights is not implemented yet (see docs/tasks/card_B_statarb.md)); experiment skipped

### binance_top100 / statarb_w90

**skipped** (kfactors.strategies.statarb:ResidualReversion): strategy kfactors.strategies.statarb:ResidualReversion is not implemented yet (NotImplementedError: Card B: ResidualReversion.weights is not implemented yet (see docs/tasks/card_B_statarb.md)); experiment skipped

### french_100_daily / minvar_w126

**skipped** (kfactors.strategies.minvar:MinVariance): strategy kfactors.strategies.minvar:MinVariance is not implemented yet (NotImplementedError: Card A: MinVariance.weights is not implemented yet (see docs/tasks/card_A_minvar.md)); experiment skipped

### jkp_usa_monthly / tangency_w120

Strategy `kfactors.strategies.sdf:TangencyPCA`, metric `sharpe_net`: oracle k = **33**, oracle set = {18,19,26,32,33,36,40}.

| rule | k_median | k_mean | metric | low | high | dist_to_oracle | in_set | regret |
|---|---|---|---|---|---|---|---|---|
| gavish_donoho | 30.0 | 30.05 | 1.816 | 1.427 | 2.201 | 3.0 | no | 0.053 |
| kaiser | 18.0 | 17.90 | 1.685 | 1.359 | 2.032 | 15.0 | yes | 0.185 |
| variance_share | 18.0 | 17.09 | 1.684 | 1.359 | 2.030 | 15.0 | yes | 0.186 |
| bai_ng | 17.0 | 19.36 | 1.592 | 1.280 | 1.922 | 16.0 | no | 0.278 |
| mp_market | 8.0 | 8.27 | 0.946 | 0.604 | 1.286 | 25.0 | no | 0.923 |
| parallel | 7.0 | 6.65 | 0.998 | 0.678 | 1.321 | 26.0 | no | 0.871 |
| mp | 6.0 | 5.68 | 0.917 | 0.645 | 1.226 | 27.0 | no | 0.952 |
| onatski | 2.0 | 2.61 | 0.242 | 0.002 | 0.496 | 31.0 | no | 1.627 |
| ahn_horenstein | 1.0 | 1.45 | 0.156 | -0.079 | 0.409 | 32.0 | no | 1.713 |

### jkp_usa_monthly / tangency_w120_cost10

Strategy `kfactors.strategies.sdf:TangencyPCA`, metric `sharpe_net`: oracle k = **18**, oracle set = {13,14,15,17,18,19,26,32,33,36}.

| rule | k_median | k_mean | metric | low | high | dist_to_oracle | in_set | regret |
|---|---|---|---|---|---|---|---|---|
| kaiser | 18.0 | 17.90 | 1.525 | 1.197 | 1.874 | 0.0 | yes | 0.070 |
| variance_share | 18.0 | 17.09 | 1.518 | 1.190 | 1.858 | 0.0 | yes | 0.077 |
| bai_ng | 17.0 | 19.36 | 1.395 | 1.074 | 1.720 | 1.0 | yes | 0.200 |
| mp_market | 8.0 | 8.27 | 0.863 | 0.530 | 1.202 | 10.0 | no | 0.732 |
| parallel | 7.0 | 6.65 | 0.909 | 0.589 | 1.239 | 11.0 | no | 0.686 |
| gavish_donoho | 30.0 | 30.05 | 1.498 | 1.102 | 1.888 | 12.0 | no | 0.097 |
| mp | 6.0 | 5.68 | 0.868 | 0.591 | 1.175 | 12.0 | no | 0.728 |
| onatski | 2.0 | 2.61 | 0.188 | -0.056 | 0.434 | 16.0 | no | 1.407 |
| ahn_horenstein | 1.0 | 1.45 | 0.132 | -0.103 | 0.383 | 17.0 | no | 1.463 |

### jkp_usa_monthly / tangency_w240

Strategy `kfactors.strategies.sdf:TangencyPCA`, metric `sharpe_net`: oracle k = **38**, oracle set = {23,27,29,30,31,36,38,40}.

| rule | k_median | k_mean | metric | low | high | dist_to_oracle | in_set | regret |
|---|---|---|---|---|---|---|---|---|
| bai_ng | 36.0 | 33.38 | 1.762 | 1.445 | 2.120 | 2.0 | yes | 0.099 |
| gavish_donoho | 40.0 | 40.00 | 1.823 | 1.478 | 2.190 | 2.0 | yes | 0.038 |
| kaiser | 18.0 | 18.25 | 1.594 | 1.214 | 1.990 | 20.0 | no | 0.267 |
| variance_share | 18.0 | 19.77 | 1.592 | 1.216 | 1.960 | 20.0 | no | 0.269 |
| mp_market | 11.0 | 11.16 | 1.127 | 0.769 | 1.535 | 27.0 | no | 0.734 |
| parallel | 8.0 | 8.52 | 1.149 | 0.812 | 1.508 | 30.0 | no | 0.713 |
| mp | 7.0 | 7.29 | 1.164 | 0.830 | 1.525 | 31.0 | no | 0.698 |
| onatski | 2.0 | 2.85 | 0.096 | -0.289 | 0.583 | 36.0 | no | 1.765 |
| ahn_horenstein | 1.0 | 1.16 | 0.067 | -0.283 | 0.446 | 37.0 | no | 1.794 |

### jkp_usa_monthly / tangency_w240_cost10

Strategy `kfactors.strategies.sdf:TangencyPCA`, metric `sharpe_net`: oracle k = **38**, oracle set = {17,23,28,29,30,31,36,38,40}.

| rule | k_median | k_mean | metric | low | high | dist_to_oracle | in_set | regret |
|---|---|---|---|---|---|---|---|---|
| bai_ng | 36.0 | 33.38 | 1.573 | 1.253 | 1.931 | 2.0 | yes | 0.138 |
| gavish_donoho | 40.0 | 40.00 | 1.662 | 1.319 | 2.034 | 2.0 | yes | 0.049 |
| kaiser | 18.0 | 18.25 | 1.501 | 1.117 | 1.898 | 20.0 | no | 0.211 |
| variance_share | 18.0 | 19.77 | 1.480 | 1.101 | 1.853 | 20.0 | no | 0.231 |
| mp_market | 11.0 | 11.16 | 1.051 | 0.694 | 1.452 | 27.0 | no | 0.661 |
| parallel | 8.0 | 8.52 | 1.094 | 0.758 | 1.445 | 30.0 | no | 0.618 |
| mp | 7.0 | 7.29 | 1.124 | 0.794 | 1.483 | 31.0 | no | 0.588 |
| onatski | 2.0 | 2.85 | 0.072 | -0.306 | 0.559 | 36.0 | no | 1.640 |
| ahn_horenstein | 1.0 | 1.16 | 0.061 | -0.290 | 0.442 | 37.0 | no | 1.651 |
