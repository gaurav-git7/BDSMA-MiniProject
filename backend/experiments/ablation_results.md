# Ablation and tuning results

Mean over all ego-networks, one run per setting (seed 42). Differences of about 0.01-0.02 are within run-to-run noise and should not be over-interpreted.

## Reference and ablation (one factor changed from default)

| Setting | ONMI | Avg F1 | Omega | Bridge F1 | Circles | Train s |
|---|---|---|---|---|---|---|
| Louvain (disjoint) | 0.3394 | 0.3843 | 0.3217 | 0.1 | 15.8 | 0.07 |
| GNN without seeds | 0.291 | 0.3963 | 0.2256 | 0.2911 | 6.4 | 1.4 |
| default | 0.2459 | 0.3923 | 0.2075 | 0.1574 | 5.3 | 1.23 |
| weak-clique seeds | 0.3195 | 0.4346 | 0.2942 | 0.1877 | 5.6 | 1.25 |
| k = 3 | 0.2602 | 0.391 | 0.2092 | 0.0644 | 5.0 | 1.3 |
| k = 5 | 0.2798 | 0.399 | 0.2465 | 0.191 | 5.0 | 1.51 |
| soft modularity loss | 0.1274 | 0.3658 | 0.0743 | 0.1704 | 2.2 | 0.8 |
| lambda_seed = 0 (no seed loss) | 0.2927 | 0.414 | 0.2249 | 0.3176 | 6.2 | 1.57 |
| lambda_seed = 0.5 | 0.2617 | 0.3831 | 0.1915 | 0.1805 | 5.2 | 1.6 |
| lambda_seed = 2 | 0.2527 | 0.3884 | 0.2099 | 0.1765 | 5.0 | 1.62 |
| lambda_seed = 5 | 0.2174 | 0.3744 | 0.1868 | 0.2124 | 5.2 | 1.63 |
| epochs = 50 | 0.23 | 0.3649 | 0.163 | 0.1434 | 4.2 | 0.63 |
| epochs = 300 | 0.2528 | 0.3887 | 0.2147 | 0.203 | 5.9 | 3.52 |
| GAT encoder (N <= 800) | 0.3301 | 0.4609 | 0.2557 | 0.2751 | 6.7 | 5.52 |

## Threshold tau (default model)

| Setting | ONMI | Avg F1 | Omega | Bridge F1 | Circles | Train s |
|---|---|---|---|---|---|---|
| tau = 0.1 | 0.2407 | 0.3987 | 0.1611 | 0.2827 | 6.2 | 1.73 |
| tau = 0.2 | 0.2449 | 0.3959 | 0.1793 | 0.2349 | 6.1 | 1.73 |
| tau = 0.3 | 0.2474 | 0.3945 | 0.1922 | 0.2077 | 5.8 | 1.73 |
| tau = 0.5 | 0.2459 | 0.3923 | 0.2075 | 0.1574 | 5.3 | 1.73 |
| tau = 0.7 | 0.2474 | 0.3923 | 0.215 | 0.1394 | 5.2 | 1.73 |
| tau = 1.0 | 0.2431 | 0.3842 | 0.2214 | 0.1227 | 5.0 | 1.73 |

## Top 10 grid-search settings (ranked by ONMI + F1 + Omega)

| Setting | ONMI | Avg F1 | Omega | Bridge F1 | Circles | Train s |
|---|---|---|---|---|---|---|
| weak_clique k=4 lam=1.0 ep=300 tau=0.5 | 0.3335 | 0.4305 | 0.3005 | 0.1728 | 6.5 | 3.51 |
| weak_clique k=4 lam=0.5 ep=300 tau=0.7 | 0.323 | 0.4308 | 0.3089 | 0.1001 | 7.1 | 3.58 |
| weak_clique k=4 lam=1.0 ep=300 tau=0.7 | 0.3271 | 0.428 | 0.3056 | 0.1324 | 6.5 | 3.51 |
| weak_clique k=4 lam=0.5 ep=150 tau=0.7 | 0.3165 | 0.4312 | 0.3054 | 0.0834 | 6.3 | 1.77 |
| weak_clique k=4 lam=1.0 ep=150 tau=0.5 | 0.3195 | 0.4346 | 0.2942 | 0.1877 | 5.6 | 1.78 |
| weak_clique k=4 lam=0.5 ep=150 tau=0.5 | 0.3166 | 0.4314 | 0.2975 | 0.1147 | 6.5 | 1.77 |
| weak_clique k=4 lam=0.5 ep=300 tau=0.5 | 0.3206 | 0.4256 | 0.2978 | 0.1298 | 7.3 | 3.58 |
| weak_clique k=4 lam=2.0 ep=300 tau=0.7 | 0.3154 | 0.4228 | 0.3017 | 0.0994 | 6.2 | 3.5 |
| weak_clique k=4 lam=1.0 ep=300 tau=0.3 | 0.3277 | 0.4204 | 0.2903 | 0.2205 | 6.7 | 3.51 |
| weak_clique k=4 lam=1.0 ep=150 tau=0.7 | 0.3176 | 0.4258 | 0.2948 | 0.125 | 5.7 | 1.78 |

Total runtime: 903 s
