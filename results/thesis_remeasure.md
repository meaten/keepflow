## Main table: Sinkhorn (Fork / Branch)

paper | old GT now | exact GT now

| Method | Fork | Branch |
|---|---|---|
| Social-GAN | 1.46 \| 1.201 \| 1.187 | 1.28 \| 1.163 \| 1.079 |
| Trajectron++ | 2.02 \| 2.062 \| 1.968 | 1.67 \| 1.738 \| 1.701 |
| MID | 1.80 \| 1.803 \| 1.721 | 0.97 \| 0.965 \| 0.892 |
| SingularTrajectory | 5.52 \| 5.321 \| 5.341 | 5.04 \| 4.716 \| 4.571 |
| MoFlow | 3.65 \| 3.732 \| 3.804 | 3.08 \| 3.129 \| 3.160 |
| FlowChain-NF | 1.62 \| 1.783 \| 1.718 | 1.44 \| 1.655 \| 1.530 |
| FlowChain-CNF (Pred at t) | 1.40 \| 1.450 \| 1.408 | 1.19 \| 1.239 \| 1.266 |
| CNF Update from t+1 | 1.03 \| 1.101 \| 0.944 | 1.43 \| 1.405 \| 1.133 |
| CNF Update from t+3 | 0.62 \| 0.649 \| 0.587 | 1.07 \| 1.101 \| 1.056 |
| CNF Update from t+5 | 0.61 \| 0.625 \| 0.629 | 1.17 \| 1.220 \| 1.089 |

### Ranking of the seven Pred-at-t rows (1 = best)

- simfork paper: FlowChain-CNF (Pred at t) (1.40) < Social-GAN (1.46) < FlowChain-NF (1.62) < MID (1.80) < Trajectron++ (2.02) < MoFlow (3.65) < SingularTrajectory (5.52)
- simfork old  : Social-GAN (1.20) < FlowChain-CNF (Pred at t) (1.45) < FlowChain-NF (1.78) < MID (1.80) < Trajectron++ (2.06) < MoFlow (3.73) < SingularTrajectory (5.32)
- simfork exact: Social-GAN (1.19) < FlowChain-CNF (Pred at t) (1.41) < FlowChain-NF (1.72) < MID (1.72) < Trajectron++ (1.97) < MoFlow (3.80) < SingularTrajectory (5.34)
- simbranch paper: MID (0.97) < FlowChain-CNF (Pred at t) (1.19) < Social-GAN (1.28) < FlowChain-NF (1.44) < Trajectron++ (1.67) < MoFlow (3.08) < SingularTrajectory (5.04)
- simbranch old  : MID (0.97) < Social-GAN (1.16) < FlowChain-CNF (Pred at t) (1.24) < FlowChain-NF (1.66) < Trajectron++ (1.74) < MoFlow (3.13) < SingularTrajectory (4.72)
- simbranch exact: MID (0.89) < Social-GAN (1.08) < FlowChain-CNF (Pred at t) (1.27) < FlowChain-NF (1.53) < Trajectron++ (1.70) < MoFlow (3.16) < SingularTrajectory (4.57)

## ETH log-probability, dopri5 (paper column was RK4)

| Row | paper (RK4) | dopri5 |
|---|---|---|
| Pred at t | -0.741 ± 0.104 | -0.770 ± 0.099 |
| Update from t+1 | -0.709 ± 0.180 | -0.711 ± 0.167 |
| Update from t+3 | -1.141 ± 0.194 | -1.178 ± 0.215 |
| Update from t+5 | -2.292 ± 0.539 | -2.338 ± 0.569 |

Ablation, ETH log-probability

| Arm | paper (RK4) | dopri5 |
|---|---|---|
| full | -0.741 ± 0.104 | -0.770 ± 0.099 |
| w/o CRF | -0.799 ± 0.214 | -0.808 ± 0.229 |
| w/o CIF | -0.836 ± 0.257 | -0.847 ± 0.278 |

## Ablation: Sinkhorn (mean ± std over 3 seeds)

paper | old GT now | exact GT now

| Arm | Fork | Branch |
|---|---|---|
| full | 1.387 ± 0.084 \| 1.387 ± 0.084 \| 1.375 ± 0.090 | 1.279 ± 0.056 \| 1.279 ± 0.056 \| 1.268 ± 0.036 |
| w/o CRF | 1.464 ± 0.156 \| 1.464 ± 0.156 \| 1.469 ± 0.145 | 1.333 ± 0.051 \| 1.333 ± 0.051 \| 1.396 ± 0.030 |
| w/o CIF | 1.253 ± 0.063 \| 1.253 ± 0.063 \| 1.242 ± 0.068 | 1.344 ± 0.074 \| 1.344 ± 0.074 \| 1.371 ± 0.018 |

## Solver table

| Solver | Fork: paper \| old GT now \| exact GT now | ETH log-prob: paper \| now |
|---|---|---|
| Dormand-Prince 5 | 1.40 \| 1.450 \| 1.408 | 0.93 \| 0.95 |
| Tsitouras 5 | 1.44 \| 1.445 \| 1.406 | 0.97 \| n/a |
| Euler 1 | 2.00 \| 2.015 \| 1.927 | 0.81 \| n/a |
| Euler 10 | 1.51 \| 1.538 \| 1.490 | 0.91 \| n/a |
| Euler 100 | 1.43 \| 1.458 \| 1.419 | 0.95 \| n/a |
| RK4 1 | 1.77 \| 1.788 \| 1.618 | 0.63 \| n/a |
| RK4 10 | 1.35 \| 1.383 \| 1.332 | 0.97 \| 0.88 |
| RK4 100 | 1.40 \| 1.421 \| 1.381 | 0.19 \| 0.95 |

