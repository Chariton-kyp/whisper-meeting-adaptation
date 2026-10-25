# Far-field penalty (SDM − IHM WER, same segments)

| System | WER IHM | WER SDM | Gap [95% CI] | Gap change vs ref [95% CI] | p(gap not smaller) |
|---|---|---|---|---|---|
| `zeroshot` | 16.03 | 36.25 | 20.22 [15.97, 24.01] | — | — |
| `both` | 9.68 | 22.38 | 12.70 [9.92, 15.29] | -7.52 [-9.28, -5.81] | 0.000 |
| `encoder` | 9.60 | 22.39 | 12.79 [9.89, 15.43] | -7.43 [-8.96, -5.88] | 0.000 |
| `decoder` | 11.47 | 27.59 | 16.12 [12.69, 19.23] | -4.10 [-5.68, -2.64] | 0.000 |

## Post hoc: relative far-field penalty (SDM/IHM WER ratio)

_Not pre-registered. Same segments, so the ratio equals the ratio of total errors._

| System | SDM/IHM ratio [95% CI] | Ratio change vs ref [95% CI] | Relative WER change SDM / IHM |
|---|---|---|---|
| `zeroshot` | 2.26 [2.08, 2.41] | — | — |
| `both` | 2.31 [2.12, 2.54] | +0.05 [-0.15, +0.24] | -38.3% / -39.6% |
| `encoder` | 2.33 [2.15, 2.53] | +0.07 [-0.10, +0.24] | -38.2% / -40.1% |
| `decoder` | 2.40 [2.17, 2.69] | +0.14 [-0.11, +0.39] | -23.9% / -28.4% |
