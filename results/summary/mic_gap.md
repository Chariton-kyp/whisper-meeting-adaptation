# Far-field penalty (SDM − IHM WER, same segments)

| System | WER IHM | WER SDM | Gap [95% CI] | Gap change vs ref [95% CI] | p(gap not smaller) |
|---|---|---|---|---|---|
| `zeroshot` | 16.03 | 36.25 | 20.22 [15.97, 24.01] | — | — |
| `both` | 9.68 | 22.38 | 12.70 [9.92, 15.29] | -7.52 [-9.28, -5.81] | 0.000 |
| `encoder` | 9.60 | 22.39 | 12.79 [9.89, 15.43] | -7.43 [-8.96, -5.88] | 0.000 |
| `decoder` | 11.47 | 27.59 | 16.12 [12.69, 19.23] | -4.10 [-5.68, -2.64] | 0.000 |
