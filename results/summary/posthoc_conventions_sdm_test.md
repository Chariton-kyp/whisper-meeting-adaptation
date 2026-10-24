# Post hoc: convention-insensitive WER (sdm_test)

_Exploratory, not pre-registered. The convention map and the spelled-letter joining are applied identically to references and hypotheses, on top of the disfluency-insensitive rule._

| System | WER (standard) | WER (convention-insensitive) | Δ vs ref, standard | Δ vs ref, convention-insensitive |
|---|---|---|---|---|
| `zeroshot` | 36.25 [30.28, 41.54] | 34.26 [28.19, 39.65] | — | — |
| `both` | 22.38 [17.98, 26.44] | 21.88 [17.53, 25.85] | -13.87 [-15.35, -12.16] | -12.38 [-13.84, -10.66] |
| `encoder` | 22.39 [17.91, 26.52] | 21.97 [17.47, 26.06] | -13.86 [-15.23, -12.24] | -12.29 [-13.62, -10.70] |
| `decoder` | 27.59 [22.24, 32.71] | 25.74 [20.65, 30.30] | -8.66 [-9.63, -7.47] | -8.52 [-9.66, -7.22] |
