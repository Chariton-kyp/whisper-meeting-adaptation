# Post hoc: repetition loops (sdm_test)

_Exploratory, not pre-registered. A loop is ≥5 identical tokens or bigrams in a row in the normalized hypothesis; output length is capped by `max_new_tokens=128`._

| System | Looping utterances | Insertions in them | Share of all errors |
|---|---|---|---|
| `zeroshot` | 1 | 60 | 0.2% |
| `both` | 6 | 123 | 0.6% |
| `encoder` | 9 | 130 | 0.6% |
| `decoder` | 30 | 1250 | 5.1% |
