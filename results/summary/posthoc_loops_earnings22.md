# Post hoc: repetition loops (earnings22)

_Exploratory, not pre-registered. A loop is ≥5 identical tokens or bigrams in a row in the normalized hypothesis; output length is capped by `max_new_tokens=128`._

| System | Looping utterances | Insertions in them | Share of all errors |
|---|---|---|---|
| `zeroshot` | 0 | 0 | 0.0% |
| `both` | 4 | 14 | 0.3% |
| `encoder` | 3 | 8 | 0.1% |
| `decoder` | 14 | 484 | 8.2% |
