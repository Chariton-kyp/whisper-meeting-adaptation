# Post hoc: repetition loops (ihm_test)

_Exploratory, not pre-registered. A loop is ≥5 identical tokens or bigrams in a row in the normalized hypothesis; output length is capped by `max_new_tokens=128`._

| System | Looping utterances | Insertions in them | Share of all errors |
|---|---|---|---|
| `zeroshot` | 2 | 24 | 0.2% |
| `both` | 5 | 198 | 2.3% |
| `encoder` | 4 | 203 | 2.4% |
| `decoder` | 11 | 794 | 7.7% |
