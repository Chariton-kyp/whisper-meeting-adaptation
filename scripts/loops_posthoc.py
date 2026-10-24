"""POST HOC, EXPLORATORY (not pre-registered): degenerate repetition loops in hypotheses.

A hypothesis "loops" if its normalized text contains at least K identical tokens in a row, or at
least K repetitions in a row of the same bigram. For each system the script reports how many
utterances loop, how many insertions they contribute, and what share of all errors that is. The
damage is bounded by max_new_tokens in the decoding config.

Usage:
    uv run python scripts/loops_posthoc.py --set sdm_test --systems zeroshot both encoder decoder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

K = 5


def loops(text: str, k: int = K) -> bool:
    t = text.split()
    run = 1
    for i in range(1, len(t)):
        run = run + 1 if t[i] == t[i - 1] else 1
        if run >= k:
            return True
    return any(all(t[i + 2 * j:i + 2 * j + 2] == t[i:i + 2] for j in range(k)) for i in range(len(t) - 2 * k + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    args = ap.parse_args()

    out = {"note": f"POST HOC, exploratory; loop = >= {K} identical tokens or bigrams in a row", "systems": {}}
    lines = [f"# Post hoc: repetition loops ({args.set})", "",
             f"_Exploratory, not pre-registered. A loop is ≥{K} identical tokens or bigrams in a row in the "
             "normalized hypothesis; output length is capped by `max_new_tokens=128`._", "",
             "| System | Looping utterances | Insertions in them | Share of all errors |", "|---|---|---|---|"]
    for s in args.systems:
        df = pd.read_json(Path("results") / f"{s}_{args.set}" / "hyps.jsonl", lines=True)
        mask = df["hyp_norm"].fillna("").map(loops)
        ins = int(df.loc[mask, "ins"].sum())
        errors = int((df["sub"] + df["del"] + df["ins"]).sum())
        rec = {"looping_utterances": int(mask.sum()), "insertions_in_loops": ins, "share_of_errors": ins / errors,
               "examples": df.loc[mask, "hyp_norm"].str.slice(0, 120).head(3).tolist()}
        out["systems"][s] = rec
        lines.append(f"| `{s}` | {rec['looping_utterances']} | {ins} | {100 * rec['share_of_errors']:.1f}% |")
    dest = Path("results/summary")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"posthoc_loops_{args.set}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dest / f"posthoc_loops_{args.set}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
