"""POST HOC, EXPLORATORY (not pre-registered): how much of the WER gain is AMI spelling conventions?

AMI references follow transcription conventions that the Whisper normalizer leaves alone
("okay" vs "ok", "cause" vs "because", "dunno", "gonna", letters spelled with spaces such as "l c d").
An adapted model learns these conventions, and that lowers WER without better recognition. The script
re-scores stored hypotheses after a convention map applied identically to references and hypotheses,
on top of the pre-registered disfluency-insensitive rule, and reports the paired deltas against the
reference system. It is labelled post hoc everywhere it is reported (docs/design.md, amendment 3).

Usage:
    uv run python scripts/conventions_posthoc.py --set sdm_test --systems zeroshot both encoder decoder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from meeting_asr.metrics import bootstrap_error_rate, error_counts, paired_bootstrap_delta
from meeting_asr.text import disfluency_insensitive

# Applied after Whisper normalization; keys and values are normalized-text tokens or phrases.
CONVENTIONS = {
    "ok": "okay", "cause": "because", "cos": "because", "dunno": "do not know", "gonna": "going to",
    "wanna": "want to", "gotta": "got to", "kinda": "kind of", "sorta": "sort of", "alright": "all right",
    "yeah": "yes", "yep": "yes", "yup": "yes", "nope": "no",
}


def join_spelled_letters(tokens: list[str]) -> list[str]:
    """Join runs of 2+ single letters ("l c d" -> "lcd"). The words "a" and "i" are left alone."""
    out, run = [], []
    for t in tokens + [""]:
        if len(t) == 1 and t.isalpha() and t not in ("a", "i"):
            run.append(t)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        if t:
            out.append(t)
    return out


def convention_insensitive(normalized: str) -> str:
    mapped = " ".join(CONVENTIONS.get(t, t) for t in normalized.split())
    return " ".join(join_spelled_letters(disfluency_insensitive(mapped).split()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="e.g. sdm_test")
    ap.add_argument("--systems", nargs="+", required=True)
    args = ap.parse_args()

    runs = {}
    for s in args.systems:
        df = pd.read_json(Path("results") / f"{s}_{args.set}" / "hyps.jsonl", lines=True,
                          dtype={"audio_id": str, "meeting_id": str}).sort_values("audio_id").reset_index(drop=True)
        runs[s] = df
    ref = args.systems[0]
    clusters = runs[ref]["meeting_id"]
    counts = {}
    for s, df in runs.items():
        counts[s] = {
            "standard": df[["sub", "del", "ins", "hit", "n_ref"]],
            "conventions": error_counts(df["ref_norm"].fillna("").map(convention_insensitive).tolist(),
                                        df["hyp_norm"].fillna("").map(convention_insensitive).tolist()),
        }
    out = {"note": "POST HOC, exploratory; not pre-registered", "set": args.set, "reference": ref, "systems": {}}
    lines = [f"# Post hoc: convention-insensitive WER ({args.set})", "",
             "_Exploratory, not pre-registered. The convention map and the spelled-letter joining are applied "
             "identically to references and hypotheses, on top of the disfluency-insensitive rule._", "",
             "| System | WER (standard) | WER (convention-insensitive) | Δ vs ref, standard | Δ vs ref, convention-insensitive |",
             "|---|---|---|---|---|"]
    for s in args.systems:
        rec = {k: bootstrap_error_rate(c, clusters).as_dict() for k, c in counts[s].items()}
        cells = [f"`{s}`"] + [f"{100 * rec[k]['value']:.2f} [{100 * rec[k]['ci_lo']:.2f}, {100 * rec[k]['ci_hi']:.2f}]"
                              for k in ("standard", "conventions")]
        if s != ref:
            for k in ("standard", "conventions"):
                d = paired_bootstrap_delta(counts[ref][k], counts[s][k], clusters)
                rec[f"delta_{k}"] = d
                cells.append(f"{100 * d['delta']:+.2f} [{100 * d['ci_lo']:+.2f}, {100 * d['ci_hi']:+.2f}]")
        else:
            cells += ["—", "—"]
        out["systems"][s] = rec
        lines.append("| " + " | ".join(cells) + " |")
    dest = Path("results/summary")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"posthoc_conventions_{args.set}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dest / f"posthoc_conventions_{args.set}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
