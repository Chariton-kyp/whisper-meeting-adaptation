"""Qualitative error analysis of one evaluated system (results/<run>/hyps.jsonl).

Reports the most frequent substitutions, deletions and insertions on normalized text, the outputs
produced for utterances whose reference is empty after normalization (a hallucination proxy), and
the share of errors that disappear under the disfluency-insensitive rule. With --compare, it also
lists the utterances that changed most between two systems.

Usage:
    uv run python scripts/error_analysis.py results/zeroshot_sdm_test
    uv run python scripts/error_analysis.py results/both_sdm_test --compare results/zeroshot_sdm_test
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jiwer
import pandas as pd

from meeting_asr.text import FILLERS


def alignment_counts(refs: list[str], hyps: list[str]) -> tuple[Counter, Counter, Counter]:
    subs, dels, ins = Counter(), Counter(), Counter()
    for r, h in zip(refs, hyps):
        rt, ht = r.split(), h.split()
        if not rt:
            ins.update(ht)
            continue
        if not ht:
            dels.update(rt)
            continue
        out = jiwer.process_words(r, h)
        for chunk in out.alignments[0]:
            if chunk.type == "substitute":
                subs.update(f"{a} -> {b}" for a, b in zip(rt[chunk.ref_start_idx:chunk.ref_end_idx],
                                                           ht[chunk.hyp_start_idx:chunk.hyp_end_idx]))
            elif chunk.type == "delete":
                dels.update(rt[chunk.ref_start_idx:chunk.ref_end_idx])
            elif chunk.type == "insert":
                ins.update(ht[chunk.hyp_start_idx:chunk.hyp_end_idx])
    return subs, dels, ins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--compare", type=Path, help="baseline run on the same utterances")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_json(args.run / "hyps.jsonl", lines=True, dtype={"audio_id": str})
    df["ref_norm"] = df["ref_norm"].fillna("")
    df["hyp_norm"] = df["hyp_norm"].fillna("")
    subs, dels, ins = alignment_counts(df["ref_norm"].tolist(), df["hyp_norm"].tolist())
    n_ref = int(df["n_ref"].sum())
    filler_dels = sum(c for w, c in dels.items() if w in FILLERS)
    # Repetitions: deleted reference tokens that repeat the previous reference token.
    repeat_dels = 0
    for r, h in zip(df["ref_norm"], df["hyp_norm"]):
        rt = r.split()
        if len(rt) > 1 and h:
            for chunk in jiwer.process_words(r, h).alignments[0]:
                if chunk.type == "delete":
                    repeat_dels += sum(1 for i in range(chunk.ref_start_idx, chunk.ref_end_idx)
                                       if i > 0 and rt[i] == rt[i - 1])
    empty = df[df["n_ref"] == 0]
    hall = Counter(empty.loc[empty["hyp_norm"] != "", "hyp_raw"].str.strip().str.lower())

    report = {
        "run": args.run.name,
        "n_ref_words": n_ref,
        "top_substitutions": subs.most_common(args.top),
        "top_deletions": dels.most_common(args.top),
        "top_insertions": ins.most_common(args.top),
        "filler_deletions": filler_dels,
        "repeated_word_deletions": repeat_dels,
        "deletions_total": int(df["del"].sum()),
        "empty_ref_utterances": int(len(empty)),
        "empty_ref_with_output": int((empty["hyp_norm"] != "").sum()),
        "top_outputs_on_empty_refs": hall.most_common(args.top),
    }

    if args.compare:
        base = pd.read_json(args.compare / "hyps.jsonl", lines=True, dtype={"audio_id": str}).set_index("audio_id")
        cur = df.set_index("audio_id")
        err_cur = cur["sub"] + cur["del"] + cur["ins"]
        err_base = (base["sub"] + base["del"] + base["ins"]).reindex(err_cur.index)
        diff = (err_cur - err_base).sort_values()
        pick = lambda ids: [{"audio_id": i, "ref": cur.at[i, "ref_norm"], "base": base.at[i, "hyp_norm"],
                             "new": cur.at[i, "hyp_norm"], "delta_errors": int(diff[i])} for i in ids]
        report["most_improved"] = pick(diff.index[: args.top // 2])
        report["most_degraded"] = pick(diff.index[::-1][: args.top // 2])
        report["utterances_improved"] = int((diff < 0).sum())
        report["utterances_degraded"] = int((diff > 0).sum())

    dest = Path("results/summary")
    dest.mkdir(parents=True, exist_ok=True)
    name = args.run.name + (f"_vs_{args.compare.name}" if args.compare else "")
    (dest / f"errors_{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("most_")}, indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
