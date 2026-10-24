"""Re-score existing runs from their stored raw hypotheses with the current scoring code (no GPU).

Rewrites hyps.jsonl and metrics.json in place, keeps the original timing block, and records the
scoring commit in run.json. Every reported number is then computed by the same scorer.

Usage:
    uv run python scripts/rescore.py results/smallen_ihm_test results/smallen_sdm_test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import score  # noqa: E402

from meeting_asr import run_meta  # noqa: E402

DROP = ["ref_norm", "hyp_norm", "sub", "del", "ins", "hit", "n_ref", "d_err", "d_n_ref", "o_err", "o_n_ref"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    for run in args.runs:
        old = pd.read_json(run / "hyps.jsonl", lines=True, dtype={"audio_id": str, "meeting_id": str})
        hyps = dict(zip(old["audio_id"], old["hyp_raw"].fillna("")))
        manifest = old.drop(columns=[c for c in DROP + ["hyp_raw"] if c in old.columns])
        df, metrics = score(manifest, hyps, args.n_boot)
        prev = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        metrics["timing"] = prev.get("timing")
        df.to_json(run / "hyps.jsonl", orient="records", lines=True, force_ascii=False)
        (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
        meta.setdefault("rescored", []).append(run_meta.collect()["git_commit"])
        (run / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        w = metrics["wer"]
        print(f"{run.name}: WER {w['value']:.4f} [{w['ci_lo']:.4f}, {w['ci_hi']:.4f}] (was {prev['wer']['value']:.4f})")


if __name__ == "__main__":
    main()
