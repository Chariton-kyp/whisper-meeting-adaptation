"""Compare evaluated systems on the same utterances: paired deltas and the pre-registered slices.

Reads results/<run>/hyps.jsonl. The first run given is the reference for paired comparisons. Writes
a markdown table and a JSON file. README tables are generated from these outputs, never typed.

Usage:
    uv run python scripts/analyze.py --name sdm_test --runs results/turbo_sdm_test results/lboth_sdm_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from meeting_asr.metrics import bootstrap_error_rate, paired_bootstrap_delta
from meeting_asr.overlap import bucketize

N_BOOT = 2000
DURATION = ([1.0, 3.0], ["<1s", "1-3s", ">3s"])
OVERLAP = ([0.0, 0.3], ["none", "(0,0.3]", ">0.3"])


def load(run: Path) -> pd.DataFrame:
    df = pd.read_json(run / "hyps.jsonl", lines=True, dtype={"audio_id": str, "meeting_id": str})
    return df.sort_values("audio_id").reset_index(drop=True)


def counts(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "wer":
        return df[["sub", "del", "ins", "hit", "n_ref"]]
    err, n = {"dis": ("d_err", "d_n_ref"), "ortho": ("o_err", "o_n_ref")}[kind]
    return pd.DataFrame({"sub": df[err], "del": 0, "ins": 0, "hit": 0, "n_ref": df[n]})


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def est(df: pd.DataFrame, kind: str) -> dict:
    return bootstrap_error_rate(counts(df, kind), df["meeting_id"], n_boot=N_BOOT).as_dict()


def fmt(e: dict) -> str:
    return f"{pct(e['value'])} [{pct(e['ci_lo'])}, {pct(e['ci_hi'])}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--orthographic", action="store_true", help="references are cased and punctuated")
    args = ap.parse_args()

    runs = {r.name: load(r) for r in args.runs}
    ref_name = args.runs[0].name
    ids = runs[ref_name]["audio_id"]
    for name, df in runs.items():
        if not df["audio_id"].equals(ids):
            raise ValueError(f"{name} was not evaluated on the same utterances as {ref_name}")

    kinds = ["wer", "dis"] + (["ortho"] if args.orthographic else [])
    out: dict = {"reference": ref_name, "n_utterances": len(ids), "systems": {}}
    lines = [f"# {args.name}", "", f"Reference for paired deltas: `{ref_name}`. "
             f"{len(ids)} utterances, {runs[ref_name]['meeting_id'].nunique()} clusters, "
             f"95% meeting-level bootstrap CIs ({N_BOOT} resamples).", ""]
    header = ["System", "WER", "Disfluency-insens. WER"] + (["Orthographic WER"] if args.orthographic else [])
    header += ["ΔWER vs ref [CI]", "p(not better)", "Ins. on empty refs"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    base = runs[ref_name]
    for name, df in runs.items():
        sys_out = {k: est(df, k) for k in kinds}
        sys_out["sub_del_ins_rates"] = {c: float(df[c].sum() / df["n_ref"].sum()) for c in ("sub", "del", "ins")}
        sys_out["ins_on_empty_refs"] = int(df.loc[df["n_ref"] == 0, "ins"].sum())
        row = [f"`{name}`"] + [fmt(sys_out[k]) for k in kinds]
        if name != ref_name:
            for k in kinds:
                sys_out[f"delta_{k}"] = paired_bootstrap_delta(counts(base, k), counts(df, k), base["meeting_id"],
                                                               n_boot=N_BOOT)
            d = sys_out["delta_wer"]
            row += [f"{100 * d['delta']:+.2f} [{100 * d['ci_lo']:+.2f}, {100 * d['ci_hi']:+.2f}]",
                    f"{d['p_b_not_better']:.3f}"]
        else:
            row += ["—", "—"]
        row.append(str(sys_out["ins_on_empty_refs"]))
        lines.append("| " + " | ".join(row) + " |")

        slices = {}
        for label, col, (edges, names) in (("duration", "duration", DURATION), ("overlap", "overlap_ratio", OVERLAP)):
            if col not in df or df[col].isna().all():
                continue
            buckets = bucketize(df[col], edges, names)
            slices[label] = {}
            for b in names:
                part = df[buckets == b]
                if len(part) and part["n_ref"].sum() > 0:
                    slices[label][b] = {"n": int(len(part)), "wer": est(part, "wer"),
                                        "ins_rate": float(part["ins"].sum() / part["n_ref"].sum())}
        sys_out["slices"] = slices
        out["systems"][name] = sys_out

    for label, (_, names) in (("duration", DURATION), ("overlap", OVERLAP)):
        if not all(label in s["slices"] for s in out["systems"].values()):
            continue
        lines += ["", f"## WER by {label}", "", "| System | " + " | ".join(names) + " |",
                  "|" + "---|" * (len(names) + 1)]
        for name, s in out["systems"].items():
            cells = [f"{fmt(s['slices'][label][b]['wer'])} (n={s['slices'][label][b]['n']})"
                     if b in s["slices"][label] else "—" for b in names]
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    dest = Path("results/summary")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dest / f"{args.name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
