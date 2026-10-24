"""Far-field penalty per system and its change relative to a reference system (H3 in docs/design.md).

IHM and SDM test contain the same segments, but their audio_ids differ, so runs are paired on
meeting|speaker|begin|end. For each system, gap = WER(SDM) - WER(IHM) on identical references. For
each system X versus the reference Z, the difference-in-differences is
(SDM_X - IHM_X) - (SDM_Z - IHM_Z). Both come with a meeting-level paired bootstrap.

Usage:
    uv run python scripts/mic_gap.py --ref zeroshot --systems both encoder decoder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from meeting_asr.data import segment_key

N_BOOT = 2000


def load_pair(system: str) -> pd.DataFrame:
    parts = {}
    for mic in ("ihm", "sdm"):
        df = pd.read_json(Path("results") / f"{system}_{mic}_test" / "hyps.jsonl", lines=True,
                          dtype={"audio_id": str, "meeting_id": str, "speaker_id": str})
        df["key"] = segment_key(df)
        df["err"] = df["sub"] + df["del"] + df["ins"]
        parts[mic] = df.set_index("key")[["meeting_id", "err", "n_ref"]]
    j = parts["ihm"].join(parts["sdm"], lsuffix="_ihm", rsuffix="_sdm", how="inner")
    if len(j) != len(parts["ihm"]) or not (j["n_ref_ihm"] == j["n_ref_sdm"]).all():
        raise ValueError(f"{system}: IHM and SDM runs do not pair one-to-one on identical references")
    return j


def meeting_sums(j: pd.DataFrame) -> pd.DataFrame:
    return j.groupby("meeting_id_ihm")[["err_ihm", "err_sdm", "n_ref_ihm"]].sum().sort_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="zeroshot")
    ap.add_argument("--systems", nargs="+", required=True)
    args = ap.parse_args()

    names = [args.ref, *args.systems]
    sums = {s: meeting_sums(load_pair(s)) for s in names}
    meetings = sums[args.ref].index
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(meetings), size=(N_BOOT, len(meetings)))

    def gap_samples(s: str) -> tuple[float, np.ndarray]:
        m = sums[s].loc[meetings]
        e_i, e_s, n = (m[c].to_numpy(dtype=float) for c in ("err_ihm", "err_sdm", "n_ref_ihm"))
        point = (e_s.sum() - e_i.sum()) / n.sum()
        boot = (e_s[idx].sum(1) - e_i[idx].sum(1)) / n[idx].sum(1)
        return point, boot

    out, lines = {}, ["# Far-field penalty (SDM − IHM WER, same segments)", "",
                      "| System | WER IHM | WER SDM | Gap [95% CI] | Gap change vs ref [95% CI] | p(gap not smaller) |",
                      "|---|---|---|---|---|---|"]
    ref_point, ref_boot = gap_samples(args.ref)
    for s in names:
        m = sums[s]
        wi, ws = m["err_ihm"].sum() / m["n_ref_ihm"].sum(), m["err_sdm"].sum() / m["n_ref_ihm"].sum()
        point, boot = gap_samples(s)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rec = {"wer_ihm": wi, "wer_sdm": ws, "gap": point, "gap_ci": [float(lo), float(hi)]}
        cells = [f"`{s}`", f"{100 * wi:.2f}", f"{100 * ws:.2f}", f"{100 * point:.2f} [{100 * lo:.2f}, {100 * hi:.2f}]"]
        if s != args.ref:
            d = boot - ref_boot
            dlo, dhi = np.quantile(d, [0.025, 0.975])
            rec.update({"gap_change": point - ref_point, "gap_change_ci": [float(dlo), float(dhi)],
                        "p_gap_not_smaller": float((d >= 0).mean())})
            cells += [f"{100 * (point - ref_point):+.2f} [{100 * dlo:+.2f}, {100 * dhi:+.2f}]",
                      f"{rec['p_gap_not_smaller']:.3f}"]
        else:
            cells += ["—", "—"]
        out[s] = rec
        lines.append("| " + " | ".join(cells) + " |")

    dest = Path("results/summary")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "mic_gap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dest / "mic_gap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
