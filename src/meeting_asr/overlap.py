"""Per-segment speaker overlap, computed from the segment timings of a whole split.

overlap_ratio is the share of a segment's duration during which at least one other speaker in the
same meeting also has a segment. It approximates overlapped speech: it relies on the corpus
segmentation, and very short backchannels may be missing from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _union_length(intervals: list[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def overlap_ratio(manifest: pd.DataFrame) -> pd.Series:
    """Return overlap_ratio per row (index aligned with `manifest`)."""
    out = pd.Series(0.0, index=manifest.index)
    for _, m in manifest.groupby("meeting_id"):
        begins, ends = m["begin_time"].to_numpy(), m["end_time"].to_numpy()
        speakers = m["speaker_id"].to_numpy()
        for i, idx in enumerate(m.index):
            b, e = begins[i], ends[i]
            if e <= b:
                continue
            mask = (speakers != speakers[i]) & (begins < e) & (ends > b)
            clipped = [(max(b, s), min(e, t)) for s, t in zip(begins[mask], ends[mask])]
            out[idx] = _union_length(clipped) / (e - b)
    return out


def bucketize(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(values, bins=[-np.inf, *edges, np.inf], labels=labels, right=True)
