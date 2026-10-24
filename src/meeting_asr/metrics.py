"""Error counting, corpus-level WER/CER and meeting-level bootstrap confidence intervals.

WER is computed at corpus level, sum(S+D+I) / sum(N), never as a mean of per-utterance WERs.
Utterances from the same meeting share speakers, room and topic, so they are not independent.
Confidence intervals therefore resample whole meetings (cluster bootstrap). Resampling individual
utterances would give intervals that are too narrow.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer
import numpy as np
import pandas as pd

COUNT_COLUMNS = ["sub", "del", "ins", "hit", "n_ref"]


def _counts(ref: str, hyp: str, unit: str) -> tuple[int, int, int, int, int]:
    ref_toks = ref.split() if unit == "word" else list(ref.replace(" ", ""))
    hyp_toks = hyp.split() if unit == "word" else list(hyp.replace(" ", ""))
    if not ref_toks:
        return 0, 0, len(hyp_toks), 0, 0
    if not hyp_toks:
        return 0, len(ref_toks), 0, 0, len(ref_toks)
    out = jiwer.process_words(" ".join(ref_toks), " ".join(hyp_toks))
    return out.substitutions, out.deletions, out.insertions, out.hits, len(ref_toks)


def error_counts(refs: list[str], hyps: list[str], unit: str = "word") -> pd.DataFrame:
    """Per-utterance S/D/I/H/N for already-normalized text. unit is 'word' or 'char'."""
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps differ in length")
    if unit not in ("word", "char"):
        raise ValueError(unit)
    return pd.DataFrame([_counts(r, h, unit) for r, h in zip(refs, hyps)], columns=COUNT_COLUMNS)


def error_rate(counts: pd.DataFrame) -> float:
    n = counts["n_ref"].sum()
    return float((counts["sub"].sum() + counts["del"].sum() + counts["ins"].sum()) / n) if n else float("nan")


@dataclass
class Estimate:
    value: float
    lo: float
    hi: float

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value, "ci_lo": self.lo, "ci_hi": self.hi}


def _cluster_sums(counts: pd.DataFrame, clusters: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    g = counts.assign(_c=clusters.to_numpy()).groupby("_c", sort=True)
    errs = (g["sub"].sum() + g["del"].sum() + g["ins"].sum()).to_numpy(dtype=float)
    return errs, g["n_ref"].sum().to_numpy(dtype=float)


def bootstrap_error_rate(
    counts: pd.DataFrame, clusters: pd.Series, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Estimate:
    """Corpus error rate with a percentile CI from resampling clusters (meetings) with replacement."""
    errs, n = _cluster_sums(counts, clusters)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(errs), size=(n_boot, len(errs)))
    boot = errs[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return Estimate(error_rate(counts), float(lo), float(hi))


def paired_bootstrap_delta(
    counts_a: pd.DataFrame,
    counts_b: pd.DataFrame,
    clusters: pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Difference B - A in corpus error rate on the same utterances, with the same resampled
    meetings applied to both systems. p_b_not_better is the share of resamples where B >= A."""
    if not counts_a["n_ref"].equals(counts_b["n_ref"]):
        raise ValueError("systems were not scored on the same references")
    ea, n = _cluster_sums(counts_a, clusters)
    eb, _ = _cluster_sums(counts_b, clusters)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ea), size=(n_boot, len(ea)))
    denom = n[idx].sum(axis=1)
    delta = eb[idx].sum(axis=1) / denom - ea[idx].sum(axis=1) / denom
    lo, hi = np.quantile(delta, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": error_rate(counts_b) - error_rate(counts_a),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_b_not_better": float((delta >= 0).mean()),
    }
