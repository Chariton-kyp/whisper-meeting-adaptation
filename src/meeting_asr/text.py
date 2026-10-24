"""Text normalization for scoring.

Scoring uses Whisper's English normalizer (case, punctuation, spelling variants, numbers, and
removal of the fillers hmm/mm/mhm/mmm/uh/um). This is the convention of the Whisper paper and of the
Open ASR Leaderboard, so baseline numbers are comparable with published ones. The same normalizer is
applied to references and hypotheses for every system.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from huggingface_hub import hf_hub_download
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

# normalizer.json (British-to-American spelling map) is identical across the Whisper English checkpoints.
_SPELLING_SOURCE = ("openai/whisper-small.en", "normalizer.json")

_PUNCT = re.compile(r"[.,?!;:]")


@lru_cache(maxsize=1)
def whisper_normalizer() -> EnglishTextNormalizer:
    path = hf_hub_download(*_SPELLING_SOURCE)
    with open(path, encoding="utf-8") as fh:
        return EnglishTextNormalizer(json.load(fh))


def normalize(text: str) -> str:
    # The Whisper normalizer leaves a bare "." from a trailing ellipsis ("so..." -> "so ."), which
    # would count as an insertion. Tokens without any letter or digit are dropped (design.md, amendment 3).
    return " ".join(t for t in whisper_normalizer()(text).split() if any(c.isalnum() for c in t))


FILLERS = frozenset({"ah", "eh", "er", "erm", "hm", "hmm", "huh", "mhm", "mm", "mmm", "uh", "uhm", "um"})


def _collapse_repeats(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        if out and out[-1] == tok:
            continue  # unigram repeat
        out.append(tok)
        if len(out) >= 4 and out[-2:] == out[-4:-2]:
            del out[-2:]  # bigram repeat
    return out


def disfluency_insensitive(normalized: str) -> str:
    """Pre-registered rule (docs/design.md): on Whisper-normalized text, delete filler tokens and
    collapse immediate unigram and bigram repetitions."""
    return " ".join(_collapse_repeats([t for t in normalized.split() if t not in FILLERS]))


def orthographic(text: str) -> str:
    """Case- and punctuation-sensitive scoring form: only whitespace is normalized."""
    return " ".join(text.split())


def formatting_profile(texts: list[str]) -> dict[str, float]:
    """How much casing and punctuation a system emits. AMI references have neither, so a model
    adapted on them can silently lose formatting. Normalized WER cannot see that regression."""
    non_empty = [t for t in texts if t.strip()]
    if not non_empty:
        return {"frac_with_punct": 0.0, "frac_with_lowercase": 0.0, "frac_starting_upper": 0.0}
    n = len(non_empty)
    return {
        "frac_with_punct": sum(bool(_PUNCT.search(t)) for t in non_empty) / n,
        "frac_with_lowercase": sum(any(c.islower() for c in t) for t in non_empty) / n,
        "frac_starting_upper": sum(t.lstrip()[0].isupper() for t in non_empty) / n,
    }
