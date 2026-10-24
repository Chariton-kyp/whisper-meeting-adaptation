"""Evaluation sources other than the AMI release used for training, read from pinned parquet files.

Each source is exposed with the same manifest columns as AMI: `audio_id`, `meeting_id` (the
bootstrap cluster) and `text`, plus `duration`. This lets the evaluation and scoring code treat all
sources alike.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    revision: str
    files: tuple[str, ...]  # allow_patterns relative to the repo root
    id_cols: tuple[str, ...]  # joined with "_" to form audio_id
    text_col: str
    cluster: str  # "column:<name>" or "ami_meeting_from_id"


# IHM audio of the AMI test meetings with cased, punctuated (orthographic) references.
ESB_AMI = Source(
    name="esb_ami",
    repo="hf-audio/esb-datasets-test-only-sorted",
    revision="de79e57adf9983993bcc72586d7313460825f5eb",
    files=("ami/test-*.parquet",),
    id_cols=("id",),
    text_col="text",
    cluster="ami_meeting_from_id",
)

EARNINGS22 = Source(
    name="earnings22",
    repo="distil-whisper/earnings22",
    revision="0a034f9ed86d33a3859d9025d3e621cf243773ab",
    files=(
        "chunked/test-00000-of-00038-*.parquet",
        "chunked/test-00012-of-00038-*.parquet",
        "chunked/test-00025-of-00038-*.parquet",
        "chunked/test-00037-of-00038-*.parquet",
    ),
    id_cols=("file_id", "segment_id"),
    text_col="transcription",
    cluster="column:file_id",
)

SOURCES = {s.name: s for s in (ESB_AMI, EARNINGS22)}

_TAG = re.compile(r"<[^>]*>")


def clean_reference(text: str) -> str:
    """Remove annotation tags such as <inaudible> from references before scoring."""
    return " ".join(_TAG.sub(" ", text).split())


def _paths(src: Source) -> list[Path]:
    root = Path(snapshot_download(src.repo, repo_type="dataset", revision=src.revision,
                                  allow_patterns=list(src.files)))
    paths = sorted(p for pattern in src.files for p in root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files for {src.name}")
    return paths


def _ids(table: pd.DataFrame, src: Source) -> pd.Series:
    return table[list(src.id_cols)].astype(str).agg("_".join, axis=1)


def _cluster(table: pd.DataFrame, ids: pd.Series, src: Source) -> pd.Series:
    if src.cluster == "ami_meeting_from_id":
        return ids.str.split("_").str[1]
    return table[src.cluster.split(":", 1)[1]].astype(str)


def _decode(audio_bytes: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly

        g = np.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    return audio


def load_source_manifest(src: Source) -> pd.DataFrame:
    frames = []
    for path in _paths(src):
        cols = [c for c in pq.ParquetFile(path).schema_arrow.names if c != "audio"]
        frames.append(pq.read_table(path, columns=cols).to_pandas())
    t = pd.concat(frames, ignore_index=True)
    ids = _ids(t, src)
    out = pd.DataFrame({"audio_id": ids, "meeting_id": _cluster(t, ids, src),
                        "text": t[src.text_col].map(clean_reference)})
    if "audio_length_s" in t:
        out["duration"] = t["audio_length_s"].astype(float)
    elif {"start_ts", "end_ts"} <= set(t.columns):
        out["duration"] = (t["end_ts"] - t["start_ts"]).astype(float)
    return out


def iter_source_audio(src: Source, ids: set[str] | None = None, batch_rows: int = 64) -> Iterator[dict]:
    for path in _paths(src):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_rows):
            t = batch.to_pandas()
            row_ids = _ids(t, src)
            for i, uid in enumerate(row_ids):
                if ids is not None and uid not in ids:
                    continue
                yield {"audio_id": uid, "array": _decode(t["audio"].iloc[i]["bytes"])}
