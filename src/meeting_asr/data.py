"""AMI corpus access from the Hugging Face parquet release (`edinburghcstr/ami`).

Audio is decoded with soundfile from the stored bytes, so no FFmpeg/torchcodec is needed.
The metadata-only manifest is what splits, slicing and error analysis work from.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download

AMI_REPO = "edinburghcstr/ami"
AMI_REVISION = "46f28f2503e2ec48f8867a84eef356c70476beab"
META_COLUMNS = ["audio_id", "meeting_id", "speaker_id", "microphone_id", "begin_time", "end_time", "text"]


def shard_paths(config: str, split: str, revision: str = AMI_REVISION) -> list[Path]:
    """Download (or reuse from cache) the parquet shards of one config/split."""
    root = snapshot_download(
        AMI_REPO, repo_type="dataset", revision=revision, allow_patterns=[f"{config}/{split}-*.parquet"]
    )
    paths = sorted(Path(root, config).glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no shards for {config}/{split}")
    return paths


def load_manifest(config: str, split: str, revision: str = AMI_REVISION) -> pd.DataFrame:
    """Metadata for every segment of a split, without audio."""
    frames = [pq.read_table(p, columns=META_COLUMNS).to_pandas() for p in shard_paths(config, split, revision)]
    df = pd.concat(frames, ignore_index=True)
    df["duration"] = df["end_time"] - df["begin_time"]
    return df


def iter_audio(
    config: str, split: str, ids: set[str] | None = None, revision: str = AMI_REVISION, batch_rows: int = 256
) -> Iterator[dict]:
    """Yield segments with decoded 16 kHz float32 audio, optionally restricted to `ids`."""
    for path in shard_paths(config, split, revision):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=META_COLUMNS + ["audio"]):
            for row in batch.to_pylist():
                if ids is not None and row["audio_id"] not in ids:
                    continue
                audio, sr = sf.read(io.BytesIO(row.pop("audio")["bytes"]), dtype="float32")
                if sr != 16000:
                    raise ValueError(f"{row['audio_id']}: expected 16 kHz, got {sr}")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                yield {**row, "array": audio}


def segment_key(df: pd.DataFrame) -> pd.Series:
    """Microphone-independent key for pairing the same segment across ihm and sdm."""
    return (
        df["meeting_id"] + "|" + df["speaker_id"] + "|"
        + (df["begin_time"] * 100).round().astype(int).astype(str) + "|"
        + (df["end_time"] * 100).round().astype(int).astype(str)
    )
