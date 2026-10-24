"""Transcribe an AMI split with a Whisper model and score it.

Writes results/<run_id>/ with:
- hyps.jsonl: per-utterance reference, raw and normalized hypothesis, and error counts
- metrics.json: corpus WER/CER with meeting-level bootstrap CIs, error breakdown, formatting profile, timing
- run.json: config and provenance

Usage:
    uv run python scripts/evaluate.py --config configs/eval/turbo_sdm_dev.yaml
    uv run python scripts/evaluate.py --config configs/eval/turbo_sdm_dev.yaml --limit-meetings 2 --run-id smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from meeting_asr import run_meta
from meeting_asr.data import AMI_REVISION, iter_audio, load_manifest
from meeting_asr.metrics import bootstrap_error_rate, error_counts
from meeting_asr.overlap import overlap_ratio
from meeting_asr.sources import SOURCES, iter_source_audio, load_source_manifest
from meeting_asr.text import disfluency_insensitive, formatting_profile, normalize, orthographic

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_model(cfg: dict):
    processor = WhisperProcessor.from_pretrained(cfg["model"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model"], dtype=DTYPES[cfg["dtype"]])
    if cfg.get("adapter"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, cfg["adapter"]).merge_and_unload()
    return processor, model.to("cuda").eval()


def generate_kwargs(cfg: dict) -> dict:
    kw = {"num_beams": cfg["num_beams"], "max_new_tokens": cfg["max_new_tokens"]}
    if not cfg["model"].endswith(".en"):
        kw.update(language="en", task="transcribe")
    return kw


def load_eval_manifest(cfg: dict) -> pd.DataFrame:
    if cfg.get("dataset", "ami") == "ami":
        df = load_manifest(cfg["ami_config"], cfg["split"])
        df["overlap_ratio"] = overlap_ratio(df)  # computed on the full split, before any subsetting
    else:
        df = load_source_manifest(SOURCES[cfg["dataset"]])
    if cfg.get("ids_file"):
        keep = set(Path(cfg["ids_file"]).read_text(encoding="utf-8").split())
        df = df[df["audio_id"].isin(keep)]
        if len(df) != len(keep):
            raise RuntimeError(f"ids_file has {len(keep)} ids, {len(df)} found")
    return df


def audio_rows(cfg: dict, ids: set[str]):
    if cfg.get("dataset", "ami") == "ami":
        return iter_audio(cfg["ami_config"], cfg["split"], ids=ids)
    return iter_source_audio(SOURCES[cfg["dataset"]], ids=ids)


def transcribe(cfg: dict, manifest: pd.DataFrame, processor, model) -> tuple[dict[str, str], dict]:
    ids = set(manifest["audio_id"])
    kw = generate_kwargs(cfg)
    hyps: dict[str, str] = {}
    gen_seconds, audio_seconds, n_batches = 0.0, 0.0, 0
    batch: list[dict] = []

    def flush(rows: list[dict], timed: bool) -> None:
        nonlocal gen_seconds, audio_seconds, n_batches
        feats = processor.feature_extractor(
            [r["array"] for r in rows], sampling_rate=16000, return_tensors="pt"
        ).input_features.to("cuda", dtype=model.dtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(feats, **kw)
        torch.cuda.synchronize()
        if timed:
            gen_seconds += time.perf_counter() - t0
            audio_seconds += sum(len(r["array"]) / 16000 for r in rows)
            n_batches += 1
        for r, text in zip(rows, processor.batch_decode(out, skip_special_tokens=True)):
            hyps[r["audio_id"]] = text.strip()

    torch.cuda.reset_peak_memory_stats()
    warmed = False
    for row in audio_rows(cfg, ids):
        batch.append(row)
        if len(batch) == cfg["batch_size"]:
            flush(batch, timed=warmed)  # first batch is warm-up and not timed
            warmed, batch = True, []
    if batch:
        flush(batch, timed=warmed)
    timing = {
        "scope": "model.generate only (feature extraction, decoding to text and I/O excluded)",
        "batch_size": cfg["batch_size"],
        "timed_batches": n_batches,
        "timed_audio_seconds": round(audio_seconds, 1),
        "generate_seconds": round(gen_seconds, 2),
        "rtf": round(gen_seconds / audio_seconds, 5) if audio_seconds else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    return hyps, timing


def score(manifest: pd.DataFrame, hyps: dict[str, str], n_boot: int) -> tuple[pd.DataFrame, dict]:
    df = manifest.copy().reset_index(drop=True)
    df["hyp_raw"] = df["audio_id"].map(hyps).fillna("")
    df["ref_norm"] = df["text"].map(normalize)
    df["hyp_norm"] = df["hyp_raw"].map(normalize)
    wc = error_counts(df["ref_norm"].tolist(), df["hyp_norm"].tolist(), unit="word")
    cc = error_counts(df["ref_norm"].tolist(), df["hyp_norm"].tolist(), unit="char")
    dc = error_counts(df["ref_norm"].map(disfluency_insensitive).tolist(),
                      df["hyp_norm"].map(disfluency_insensitive).tolist(), unit="word")
    oc = error_counts(df["text"].map(orthographic).tolist(), df["hyp_raw"].map(orthographic).tolist(), unit="word")
    df = pd.concat([df, wc], axis=1)
    df[["d_err", "d_n_ref"]] = pd.DataFrame({"d_err": dc["sub"] + dc["del"] + dc["ins"], "d_n_ref": dc["n_ref"]})
    df[["o_err", "o_n_ref"]] = pd.DataFrame({"o_err": oc["sub"] + oc["del"] + oc["ins"], "o_n_ref": oc["n_ref"]})
    clusters = df["meeting_id"]
    n_ref = int(wc["n_ref"].sum())
    metrics = {
        "n_utterances": len(df),
        "n_meetings": int(df["meeting_id"].nunique()),
        "n_ref_words": n_ref,
        "n_empty_ref_after_norm": int((wc["n_ref"] == 0).sum()),
        "wer": bootstrap_error_rate(wc, clusters, n_boot=n_boot).as_dict(),
        "cer": bootstrap_error_rate(cc, clusters, n_boot=n_boot).as_dict(),
        "wer_disfluency_insensitive": bootstrap_error_rate(dc, clusters, n_boot=n_boot).as_dict(),
        # Case- and punctuation-sensitive; only meaningful when references are orthographic.
        "wer_orthographic": bootstrap_error_rate(oc, clusters, n_boot=n_boot).as_dict(),
        "sub_rate": wc["sub"].sum() / n_ref,
        "del_rate": wc["del"].sum() / n_ref,
        "ins_rate": wc["ins"].sum() / n_ref,
        "insertions_on_empty_refs": int(wc.loc[wc["n_ref"] == 0, "ins"].sum()),
        "formatting": formatting_profile(df["hyp_raw"].tolist()),
    }
    return df, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--limit-meetings", type=int, help="debug: only the first N meetings (sorted)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value (YAML-parsed), e.g. --set split=test")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for item in args.set:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    run_id = args.run_id or Path(args.config).stem
    out = Path("results") / run_id
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_eval_manifest(cfg)
    if args.limit_meetings:
        keep = sorted(manifest["meeting_id"].unique())[: args.limit_meetings]
        manifest = manifest[manifest["meeting_id"].isin(keep)]
    manifest = manifest.sort_values("audio_id").reset_index(drop=True)

    meta = {"config": cfg, "config_path": args.config, "limit_meetings": args.limit_meetings,
            "ami_revision": AMI_REVISION, "provenance": run_meta.collect()}
    processor, model = load_model(cfg)
    t0 = time.perf_counter()
    hyps, timing = transcribe(cfg, manifest, processor, model)
    meta["wall_seconds_total"] = round(time.perf_counter() - t0, 1)

    df, metrics = score(manifest, hyps, args.n_boot)
    metrics["timing"] = timing
    cols = ["audio_id", "meeting_id", "speaker_id", "begin_time", "end_time", "duration", "overlap_ratio",
            "text", "hyp_raw", "ref_norm", "hyp_norm", "sub", "del", "ins", "hit", "n_ref",
            "d_err", "d_n_ref", "o_err", "o_n_ref"]
    df[[c for c in cols if c in df.columns]].to_json(out / "hyps.jsonl", orient="records", lines=True, force_ascii=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    w = metrics["wer"]
    print(f"{run_id}: WER {w['value']:.4f} [{w['ci_lo']:.4f}, {w['ci_hi']:.4f}] "
          f"n={metrics['n_utterances']} meetings={metrics['n_meetings']} rtf={timing['rtf']}")


if __name__ == "__main__":
    main()
