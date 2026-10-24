"""Environment smoke test.

Checks, in order: CUDA, AMI data access and audio decoding (soundfile, no FFmpeg), Whisper
inference with WER on a few utterances, and the peak VRAM of LoRA training steps.

Usage:
    uv run python scripts/smoke_test.py --model openai/whisper-small.en --config sdm --n 8
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import time
from pathlib import Path

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

AMI_REPO = "edinburghcstr/ami"


def env_info() -> dict:
    import datasets, peft, transformers

    free, total = torch.cuda.mem_get_info()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "vram_free_gb": round(free / 1e9, 2),
        "vram_total_gb": round(total / 1e9, 2),
        "bf16": torch.cuda.is_bf16_supported(),
    }


def load_ami_rows(config: str, n: int, min_dur: float) -> tuple[list[dict], dict]:
    path = hf_hub_download(AMI_REPO, f"{config}/test-00000-of-00004.parquet", repo_type="dataset")
    table = pq.read_table(path)
    info = {"shard": path, "num_rows": table.num_rows, "schema": str(table.schema)}
    rows = []
    for row in table.to_pylist():
        if row["end_time"] - row["begin_time"] < min_dur:
            continue
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        rows.append({**{k: v for k, v in row.items() if k != "audio"}, "array": audio, "sr": sr})
        if len(rows) == n:
            break
    return rows, info


def transcribe(model, processor, rows: list[dict], device: str) -> tuple[list[str], float]:
    feats = processor(
        [r["array"] for r in rows], sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device, dtype=model.dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lang = {} if model.config.name_or_path.endswith(".en") else {"language": "en", "task": "transcribe"}
    with torch.inference_mode():
        ids = model.generate(feats, num_beams=1, max_new_tokens=200, **lang)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return processor.batch_decode(ids, skip_special_tokens=True), elapsed


def lora_step_memory(model_id: str, processor, rows: list[dict], batch: int, steps: int) -> dict:
    model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32).cuda()
    model.config.use_cache = False
    # Non-reentrant checkpointing: with the reentrant variant the encoder LoRA weights can silently
    # receive no gradient, because only the decoder input embeddings get requires_grad hooks.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
    model = get_peft_model(model, lora)
    model.train()  # from_pretrained returns eval mode; HF checkpointing only runs when training
    trainable, total = model.get_nb_trainable_parameters()

    batch_rows = (rows * ((batch // len(rows)) + 1))[:batch]
    feats = processor.feature_extractor(
        [r["array"] for r in batch_rows], sampling_rate=16000, return_tensors="pt"
    ).input_features
    tok = processor.tokenizer([r["text"] for r in batch_rows], padding=True, return_tensors="pt")
    labels = tok["input_ids"].masked_fill(tok["attention_mask"].ne(1), -100)
    if (labels[:, 0] == model.config.decoder_start_token_id).all():
        labels = labels[:, 1:]
    feats, labels = feats.cuda(), labels.cuda()

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    losses, step_times, grad_check = [], [], {}
    for step in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_features=feats, labels=labels).loss
        loss.backward()
        if step == 0:
            for part in ("encoder", "decoder"):
                lora = [p for n, p in model.named_parameters() if f".{part}." in n and "lora_" in n]
                grads = sum(1 for p in lora if p.grad is not None and p.grad.abs().sum() > 0)
                grad_check[part] = f"{grads}/{len(lora)} lora tensors with nonzero grad"
        optim.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        losses.append(round(loss.item(), 4))
    timed = step_times[1:] or step_times  # first step includes warm-up
    peak = torch.cuda.max_memory_allocated()
    result = {
        "attn_implementation": model.config._attn_implementation,
        # On Windows (WDDM) allocations beyond physical VRAM silently spill to system RAM.
        "spilled_to_sysmem": peak > torch.cuda.mem_get_info()[1],
        "batch": batch,
        "steps": steps,
        "sec_per_step": round(sum(timed) / len(timed), 3),
        "first_step_sec": round(step_times[0], 3),
        "grad_check": grad_check,
        "peak_vram_gb": round(peak / 1e9, 2),
        "trainable_params": trainable,
        "total_params": total,
        "losses": losses,
    }
    del model, optim
    torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-small.en")
    ap.add_argument("--config", default="sdm", choices=["ihm", "sdm"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--train-batch", type=int, default=16)
    ap.add_argument("--train-steps", type=int, default=3)
    ap.add_argument("--out", default="outputs/smoke")
    args = ap.parse_args()

    report: dict = {"args": vars(args), "env": env_info()}
    print(json.dumps(report["env"], indent=2))

    rows, info = load_ami_rows(args.config, args.n, args.min_dur)
    report["data"] = {**info, "sample_rates": sorted({r["sr"] for r in rows}),
                      "durations_s": [round(len(r["array"]) / r["sr"], 2) for r in rows],
                      "ids": [r["audio_id"] for r in rows]}
    print(info["schema"])
    print(f"rows in shard: {info['num_rows']}, sampled: {len(rows)}")

    processor = WhisperProcessor.from_pretrained(args.model)
    if not args.model.endswith(".en"):
        processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16).cuda()
    hyps, elapsed = transcribe(model, processor, rows, "cuda")
    audio_s = sum(len(r["array"]) / r["sr"] for r in rows)
    norm = processor.tokenizer.normalize
    refs_n, hyps_n = [norm(r["text"]) for r in rows], [norm(h) for h in hyps]
    report["inference"] = {
        "audio_seconds": round(audio_s, 2),
        "wall_seconds": round(elapsed, 3),
        "rtf_batched": round(elapsed / audio_s, 4),
        "wer_normalized": round(jiwer.wer(refs_n, hyps_n), 4),
        "examples": [{"ref": r["text"], "hyp": h} for r, h in zip(rows, hyps)],
    }
    for ex in report["inference"]["examples"]:
        print(f"REF: {ex['ref']}\nHYP: {ex['hyp']}\n")
    print({k: v for k, v in report["inference"].items() if k != "examples"})
    del model
    torch.cuda.empty_cache()

    report["lora"] = lora_step_memory(args.model, processor, rows, args.train_batch, args.train_steps)
    print(report["lora"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"{args.model.split('/')[-1]}_{args.config}.json"
    out_file.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"report -> {out_file}")


if __name__ == "__main__":
    main()
