"""LoRA adaptation of Whisper on AMI, with matched-budget arms (encoder / decoder / both).

All arms share the data subset and order, steps, batch, schedule, rank and seed (docs/design.md).
Outputs:
- outputs/runs/<run_id>/{best,last}/: PEFT adapters (gitignored)
- results/<run_id>/train_log.jsonl and train_run.json: losses, dev-select WER, timing, provenance

Usage:
    uv run python scripts/train_lora.py --config configs/train/turbo_lora_both.yaml
    uv run python scripts/train_lora.py --config configs/train/turbo_lora_both.yaml --pilot-steps 200
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

from meeting_asr import run_meta
from meeting_asr.data import AMI_REVISION, iter_audio, load_manifest
from meeting_asr.metrics import error_counts, error_rate
from meeting_asr.text import normalize

_LINEAR = r"(self_attn\.(q_proj|k_proj|v_proj|out_proj)|encoder_attn\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)"
TARGETS = {
    "encoder": rf"model\.encoder\.layers\.\d+\.{_LINEAR}",
    "decoder": rf"model\.decoder\.layers\.\d+\.{_LINEAR}",
    "both": rf"model\.(encoder|decoder)\.layers\.\d+\.{_LINEAR}",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_ids(path: str, n: int | None = None) -> list[str]:
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids[:n] if n else ids


def preload(config: str, split: str, ids: list[str]) -> dict[str, np.ndarray]:
    """Decode the needed utterances once and keep them in RAM as int16."""
    wanted = set(ids)
    audio = {}
    for row in iter_audio(config, split, ids=wanted):
        audio[row["audio_id"]] = (np.clip(row["array"], -1, 1) * 32767).astype(np.int16)
    missing = wanted - audio.keys()
    if missing:
        raise RuntimeError(f"{len(missing)} ids not found, e.g. {sorted(missing)[:3]}")
    return audio


def features(processor, arrays: list[np.ndarray]) -> torch.Tensor:
    floats = [a.astype(np.float32) / 32767 for a in arrays]
    return processor.feature_extractor(
        floats, sampling_rate=16000, return_tensors="pt", device="cuda"
    ).input_features.to("cuda")


def make_labels(processor, texts: list[str], decoder_start_id: int) -> torch.Tensor:
    tok = processor.tokenizer(texts, padding=True, return_tensors="pt")
    labels = tok["input_ids"].masked_fill(tok["attention_mask"].ne(1), -100)
    if (labels[:, 0] == decoder_start_id).all():
        labels = labels[:, 1:]  # the model prepends decoder_start_token_id when shifting labels
    return labels.to("cuda")


@torch.no_grad()
def dev_wer(model, processor, dev_ids, dev_audio, dev_text, batch: int, gen_kw: dict) -> float:
    model.eval()
    hyps = []
    for i in range(0, len(dev_ids), batch):
        chunk = dev_ids[i : i + batch]
        feats = features(processor, [dev_audio[u] for u in chunk])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(input_features=feats, use_cache=True, **gen_kw)
        hyps.extend(processor.batch_decode(out, skip_special_tokens=True))
    model.train()
    refs = [normalize(dev_text[u]) for u in dev_ids]
    return error_rate(error_counts(refs, [normalize(h) for h in hyps]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--pilot-steps", type=int, help="run only this many optimizer steps, no dev eval")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or Path(args.config).stem + (f"_pilot{args.pilot_steps}" if args.pilot_steps else "")
    res_dir, ckpt_dir = Path("results") / run_id, Path("outputs/runs") / run_id
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])

    eff_batch = cfg["micro_batch"] * cfg["grad_accum"]
    total_steps = cfg["n_train"] // eff_batch
    run_steps = min(total_steps, args.pilot_steps or total_steps)
    train_ids = read_ids(cfg["train_order"], cfg["n_train"])[: run_steps * eff_batch]

    t_load = time.perf_counter()
    train_meta = load_manifest("sdm", "train").set_index("audio_id")
    train_audio = preload(cfg["ami_config"], "train", train_ids)
    dev_ids, dev_audio, dev_text = [], {}, {}
    if not args.pilot_steps:
        dev_ids = read_ids(cfg["dev_select"])
        dev_meta = load_manifest(cfg["ami_config"], "validation").set_index("audio_id")
        dev_audio = preload(cfg["ami_config"], "validation", dev_ids)
        dev_text = dev_meta.loc[dev_ids, "text"].to_dict()
    load_seconds = time.perf_counter() - t_load

    processor = WhisperProcessor.from_pretrained(cfg["model"])
    multilingual = not cfg["model"].endswith(".en")
    if multilingual:
        processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")
    gen_kw = {"num_beams": 1, "max_new_tokens": 128}
    if multilingual:
        gen_kw.update(language="en", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(cfg["model"], dtype=torch.float32).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=TARGETS[cfg["lora_target"]], bias="none")
    model = get_peft_model(model, lora)
    model.train()
    trainable, total = model.get_nb_trainable_parameters()
    decoder_start = model.config.decoder_start_token_id

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(optim, cfg["warmup_steps"], total_steps)

    meta = {"run_id": run_id, "config": cfg, "config_path": args.config, "pilot_steps": args.pilot_steps,
            "ami_revision": AMI_REVISION, "total_steps": total_steps, "run_steps": run_steps,
            "effective_batch": eff_batch, "trainable_params": trainable, "total_params": total,
            "lora_target_regex": TARGETS[cfg["lora_target"]], "data_load_seconds": round(load_seconds, 1),
            "provenance": run_meta.collect()}
    (res_dir / "train_run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log = open(res_dir / "train_log.jsonl", "w", encoding="utf-8")

    best = math.inf
    if dev_ids and cfg.get("eval_at_start", True):
        best_step0 = dev_wer(model, processor, dev_ids, dev_audio, dev_text, cfg["eval_batch"], gen_kw)
        log.write(json.dumps({"step": 0, "dev_select_wer": round(best_step0, 5)}) + "\n")
        log.flush()
        print(f"step 0 dev-select WER {best_step0:.4f}")

    torch.cuda.reset_peak_memory_stats()
    t_train = time.perf_counter()
    running, n_utts = [], 0
    for step in range(1, run_steps + 1):
        for micro in range(cfg["grad_accum"]):
            start = ((step - 1) * cfg["grad_accum"] + micro) * cfg["micro_batch"]
            ids = train_ids[start : start + cfg["micro_batch"]]
            feats = features(processor, [train_audio[u] for u in ids])
            labels = make_labels(processor, [train_meta.at[u, "text"].lower() for u in ids], decoder_start)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=feats, labels=labels).loss / cfg["grad_accum"]
            loss.backward()
            running.append(loss.item() * cfg["grad_accum"])
            n_utts += len(ids)
        torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

        if step % cfg["log_every"] == 0 or step == run_steps:
            elapsed = time.perf_counter() - t_train
            rec = {"step": step, "loss": round(float(np.mean(running)), 4), "lr": sched.get_last_lr()[0],
                   "utt_per_s": round(n_utts / elapsed, 2), "elapsed_s": round(elapsed, 1),
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            running = []
            if dev_ids and (step % cfg["eval_every"] == 0 or step == run_steps):
                wer = dev_wer(model, processor, dev_ids, dev_audio, dev_text, cfg["eval_batch"], gen_kw)
                rec["dev_select_wer"] = round(wer, 5)
                if wer < best:
                    best = wer
                    model.save_pretrained(ckpt_dir / "best")
                    rec["saved_best"] = True
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec)

    model.save_pretrained(ckpt_dir / "last")
    meta.update({"train_seconds": round(time.perf_counter() - t_train, 1), "best_dev_select_wer": best,
                 "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                 "spilled_to_sysmem": torch.cuda.max_memory_allocated() > torch.cuda.mem_get_info()[1]})
    (res_dir / "train_run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log.close()


if __name__ == "__main__":
    main()
