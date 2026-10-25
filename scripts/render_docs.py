"""Fill the generated tables in README.md and docs/report.md from results/.

A generated block looks like:

    <!-- BEGIN GENERATED: name -->
    ...replaced on every run...
    <!-- END GENERATED: name -->

`name` is either a file in results/summary/ (`<name>.md`; its title line is dropped and its
headings are demoted) or one of the built-in tables below, which are computed from the run
JSON files. No number in these blocks is typed by hand.

Usage:
    uv run python scripts/render_docs.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RESULTS = Path("results")
SUMMARY = RESULTS / "summary"
DOCS = [Path("README.md"), Path("docs/report.md")]
BLOCK = re.compile(r"(<!-- BEGIN GENERATED: (?P<name>[\w-]+) -->\n)(?P<body>.*?)(<!-- END GENERATED: (?P=name) -->)", re.S)

# Published greedy zero-shot results for whisper-small.en on AMI test (normalized WER, %).
PUBLISHED_SMALL_EN = {"ihm": (17.4, 17.5), "sdm": (38.1, 38.0)}
SYSTEMS = [("zeroshot", "Zero-shot"), ("both", "LoRA encoder+decoder"), ("encoder", "LoRA encoder only"),
           ("decoder", "LoRA decoder only")]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def ci(e: dict) -> str:
    return f"{pct(e['value'])} [{pct(e['ci_lo'])}, {pct(e['ci_hi'])}]"


def delta(d: dict) -> str:
    return f"{100 * d['delta']:+.2f} [{100 * d['ci_lo']:+.2f}, {100 * d['ci_hi']:+.2f}]"


def harness() -> str:
    rows = ["| Mic | This harness [95% CI] | Distil-Whisper Table 16 | Whisper Table 8 | Δ |", "|---|---|---|---|---|"]
    for mic in ("ihm", "sdm"):
        m = load(RESULTS / f"smallen_{mic}_test" / "metrics.json")
        a, b = PUBLISHED_SMALL_EN[mic]
        v = 100 * m["wer"]["value"]
        lo, hi = sorted((v - a, v - b))
        rows.append(f"| {mic.upper()} | {ci(m['wer'])} | {a} | {b} | {lo:+.1f} to {hi:+.1f} |")
    return "\n".join(rows)


def headline() -> str:
    sdm, ihm = load(SUMMARY / "sdm_test.json")["systems"], load(SUMMARY / "ihm_test.json")["systems"]
    esb, e22 = load(SUMMARY / "esb_ami.json")["systems"], load(SUMMARY / "earnings22.json")["systems"]
    rows = ["| System | SDM test WER | Δ SDM vs zero-shot | IHM test WER | Δ IHM | Orthographic WER (cased, punctuated) | "
            "Earnings-22 WER (out of domain) |", "|---|---|---|---|---|---|---|"]
    for key, label in SYSTEMS:
        s, i = sdm[f"{key}_sdm_test"], ihm[f"{key}_ihm_test"]
        o, e = esb[f"{key}_esb_ami"], e22[f"{key}_earnings22"]
        rows.append(" | ".join([
            f"| {label}", ci(s["wer"]), delta(s["delta_wer"]) if "delta_wer" in s else "—",
            ci(i["wer"]), delta(i["delta_wer"]) if "delta_wer" in i else "—",
            ci(o["ortho"]) + (f" ({100 * o['delta_ortho']['delta']:+.2f})" if "delta_ortho" in o else ""),
            ci(e["wer"]) + (f" ({100 * e['delta_wer']['delta']:+.2f})" if "delta_wer" in e else ""),
        ]) + " |")
    return "\n".join(rows)


def training() -> str:
    rows = ["| Arm | Trainable params | Train time | Peak VRAM | Best dev-select WER (step) |", "|---|---|---|---|---|"]
    for key, label in SYSTEMS[1:]:
        run = load(RESULTS / f"turbo_lora_{key}" / "train_run.json")
        log = [json.loads(line) for line in (RESULTS / f"turbo_lora_{key}" / "train_log.jsonl").read_text().splitlines()]
        best = min((r for r in log if "dev_select_wer" in r and r["step"] > 0), key=lambda r: r["dev_select_wer"])
        rows.append(f"| {label} | {run['trainable_params'] / 1e6:.1f}M | {run['train_seconds'] / 3600:.2f} h | "
                    f"{run['peak_vram_gb']:.2f} GB | {100 * best['dev_select_wer']:.2f} ({best['step']}) |")
    return "\n".join(rows)


def errors() -> str:
    z, e = load(SUMMARY / "errors_zeroshot_sdm_test.json"), load(SUMMARY / "errors_encoder_sdm_test.json")
    c = load(SUMMARY / "errors_encoder_sdm_test_vs_zeroshot_sdm_test.json")
    top = lambda r: ", ".join(f'"{t}" ×{n}' for t, n in r["top_outputs_on_empty_refs"][:4])
    rows = ["| SDM test | Zero-shot | LoRA encoder only |", "|---|---|---|",
            f"| Filler-only segments (empty reference after normalization) that get any output | "
            f"{z['empty_ref_with_output']} / {z['empty_ref_utterances']} | {e['empty_ref_with_output']} / {e['empty_ref_utterances']} |",
            f"| Most frequent outputs on them | {top(z)} | {top(e)} |",
            f"| Deletions (all) | {z['deletions_total']} | {e['deletions_total']} |",
            f"| Deletions of an immediately repeated reference word | {z['repeated_word_deletions']} | {e['repeated_word_deletions']} |",
            f"| Utterances with fewer / more errors than zero-shot | — | {c['utterances_improved']} / {c['utterances_degraded']} |"]
    return "\n".join(rows)


def arm_agreement() -> str:
    import pandas as pd

    a = pd.read_json(RESULTS / "both_sdm_test" / "hyps.jsonl", lines=True, dtype={"audio_id": str}).set_index("audio_id")
    b = pd.read_json(RESULTS / "encoder_sdm_test" / "hyps.jsonl", lines=True, dtype={"audio_id": str}).set_index("audio_id")
    b = b.loc[a.index]
    ea, eb = a["sub"] + a["del"] + a["ins"], b["sub"] + b["del"] + b["ins"]
    differ = (a["hyp_norm"].fillna("") != b["hyp_norm"].fillna("")).mean()
    return "\n".join([
        "| SDM test: encoder-only vs encoder+decoder | Value |", "|---|---|",
        f"| Normalized hypotheses that differ | {100 * differ:.1f}% |",
        f"| Utterances where encoder-only has fewer / more errors | {int((eb < ea).sum())} / {int((eb > ea).sum())} |",
        f"| Total errors, encoder-only / encoder+decoder | {int(eb.sum())} / {int(ea.sum())} |",
    ])


def formatting_compute() -> str:
    rows = ["| System | Hyps with punctuation (cased AMI test) | Hyps starting uppercase | RTF, SDM test (generate only, batch 16) | "
            "Peak inference VRAM |", "|---|---|---|---|---|"]
    for key, label in SYSTEMS:
        f = load(RESULTS / f"{key}_esb_ami" / "metrics.json")["formatting"]
        t = load(RESULTS / f"{key}_sdm_test" / "metrics.json")["timing"]
        rows.append(f"| {label} | {100 * f['frac_with_punct']:.1f}% | {100 * f['frac_starting_upper']:.1f}% | "
                    f"{t['rtf']:.4f} | {t['peak_vram_gb']:.2f} GB |")
    return "\n".join(rows)


BUILTIN = {"harness": harness, "headline": headline, "training": training, "errors": errors,
           "arm_agreement": arm_agreement, "formatting_compute": formatting_compute}


def summary_md(name: str) -> str:
    text = (SUMMARY / f"{name}.md").read_text(encoding="utf-8").strip().splitlines()
    if text and text[0].startswith("# "):
        text = text[1:]
    return "\n".join(("###" + line[1:]) if line.startswith("## ") else line for line in text).strip()


def render(name: str) -> str:
    return BUILTIN[name]() if name in BUILTIN else summary_md(name)


def main() -> None:
    for doc in DOCS:
        src = doc.read_text(encoding="utf-8")
        out = BLOCK.sub(lambda m: f"{m.group(1)}{render(m.group('name'))}\n{m.group(4)}", src)
        doc.write_text(out, encoding="utf-8")
        print(f"{doc}: {len(BLOCK.findall(src))} blocks rendered")


if __name__ == "__main__":
    main()
