# whisper-meeting-adaptation

**What LoRA adaptation of Whisper on far-field meeting audio gains, and what it costs.**

A controlled study on the AMI Meeting Corpus. `whisper-large-v3-turbo` is adapted with LoRA on
single-distant-microphone (SDM) audio, as three matched-budget arms: encoder+decoder, encoder only, and
decoder only. Each arm is evaluated on held-out meetings, on both microphones, and on cased and
punctuated references, and on an out-of-domain set.

The protocol (hypotheses, metrics, splits, decoding) was committed to [`docs/design.md`](docs/design.md)
before any adapted model was trained or evaluated, and every later change is logged there as an
amendment. Every table below is generated from files in [`results/`](results/) by
`scripts/render_docs.py`, and every number quoted in the text comes from those tables. The full
write-up is in [`docs/report.md`](docs/report.md).

## Results at a glance

AMI test: 16 held-out meetings and 12,643 segments. The IHM and SDM versions contain the same
segments. 95% CIs come from a meeting-level cluster bootstrap. Δ is a paired difference against the
zero-shot model on identical utterances. WER is normalized with Whisper's English normalizer unless
the column says otherwise.

<!-- BEGIN GENERATED: headline -->
| System | SDM test WER | Δ SDM vs zero-shot | IHM test WER | Δ IHM | Orthographic WER (cased, punctuated) | Earnings-22 WER (out of domain) |
|---|---|---|---|---|---|---|
| Zero-shot | 36.25 [30.28, 41.54] | — | 16.03 [14.00, 17.92] | — | 30.39 [28.76, 31.74] | 14.94 [13.14, 17.20] |
| LoRA encoder+decoder | 22.38 [17.98, 26.44] | -13.87 [-15.35, -12.16] | 9.68 [7.72, 11.66] | -6.35 [-7.17, -5.47] | 35.53 [31.20, 39.57] (+5.14) | 15.78 [13.51, 18.18] (+0.83) |
| LoRA encoder only | 22.39 [17.91, 26.52] | -13.86 [-15.23, -12.24] | 9.60 [7.72, 11.40] | -6.43 [-7.26, -5.60] | 35.55 [31.22, 39.60] (+5.16) | 15.99 [13.62, 18.49] (+1.05) |
| LoRA decoder only | 27.59 [22.24, 32.71] | -8.66 [-9.63, -7.47] | 11.47 [8.99, 14.14] | -4.56 [-5.71, -3.01] | 37.35 [32.37, 42.19] (+6.96) | 17.52 [14.63, 20.97] (+2.58) |
<!-- END GENERATED: headline -->

What this supports:

- **A large, well-measured gain.**
  - On held-out SDM meetings: −13.9 WER points (−38% relative).
  - On close-talk IHM audio of the same meetings, which the model never trained on: −6.4 points
    (−40% relative).
  - The relative improvement is about the same on both microphones; the SDM/IHM WER ratio does not
    change significantly (post hoc). The data therefore show a broad meeting-domain improvement, not
    a gain specific to the far-field condition.
- **Encoder LoRA is sufficient at this budget.**
  - Encoder-only and encoder+decoder LoRA are indistinguishable (paired Δ −0.01 [−0.24, +0.26], one
    seed).
  - Decoder-only LoRA gains less (−8.7 points). Caveat: it also trains about 5× fewer parameters,
    because turbo's decoder has only 4 layers.
- **The mechanism is not isolated by this design.**
  - The encoder-only model, with its decoder frozen, also adopts AMI's transcription style: lowercase,
    no punctuation, "uh". The encoder/decoder split therefore does not separate acoustics from style.
  - Rule-captured style and spelling conventions account for at least about 10% of the gain
    (post hoc). This is a lower bound.
- **Costs that normalized WER does not show:**
  - **Formatting.** Adapted models are worse on case- and punctuation-sensitive WER (+5.1 to +7.0
    points), because they stop producing casing and punctuation.
  - **Out of domain.** No significant regression was detected on Earnings-22 for the encoder and
    encoder+decoder arms. The CIs do not rule out about +2.5 points. The decoder-only arm regresses
    significantly.
  - **Degenerate repetition loops** become more frequent after adaptation, most of all in the
    decoder-only arm (post hoc).
  - **Inference speed** is unchanged by construction: the merged LoRA leaves the architecture as it
    was.
- **A hallucination proxy improves.** Insertions on segments whose reference contains only fillers
  ("UH", "UM") fall by about 75%. The zero-shot model's most frequent output on them is "Thank you."

## Harness validation

Before anything else, the evaluation code had to reproduce published zero-shot numbers.
`whisper-small.en`, greedy, AMI test:

<!-- BEGIN GENERATED: harness -->
| Mic | This harness [95% CI] | Distil-Whisper Table 16 | Whisper Table 8 | Δ |
|---|---|---|---|---|
| IHM | 17.61 [15.46, 19.53] | 17.4 | 17.5 | +0.1 to +0.2 |
| SDM | 39.32 [33.40, 44.62] | 38.1 | 38.0 | +1.2 to +1.3 |
<!-- END GENERATED: harness -->

Pre-registered tolerance: ±2 absolute. As a further check, zero-shot turbo on IHM scores 16.03 here,
against 16.13 for turbo on the Open ASR Leaderboard AMI set (May 2026). That set uses the same audio
with differently formatted references.

## Setup

| | |
|---|---|
| Data | AMI Full-corpus-ASR partition (`edinburghcstr/ami` @ `46f28f2`). Train: 135 meetings, 76.9 h. Validation: 18 meetings. Test: 16 meetings. Train and test share **no meetings and no speakers** ([`splits/README.md`](splits/README.md)). |
| Training | 16,000 SDM training utterances, the same seeded subset and order for every arm; 1,000 steps; effective batch 16; lr 1e-4 with linear decay; LoRA r=32, α=64 on all attention and MLP projections of the adapted part. One seed. |
| Checkpoint selection | 2,000 validation utterances (`dev-select`); test is run once per system |
| Decoding | Greedy, no timestamps, `max_new_tokens=128`, identical for every system, no tuning |
| Other test sets | Cased and punctuated AMI test (`hf-audio/esb-datasets-test-only-sorted`, `ami`). Earnings-22 (2,000 chunked segments from 16 calls). |
| Hardware | One RTX 4080 Laptop GPU (12 GB); about 1.2 to 2 h of training per arm |

<!-- BEGIN GENERATED: training -->
| Arm | Trainable params | Train time | Peak VRAM | Best dev-select WER (step) |
|---|---|---|---|---|
| LoRA encoder+decoder | 27.9M | 1.97 h | 7.46 GB | 19.45 (1000) |
| LoRA encoder only | 23.6M | 1.94 h | 7.39 GB | 19.36 (1000) |
| LoRA decoder only | 4.3M | 1.24 h | 6.20 GB | 24.46 (750) |
<!-- END GENERATED: training -->

`dev-select` WER serves only to choose a checkpoint within each run. The encoder+decoder arm computed
it with the pre-fix normalizer (amendment 3), so the column is not comparable across arms. The arm's
best checkpoint was its last step anyway, so its selection was not affected.

## Findings from building the harness

Each of these would otherwise have silently biased or slowed the study:

- **Normalizer artifact.** Whisper's English normalizer leaves a bare "." from a trailing ellipsis
  ("so..." becomes "so ."), and that token is scored as an insertion. Models adapted on unpunctuated
  labels stop producing ellipses, so the artifact would have inflated the measured benefit of
  adaptation. It was fixed for all systems before any adapted model was evaluated
  ([amendment 3](docs/design.md#amendments)).
- **Gradient checkpointing silently inactive.** `from_pretrained` returns a model in eval mode, and
  Hugging Face only checkpoints in train mode.
- **Silent VRAM spill on Windows (WDDM).** Allocations above physical VRAM spill to system RAM with no
  error. One run peaked at 16.5 GB on a 12.9 GB card and was 25× slower ([`results/smoke/`](results/smoke/)).
  Training runs record `spilled_to_sysmem`.
- **Missing dependency for resampling.** Earnings-22 audio is 24 kHz; the out-of-domain evaluation
  surfaced the missing resampler.

## Reproduce

```bash
uv sync                                  # Python 3.12, torch 2.14 (CUDA 13.0), transformers 5.20
uv run pytest                            # unit tests: metrics, bootstrap, normalizer, overlap
uv run python scripts/make_splits.py     # regenerates splits/ (seeded) and the partition check
bash scripts/run_all.sh                  # validation, 3 training arms, all evaluations (resumable)
uv run python scripts/analyze.py --name sdm_test --runs results/zeroshot_sdm_test results/both_sdm_test \
    results/encoder_sdm_test results/decoder_sdm_test
uv run python scripts/mic_gap.py --ref zeroshot --systems both encoder decoder
uv run python scripts/render_docs.py     # regenerates the tables in README.md and docs/report.md
```

Run provenance (config, commit, library versions, GPU state) is in each `results/<run>/run.json`. The
history was rewritten once before publication; [`docs/commit-map.md`](docs/commit-map.md) maps the
recorded commits to the published ones.

## Layout

```
configs/          evaluation and training configs (one YAML per run)
docs/             design.md (protocol and amendments), report.md, commit-map.md
scripts/          evaluate, train_lora, analyze, mic_gap, error_analysis, *_posthoc, rescore, make_splits, run_all
splits/           fixed ID lists (training order, dev-select, Earnings-22 subset) and the partition check
src/meeting_asr/  data access, normalization, metrics and bootstrap, overlap, provenance
results/          per-run metrics.json, hyps.jsonl, run.json; summary/ holds the comparison tables
tests/            unit tests
```

## Limitations

- **"Zero-shot" means not adapted by us.** Whisper's pre-training data is undisclosed and was not
  de-duplicated against AMI.
- **One seed per arm.** Bootstrap CIs cover meeting sampling, not training variance.
- **Few test meetings.** There are 16, so the CIs are wide.
- **Segmented utterances only.** No long-form transcription, no speaker attribution, no streaming.
- **Arms are matched in data and steps, not in parameters.** The decoder-only result reflects turbo's
  4-layer decoder.
- **Normalization.** It does not unify every AMI convention. The style and convention analyses are
  post hoc and give lower bounds.

## Data and licenses

- Code: MIT ([`LICENSE`](LICENSE)).
- AMI Meeting Corpus: CC BY 4.0 (Carletta et al., 2005).
- Earnings-22: CC BY-SA 4.0 (Del Rio et al., 2022), via `distil-whisper/earnings22`.
- Hypotheses and references in `results/` are derived from these corpora under their licenses.
