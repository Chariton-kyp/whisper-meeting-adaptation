# Report: far-field meeting adaptation of Whisper

> Status: **draft**. The methods and harness validation are final. The results sections are filled
> from `results/summary/` once the queue (`scripts/run_all.sh`) has finished. Hypotheses and metrics
> were fixed in advance in [`design.md`](design.md); deviations are listed there under Amendments.

## 1. Summary

_Pending._ To be written last: a few sentences per question Q1–Q4, each with an effect size and its
95% CI.

## 2. Setup

- **Data.** AMI, Full-corpus-ASR partition (`edinburghcstr/ami` @ `46f28f2`).
  - Training: the first 16,000 utterances of a seeded permutation of SDM train (135 meetings,
    76.9 h in total).
  - Checkpoint selection: `dev-select`, 2,000 SDM validation utterances stratified by meeting.
  - Test: 16 meetings and 12,643 segments. The IHM and SDM versions are identical segment for
    segment.
  - Train and test share no meetings and no speakers ([`splits/README.md`](../splits/README.md)).
- **Systems.** `whisper-large-v3-turbo` zero-shot (Z), and three matched-budget LoRA arms:
  - L-both: encoder and decoder
  - L-enc: encoder only
  - L-dec: decoder only

  Every arm uses r=32, α=64, 1,000 steps at effective batch 16, lr 1e-4 with linear decay, and
  seed 0.
- **Scoring.**
  - Corpus WER after Whisper's English normalizer. Punctuation-only tokens left by the normalizer
    are dropped (amendment 3).
  - 95% CIs from a meeting-level cluster bootstrap (2,000 resamples).
  - Paired bootstrap for differences between systems.
- **Hardware.** One RTX 4080 Laptop GPU (12 GB).

## 3. Harness validation

`whisper-small.en`, greedy, zero-shot, AMI test:

| | This harness [95% CI] | Distil-Whisper Table 16 | Whisper Table 8 |
|---|---|---|---|
| IHM | 17.61 [15.46, 19.53] | 17.4 | 17.5 |
| SDM | 39.32 [33.40, 44.62] | 38.1 | 38.0 |

Both are within the pre-registered ±2 tolerance. The residual on SDM (+1.2) has three candidate
explanations, none of them tested: decoding details (fp16; `max_new_tokens=128`), the
generation-config defaults of `transformers` 5.20, and the normalizer version.

## 4. Q1: gain on held-out meetings

_Pending: `results/summary/sdm_test.md` and `ihm_test.md`._

## 5. Q2: encoder vs decoder (matched-budget ablation)

_Pending: the arm comparison on SDM and IHM, and the far-field penalty with its change against Z
(`results/summary/mic_gap.md`)._

## 6. Q3: style vs content

_Pending: WER vs disfluency-insensitive WER per system._

A post hoc, exploratory analysis (not pre-registered) looks at spelling conventions that the
normalizer leaves untouched: "okay"/"ok", "cause"/"because", "dunno", "t v"/"tv".

## 7. Q4: costs

_Pending:_
- Formatting: orthographic WER on the cased and punctuated AMI test (`esb_ami`), plus the share of
  hypotheses that contain punctuation or casing.
- Out-of-domain: Earnings-22, 2,000 segments.
- Hallucinations: insertions on references that are empty after normalization, and the insertion
  rate in the <1 s slice.
- Compute: GPU-hours, peak VRAM, and RTF after the LoRA merge.

## 8. Error analysis

_Pending: the most frequent substitutions, deletions and insertions, outputs on filler-only
segments, and the utterances that improved or degraded most (`scripts/error_analysis.py`)._

Already observed on zero-shot validation (turbo). This was a preliminary run that pre-dates the first
commit; these figures will be replaced by `results/zeroshot_sdm_dev` and `results/zeroshot_ihm_dev`
when the queue's final step runs:
- On SDM, the model produces text for 66% of the 891 filler-only segments, most often
  "Thank you."; on IHM it does so for 23%.
- Repeated-word deletions are 8% of all deletions.

## 9. Engineering notes

These are problems that would have biased or invalidated results without raising any error:

1. **Normalizer artifact** (amendment 3). A trailing "..." left a bare "." that was scored as an
   insertion, and the effect favoured adapted models.
2. **Gradient checkpointing silently inactive.** `from_pretrained` returns eval mode, and HF only
   checkpoints in train mode.
3. **Silent VRAM spill on Windows/WDDM.** Training ran up to 25× slower, with no OOM error.
4. **Orphaned processes.** When the `uv run` wrapper exits on Windows, Python keeps running, and
   two trainers ended up sharing the GPU. Fixed by launching the venv interpreter as one detached
   queue.

## 10. Limitations

See [`design.md`](design.md#known-limitations). The main ones:
- Whisper's pre-training data is undisclosed.
- Evaluation covers segmented utterances only.
- One seed per arm.
- There are 16 test meetings.
- `dev-select` WER in the training logs uses the pre-fix normalizer for L-both only (amendment 3).
  It is not used for any reported comparison.
