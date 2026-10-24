# Design: far-field meeting adaptation of Whisper

**What does the gain consist of, and what does it cost?**

Status: **pre-registered on 2026-10-24, before any adaptation run.** Only the training-subset size N
is left open; it is fixed after a throughput pilot (see Training). Any later change is recorded under
[Amendments](#amendments) with its reason.

## Questions

- **Q1.** How much does LoRA adaptation of `whisper-large-v3-turbo` on single-distant-microphone
  (SDM) AMI meetings reduce WER on held-out meetings, on SDM and on close-talk (IHM) audio?
- **Q2.** Where does the gain come from: the encoder (acoustics) or the decoder (language and
  transcription style)? This is answered by a matched-budget ablation: LoRA on encoder only,
  decoder only, and both.
- **Q3.** How much of the gain is adaptation to the verbatim transcription style of AMI (fillers,
  repetitions), and how much is better recognition of content? This is measured with a
  disfluency-insensitive WER whose rule is fixed below.
- **Q4.** What does adaptation cost? Formatting (casing, punctuation), out-of-domain accuracy,
  hallucinations on short or filler-only segments, and compute.

## Hypotheses (directional, stated before results)

- **H1.** L-both reduces SDM test WER relative to zero-shot. Paired meeting bootstrap,
  p_not_better < 0.05.
- **H2.** Part of that gain disappears under the disfluency-insensitive WER, i.e. style adaptation
  contributes.
- **H3 (exploratory).** L-dec captures most of the style component. L-enc captures relatively more
  of the SDM-specific component, visible as a larger reduction of the SDM–IHM gap.
- **H4.** Adapted models lose formatting: orthographic WER on cased and punctuated references rises.
  They also regress somewhat on the out-of-domain set (Earnings-22).

Negative or mixed outcomes will be reported as they are.

## Data

- **AMI**
  - Source: `edinburghcstr/ami`, revision `46f28f2503e2ec48f8867a84eef356c70476beab`.
  - Partition: Full-corpus-ASR (Kaldi s5/s5b).
  - Validation (18 meetings) and test (16 meetings) share no meetings and no speakers. The
    train/test speaker check is recorded in `splits/README.md`.
  - IHM and SDM validation/test contain the same segments with identical references
    (12,643 / 12,643 matched on meeting|speaker|begin|end). This gives a paired comparison of the
    acoustic condition.
- **Training data:** SDM train. A seeded random subset of N utterances, with seed 0. The same subset
  and the same order are used for every arm. Utterances whose text is empty are dropped. Filler-only
  segments are kept.
- **Labels:** AMI reference text, lowercased. Apostrophes are kept and nothing else changes.
  - AMI has no casing or punctuation.
  - Lowercasing avoids teaching all-caps output.
  - The resulting formatting loss is measured, not prevented.
- **Checkpoint selection:** `dev-select` holds 2,000 SDM validation utterances, stratified by
  meeting, with seed 0. The IDs are in `splits/`.
- **Choosing among final candidates:** full SDM and IHM validation.
- **Test:** run once per final system.
- **Formatting set:** `hf-audio/esb-datasets-test-only-sorted`, config `ami`, test split. This is
  IHM audio of the AMI test meetings with cased and punctuated references.
- **Out-of-domain set:** `distil-whisper/earnings22`, config `chunked`, test split. 2,000 segments
  sampled from 4 fixed shards (indices 0, 12, 25, 37 of the sorted shard list), stratified by
  `file_id`, with seed 0. The references are cased and punctuated.

## Systems

| ID | Model | Adaptation |
|---|---|---|
| V | `openai/whisper-small.en` | none. Harness validation only. |
| Z | `openai/whisper-large-v3-turbo` | none (zero-shot) |
| L-both | turbo | LoRA on encoder and decoder |
| L-enc | turbo | LoRA on encoder only |
| L-dec | turbo | LoRA on decoder only |

- **Harness check:** V must reproduce the published greedy results, IHM ≈ 17.4 and SDM ≈ 38.1
  (Distil-Whisper Table 16; Whisper paper Table 8: 17.5 / 38.0). An absolute deviation above 2 points
  means the harness is debugged before anything else runs.
- **Matched budget:** every arm uses the same data subset and order, number of steps, batch size,
  learning-rate schedule, LoRA rank and seed. LoRA is applied to every linear layer
  (`q_proj, k_proj, v_proj, out_proj, fc1, fc2`) of the adapted part, with r = 32, alpha = 64,
  dropout 0.05. The trainable-parameter count differs by construction and is reported.

## Decoding

The same decoding is used for every system, with no tuning:
- greedy, temperature 0
- no timestamps
- `max_new_tokens = 128`
- `language="en"` and `task="transcribe"` for multilingual checkpoints
- fp16 weights, LoRA merged before evaluation

## Metrics

- **Primary:** corpus-level WER, sum(S+D+I) / sum(N), after Whisper's `EnglishTextNormalizer`
  applied to both references and hypotheses.
- **Uncertainty:** 95% percentile cluster bootstrap over meetings, 2,000 resamples.
- **Comparisons:** paired bootstrap on identical utterances. Report Δ, its CI, and the share of
  resamples in which the adapted system is not better.
- **Disfluency-insensitive WER** (Q3). This rule is fixed now. After Whisper normalization, apply to
  both reference and hypothesis:
  1. delete the filler tokens `{ah, eh, er, erm, hm, hmm, huh, mhm, mm, mmm, uh, uhm, um}`;
  2. collapse immediate repetitions of the same unigram or bigram ("the the the" becomes "the";
     "i think i think" becomes "i think").

  Partial words are not marked in this release of the corpus, so they cannot be handled reliably.
  This is a stated limitation.
- **Fixed slices:**
  - utterance duration: <1 s, 1–3 s, >3 s
  - speaker-overlap ratio: 0, (0, 0.3], >0.3, computed from the segment timings of the split
  - microphone: IHM vs SDM
  - per-meeting WER
- **Hallucination proxies:**
  - insertions on utterances whose normalized reference is empty
  - insertion rate in the <1 s slice
- **Formatting:**
  - Orthographic WER: case- and punctuation-sensitive. Tokens are whitespace-split, punctuation stays
    attached, and only whitespace is normalized. Computed on the formatting set and on Earnings-22.
  - Share of hypotheses that contain punctuation or lowercase letters.
- **Efficiency:**
  - training GPU-hours, peak VRAM, seconds per step
  - inference RTF: `model.generate` only, batch 16, warm-up batch excluded, measured after the LoRA
    merge. The architecture is unchanged, so RTF is expected to stay the same.

## Training

The following is fixed before the pilot:
- AdamW, learning rate 1e-4. Linear warm-up of 100 steps, then linear decay to 0.
- Weight decay 0.0, gradient clipping 1.0.
- Micro-batch 8, gradient accumulation 2, so the effective batch is 16 utterances.
- One pass over the subset.
- bf16 autocast with fp32 master weights.
- Non-reentrant gradient checkpointing.
- SpecAugment off, which is the checkpoint default.
- Seed 0.

Checkpoints:
- Evaluate on `dev-select` every 250 optimizer steps.
- Keep the adapter with the best `dev-select` WER, and also keep the last one.

The pilot runs 200 steps of L-both on real batches and measures end-to-end utterances per second
(including data loading and feature extraction). N is then set so that each arm takes about
2 GPU-hours, and is recorded in [Amendments](#amendments).

## Fallback

If the pilot has not passed and L-both has not been launched by 19:00 on 2026-10-24, the deliverable
becomes the baseline benchmark (V and Z) with error analysis.

## Known limitations

- **Pre-training contamination.** Whisper's pre-training data is undisclosed and was not
  de-duplicated against AMI. "Zero-shot" here means "not adapted by us".
- **Scope of evaluation.** Segmented utterances only. No long-form transcription, no speaker
  attribution, no streaming.
- **Statistical power.** One seed per arm, and 16 test meetings, so confidence intervals are wide.
- **References.** They are verbatim, and the Whisper normalizer has quirks (for example "L. C. D."
  becomes "l c d").
- **Out-of-domain coverage.** The Earnings-22 subset comes from 4 shards only.

## Amendments

1. **2026-10-24, before any adaptation run. Full validation sets are now optional.**
   - What changed: full SDM/IHM validation is no longer required for adapted systems. It runs at the
     end of the queue, if time allows.
   - Reason: all three arms are reported, so nothing needs to be chosen among them, and each arm's
     checkpoint is already selected on `dev-select`.
   - Unchanged: the test set is still run exactly once per system.
   - Evaluation details: annotation tags in angle brackets (e.g. `<inaudible>` in Earnings-22) are
     removed from references before scoring, for every system.
2. **2026-10-24, from the throughput pilot. N = 16,000.**
   - The L-both pilot was stopped after 50 of the planned 200 steps. Throughput and memory were
     already stable, and stopping saved time.
   - Measured: 2.59 utt/s end to end, peak VRAM 7.47 GB with no spill. Loss went from 1.87 at
     step 25 to 0.97 at step 50.
   - N = 16,000 utterances (the first 16,000 lines of `splits/ami_sdm_train_order.txt`). That gives
     1,000 optimizer steps per arm, about 1.7 h of training per arm plus the `dev-select` evaluations.
   - The pilot did not exercise the `dev-select` evaluation path, which runs first, at step 0 of the
     first arm.
   - Log: `results/turbo_lora_both_pilot200/`.
