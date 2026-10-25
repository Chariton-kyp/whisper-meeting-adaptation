# Report: LoRA adaptation of Whisper on far-field meeting audio

The protocol is in [`design.md`](design.md). It was committed before any adapted model was trained or
evaluated, and every later change is logged there as an amendment.

- All tables are generated from `results/` by `scripts/render_docs.py`, and numbers quoted in the text
  come from those tables.
- Analyses marked **post hoc** were not part of the protocol and are exploratory.

## 1. Summary

- **Q1, gain.** LoRA adaptation of `whisper-large-v3-turbo` on 16k SDM utterances cuts WER on 16
  held-out AMI meetings:
  - SDM: from 36.25 to 22.38, a paired Δ of −13.87 [−15.35, −12.16];
  - IHM, the close-talk audio of the same meetings, which the model never trained on: from 16.03 to
    9.68.
- **Q2, where the gain lives.**
  - At matched data and steps, encoder-only LoRA is indistinguishable from encoder+decoder LoRA
    (Δ −0.01 [−0.24, +0.26], one seed).
  - Decoder-only LoRA gains less (−8.66). It also has about 5× fewer trainable parameters, because
    turbo's decoder has 4 layers.
  - The relative improvement is about the same on both microphones. The far-field penalty therefore
    shrinks in absolute terms but not as a ratio (post hoc).
- **Q3, style vs content.** The pre-registered disfluency rule and a post hoc convention map together
  account for at least about 10% of the gain. This is a lower bound, and the design does not isolate
  the rest. The encoder-only model, with its decoder frozen, also adopts AMI's transcription style.
- **Q4, costs.**
  - Casing and punctuation are lost: orthographic WER rises by 5.1 to 7.0 points.
  - No significant out-of-domain regression was detected for the encoder and encoder+decoder arms on
    Earnings-22; the CIs do not rule out about +2.5. Decoder-only regresses significantly.
  - Repetition loops become more frequent (post hoc).
  - Insertions on filler-only segments, a hallucination proxy, fall by about 75%.

## 2. Setup

- **Data.** AMI, Full-corpus-ASR partition (`edinburghcstr/ami` @ `46f28f2`).
  - Training: the first 16,000 utterances of a seeded permutation of SDM train (135 meetings, 76.9 h).
  - Checkpoint selection: `dev-select`, 2,000 SDM validation utterances stratified by meeting.
  - Test: 16 meetings and 12,643 segments. The IHM and SDM versions contain identical segments and
    references. Train and test share no meetings and no speakers.
- **Systems.** `whisper-large-v3-turbo` zero-shot, and three LoRA arms (encoder+decoder, encoder only,
  decoder only).
  - Shared settings: r=32, α=64, dropout 0.05 on all attention and MLP projections of the adapted
    part; 1,000 steps at effective batch 16; lr 1e-4 with 100 warm-up steps and linear decay; seed 0.
  - Arms are matched in data, order, steps and schedule, not in parameter count.
- **Decoding.** Greedy, no timestamps, `max_new_tokens=128`, fp16, LoRA merged. Identical for all
  systems.
- **Scoring.**
  - Corpus WER after Whisper's English normalizer. Tokens left with no letter or digit are dropped
    (amendment 3).
  - 95% percentile CIs from a meeting-level cluster bootstrap (2,000 resamples); paired bootstrap for
    differences.
  - Orthographic WER is case- and punctuation-sensitive.

<!-- BEGIN GENERATED: training -->
| Arm | Trainable params | Train time | Peak VRAM | Best dev-select WER (step) |
|---|---|---|---|---|
| LoRA encoder+decoder | 27.9M | 1.97 h | 7.46 GB | 19.45 (1000) |
| LoRA encoder only | 23.6M | 1.94 h | 7.39 GB | 19.36 (1000) |
| LoRA decoder only | 4.3M | 1.24 h | 6.20 GB | 24.46 (750) |
<!-- END GENERATED: training -->

The encoder+decoder arm computed its `dev-select` WER with the pre-fix normalizer. That figure is used
only to choose a checkpoint within the run, and the best checkpoint was the last one, so selection was
not affected.

## 3. Harness validation

<!-- BEGIN GENERATED: harness -->
| Mic | This harness [95% CI] | Distil-Whisper Table 16 | Whisper Table 8 | Δ |
|---|---|---|---|---|
| IHM | 17.61 [15.46, 19.53] | 17.4 | 17.5 | +0.1 to +0.2 |
| SDM | 39.32 [33.40, 44.62] | 38.1 | 38.0 | +1.2 to +1.3 |
<!-- END GENERATED: harness -->

Both rows are within the ±2 tolerance set in the protocol. The SDM residual (+1.2) has candidate
causes that were not tested: fp16, `max_new_tokens=128`, the `transformers` 5.20 generation defaults,
and the normalizer version.

## 4. Q1: gain on held-out meetings

### SDM test

<!-- BEGIN GENERATED: sdm_test -->
Reference for paired deltas: `zeroshot_sdm_test`. 12643 utterances, 16 clusters, 95% meeting-level bootstrap CIs (2000 resamples).

| System | WER | Disfluency-insens. WER | ΔWER vs ref [CI] | p(not better) | Ins. on empty refs |
|---|---|---|---|---|---|
| `zeroshot_sdm_test` | 36.25 [30.28, 41.54] | 35.22 [29.20, 40.60] | — | — | 1316 |
| `both_sdm_test` | 22.38 [17.98, 26.44] | 22.25 [17.85, 26.29] | -13.87 [-15.35, -12.16] | 0.000 | 334 |
| `encoder_sdm_test` | 22.39 [17.91, 26.52] | 22.24 [17.72, 26.37] | -13.86 [-15.23, -12.24] | 0.000 | 340 |
| `decoder_sdm_test` | 27.59 [22.24, 32.71] | 26.11 [20.96, 30.75] | -8.66 [-9.63, -7.47] | 0.000 | 368 |

#### WER by duration

| System | <1s | 1-3s | >3s |
|---|---|---|---|
| `zeroshot_sdm_test` | 100.53 [94.11, 111.12] (n=4865) | 50.34 [44.70, 54.48] (n=4047) | 24.05 [19.42, 28.76] (n=3731) |
| `both_sdm_test` | 54.15 [51.65, 56.39] (n=4865) | 34.85 [30.71, 38.07] (n=4047) | 14.32 [10.99, 17.81] (n=3731) |
| `encoder_sdm_test` | 54.47 [51.64, 56.91] (n=4865) | 34.99 [30.86, 38.10] (n=4047) | 14.25 [10.97, 17.76] (n=3731) |
| `decoder_sdm_test` | 63.17 [60.20, 66.14] (n=4865) | 42.07 [37.06, 46.10] (n=4047) | 18.39 [14.16, 22.71] (n=3731) |

#### WER by overlap

| System | none | (0,0.3] | >0.3 |
|---|---|---|---|
| `zeroshot_sdm_test` | 20.10 [17.07, 23.25] (n=3969) | 21.76 [18.83, 24.37] (n=2075) | 71.46 [67.26, 74.75] (n=6599) |
| `both_sdm_test` | 11.96 [9.88, 14.33] (n=3969) | 12.81 [10.73, 14.89] (n=2075) | 45.33 [41.06, 48.99] (n=6599) |
| `encoder_sdm_test` | 11.92 [9.73, 14.34] (n=3969) | 13.09 [10.77, 15.51] (n=2075) | 45.11 [40.63, 48.60] (n=6599) |
| `decoder_sdm_test` | 14.18 [11.67, 16.95] (n=3969) | 15.59 [12.45, 18.73] (n=2075) | 56.80 [51.13, 61.71] (n=6599) |
<!-- END GENERATED: sdm_test -->

### IHM test (same segments, close-talk microphones)

<!-- BEGIN GENERATED: ihm_test -->
Reference for paired deltas: `zeroshot_ihm_test`. 12643 utterances, 16 clusters, 95% meeting-level bootstrap CIs (2000 resamples).

| System | WER | Disfluency-insens. WER | ΔWER vs ref [CI] | p(not better) | Ins. on empty refs |
|---|---|---|---|---|---|
| `zeroshot_ihm_test` | 16.03 [14.00, 17.92] | 14.63 [12.57, 16.50] | — | — | 326 |
| `both_ihm_test` | 9.68 [7.72, 11.66] | 9.44 [7.63, 11.12] | -6.35 [-7.17, -5.47] | 0.000 | 115 |
| `encoder_ihm_test` | 9.60 [7.72, 11.40] | 9.34 [7.62, 10.96] | -6.43 [-7.26, -5.60] | 0.000 | 104 |
| `decoder_ihm_test` | 11.47 [8.99, 14.14] | 10.54 [8.69, 12.28] | -4.56 [-5.71, -3.01] | 0.000 | 97 |

#### WER by duration

| System | <1s | 1-3s | >3s |
|---|---|---|---|
| `zeroshot_ihm_test` | 32.10 [29.61, 34.61] (n=4865) | 18.04 [15.52, 20.18] (n=4047) | 13.54 [11.75, 15.39] (n=3731) |
| `both_ihm_test` | 21.72 [20.34, 23.12] (n=4865) | 12.86 [10.95, 14.36] (n=4047) | 7.20 [5.48, 9.30] (n=3731) |
| `encoder_ihm_test` | 21.31 [20.12, 22.65] (n=4865) | 12.91 [11.01, 14.40] (n=4047) | 7.11 [5.46, 8.93] (n=3731) |
| `decoder_ihm_test` | 25.17 [23.38, 26.99] (n=4865) | 14.99 [12.60, 17.22] (n=4047) | 8.68 [6.37, 11.91] (n=3731) |

#### WER by overlap

| System | none | (0,0.3] | >0.3 |
|---|---|---|---|
| `zeroshot_ihm_test` | 11.82 [10.73, 13.07] (n=3969) | 13.51 [11.80, 15.17] (n=2075) | 23.88 [21.77, 25.35] (n=6599) |
| `both_ihm_test` | 6.54 [5.69, 7.45] (n=3969) | 7.72 [5.74, 10.47] (n=2075) | 15.62 [13.36, 17.17] (n=6599) |
| `encoder_ihm_test` | 6.44 [5.65, 7.28] (n=3969) | 7.28 [5.63, 9.32] (n=2075) | 15.95 [13.65, 17.67] (n=6599) |
| `decoder_ihm_test` | 7.41 [6.55, 8.34] (n=3969) | 8.88 [6.64, 11.61] (n=2075) | 19.22 [15.60, 22.64] (n=6599) |
<!-- END GENERATED: ihm_test -->

The gain appears in every pre-registered slice: duration under 1 s, 1–3 s and over 3 s, and every
overlap bucket.

- **Short segments.** Zero-shot WER under 1 s is about 100% on SDM, because insertions on short and
  filler-only segments exceed the reference length.
- **Heavy overlap.** Segments with more than 30% overlap remain the hardest after adaptation, at about
  45% WER on SDM.

## 5. Q2: encoder vs decoder, and the far-field penalty

### Arms compared with encoder-only LoRA (SDM test)

<!-- BEGIN GENERATED: sdm_test_arms_vs_encoder -->
Reference for paired deltas: `encoder_sdm_test`. 12643 utterances, 16 clusters, 95% meeting-level bootstrap CIs (2000 resamples).

| System | WER | Disfluency-insens. WER | ΔWER vs ref [CI] | p(not better) | Ins. on empty refs |
|---|---|---|---|---|---|
| `encoder_sdm_test` | 22.39 [17.91, 26.52] | 22.24 [17.72, 26.37] | — | — | 340 |
| `both_sdm_test` | 22.38 [17.98, 26.44] | 22.25 [17.85, 26.29] | -0.01 [-0.24, +0.26] | 0.476 | 334 |
| `decoder_sdm_test` | 27.59 [22.24, 32.71] | 26.11 [20.96, 30.75] | +5.20 [+3.90, +6.40] | 1.000 | 368 |

#### WER by duration

| System | <1s | 1-3s | >3s |
|---|---|---|---|
| `encoder_sdm_test` | 54.47 [51.64, 56.91] (n=4865) | 34.99 [30.86, 38.10] (n=4047) | 14.25 [10.97, 17.76] (n=3731) |
| `both_sdm_test` | 54.15 [51.65, 56.39] (n=4865) | 34.85 [30.71, 38.07] (n=4047) | 14.32 [10.99, 17.81] (n=3731) |
| `decoder_sdm_test` | 63.17 [60.20, 66.14] (n=4865) | 42.07 [37.06, 46.10] (n=4047) | 18.39 [14.16, 22.71] (n=3731) |

#### WER by overlap

| System | none | (0,0.3] | >0.3 |
|---|---|---|---|
| `encoder_sdm_test` | 11.92 [9.73, 14.34] (n=3969) | 13.09 [10.77, 15.51] (n=2075) | 45.11 [40.63, 48.60] (n=6599) |
| `both_sdm_test` | 11.96 [9.88, 14.33] (n=3969) | 12.81 [10.73, 14.89] (n=2075) | 45.33 [41.06, 48.99] (n=6599) |
| `decoder_sdm_test` | 14.18 [11.67, 16.95] (n=3969) | 15.59 [12.45, 18.73] (n=2075) | 56.80 [51.13, 61.71] (n=6599) |
<!-- END GENERATED: sdm_test_arms_vs_encoder -->

The encoder-only and encoder+decoder models are different systems that reach the same accuracy:

<!-- BEGIN GENERATED: arm_agreement -->
| SDM test: encoder-only vs encoder+decoder | Value |
|---|---|
| Normalized hypotheses that differ | 26.8% |
| Utterances where encoder-only has fewer / more errors | 1224 / 1214 |
| Total errors, encoder-only / encoder+decoder | 20050 / 20040 |
<!-- END GENERATED: arm_agreement -->

### Far-field penalty

<!-- BEGIN GENERATED: mic_gap -->
| System | WER IHM | WER SDM | Gap [95% CI] | Gap change vs ref [95% CI] | p(gap not smaller) |
|---|---|---|---|---|---|
| `zeroshot` | 16.03 | 36.25 | 20.22 [15.97, 24.01] | — | — |
| `both` | 9.68 | 22.38 | 12.70 [9.92, 15.29] | -7.52 [-9.28, -5.81] | 0.000 |
| `encoder` | 9.60 | 22.39 | 12.79 [9.89, 15.43] | -7.43 [-8.96, -5.88] | 0.000 |
| `decoder` | 11.47 | 27.59 | 16.12 [12.69, 19.23] | -4.10 [-5.68, -2.64] | 0.000 |

#### Post hoc: relative far-field penalty (SDM/IHM WER ratio)

_Not pre-registered. Same segments, so the ratio equals the ratio of total errors._

| System | SDM/IHM ratio [95% CI] | Ratio change vs ref [95% CI] | Relative WER change SDM / IHM |
|---|---|---|---|
| `zeroshot` | 2.26 [2.08, 2.41] | — | — |
| `both` | 2.31 [2.12, 2.54] | +0.05 [-0.15, +0.24] | -38.3% / -39.6% |
| `encoder` | 2.33 [2.15, 2.53] | +0.07 [-0.10, +0.24] | -38.2% / -40.1% |
| `decoder` | 2.40 [2.17, 2.69] | +0.14 [-0.11, +0.39] | -23.9% / -28.4% |
<!-- END GENERATED: mic_gap -->

The pre-registered H3 metric, the absolute gap SDM − IHM, shrinks by 7.4 to 7.5 points for the
encoder and encoder+decoder arms. The post hoc ratio does not change significantly: relative WER falls
by about 38–40% on both microphones. The data therefore show a broad meeting-domain improvement that
carries over to close-talk audio. They do not show that adaptation specifically closed the far-field
penalty. H3's prediction that the encoder reduces the gap more than the decoder holds in absolute
terms.

## 6. Q3: style vs content

The pre-registered disfluency-insensitive WER (fillers removed, immediate repetitions collapsed)
changes the gain only slightly (§4). A post hoc convention map goes further. It unifies spellings that
the normalizer leaves alone ("ok"/"okay", "cause"/"because", "dunno", "gonna", and spelled-out letters
such as "l c d"):

<!-- BEGIN GENERATED: posthoc_conventions_sdm_test -->
_Exploratory, not pre-registered. The convention map and the spelled-letter joining are applied identically to references and hypotheses, on top of the disfluency-insensitive rule._

| System | WER (standard) | WER (convention-insensitive) | Δ vs ref, standard | Δ vs ref, convention-insensitive |
|---|---|---|---|---|
| `zeroshot` | 36.25 [30.28, 41.54] | 34.26 [28.19, 39.65] | — | — |
| `both` | 22.38 [17.98, 26.44] | 21.88 [17.53, 25.85] | -13.87 [-15.35, -12.16] | -12.38 [-13.84, -10.66] |
| `encoder` | 22.39 [17.91, 26.52] | 21.97 [17.47, 26.06] | -13.86 [-15.23, -12.24] | -12.29 [-13.62, -10.70] |
| `decoder` | 27.59 [22.24, 32.71] | 25.74 [20.65, 30.30] | -8.66 [-9.63, -7.47] | -8.52 [-9.66, -7.22] |
<!-- END GENERATED: posthoc_conventions_sdm_test -->

Rule-captured style and conventions therefore account for at least about 10% of the gain. This is a
lower bound, because the rules miss:
- partial words, which this corpus release does not mark;
- discourse markers;
- other verbatim conventions;
- fillers that the normalizer already removes on both sides.

The encoder-only model with a frozen decoder emits AMI-style text: lowercase, no punctuation, and
"uh". The style that reaches the output is therefore not controlled by the decoder weights alone. The
adapted encoder steers the frozen decoder, which is consistent with Whisper conditioning its output
format on the audio. This is an observation, not a tested mechanism.

## 7. Q4: costs

### Formatting: cased and punctuated AMI test

<!-- BEGIN GENERATED: esb_ami -->
Reference for paired deltas: `zeroshot_esb_ami`. 12643 utterances, 16 clusters, 95% meeting-level bootstrap CIs (2000 resamples).

| System | WER | Disfluency-insens. WER | Orthographic WER | ΔWER vs ref [CI] | p(not better) | Ins. on empty refs |
|---|---|---|---|---|---|---|
| `zeroshot_esb_ami` | 16.03 [14.00, 17.92] | 14.63 [12.57, 16.50] | 30.39 [28.76, 31.74] | — | — | 326 |
| `both_esb_ami` | 9.68 [7.72, 11.66] | 9.44 [7.63, 11.12] | 35.53 [31.20, 39.57] | -6.35 [-7.16, -5.47] | 0.000 | 115 |
| `encoder_esb_ami` | 9.60 [7.73, 11.40] | 9.34 [7.62, 10.96] | 35.55 [31.22, 39.60] | -6.43 [-7.26, -5.60] | 0.000 | 104 |
| `decoder_esb_ami` | 11.47 [8.99, 14.14] | 10.54 [8.69, 12.28] | 37.35 [32.37, 42.19] | -4.56 [-5.71, -3.01] | 0.000 | 97 |

#### WER by duration

| System | <1s | 1-3s | >3s |
|---|---|---|---|
| `zeroshot_esb_ami` | 32.09 [29.61, 34.61] (n=4865) | 18.04 [15.52, 20.18] (n=4047) | 13.54 [11.75, 15.39] (n=3731) |
| `both_esb_ami` | 21.73 [20.36, 23.13] (n=4865) | 12.86 [10.95, 14.36] (n=4047) | 7.20 [5.48, 9.30] (n=3731) |
| `encoder_esb_ami` | 21.33 [20.14, 22.66] (n=4865) | 12.91 [11.01, 14.40] (n=4047) | 7.11 [5.46, 8.94] (n=3731) |
| `decoder_esb_ami` | 25.18 [23.40, 27.00] (n=4865) | 14.99 [12.60, 17.22] (n=4047) | 8.69 [6.37, 11.91] (n=3731) |
<!-- END GENERATED: esb_ami -->

<!-- BEGIN GENERATED: formatting_compute -->
| System | Hyps with punctuation (cased AMI test) | Hyps starting uppercase | RTF, SDM test (generate only, batch 16) | Peak inference VRAM |
|---|---|---|---|---|
| Zero-shot | 93.9% | 84.7% | 0.0284 | 2.39 GB |
| LoRA encoder+decoder | 2.5% | 0.0% | 0.0284 | 2.39 GB |
| LoRA encoder only | 2.4% | 0.0% | 0.0284 | 2.39 GB |
| LoRA decoder only | 2.4% | 0.0% | 0.0286 | 2.39 GB |
<!-- END GENERATED: formatting_compute -->

Training on AMI's uppercase, unpunctuated references removes casing and punctuation (table above). On
case- and punctuation-sensitive WER, every adapted model is worse than zero-shot, even though it is
about 40% better on normalized WER.

Practical mitigations, none of them tested here:
- training on orthographic references;
- a punctuation and truecasing stage;
- regularizing toward the base model's formatting.

### Out of domain: Earnings-22

<!-- BEGIN GENERATED: earnings22 -->
Reference for paired deltas: `zeroshot_earnings22`. 2000 utterances, 16 clusters, 95% meeting-level bootstrap CIs (2000 resamples).

| System | WER | Disfluency-insens. WER | Orthographic WER | ΔWER vs ref [CI] | p(not better) | Ins. on empty refs |
|---|---|---|---|---|---|---|
| `zeroshot_earnings22` | 14.94 [13.14, 17.20] | 13.79 [12.00, 15.91] | 26.42 [24.31, 28.66] | — | — | 8 |
| `both_earnings22` | 15.78 [13.51, 18.18] | 15.17 [13.12, 17.40] | 42.80 [40.67, 45.23] | +0.83 [-0.56, +2.38] | 0.864 | 7 |
| `encoder_earnings22` | 15.99 [13.62, 18.49] | 15.41 [13.21, 17.73] | 42.78 [40.73, 45.17] | +1.05 [-0.41, +2.57] | 0.919 | 6 |
| `decoder_earnings22` | 17.52 [14.63, 20.97] | 15.50 [13.32, 17.90] | 44.28 [41.71, 47.38] | +2.58 [+0.52, +4.59] | 0.991 | 6 |

#### WER by duration

| System | <1s | 1-3s | >3s |
|---|---|---|---|
| `zeroshot_earnings22` | 50.00 [41.81, 58.78] (n=120) | 23.47 [20.36, 26.50] (n=359) | 14.02 [12.21, 16.25] (n=1521) |
| `both_earnings22` | 49.02 [40.26, 58.05] (n=120) | 23.33 [19.75, 27.53] (n=359) | 14.94 [12.79, 17.31] (n=1521) |
| `encoder_earnings22` | 50.65 [43.13, 59.57] (n=120) | 22.95 [18.71, 27.51] (n=359) | 15.18 [12.95, 17.63] (n=1521) |
| `decoder_earnings22` | 57.84 [50.00, 66.22] (n=120) | 23.00 [19.66, 26.55] (n=359) | 16.75 [13.94, 20.21] (n=1521) |
<!-- END GENERATED: earnings22 -->

- For the encoder and encoder+decoder arms, no significant regression was detected; the CIs reach
  about +2.5.
- Decoder-only regresses significantly (+2.58 [+0.52, +4.59]), partly through repetition loops (below).
- Orthographic WER rises by 16 to 18 points, for the formatting reason above.

### Post hoc: repetition loops

<!-- BEGIN GENERATED: posthoc_loops_sdm_test -->
_Exploratory, not pre-registered. A loop is ≥5 identical tokens or bigrams in a row in the normalized hypothesis; output length is capped by `max_new_tokens=128`._

| System | Looping utterances | Insertions in them | Share of all errors |
|---|---|---|---|
| `zeroshot` | 1 | 60 | 0.2% |
| `both` | 6 | 123 | 0.6% |
| `encoder` | 9 | 130 | 0.6% |
| `decoder` | 30 | 1250 | 5.1% |
<!-- END GENERATED: posthoc_loops_sdm_test -->

<!-- BEGIN GENERATED: posthoc_loops_earnings22 -->
_Exploratory, not pre-registered. A loop is ≥5 identical tokens or bigrams in a row in the normalized hypothesis; output length is capped by `max_new_tokens=128`._

| System | Looping utterances | Insertions in them | Share of all errors |
|---|---|---|---|
| `zeroshot` | 0 | 0 | 0.0% |
| `both` | 4 | 14 | 0.3% |
| `encoder` | 3 | 8 | 0.1% |
| `decoder` | 14 | 484 | 8.2% |
<!-- END GENERATED: posthoc_loops_earnings22 -->

Adaptation makes rare, severe repetition loops more frequent, most of all in the decoder-only arm.
`max_new_tokens=128` caps the damage. A production decoder would add a compression-ratio fallback or a
repetition guard.

### Compute

- **Training.** 1.2 to 2 GPU-hours per arm on one 12 GB laptop GPU, peak 6.2 to 7.5 GB (§2).
- **Inference.** RTF is unchanged after merging, because the architecture does not change. The
  measured RTF covers batched `generate` only, excluding I/O (formatting table, §7); it is not a
  production latency figure.

## 8. Error analysis

<!-- BEGIN GENERATED: errors -->
| SDM test | Zero-shot | LoRA encoder only |
|---|---|---|
| Filler-only segments (empty reference after normalization) that get any output | 615 / 990 | 195 / 990 |
| Most frequent outputs on them | "thank you." ×151, "yeah." ×46, "you" ×42, "yes." ×15 | "yeah" ×74, "okay" ×16, "oh" ×9, "no" ×8 |
| Deletions (all) | 11451 | 5769 |
| Deletions of an immediately repeated reference word | 615 | 141 |
| Utterances with fewer / more errors than zero-shot | — | 5758 / 1241 |
<!-- END GENERATED: errors -->

- **Hallucinations.** The zero-shot model's hallucinations on filler-only segments are mostly
  "Thank you.". After adaptation, the outputs on these segments are backchannel-like ("yeah", "okay").
  Some of them may be real crosstalk that the reference does not attribute to that segment. This is
  why insertions on empty references are reported as a proxy.
- **Repetitions.** Deletions of verbatim repetitions fall sharply, which is consistent with learning
  the corpus's verbatim style.

## 9. Engineering notes

These problems would have biased or invalidated results without raising any error:

1. **Normalizer artifact** (amendment 3). A trailing "..." left a bare "." that was scored as an
   insertion, and the artifact favoured adapted models.
2. **Gradient checkpointing inactive in eval mode.** `from_pretrained` returns eval mode.
3. **Silent VRAM spill to system RAM under Windows WDDM.** Training ran up to 25× slower, with no
   error.
4. **Orphaned processes.** On Windows, Python kept running after its `uv run` wrapper exited, and two
   trainers ended up sharing the GPU.
5. **Resampling dependency.** Earnings-22 audio is 24 kHz. The missing resampler failed loudly and was
   fixed before those evaluations ran (amendment 4).

## 10. Limitations

- **Pre-training contamination.** Whisper's pre-training data is undisclosed and was not de-duplicated
  against AMI.
- **Statistical power.**
  - One seed per arm; the CIs reflect meeting sampling, not training variance.
  - 16 test meetings.
- **Scope.** Segmented utterances only: no long-form transcription, speaker attribution or streaming.
- **Arm design.** Arms are matched in data and steps, not in parameters.
- **Post hoc analyses.** The conventions, loops and ratio analyses are exploratory.
