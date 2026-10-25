# Commit map (history rewrite on 2026-10-25)

Before publication, the local history was rewritten for two reasons: to drop session-link trailers from
commit messages, and to remove a local cache path from three smoke-test reports. File contents at HEAD
are unchanged.

Each run recorded the commit it ran on, in `provenance.git_commit` inside `run.json` or
`train_run.json`. Those are pre-rewrite SHAs. The table below maps them to the published commits; order
and subjects are the same.

| Pre-rewrite | Published | Subject | Runs recorded on it |
|---|---|---|---|
| `d24e39e` | `6e83c63` | Add evaluation harness, LoRA training script and pre-registered design | — |
| `e2930aa` | `290dc41` | Ignore untracked files when recording git_dirty in run provenance | — |
| `91d0560` | `b5076b0` | Record throughput pilot and fix N=16000 (design amendment 2) | — |
| `20df359` | `7807e6a` | Call the project venv interpreter directly in the queue | `smallen_ihm_test`, `smallen_sdm_test`, `turbo_lora_both` |
| `61934ee` | `307adab` | Drop punctuation-only tokens after normalization; add rescore and error analysis | — |
| `aaee128` | `4a9a973` | README with harness validation; add smoke-test and validation results | — |
| `f6e1fca` | `b6c2ef9` | Add paired SDM-IHM far-field penalty with difference-in-differences bootstrap | — |
| `409acb2` | `642a366` | Add L-both training log (best dev-select WER 19.45 at step 1000) | `turbo_lora_encoder` |
| `856935f` | `08df610` | Add report skeleton (methods and harness validation; results pending) | — |
| `7bb2757` | `7a1e52c` | Add L-encoder training log (best dev-select WER 19.36 at step 1000) | `turbo_lora_decoder` |
| `106f2c3` | `da8e716` | Add L-decoder training log (best dev-select WER 24.46 at step 750) | — |
| `90f4b63` | `dc2f944` | Make the experiment queue resumable (skip finished steps) | — |
| `becdbad` | `ccc0703` | Run evaluations set by set so SDM test completes first | `both_sdm_test`, `encoder_sdm_test`, `zeroshot_sdm_test` |
| `a70c0f9` | `4449fe1` | Add post hoc convention-insensitive WER analysis (exploratory, labelled as such) | `decoder_sdm_test`, `zeroshot_ihm_test` |
| `1aa83dd` | `2ab8b98` | SDM test results for zero-shot and three LoRA arms; summaries | — |
| `49f1ca9` | `7f6cf7e` | Add post hoc repetition-loop analysis and SDM error-analysis summaries | `both_ihm_test`, `decoder_ihm_test`, `encoder_ihm_test` |
| `eb465df` | `589bd38` | IHM test results for zero-shot, L-both, L-encoder | `zeroshot_esb_ami` |
| `1d341e4` | `5dbbf93` | IHM test for L-decoder; far-field gap (difference-in-differences) and IHM summaries | `both_esb_ami`, `decoder_esb_ami`, `encoder_esb_ami` |
| `8c5f812` | `a17a8c9` | Add scipy (Earnings-22 is 24 kHz and needs resampling); ESB AMI orthographic results | `both_earnings22`, `decoder_earnings22`, `encoder_earnings22`, `zeroshot_earnings22`, `zeroshot_sdm_dev` |
| `d70adb0` | `2a74324` | Earnings-22 (out-of-domain) results for all systems | `both_sdm_dev`, `zeroshot_ihm_dev` |
| `e183afa` | `ecee0fd` | Optional zero-shot SDM validation run | — |
| `8cbf8c6` | `2c005c6` | Optional validation runs: zero-shot IHM, L-both SDM | — |
| `c078f21` | `60e0629` | Add post hoc SDM/IHM ratio to far-field analysis; strip local cache paths from smoke reports | — |

Unmapped recorded commits: none
