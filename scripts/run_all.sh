#!/usr/bin/env bash
# Full experiment queue on one GPU, in the order fixed in docs/design.md:
#   1. harness validation (small.en zero-shot on AMI test)
#   2. the three matched-budget LoRA arms
#   3. test-set and out-of-domain evaluation of the zero-shot model and of each arm (once each)
#   4. optional: full validation sets, if time allows
# Each step logs to outputs/queue.log and the queue continues if a step fails.
# Resumable: steps whose outputs already exist are skipped, so re-running continues the queue.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/queue.log
mkdir -p outputs
run() {
  echo "[$(date '+%F %T')] START $*" >> "$LOG"
  PYTHONIOENCODING=utf-8 "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END rc=$rc $*" >> "$LOG"
}
skip() { echo "[$(date '+%F %T')] SKIP (done) $1" >> "$LOG"; }
eval_done() { [ -f "results/$1/metrics.json" ]; }
train_done() { [ -d "outputs/runs/turbo_lora_$1/last" ]; }
PY=.venv/Scripts/python.exe
EVAL="$PY -u scripts/evaluate.py"

eval_done smallen_ihm_test && skip smallen_ihm_test || run $EVAL --config configs/eval/smallen_ihm.yaml --run-id smallen_ihm_test
eval_done smallen_sdm_test && skip smallen_sdm_test || run $EVAL --config configs/eval/smallen_sdm.yaml --run-id smallen_sdm_test

for arm in both encoder decoder; do
  train_done $arm && skip "train $arm" || run $PY -u scripts/train_lora.py --config configs/train/turbo_lora_$arm.yaml
done

adapter_for() { if [ "$1" = zeroshot ]; then echo null; else echo outputs/runs/turbo_lora_$1/best; fi; }
# Evaluation set by evaluation set, so the primary comparison (SDM test) is complete first.
# Each evaluation is independent, so the order does not affect any result.
for set in sdm_test ihm_test esb_ami earnings22; do
  case $set in
    sdm_test)   cfg="configs/eval/turbo_sdm_dev.yaml --set split=test" ;;
    ihm_test)   cfg="configs/eval/turbo_ihm_dev.yaml --set split=test" ;;
    esb_ami)    cfg="configs/eval/turbo_esb_ami.yaml" ;;
    earnings22) cfg="configs/eval/turbo_earnings22.yaml" ;;
  esac
  for sys in zeroshot both encoder decoder; do
    eval_done ${sys}_${set} && skip ${sys}_${set} || \
      run $EVAL --config $cfg --set adapter=$(adapter_for $sys) --run-id ${sys}_${set}
  done
done
echo "[$(date '+%F %T')] CORE DONE" >> "$LOG"

for sys in zeroshot both encoder decoder; do
  if [ "$sys" = zeroshot ]; then ad=null; else ad=outputs/runs/turbo_lora_$sys/best; fi
  eval_done ${sys}_sdm_dev && skip ${sys}_sdm_dev || \
    run $EVAL --config configs/eval/turbo_sdm_dev.yaml --set adapter=$ad --run-id ${sys}_sdm_dev
  eval_done ${sys}_ihm_dev && skip ${sys}_ihm_dev || \
    run $EVAL --config configs/eval/turbo_ihm_dev.yaml --set adapter=$ad --run-id ${sys}_ihm_dev
done
echo "[$(date '+%F %T')] QUEUE DONE" >> "$LOG"
