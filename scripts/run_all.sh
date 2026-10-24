#!/usr/bin/env bash
# Full experiment queue on one GPU, in the order fixed in docs/design.md:
#   1. harness validation (small.en zero-shot on AMI test)
#   2. the three matched-budget LoRA arms
#   3. test-set and out-of-domain evaluation of the zero-shot model and of each arm (once each)
#   4. optional: full validation sets, if time allows
# Each step logs to outputs/queue.log and the queue continues if a step fails.
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
PY=.venv/Scripts/python.exe
EVAL="$PY -u scripts/evaluate.py"

run $EVAL --config configs/eval/smallen_ihm.yaml --run-id smallen_ihm_test
run $EVAL --config configs/eval/smallen_sdm.yaml --run-id smallen_sdm_test

for arm in both encoder decoder; do
  run $PY -u scripts/train_lora.py --config configs/train/turbo_lora_$arm.yaml
done

for sys in zeroshot both encoder decoder; do
  if [ "$sys" = zeroshot ]; then ad=null; else ad=outputs/runs/turbo_lora_$sys/best; fi
  run $EVAL --config configs/eval/turbo_sdm_dev.yaml --set split=test --set adapter=$ad --run-id ${sys}_sdm_test
  run $EVAL --config configs/eval/turbo_ihm_dev.yaml --set split=test --set adapter=$ad --run-id ${sys}_ihm_test
  run $EVAL --config configs/eval/turbo_esb_ami.yaml --set adapter=$ad --run-id ${sys}_esb_ami
  run $EVAL --config configs/eval/turbo_earnings22.yaml --set adapter=$ad --run-id ${sys}_earnings22
done

for sys in zeroshot both encoder decoder; do
  if [ "$sys" = zeroshot ]; then ad=null; else ad=outputs/runs/turbo_lora_$sys/best; fi
  run $EVAL --config configs/eval/turbo_sdm_dev.yaml --set adapter=$ad --run-id ${sys}_sdm_dev
  run $EVAL --config configs/eval/turbo_ihm_dev.yaml --set adapter=$ad --run-id ${sys}_ihm_dev
done
echo "[$(date '+%F %T')] QUEUE DONE" >> "$LOG"
