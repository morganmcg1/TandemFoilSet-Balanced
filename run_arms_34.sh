#!/bin/bash
# Sequential runner for PR #4216 arms 3 + optional arm 4 (lr=2e-3, 3e-3) with --use_bf16.
# Launched after arm 2 finishes; uses the new canonical with bf16.

set -u
cd /workspace/senpai/target

LOG_DIR=logs
mkdir -p "$LOG_DIR"

run_arm() {
  local arm_id="$1"
  local lr="$2"
  local wname="$3"
  local lrtag
  lrtag="${lr//./}"
  lrtag="${lrtag//-/}"
  local logfile="$LOG_DIR/${arm_id}_lr${lrtag}_bf16.log"
  echo "=== [$(date -Is)] Launching $arm_id (lr=$lr, bf16) -> $logfile ===" | tee -a "$LOG_DIR/runner.log"
  python train.py \
    --optimizer soap --precondition_frequency 5 \
    --lr "$lr" --warmup_epochs 3 \
    --huber_beta 0.01 \
    --surf_weight 10.0 --seed 42 --ema_decay 0.99 \
    --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
    --grad_clip 1.0 \
    --use_bf16 \
    --wandb_group lr-sweep-on-canonical \
    --wandb_name "willowpai2i48h3-fern/${wname}" \
    --agent willowpai2i48h3-fern \
    > "$logfile" 2>&1
  local rc=$?
  echo "=== [$(date -Is)] $arm_id finished rc=$rc ===" | tee -a "$LOG_DIR/runner.log"
  return $rc
}

run_arm arm3 2e-3 variant-lr2e3-bf16
run_arm arm4 3e-3 variant-lr3e3-bf16

echo "=== [$(date -Is)] ARMS 3+4 DONE ===" | tee -a "$LOG_DIR/runner.log"
