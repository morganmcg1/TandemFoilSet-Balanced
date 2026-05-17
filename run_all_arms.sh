#!/bin/bash
# Sequential runner for PR #4216: LR sweep on 12-winner canonical.
# Runs 3 arms (1e-3, 5e-4, 2e-3) one at a time on a single GPU.

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
  local logfile="$LOG_DIR/${arm_id}_lr${lrtag}.log"
  echo "=== [$(date -Is)] Launching $arm_id (lr=$lr) -> $logfile ===" | tee -a "$LOG_DIR/runner.log"
  python train.py \
    --optimizer soap --precondition_frequency 5 \
    --lr "$lr" --warmup_epochs 3 \
    --huber_beta 0.01 \
    --surf_weight 10.0 --seed 42 --ema_decay 0.99 \
    --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
    --grad_clip 1.0 \
    --wandb_group lr-sweep-on-canonical \
    --wandb_name "willowpai2i48h3-fern/${wname}" \
    --agent willowpai2i48h3-fern \
    > "$logfile" 2>&1
  local rc=$?
  echo "=== [$(date -Is)] $arm_id finished rc=$rc ===" | tee -a "$LOG_DIR/runner.log"
  return $rc
}

run_arm arm1 1e-3 baseline-lr1e3
run_arm arm2 5e-4 variant-lr5e4
run_arm arm3 2e-3 variant-lr2e3

echo "=== [$(date -Is)] ALL ARMS DONE ===" | tee -a "$LOG_DIR/runner.log"
