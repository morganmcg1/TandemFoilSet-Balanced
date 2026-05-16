#!/usr/bin/env bash
# PR #4099 — grad-clip lower bound, NEW CANONICAL β=0.01 stack.
# Tests clip=0.5 and clip=0.1 against the new clip=1.0+β=0.01 baseline (val=45.9199, fern PR #4037).
set -u
cd "$(dirname "$0")"
mkdir -p logs

# Arm A — β=0.01 + clip=0.5
echo "[$(date -u +%H:%M:%S)] Starting Arm A: variant-clip05-huber001" | tee -a logs/run_arms_huber001_grad_clip_lower_bound_driver.log
CUDA_VISIBLE_DEVICES=0 python train.py \
  --optimizer soap --precondition_frequency 5 \
  --lr 1e-3 --warmup_epochs 3 \
  --huber_beta 0.01 \
  --surf_weight 10.0 --seed 42 --ema_decay 0.99 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --grad_clip 0.5 \
  --wandb_group grad-clip-lower-bound \
  --wandb_name willowpai2i48h3-tanjiro/variant-clip05-huber001 \
  --agent willowpai2i48h3-tanjiro \
  >> logs/arm_clip05_huber001.log 2>&1
armA_rc=$?
echo "[$(date -u +%H:%M:%S)] Arm A finished rc=$armA_rc" | tee -a logs/run_arms_huber001_grad_clip_lower_bound_driver.log

# Arm B — β=0.01 + clip=0.1
echo "[$(date -u +%H:%M:%S)] Starting Arm B: variant-clip01-huber001" | tee -a logs/run_arms_huber001_grad_clip_lower_bound_driver.log
CUDA_VISIBLE_DEVICES=0 python train.py \
  --optimizer soap --precondition_frequency 5 \
  --lr 1e-3 --warmup_epochs 3 \
  --huber_beta 0.01 \
  --surf_weight 10.0 --seed 42 --ema_decay 0.99 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --grad_clip 0.1 \
  --wandb_group grad-clip-lower-bound \
  --wandb_name willowpai2i48h3-tanjiro/variant-clip01-huber001 \
  --agent willowpai2i48h3-tanjiro \
  >> logs/arm_clip01_huber001.log 2>&1
armB_rc=$?
echo "[$(date -u +%H:%M:%S)] Arm B finished rc=$armB_rc" | tee -a logs/run_arms_huber001_grad_clip_lower_bound_driver.log

echo "[$(date -u +%H:%M:%S)] ALL DONE armA=$armA_rc armB=$armB_rc" | tee -a logs/run_arms_huber001_grad_clip_lower_bound_driver.log
