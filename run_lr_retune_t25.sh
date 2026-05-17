#!/bin/bash
# PR #4336: LR re-tune on T_max=25 canonical — Arms 1 (lr=1.5e-3) and 2 (lr=2e-3)
set -u
cd /workspace/senpai/target
mkdir -p logs/lr-retune-cosine-t25

ARM1_LOG=logs/lr-retune-cosine-t25/arm1-lr15e3-tmax25.log
ARM2_LOG=logs/lr-retune-cosine-t25/arm2-lr2e3-tmax25.log

echo "[$(date -u +%FT%TZ)] Launching Arm 1 (lr=1.5e-3, T_max=25)..."
python train.py \
  --optimizer soap --precondition_frequency 5 \
  --lr 1.5e-3 --warmup_epochs 3 \
  --huber_beta 0.01 \
  --surf_weight 10.0 --seed 42 --ema_decay 0.99 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --grad_clip 1.0 \
  --use_bf16 \
  --cosine_t_max 25 \
  --agent willowpai2i48h3-tanjiro \
  --wandb_group lr-retune-cosine-t25 \
  --wandb_name willowpai2i48h3-tanjiro/variant-lr15e3-tmax25 \
  >"$ARM1_LOG" 2>&1
ec1=$?
echo "[$(date -u +%FT%TZ)] Arm 1 exit=$ec1"
grep -E "Best val|TEST  avg_surf_p|test_single_in_dist|test_geom_camber_rc|test_geom_camber_cruise|test_re_rand|Peak|Cosine T_max|Timeout" "$ARM1_LOG" | tail -30

echo "[$(date -u +%FT%TZ)] Launching Arm 2 (lr=2e-3, T_max=25)..."
python train.py \
  --optimizer soap --precondition_frequency 5 \
  --lr 2e-3 --warmup_epochs 3 \
  --huber_beta 0.01 \
  --surf_weight 10.0 --seed 42 --ema_decay 0.99 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --grad_clip 1.0 \
  --use_bf16 \
  --cosine_t_max 25 \
  --agent willowpai2i48h3-tanjiro \
  --wandb_group lr-retune-cosine-t25 \
  --wandb_name willowpai2i48h3-tanjiro/variant-lr2e3-tmax25 \
  >"$ARM2_LOG" 2>&1
ec2=$?
echo "[$(date -u +%FT%TZ)] Arm 2 exit=$ec2"
grep -E "Best val|TEST  avg_surf_p|test_single_in_dist|test_geom_camber_rc|test_geom_camber_cruise|test_re_rand|Peak|Cosine T_max|Timeout" "$ARM2_LOG" | tail -30

echo "[$(date -u +%FT%TZ)] Both arms done. Exit codes: arm1=$ec1 arm2=$ec2"
