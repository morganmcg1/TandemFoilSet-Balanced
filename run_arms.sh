#!/bin/bash
set -e
cd /workspace/senpai/target

common_args=(
  --agent willowpai2i48h5-edward
  --epochs 50
  --wandb_group round7-huber-beta-edward
  --loss_type smooth_l1
  --n_fourier 16 --fourier_sigma 10.0
  --cosine_t_max 14
  --optimizer_name lion --lr 5e-5 --weight_decay 1e-3
  --ema_decay 0.997
  --use_film
)

mkdir -p logs

echo "=== Arm A (beta=0.05 control) starting $(date) ==="
python train.py "${common_args[@]}" --loss_beta 0.05 \
  --wandb_name edward-r7-huber-arm-A-beta005 2>&1 | tee logs/arm_A.log
echo "=== Arm A done $(date) ==="

echo "=== Arm B (beta=0.10) starting $(date) ==="
python train.py "${common_args[@]}" --loss_beta 0.10 \
  --wandb_name edward-r7-huber-arm-B-beta010 2>&1 | tee logs/arm_B.log
echo "=== Arm B done $(date) ==="

echo "=== Arm C (beta=0.20) starting $(date) ==="
python train.py "${common_args[@]}" --loss_beta 0.20 \
  --wandb_name edward-r7-huber-arm-C-beta020 2>&1 | tee logs/arm_C.log
echo "=== Arm C done $(date) ==="
echo "=== ALL ARMS COMPLETE $(date) ==="
