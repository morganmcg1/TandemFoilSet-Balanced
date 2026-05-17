#!/bin/bash
# Queue Arm B (T_max=16) after Arm A (PID $1) finishes.
set -u
ARMA_PID="${1:-72570}"
cd /workspace/senpai/target

echo "[$(date '+%H:%M:%S')] Waiting for Arm A PID=$ARMA_PID to finish..."
while kill -0 "$ARMA_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(date '+%H:%M:%S')] Arm A finished. Launching Arm B (T_max=16)."

mkdir -p logs
python train.py --agent willowpai2i48h5-thorfinn --epochs 50 \
  --wandb_group round12-tmax-newbl-thorfinn \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 0 --cosine_t_max 16 \
  --optimizer_name lion --lr 2e-4 --weight_decay 1e-3 \
  --ema_decay 0.997 --use_film \
  --layer_scale_init 1e-4 \
  --grad_clip 1.0 \
  --wandb_name thorfinn-r12-tmax16-newbl > logs/armB_tmax16.log 2>&1 &
ARMB_PID=$!
echo "$ARMB_PID" > logs/armB_tmax16.pid
echo "[$(date '+%H:%M:%S')] Arm B PID=$ARMB_PID"
wait "$ARMB_PID"
echo "[$(date '+%H:%M:%S')] Arm B finished."
