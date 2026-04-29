#!/bin/bash
# Wave 3 second batch — launches when wave 1 GPUs (0, 3, 4, 6, 7) free up.
# Run this after wave 1 finishes; it will overwrite GPU assignments.
set -e
cd "$(dirname "$0")/.."
mkdir -p session_logs

GROUP="mlintern-pai2-24h-v2-r1"
AGENT="ml-intern-r1"
WC="--warmup_frac 0.05 --grad_clip 1.0"
LR="--lr 1e-3"

start_run() {
  local gpu="$1"; shift
  local name="$1"; shift
  local logfile="session_logs/wave3-gpu${gpu}-${name}.log"
  echo "[$(date +%H:%M:%S)] Launching GPU $gpu: $name → $logfile"
  CUDA_VISIBLE_DEVICES="$gpu" SENPAI_TIMEOUT_MINUTES=720 \
    nohup python train.py \
      --agent "$AGENT" \
      --wandb_group "$GROUP" \
      --wandb_name "${GROUP}/${name}" \
      "$@" > "$logfile" 2>&1 &
  echo $! > "session_logs/wave3-gpu${gpu}-${name}.pid"
}

# GPU 0: best small + 100 epochs (no extras, single seed = control).
start_run 0 wc-lr1e3-100ep-seed0 \
  --epochs 100 $WC $LR --seed 0

# GPU 3: best small + Fourier + surf_weight=50 (combined extras).
start_run 3 wc-lr1e3-sw50-fourier-100ep-seed0 \
  --epochs 100 $WC $LR --seed 0 \
  --fourier_freq 16 --fourier_scale 5.0 \
  --surf_weight 50

# GPU 4: bigger model + sub32k + warmup-clip recipe (no Fourier).
start_run 4 wc-lr1e3-sub32k-h256-l8-mlp4-100ep \
  --epochs 100 $WC $LR \
  --subsample 32000 --surf_oversample 0.3 \
  --n_hidden 256 --n_layers 8 --n_head 8 --mlp_ratio 4

# GPU 6: bigger model + sub32k + Fourier + surf_weight=50 (full combo).
start_run 6 wc-lr1e3-sub32k-h256-l8-mlp4-fourier-sw50-100ep \
  --epochs 100 $WC $LR \
  --subsample 32000 --surf_oversample 0.3 \
  --n_hidden 256 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --fourier_freq 16 --fourier_scale 5.0 \
  --surf_weight 50

# GPU 7: best small + multi-seed (variance check).
start_run 7 wc-lr1e3-100ep-seed1 \
  --epochs 100 $WC $LR --seed 1

echo "Wave 3 second batch launched."
