#!/bin/bash
# Wave 3: build on the warmup-clip + lr=1e-3 win.
#
# Best run so far: default-lr1e3-warmup-clip ep22 = 85.07
# Best small config: --lr 1e-3 --warmup_frac 0.05 --grad_clip 1.0
#
# Goals:
#   1. Confirm warmup-clip + lr=1e-3 is robust → multi-seed.
#   2. Compose with Fourier features.
#   3. Combine with subsampling+bigger model to test capacity ceiling.
#   4. Try surf_weight=50 (paper-recommended ~|V|/|S| ≈ 50).
#   5. Push to 100 epochs on the best small config.
set -e
cd "$(dirname "$0")/.."
mkdir -p session_logs

GROUP="mlintern-pai2-24h-v2-r1"
AGENT="ml-intern-r1"

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

# Common best-small recipe: --lr 1e-3 --warmup_frac 0.05 --grad_clip 1.0
WC="--warmup_frac 0.05 --grad_clip 1.0"
LR="--lr 1e-3"

# GPU 0: best-small + 100 epochs (extend past 50, see if cosine continues to gain).
start_run 0 wc-lr1e3-100ep-seed0 \
  --epochs 100 $WC $LR --seed 0

# GPU 1: best-small + 100 epochs + Fourier features.
start_run 1 wc-lr1e3-fourier-100ep-seed0 \
  --epochs 100 $WC $LR --seed 0 \
  --fourier_freq 16 --fourier_scale 5.0

# GPU 2: best-small + surf_weight=50 (per AirfRANS / DoMINO).
start_run 2 wc-lr1e3-sw50-100ep-seed0 \
  --epochs 100 $WC $LR --seed 0 \
  --surf_weight 50

# GPU 3: best-small + Fourier + surf_weight=50.
start_run 3 wc-lr1e3-sw50-fourier-100ep-seed0 \
  --epochs 100 $WC $LR --seed 0 \
  --fourier_freq 16 --fourier_scale 5.0 \
  --surf_weight 50

# GPU 4: bigger model (h256/l8/mlp4) + sub32k + warmup-clip recipe.
# surf_oversample 0.3 (closer to eval distribution's 2% than 0.5).
start_run 4 wc-lr1e3-sub32k-h256-l8-mlp4-100ep \
  --epochs 100 $WC $LR \
  --subsample 32000 --surf_oversample 0.3 \
  --n_hidden 256 --n_layers 8 --n_head 8 --mlp_ratio 4

# GPU 5: bigger model + Fourier.
start_run 5 wc-lr1e3-sub32k-h256-l8-mlp4-fourier-100ep \
  --epochs 100 $WC $LR \
  --subsample 32000 --surf_oversample 0.3 \
  --n_hidden 256 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --fourier_freq 16 --fourier_scale 5.0

# GPU 6: bigger model + Fourier + surf_weight=50.
start_run 6 wc-lr1e3-sub32k-h256-l8-mlp4-fourier-sw50-100ep \
  --epochs 100 $WC $LR \
  --subsample 32000 --surf_oversample 0.3 \
  --n_hidden 256 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --fourier_freq 16 --fourier_scale 5.0 \
  --surf_weight 50

# GPU 7: best-small + multi-seed (variance check).
start_run 7 wc-lr1e3-100ep-seed1 \
  --epochs 100 $WC $LR --seed 1

echo "Wave 3 launched."
