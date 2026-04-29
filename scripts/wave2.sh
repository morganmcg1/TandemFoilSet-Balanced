#!/bin/bash
# Wave 2: bigger Transolver via training-time subsampling + Fourier PE.
#
# Logic:
#   - Subsampling 32K nodes per sample (vs full mesh ~140K-240K) reduces
#     activation memory ~7x. This lets us run n_hidden=256, n_layers=10,
#     mlp_ratio=4 comfortably with batch_size=4.
#   - Eval still runs on the full mesh so the metric is identical.
#   - Surface oversampling (50% of subsample budget) ensures plenty of
#     surface gradient signal even with random subsampling.
#
# Each run: 50 epochs. With subsampling, each epoch should be ~50% of the
# baseline epoch time, so ~70 min per run, leaving plenty of buffer.
set -e
cd "$(dirname "$0")/.."
mkdir -p session_logs

GROUP="mlintern-pai2-24h-v2-r1"
AGENT="ml-intern-r1"
EPOCHS=50

start_run() {
  local gpu="$1"; shift
  local name="$1"; shift
  local logfile="session_logs/wave2-gpu${gpu}-${name}.log"
  echo "[$(date +%H:%M:%S)] Launching GPU $gpu: $name → $logfile"
  CUDA_VISIBLE_DEVICES="$gpu" SENPAI_TIMEOUT_MINUTES=720 \
    nohup python train.py \
      --epochs "$EPOCHS" \
      --agent "$AGENT" \
      --wandb_group "$GROUP" \
      --wandb_name "${GROUP}/${name}" \
      "$@" > "$logfile" 2>&1 &
  echo $! > "session_logs/wave2-gpu${gpu}-${name}.pid"
}

# GPU 0: subsample baseline (controls for train/eval distribution shift).
start_run 0 sub32k-default \
  --subsample 32000 --surf_oversample 0.5

# GPU 1: subsample baseline + Fourier (small-model fourier control with sub).
start_run 1 sub32k-default-fourier \
  --subsample 32000 --surf_oversample 0.5 \
  --fourier_freq 16 --fourier_scale 5.0

# GPU 2: subsample h256/l8/mlp4 (mid model, no extras).
start_run 2 sub32k-h256-l8-mlp4 \
  --subsample 32000 --surf_oversample 0.5 \
  --n_hidden 256 --n_layers 8 --n_head 8 --mlp_ratio 4

# GPU 3: subsample h256/l8/mlp4 + Fourier.
start_run 3 sub32k-h256-l8-mlp4-fourier \
  --subsample 32000 --surf_oversample 0.5 \
  --n_hidden 256 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --fourier_freq 16 --fourier_scale 5.0

# GPU 4: subsample h256/l8/mlp4 + Fourier + Transolver++ (Ada-Temp + Rep-Slice).
start_run 4 sub32k-h256-l8-mlp4-fourier-tpp \
  --subsample 32000 --surf_oversample 0.5 \
  --n_hidden 256 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --fourier_freq 16 --fourier_scale 5.0 \
  --use_ada_temp True --use_rep_slice True

# GPU 5: subsample h256/l8/mlp4 + Fourier + Ada-Temp ONLY (no Rep-Slice).
# Test whether the noise from Rep-Slice is helpful or just adds variance.
start_run 5 sub32k-h256-l8-mlp4-fourier-ada \
  --subsample 32000 --surf_oversample 0.5 \
  --n_hidden 256 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --fourier_freq 16 --fourier_scale 5.0 \
  --use_ada_temp True

# GPU 6: deeper model — n_hidden=256, n_layers=12, mlp_ratio=4.
start_run 6 sub32k-h256-l12-mlp4-fourier-ada \
  --subsample 32000 --surf_oversample 0.5 \
  --n_hidden 256 --n_layers 12 --n_head 8 --mlp_ratio 4 \
  --fourier_freq 16 --fourier_scale 5.0 \
  --use_ada_temp True

# GPU 7: wider model — n_hidden=384, n_layers=8, mlp_ratio=4.
start_run 7 sub32k-h384-l8-mlp4-fourier-ada \
  --subsample 32000 --surf_oversample 0.5 \
  --n_hidden 384 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --fourier_freq 16 --fourier_scale 5.0 \
  --use_ada_temp True

echo "Wave 2 launched."
