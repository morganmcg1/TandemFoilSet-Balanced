#!/bin/bash
# Wave 1: explore key levers in parallel on 8 GPUs.
# Each run: 50 epochs ≈ 2h.
set -e
cd "$(dirname "$0")/.."
mkdir -p session_logs

GROUP="mlintern-pai2-24h-v2-r1"
AGENT="ml-intern-r1"
EPOCHS=50

start_run() {
  local gpu="$1"; shift
  local name="$1"; shift
  local logfile="session_logs/wave1-gpu${gpu}-${name}.log"
  echo "[$(date +%H:%M:%S)] Launching GPU $gpu: $name → $logfile"
  CUDA_VISIBLE_DEVICES="$gpu" SENPAI_TIMEOUT_MINUTES=720 \
    nohup python train.py \
      --epochs "$EPOCHS" \
      --agent "$AGENT" \
      --wandb_group "$GROUP" \
      --wandb_name "${GROUP}/${name}" \
      "$@" > "$logfile" 2>&1 &
  echo $! > "session_logs/wave1-gpu${gpu}-${name}.pid"
}

# GPU 0: control — vanilla Transolver default config (with my mask=True correction).
start_run 0 baseline-default

# GPU 1: Transolver++ on default (Ada-Temp + Rep-Slice).
start_run 1 default-tpp \
  --use_ada_temp True --use_rep_slice True

# GPU 2: Pressure-weighted surface loss (10x extra on p channel inside surf_loss).
start_run 2 default-pw10 \
  --surf_p_weight 10.0

# GPU 3: Fourier features on (x, z) coordinates.
start_run 3 default-fourier16 \
  --fourier_freq 16 --fourier_scale 10.0

# GPU 4: Bigger model (n_hidden=192, n_layers=8, n_head=8, mlp_ratio=4).
start_run 4 mid-h192-l8-mlp4 \
  --n_hidden 192 --n_layers 8 --n_head 8 --mlp_ratio 4

# GPU 5: Bigger model with Transolver++ + bf16 to ease memory.
start_run 5 mid-h192-l8-mlp4-tpp-bf16 \
  --n_hidden 192 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --use_ada_temp True --use_rep_slice True \
  --amp bf16

# GPU 6: Bigger + everything combined (TPP + pressure weight + fourier) bf16.
start_run 6 mid-h192-l8-mlp4-tpp-pw10-fourier-bf16 \
  --n_hidden 192 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --use_ada_temp True --use_rep_slice True \
  --surf_p_weight 10.0 \
  --fourier_freq 16 --fourier_scale 10.0 \
  --amp bf16

# GPU 7: Big bf16 model — n_hidden=256, n_layers=8, n_head=8, mlp_ratio=4 + TPP.
start_run 7 big-h256-l8-mlp4-tpp-bf16 \
  --n_hidden 256 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --use_ada_temp True --use_rep_slice True \
  --amp bf16

echo
echo "All 8 jobs launched. Monitor with:"
echo "  tail -f session_logs/wave1-gpu*-*.log"
echo "  nvidia-smi"
