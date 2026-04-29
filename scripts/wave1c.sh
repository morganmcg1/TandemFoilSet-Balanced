#!/bin/bash
# Wave 1c: relaunch killed/NaN'd runs with TPP fix + add some training-recipe variants.
set -e
cd "$(dirname "$0")/.."
mkdir -p session_logs

GROUP="mlintern-pai2-24h-v2-r1"
AGENT="ml-intern-r1"
EPOCHS=50

start_run() {
  local gpu="$1"; shift
  local name="$1"; shift
  local logfile="session_logs/wave1c-gpu${gpu}-${name}.log"
  echo "[$(date +%H:%M:%S)] Launching GPU $gpu: $name → $logfile"
  CUDA_VISIBLE_DEVICES="$gpu" SENPAI_TIMEOUT_MINUTES=720 \
    nohup python train.py \
      --epochs "$EPOCHS" \
      --agent "$AGENT" \
      --wandb_group "$GROUP" \
      --wandb_name "${GROUP}/${name}" \
      "$@" > "$logfile" 2>&1 &
  echo $! > "session_logs/wave1c-gpu${gpu}-${name}.pid"
}

# GPU 1: default + Transolver++ (Ada-Temp + Rep-Slice), with NaN-safe clamps.
start_run 1 default-tpp-fix \
  --use_ada_temp True --use_rep_slice True

# GPU 2: default + pressure weight (re-do).
start_run 2 default-pw10-fix \
  --surf_p_weight 10.0

# GPU 5: default + TPP + pressure weight + Fourier (combined small model).
start_run 5 default-tpp-pw10-fourier \
  --use_ada_temp True --use_rep_slice True \
  --surf_p_weight 10.0 \
  --fourier_freq 16 --fourier_scale 10.0

# GPU 6: default + higher LR (1e-3 like Transolver paper) + warmup + grad clip.
start_run 6 default-lr1e3-warmup-clip \
  --lr 1e-3 --warmup_frac 0.05 --grad_clip 1.0

# GPU 7: default + warmup + grad clip + LR 5e-4 (control for the LR effect).
start_run 7 default-warmup-clip \
  --warmup_frac 0.05 --grad_clip 1.0

echo "Wave 1c launched."
