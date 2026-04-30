#!/bin/bash
# Wave 4: launched when wave 3 finishes (~01:00 UTC).
#
# What we know from waves 1-3:
#   - default model + warmup_frac=0.05 + grad_clip=1.0 + lr=1e-3 → val 51 / test 44
#   - The recipe is robust; warmup-clip without lr=1e-3 (default 5e-4) → val 54 / test 47
#   - Bigger model (sub32k+h256+l8+mlp4) on 100 epochs converges slower; ep46≈64
#   - 100-epoch schedules of small model are still in early phase at ep10-13
#
# Wave 4 priorities:
#   1. Multi-seed of the proven 50-epoch best (seeds 2, 3, 4) — variance + ensemble.
#   2. Best 50ep config + Fourier (with the same warmup-clip recipe).
#   3. 200-epoch run of best small config (seed 0) on one GPU.
#   4. Lion optimizer exploration on best small recipe (lr=5e-5).
#   5. Mid-size model (h160, l6, mlp3) without subsampling — capacity bump.
#   6. Reserve 1 GPU for a final ensemble re-evaluation script.
set -e
cd "$(dirname "$0")/.."
mkdir -p session_logs

GROUP="mlintern-pai2-24h-v2-r1"
AGENT="ml-intern-r1"
WC="--warmup_frac 0.05 --grad_clip 1.0"
LR1E3="--lr 1e-3"

start_run() {
  local gpu="$1"; shift
  local name="$1"; shift
  local logfile="session_logs/wave4-gpu${gpu}-${name}.log"
  echo "[$(date +%H:%M:%S)] Launching GPU $gpu: $name → $logfile"
  CUDA_VISIBLE_DEVICES="$gpu" SENPAI_TIMEOUT_MINUTES=720 \
    nohup python train.py \
      --agent "$AGENT" \
      --wandb_group "$GROUP" \
      --wandb_name "${GROUP}/${name}" \
      "$@" > "$logfile" 2>&1 &
  echo $! > "session_logs/wave4-gpu${gpu}-${name}.pid"
}

# ---------- Multi-seed of proven 50-ep best (variance + ensemble) ----------
start_run 0 wc-lr1e3-50ep-seed2 \
  --epochs 50 $WC $LR1E3 --seed 2

start_run 1 wc-lr1e3-50ep-seed3 \
  --epochs 50 $WC $LR1E3 --seed 3

start_run 2 wc-lr1e3-50ep-seed4 \
  --epochs 50 $WC $LR1E3 --seed 4

# ---------- 50-ep best + Fourier features ----------
start_run 3 wc-lr1e3-50ep-fourier-seed0 \
  --epochs 50 $WC $LR1E3 --seed 0 \
  --fourier_freq 16 --fourier_scale 5.0

# ---------- 50-ep best + a moderate surf_weight bump (sw=20) ----------
start_run 4 wc-lr1e3-50ep-sw20-seed0 \
  --epochs 50 $WC $LR1E3 --seed 0 \
  --surf_weight 20

# ---------- Lion optimizer experiment ----------
# AB-UPT recipe: lr=5e-5, weight_decay=0.05, grad_clip=0.25.
start_run 5 wc-lion-lr5e5-50ep \
  --epochs 50 \
  --warmup_frac 0.05 --grad_clip 0.25 \
  --optimizer lion --lr 5e-5 --weight_decay 0.05 \
  --seed 0

# ---------- 200-epoch run of best small config (seed 0) ----------
start_run 6 wc-lr1e3-200ep-seed0 \
  --epochs 200 $WC $LR1E3 --seed 0

# ---------- best recipe + h128 l8 mlp2 (slight capacity bump, no subsampling, batch=2) ----------
start_run 7 wc-lr1e3-h128-l8-mlp2-bs2-50ep \
  --epochs 50 $WC $LR1E3 --seed 0 \
  --n_hidden 128 --n_layers 8 --n_head 4 --mlp_ratio 2 \
  --batch_size 2

echo "Wave 4 launched."
