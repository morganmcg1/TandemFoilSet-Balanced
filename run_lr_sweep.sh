#!/usr/bin/env bash
# Sequential 4-arm LR sweep at new canonical (n_layers=3, sf_betas=(0.95, 0.99))
set -uo pipefail

cd "$(dirname "$0")"

TS=$(date +"%Y%m%d-%H%M%S")
LOGDIR="logs/lr_sweep_${TS}"
mkdir -p "$LOGDIR"

COMMON_ARGS=(
  --amp_dtype bf16
  --use_ema --ema_decay 0.999
  --film_cond --two_shot_film
  --grad_clip_norm 1.0
  --use_schedule_free
  --sf_beta1 0.95 --sf_beta2 0.99
  --n_layers 3
  --seed 1
)

run_arm() {
  local arm=$1
  local lr=$2
  local lr_tag=$3
  local name="charliepai2i48h4-alphonse-lr-retune-n-layers3-r1-arm${arm}-lr${lr_tag}-${TS}"
  local log="${LOGDIR}/arm_${arm}_lr${lr_tag}.log"
  echo "=========================================="
  echo "Arm ${arm}: lr=${lr}  exp=${name}"
  echo "log=${log}"
  echo "started=$(date -Is)"
  echo "=========================================="
  python train.py "${COMMON_ARGS[@]}" --lr "${lr}" --experiment_name "${name}" >"${log}" 2>&1
  local rc=$?
  echo "Arm ${arm} finished rc=${rc} at $(date -Is)" | tee -a "${log}"
  return $rc
}

run_arm A 3e-3 3e3
run_arm B 4e-3 4e3
run_arm C 5e-3 5e3
run_arm D 6e-3 6e3

echo "ALL ARMS DONE at $(date -Is)"
