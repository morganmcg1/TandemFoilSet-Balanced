#!/usr/bin/env bash
# Chain arms B, C, D after the in-flight Arm A finishes.
# Arm A is already running (PID set externally). Each arm respects
# SENPAI_TIMEOUT_MINUTES inside train.py.
set -uo pipefail

cd "$(dirname "$0")"

ARM_A_PID=${ARM_A_PID:-206420}
LOGDIR=logs/lr_sweep_bcd_$(date +"%Y%m%d-%H%M%S")
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.log"

note() {
  echo "[$(date -Is)] $*" | tee -a "$SUMMARY"
}

note "Waiting for Arm A pid=$ARM_A_PID to exit..."
while kill -0 "$ARM_A_PID" 2>/dev/null; do
  sleep 30
done
note "Arm A pid=$ARM_A_PID exited."

# Pause a few seconds to let the GPU memory release fully.
sleep 15

COMMON_ARGS=(
  --amp_dtype bf16
  --use_ema --ema_decay 0.999
  --film_cond --two_shot_film
  --grad_clip_norm 1.0
  --use_schedule_free
  --sf_beta1 0.95 --sf_beta2 0.99
  --n_layers 3
  --seed 1
  --agent charliepai2i48h4-alphonse
)

run_arm() {
  local arm=$1
  local lr=$2
  local lr_tag=$3
  local ts=$(date +"%Y%m%d-%H%M%S")
  local name="charliepai2i48h4-alphonse-lr-retune-n-layers3-r1-arm${arm}-lr${lr_tag}-${ts}"
  local log="${LOGDIR}/arm_${arm}_lr${lr_tag}.log"
  note "Arm ${arm}: lr=${lr}  exp=${name}  log=${log}"
  python -u train.py "${COMMON_ARGS[@]}" --lr "${lr}" --experiment_name "${name}" >"${log}" 2>&1
  local rc=$?
  note "Arm ${arm} finished rc=${rc}"

  # Echo last few jsonl entries so we can sanity-check
  local metrics="models/model-${name}-${ts}/metrics.jsonl"
  if [[ -f "$metrics" ]]; then
    note "Arm ${arm} metrics file: $metrics (lines=$(wc -l <"$metrics"))"
  fi
  return $rc
}

# Sequential: B → C → D
run_arm B 4e-3 4e3
run_arm C 5e-3 5e3

# Arm D divergence guard: PR says if val_avg NaN or >200 at epoch 3, stop early.
# train.py itself does not implement this — we just run it and decide post-hoc.
run_arm D 6e-3 6e3

note "All arms done."
