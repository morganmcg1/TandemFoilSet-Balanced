#!/usr/bin/env bash
# Runner for the ema_decay sweep arms B, C, D (Arm A already running)
# Waits for current Arm A PID, then runs B/C/D sequentially on the single GPU.
set -u
cd "$(dirname "$0")"

ARM_A_PID="${ARM_A_PID:-132513}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "[$(date -u +%FT%TZ)] Waiting for Arm A pid=$ARM_A_PID to exit..."
while kill -0 "$ARM_A_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(date -u +%FT%TZ)] Arm A exited."

run_arm() {
  local name="$1" ; shift
  local decay="$1" ; shift
  local log="$LOG_DIR/sf-ema-r1-${name}.log"
  echo "[$(date -u +%FT%TZ)] Launching arm ${name} ema_decay=${decay} -> $log"
  python train.py \
    --amp_dtype bf16 --use_ema --ema_decay "$decay" \
    --film_cond --two_shot_film --grad_clip_norm 1.0 \
    --use_schedule_free \
    --agent charliepai2i48h4-tanjiro \
    --experiment_name "charliepai2i48h4-tanjiro/sf-ema-r1-${name}-d${decay//./_}" \
    > "$log" 2>&1
  local rc=$?
  echo "[$(date -u +%FT%TZ)] Arm ${name} exit code=${rc}"
  return 0
}

run_arm armb 0.99
run_arm armc 0.9995
run_arm armd 0.9999

echo "[$(date -u +%FT%TZ)] Sweep complete."
