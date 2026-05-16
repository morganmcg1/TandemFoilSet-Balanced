#!/usr/bin/env bash
# Sequentially run all 4 arms of the SF-AdamW weight-decay sweep on the single available GPU.
set -e
cd "$(dirname "$0")/.."

LOG_DIR="$(pwd)/logs"

run_arm() {
    local name="$1"; local wd="$2"; local logfile="$3"
    echo "===== Starting arm $name (wd=$wd) at $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$logfile"
    python train.py \
        --amp_dtype bf16 --use_ema --ema_decay 0.999 \
        --film_cond --two_shot_film --grad_clip_norm 1.0 \
        --use_schedule_free \
        --weight_decay "$wd" \
        --experiment_name "charliepai2i48h4-thorfinn/$name" \
        >>"$logfile" 2>&1
    echo "===== Finished arm $name at $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$logfile"
}

run_arm sf-wd-r1-arma-wd1e-4 1e-4 "$LOG_DIR/sf_wd_arma.log"
run_arm sf-wd-r1-armb-wd3e-4 3e-4 "$LOG_DIR/sf_wd_armb.log"
run_arm sf-wd-r1-armc-wd1e-3 1e-3 "$LOG_DIR/sf_wd_armc.log"
run_arm sf-wd-r1-armd-wd1e-2 1e-2 "$LOG_DIR/sf_wd_armd.log"

echo "===== All 4 arms complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$LOG_DIR/sf_wd_sweep_overall.log"
