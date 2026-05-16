#!/usr/bin/env bash
# SF-warmup-steps sweep — 4 sequential arms on a single GPU.
# Each arm: identical SF+EMA+FiLM+clip1 stack, only --sf_warmup_steps differs.
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0

run_arm() {
    local arm_letter="$1"
    local warmup_steps="$2"
    local tag="sf-warmup-r1-arm${arm_letter}-w${warmup_steps}"
    local log="logs/${tag}.log"
    echo "=== Arm ${arm_letter} (warmup=${warmup_steps}) start: $(date -u +%FT%TZ) ===" | tee -a logs/sweep_meta.log
    uv run python train.py \
        --amp_dtype bf16 --use_ema --ema_decay 0.999 \
        --film_cond --two_shot_film --grad_clip_norm 1.0 \
        --use_schedule_free --sf_warmup_steps "${warmup_steps}" \
        --agent charliepai2i48h4-edward \
        --experiment_name "charliepai2i48h4-edward/${tag}" \
        > "${log}" 2>&1
    local rc=$?
    echo "=== Arm ${arm_letter} (warmup=${warmup_steps}) end:   $(date -u +%FT%TZ) rc=${rc} ===" | tee -a logs/sweep_meta.log
    return $rc
}

run_arm A 500
run_arm B 100
run_arm C 1000
run_arm D 2000

echo "=== SWEEP COMPLETE: $(date -u +%FT%TZ) ===" | tee -a logs/sweep_meta.log
