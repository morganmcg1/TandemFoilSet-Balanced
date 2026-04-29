#!/usr/bin/env bash
# Round-5 final long runs: 8 seeds / mild variants of the R4 winner config
# (recipe + sw5 + warmup_pct=0.3, default arch). Goal: a bigger ensemble
# +  push the single-model best by training even longer.
#
# Trains for ~3.5 h each so it fits inside the remaining pod budget.

set -u
cd "$(dirname "$0")"
GROUP="mlintern-pai2-r4"
AGENT="ml-intern-r4"
TIMEOUT=210   # 3.5 hours
mkdir -p logs

launch() {
    local gpu="$1"; local name="$2"; shift 2
    local logfile="logs/r5-gpu${gpu}-${name}.log"
    echo "[gpu${gpu}] launching ${name} -> ${logfile}"
    CUDA_VISIBLE_DEVICES="${gpu}" SENPAI_TIMEOUT_MINUTES="${TIMEOUT}" nohup \
        python train.py --skip_test \
            --agent "${AGENT}" --wandb_group "${GROUP}" \
            --wandb_name "${GROUP}/r5-${name}" "$@" \
        > "${logfile}" 2>&1 &
    echo "  pid=$!"
}

# 3.5 h budget @ ~132 s/ep ≈ 95 epochs default arch.

# GPUs 0..3 — recipe + sw5 + warmup30 with new seeds + ep=85.
launch 0 sw5-w30-85-s10 \
    --epochs 85 --warmup_pct 0.3 --surf_weight 5.0 --seed 10 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0
launch 1 sw5-w30-85-s11 \
    --epochs 85 --warmup_pct 0.3 --surf_weight 5.0 --seed 11 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0
launch 2 sw5-w30-85-s12 \
    --epochs 85 --warmup_pct 0.3 --surf_weight 5.0 --seed 12 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0
launch 3 sw5-w30-85-s13 \
    --epochs 85 --warmup_pct 0.3 --surf_weight 5.0 --seed 13 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 4 — slightly different schedule: ep=100 with same recipe.
launch 4 sw5-w30-100 \
    --epochs 100 --warmup_pct 0.3 --surf_weight 5.0 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 5 — surf_weight=3 (between 2 and 5).
launch 5 sw3-w30-85 \
    --epochs 85 --warmup_pct 0.3 --surf_weight 3.0 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 6 — surf_weight=8 (between 5 and 10).
launch 6 sw8-w30-85 \
    --epochs 85 --warmup_pct 0.3 --surf_weight 8.0 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 7 — sw5 + warmup_pct=0.4 (longer warmup).
launch 7 sw5-w40-85 \
    --epochs 85 --warmup_pct 0.4 --surf_weight 5.0 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

echo "All round-5 jobs launched."
