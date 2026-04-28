#!/bin/bash
# Sequential higher-LR sweep on top of warmup3+β=0.5, with --nofilm to match #549 baseline.
set -uo pipefail
cd /workspace/senpai/target

mkdir -p logs

# Run 1: baseline-ref-lr5e-4 (matches merged #549 — warmup=3, lr=5e-4 default, film=False)
echo "[$(date)] Launching baseline-ref-lr5e-4..."
python train.py --epochs 50 --huber_beta 0.5 --warmup_epochs 3 --nofilm \
    --agent charliepai2d4-alphonse \
    --experiment_name "charliepai2d4-alphonse/baseline-ref-lr5e-4" \
    > logs/baseline-ref-lr5e-4.log 2>&1
echo "[$(date)] baseline-ref-lr5e-4 finished (exit=$?)."

# Run 2: lr7e-4-warmup3 (sqrt-2 above current)
echo "[$(date)] Launching lr7e-4-warmup3..."
python train.py --epochs 50 --huber_beta 0.5 --warmup_epochs 3 --nofilm --lr 7e-4 \
    --agent charliepai2d4-alphonse \
    --experiment_name "charliepai2d4-alphonse/lr7e-4-warmup3" \
    > logs/lr7e-4-warmup3.log 2>&1
echo "[$(date)] lr7e-4-warmup3 finished (exit=$?)."

# Run 3: lr1e-3-warmup3 (aggressive)
echo "[$(date)] Launching lr1e-3-warmup3..."
python train.py --epochs 50 --huber_beta 0.5 --warmup_epochs 3 --nofilm --lr 1e-3 \
    --agent charliepai2d4-alphonse \
    --experiment_name "charliepai2d4-alphonse/lr1e-3-warmup3" \
    > logs/lr1e-3-warmup3.log 2>&1
echo "[$(date)] lr1e-3-warmup3 finished (exit=$?)."

echo "[$(date)] All 3 runs done."
