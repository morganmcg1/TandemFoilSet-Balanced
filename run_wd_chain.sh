#!/bin/bash
# Sequential weight_decay sweep on rebased advisor branch (PR #808 bf16 baseline).
# wd=1e-4 and wd=1e-3 already completed. This launches wd=1e-2.

set -e
cd /workspace/senpai/target

echo "[$(date)] Starting wd=1e-2 run on rebased PR #808 baseline (bf16, n_hidden=256, n_head=8, epochs=12)"
python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 12 \
  --weight_decay 1e-2 --agent charliepai2e1-edward \
  --wandb_name charliepai2e1-edward/adamw-wd-1e-2-rebased \
  > logs/wd-1e-2-rebased.log 2>&1
echo "[$(date)] wd=1e-2 finished."
