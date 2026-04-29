#!/bin/bash
# Phase 2A: launch 3 long runs on GPUs that finished Phase 1 already.
# Strategy: extend best Phase 1 ideas to longer epoch budgets.

set -u
cd /workspace/ml-intern-benchmark/target

GROUP="mlintern-pai2-r3-retry-r1"
AGENT="ml-intern-r1"
TIMEOUT_MIN=200   # 3.3h max per run (cosine fully anneals before this)

mkdir -p run_logs/phase2

# (gpu, name, epochs, extra_flags)
declare -a CONFIGS=(
  "0|p2-baseline-100ep|100|"
  "5|p2-warmup3-clip-100ep|100|--warmup_epochs 3 --grad_clip 1.0"
  "6|p2-cap-h192-l6-s128-warm-80|80|--n_hidden 192 --n_layers 6 --slice_num 128 --warmup_epochs 3 --grad_clip 1.0 --lr 4e-4"
)

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r GPU NAME EPOCHS EXTRA <<< "$entry"
  CMD="CUDA_VISIBLE_DEVICES=$GPU SENPAI_TIMEOUT_MINUTES=$TIMEOUT_MIN \
    python -u train.py --epochs $EPOCHS --skip_test true \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name $GROUP/$NAME $EXTRA"
  LOG="run_logs/phase2/$NAME.log"
  echo "[GPU $GPU] $NAME -> $LOG"
  echo "  $CMD"
  nohup bash -c "$CMD" > "$LOG" 2>&1 &
  echo "  PID=$!"
  sleep 6
done
echo "Phase 2A launched"
