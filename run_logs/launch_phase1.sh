#!/bin/bash
# Phase 1: 8-way parallel architecture/loss/optimizer discovery sweep.
# Each run gets 1 GPU, ~70min wall time, --epochs sized so cosine LR fully anneals.

set -u
cd /workspace/ml-intern-benchmark/target

GROUP="mlintern-pai2-r3-retry-r1"
AGENT="ml-intern-r1"
TIMEOUT_MIN=80   # hard ceiling per run; configured epochs land below this

declare -a NAMES=(
  "p1-baseline"
  "p1-wider-h256"
  "p1-deeper-l8"
  "p1-slice-256"
  "p1-cap-h192-l6-s128"
  "p1-surf-w30"
  "p1-warmup-lr8e4"
  "p1-amp-bs8"
)

# Shared base flags
BASE_FLAGS=(
  --epochs 30
  --skip_test true
  --agent "$AGENT"
  --wandb_group "$GROUP"
)

# Per-config args. Index matches NAMES.
declare -a EPOCHS=(  30  22  22  28  18  30  30  60 )
declare -a EXTRA=(
  ""                                                                # 0 baseline
  "--n_hidden 256"                                                  # 1 wider
  "--n_layers 8"                                                    # 2 deeper
  "--slice_num 256"                                                 # 3 more slices
  "--n_hidden 192 --n_layers 6 --slice_num 128"                     # 4 capacity bump combined
  "--surf_weight 30"                                                # 5 surf-heavy
  "--warmup_epochs 2 --lr 8e-4"                                     # 6 warmup + higher peak LR
  "--use_amp true --batch_size 8"                                   # 7 AMP + bigger batch
)

mkdir -p run_logs/phase1

# Launch each on its own GPU
for i in "${!NAMES[@]}"; do
  NAME="${NAMES[$i]}"
  EXTRA_FLAGS="${EXTRA[$i]}"
  EPOCH_COUNT="${EPOCHS[$i]}"

  CMD="CUDA_VISIBLE_DEVICES=$i SENPAI_TIMEOUT_MINUTES=$TIMEOUT_MIN \
    python -u train.py ${BASE_FLAGS[@]} --epochs $EPOCH_COUNT \
    $EXTRA_FLAGS \
    --wandb_name $GROUP/$NAME"

  LOG="run_logs/phase1/$NAME.log"
  echo "[GPU $i] $CMD"
  echo "    -> $LOG"
  nohup bash -c "$CMD" > "$LOG" 2>&1 &
  PID=$!
  echo "    PID=$PID"
  sleep 8  # stagger so wandb runs do not init at the same instant
done

wait
echo "Phase 1 complete"
