#!/bin/bash
# Phase 2B: launch on a fixed GPU when it becomes free.
# Argument 1: GPU index. The script picks the next config from the queue.

set -u
cd /workspace/ml-intern-benchmark/target

GROUP="mlintern-pai2-r3-retry-r1"
AGENT="ml-intern-r1"
TIMEOUT_MIN=200

mkdir -p run_logs/phase2

# Format: name|epochs|extra_flags
declare -a QUEUE=(
  "p2-amp-bs8-100ep|100|--use_amp true --batch_size 8"
  "p2-amp-bs4-100ep|100|--use_amp true --batch_size 4"
  "p2-bs8-fp32-80ep|80|--batch_size 8"
  "p2-mlp4-100ep|100|--mlp_ratio 4"
  "p2-heads8-100ep|100|--n_head 8"
  "p2-wider-h256-80ep|80|--n_hidden 256 --warmup_epochs 3 --grad_clip 1.0"
  "p2-deeper-l8-80ep|80|--n_layers 8 --warmup_epochs 3 --grad_clip 1.0"
  "p2-amp-bs8-warmup-100ep|100|--use_amp true --batch_size 8 --warmup_epochs 3 --grad_clip 1.0"
)

QUEUE_FILE=run_logs/phase2/queue.txt
LOCK_FILE=run_logs/phase2/queue.lock
[ ! -f "$QUEUE_FILE" ] && printf "%s\n" "${QUEUE[@]}" > "$QUEUE_FILE"

GPU=$1

# Pop one entry from the queue with a flock to avoid races
ENTRY=$(
  flock -x 200
  HEAD=$(head -1 "$QUEUE_FILE")
  if [ -z "$HEAD" ]; then echo ""; else
    tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
    echo "$HEAD"
  fi
) 200>"$LOCK_FILE"

if [ -z "$ENTRY" ]; then
  echo "Queue empty"
  exit 0
fi

IFS='|' read -r NAME EPOCHS EXTRA <<< "$ENTRY"
LOG="run_logs/phase2/$NAME.log"

CMD="CUDA_VISIBLE_DEVICES=$GPU SENPAI_TIMEOUT_MINUTES=$TIMEOUT_MIN \
  python -u train.py --epochs $EPOCHS --skip_test true \
  --agent $AGENT --wandb_group $GROUP \
  --wandb_name $GROUP/$NAME $EXTRA"

echo "[GPU $GPU] -> $NAME"
nohup bash -c "$CMD" > "$LOG" 2>&1 &
echo "  PID=$!  log=$LOG"
