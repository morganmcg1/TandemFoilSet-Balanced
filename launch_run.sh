#!/bin/bash
# Launch one train.py run on a specified GPU and tee output to a log file.
# Usage: launch_run.sh <gpu_id> <run_name> [extra train.py args...]
set -e
GPU=$1
NAME=$2
shift 2
LOG="/workspace/ml-intern-benchmark/target/logs/${NAME//\//__}.log"
mkdir -p "$(dirname "$LOG")"

cd /workspace/ml-intern-benchmark/target
CUDA_VISIBLE_DEVICES="$GPU" SENPAI_TIMEOUT_MINUTES=720 \
  nohup python -u train.py \
    --agent ml-intern-r5 \
    --wandb_group mlintern-pai2-r5 \
    --wandb_name "$NAME" \
    "$@" \
  > "$LOG" 2>&1 &
echo "GPU=$GPU PID=$! NAME=$NAME LOG=$LOG"
