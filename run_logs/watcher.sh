#!/bin/bash
# GPU watcher: every 30s, check each GPU; if it's free (mem<1500MB),
# pop the next entry from QUEUE_FILE and launch a job on it.
# Queue entry format:  name|epochs|extra_flags|timeout_min
#   timeout_min is optional, defaults to DEFAULT_TIMEOUT_MIN.

set -u
cd /workspace/ml-intern-benchmark/target

GROUP="mlintern-pai2-r3-retry-r1"
AGENT="ml-intern-r1"
DEFAULT_TIMEOUT_MIN=360   # 6 hours per run by default

QUEUE_FILE=run_logs/phase2/queue.txt
mkdir -p run_logs/phase2 run_logs/phase3

# Pick output subdir from the run name's first 2 chars (p1/p2/p3/p4)
slot_dir() {
  case "$1" in
    p1-*) echo run_logs/phase1 ;;
    p2-*) echo run_logs/phase2 ;;
    p3-*) echo run_logs/phase3 ;;
    p4-*) echo run_logs/phase4 ;;
    *)    echo run_logs/phase2 ;;
  esac
}

while true; do
  if [ ! -s "$QUEUE_FILE" ]; then
    echo "$(date '+%H:%M:%S') Queue empty, watcher exiting"
    break
  fi
  for GPU in 0 1 2 3 4 5 6 7; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ')
    if [ -z "$MEM" ]; then continue; fi
    if [ "$MEM" -lt 1500 ]; then
      HEAD=$(head -1 "$QUEUE_FILE")
      if [ -z "$HEAD" ]; then break; fi
      tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
      IFS='|' read -r NAME EPOCHS EXTRA TIMEOUT <<< "$HEAD"
      [ -z "$TIMEOUT" ] && TIMEOUT="$DEFAULT_TIMEOUT_MIN"
      LOG_DIR=$(slot_dir "$NAME")
      mkdir -p "$LOG_DIR"
      LOG="$LOG_DIR/$NAME.log"
      CMD="CUDA_VISIBLE_DEVICES=$GPU SENPAI_TIMEOUT_MINUTES=$TIMEOUT \
        python -u train.py --epochs $EPOCHS --skip_test true \
        --agent $AGENT --wandb_group $GROUP \
        --wandb_name $GROUP/$NAME $EXTRA"
      echo "$(date '+%H:%M:%S') [GPU $GPU] launching $NAME (epochs=$EPOCHS, timeout=${TIMEOUT}min)"
      nohup bash -c "$CMD" > "$LOG" 2>&1 &
      sleep 8
    fi
  done
  sleep 30
done
