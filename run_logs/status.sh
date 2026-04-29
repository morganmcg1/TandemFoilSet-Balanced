#!/bin/bash
# Cheap status check: per-run last "Epoch ... val_avg_surf_p" line and elapsed.
cd /workspace/ml-intern-benchmark/target
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "GPU usage:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F',' '{printf "  GPU%s mem=%s util=%s\n", $1, $2, $3}'
echo
echo "Run progress:"
shopt -s nullglob
for f in run_logs/phase1/*.log run_logs/phase2/*.log run_logs/phase3/*.log run_logs/phase4/*.log; do
  [ -f "$f" ] || continue
  NAME=$(basename "$f" .log)
  # Last summary line per epoch
  LAST=$(grep -aE "^Epoch [0-9 ]+\([0-9.]+s\)" "$f" 2>/dev/null | tail -1)
  STATUS="(running)"
  if grep -aq "Training done in" "$f"; then STATUS="(done)"; fi
  if grep -aq "Traceback\|Error\|FAILED\|CUDA out of memory" "$f"; then STATUS="(ERROR)"; fi
  echo "  $NAME  $STATUS"
  [ -n "$LAST" ] && echo "    $LAST"
done
