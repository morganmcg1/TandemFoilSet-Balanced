#!/bin/bash
# Driver: run 4 sequential baseline replicates with seeds {0,1,2,3} for PR #3546.
set -e
cd "$(dirname "$0")"
for seed in 0 1 2 3; do
  log="logs/baseline_pr3546_seed${seed}.log"
  echo "=== Launching seed=${seed} at $(date -Is) ==="
  python train.py \
    --agent willowpai2i48h1-alphonse \
    --seed "${seed}" \
    --wandb_name "willowpai2i48h1-alphonse/baseline_seed${seed}" \
    --wandb_group baseline_variance_canonical \
    >"${log}" 2>&1
  rc=$?
  echo "=== seed=${seed} done at $(date -Is) (rc=${rc}) ==="
done
echo "=== All replicates complete at $(date -Is) ==="
