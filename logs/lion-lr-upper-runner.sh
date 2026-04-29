#!/usr/bin/env bash
set -o pipefail

cd "$(dirname "$0")/.."

LOG_DIR="logs"
COMMON_ARGS="--n_layers 3 --bf16 --surf_weight 28.0 --weight_decay 1e-2 --batch_size 4 --agent charliepai2e5-frieren"

run_one() {
  tag="$1"
  lr="$2"
  log="${LOG_DIR}/lion-lr-${tag}.log"
  echo "[runner] starting ${tag} (lr=${lr}) at $(date -u)" | tee -a "${LOG_DIR}/lion-lr-upper-runner.log"
  python train.py ${COMMON_ARGS} --lr "${lr}" \
    --wandb_name "charliepai2e5-frieren/lion-lr-${tag}" \
    > "${log}" 2>&1
  echo "[runner] finished ${tag} (lr=${lr}) at $(date -u) status=$?" | tee -a "${LOG_DIR}/lion-lr-upper-runner.log"
}

run_one "control-3e-4" 3e-4
run_one "upper1-4e-4" 4e-4
run_one "upper2-5e-4" 5e-4

echo "[runner] ALL DONE at $(date -u)" | tee -a "${LOG_DIR}/lion-lr-upper-runner.log"
