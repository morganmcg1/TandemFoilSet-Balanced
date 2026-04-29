#!/bin/bash
# Wave 1b: relaunch the 4 OOM'd configs with batch_size=2 (still effective batch
# = same in expectation due to weighted sampler; trades half the throughput for
# half the activation memory).
set -e
cd "$(dirname "$0")/.."
mkdir -p session_logs

GROUP="mlintern-pai2-24h-v2-r1"
AGENT="ml-intern-r1"
EPOCHS=50

start_run() {
  local gpu="$1"; shift
  local name="$1"; shift
  local logfile="session_logs/wave1b-gpu${gpu}-${name}.log"
  echo "[$(date +%H:%M:%S)] Launching GPU $gpu: $name → $logfile"
  CUDA_VISIBLE_DEVICES="$gpu" SENPAI_TIMEOUT_MINUTES=720 \
    nohup python train.py \
      --epochs "$EPOCHS" \
      --batch_size 2 \
      --agent "$AGENT" \
      --wandb_group "$GROUP" \
      --wandb_name "${GROUP}/${name}" \
      "$@" > "$logfile" 2>&1 &
  echo $! > "session_logs/wave1b-gpu${gpu}-${name}.pid"
}

# GPU 4: BIG mid model fp32 (n_hidden=192, n_layers=8, n_head=8, mlp_ratio=4) bs=2.
start_run 4 mid-h192-l8-mlp4-bs2 \
  --n_hidden 192 --n_layers 8 --n_head 8 --mlp_ratio 4

# GPU 5: BIG mid + Transolver++ + bf16 + bs=2.
start_run 5 mid-h192-l8-mlp4-tpp-bf16-bs2 \
  --n_hidden 192 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --use_ada_temp True --use_rep_slice True \
  --amp bf16

# GPU 6: BIG mid + TPP + pressure-weight + Fourier + bf16 + bs=2.
start_run 6 mid-h192-l8-mlp4-tpp-pw10-fourier-bf16-bs2 \
  --n_hidden 192 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --use_ada_temp True --use_rep_slice True \
  --surf_p_weight 10.0 \
  --fourier_freq 16 --fourier_scale 10.0 \
  --amp bf16

# GPU 7: BIG h256 + TPP + bf16 + bs=2.
start_run 7 big-h256-l8-mlp4-tpp-bf16-bs2 \
  --n_hidden 256 --n_layers 8 --n_head 8 --mlp_ratio 4 \
  --use_ada_temp True --use_rep_slice True \
  --amp bf16

echo
echo "Wave 1b launched."
