#!/bin/bash
cd /workspace/senpai/target
CUDA_VISIBLE_DEVICES=0 python train.py \
  --epochs 50 \
  --warmup_epochs 1 \
  --stable_epochs 10 \
  --experiment_name h21-wsd-warmup1-stable10-h19 \
  --agent charliepai2i48h3-thorfinn \
  --huber_delta 0.5 2>&1
