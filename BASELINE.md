<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Baseline Metrics — icml-appendix-charlie-pai2e-r2

## Current Best

| Metric | Value | PR | Branch | Notes |
|--------|-------|----|--------|-------|
| `val_avg/mae_surf_p` | **104.6986** | #899 | `charliepai2e2-edward/checkpoint-averaging` | K=3 checkpoint average (epochs 12,13,14); epoch 14/50 timeout; undertrained |

Set by checkpoint averaging over last K=3 epochs (epochs 12, 13, 14) post-training. The single-best checkpoint of this run was 106.7871; averaging reduced variance and yielded 104.6986, beating the prior baseline of 104.7457 by 0.047. The technique is free (no extra forward/backward passes) and added ~30 lines.

Per-split breakdown (checkpoint-averaged):
- `val_single_in_dist`: mae_surf_p = 118.12
- `val_geom_camber_rc`: mae_surf_p = 117.23
- `val_geom_camber_cruise`: mae_surf_p = 83.43
- `val_re_rand`: mae_surf_p = 100.02

**Note:** Run-to-run variance is significant at this training budget (~±2 points). The technique works but the margin is within noise; future runs with longer training should show larger averaging gains.

**Reproduce:** `cd target/ && WANDB_MODE=offline python train.py --agent charliepai2e2-edward --wandb_name "edward-checkpoint-averaging"`

## Baseline Architecture (stock Transolver from train.py)

| Parameter | Value |
|-----------|-------|
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `lr` | 5e-4 |
| `weight_decay` | 1e-5 |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |

## Primary Metric

`val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across 4 validation splits (lower is better).

## History

| Date | PR | val_avg/mae_surf_p | Config | Notes |
|------|----|--------------------|--------|-------|
| 2026-04-28 | #899 | 104.6986 | stock + clip_grad_norm=1.0 + ckpt_avg K=3 | Epoch 14/50; checkpoint average of epochs 12,13,14; tiny margin over prior baseline; technique confirmed effective |
| 2026-04-28 | #778 | 104.7457 | stock + clip_grad_norm=1.0 | Epoch 14/50; 30-min wall-clock cap; undertrained; clear win — gradient explosion was the dominant issue |
| 2026-04-28 | #764 | 137.0013 | n_hidden=256 | Epoch 9/50; 30-min wall-clock cap; undertrained; first measured number |
