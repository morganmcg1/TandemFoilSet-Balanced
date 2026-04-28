<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Baseline Metrics — icml-appendix-charlie-pai2e-r2

## Current Best

| Metric | Value | PR | Branch | Notes |
|--------|-------|----|--------|-------|
| `val_avg/mae_surf_p` | **104.7457** | #778 | `charliepai2e2-tanjiro/gradient-clipping` | epoch 14/50 only (30-min timeout); still improving at cutoff; **undertrained** |

Set by gradient clipping (`clip_grad_norm_(params, 1.0)`) applied after `loss.backward()`. Pre-clip gradient norms were 40–900× above the 1.0 threshold on every step, confirming that high-Re samples generate extreme gradients. Clipping alone cut val_avg/mae_surf_p by ~24% from the previous best (137.0013 → 104.7457).

Per-split breakdown (epoch 14):
- `val_single_in_dist`: mae_surf_p = 105.24
- `val_geom_camber_rc`: mae_surf_p = 97.21
- `val_geom_camber_cruise`: mae_surf_p = 98.39 (NaN bug in test_geom_camber_cruise worked around)
- `val_re_rand`: mae_surf_p = 118.15

**Effective working baseline for third-wave assignment:** 104.7457 (gradient clipping, stock architecture otherwise, epoch 14/50, undertrained).

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
| 2026-04-28 | #778 | 104.7457 | stock + clip_grad_norm=1.0 | Epoch 14/50; 30-min wall-clock cap; undertrained; clear win — gradient explosion was the dominant issue |
| 2026-04-28 | #764 | 137.0013 | n_hidden=256 | Epoch 9/50; 30-min wall-clock cap; undertrained; first measured number |
