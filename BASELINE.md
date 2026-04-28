<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Baseline Metrics — icml-appendix-charlie-pai2e-r2

## Current Best

**No completed runs yet.** This is a fresh research track. The first student experiments are in-flight (PRs #764–#780) and will establish the baseline once they complete.

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

| Date | PR | val_avg/mae_surf_p | Notes |
|------|----|--------------------|-------|
| — | — | TBD | Awaiting first student runs |
