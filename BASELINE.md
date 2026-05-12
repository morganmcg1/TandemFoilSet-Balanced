# Baseline — TandemFoilSet (Charlie pai2g 24h r3)

Advisor branch: `icml-appendix-charlie-pai2g-24h-r3` (base: `icml-appendix-charlie`).
Fresh research track. No prior PRs merged on this branch yet — baseline
numbers will be established from the first round of experiments.

## Reference configuration (from `target/train.py`)

Transolver (~1.5M params):

| Field | Value |
|-------|-------|
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `space_dim` / `fun_dim` / `out_dim` | 2 / 22 / 3 |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| `epochs` | 50 |
| Scheduler | CosineAnnealingLR (T_max = epochs) |
| Optimizer | AdamW |

## Hard limits

- `SENPAI_TIMEOUT_MINUTES = 30` per training execution (wall clock).
- `SENPAI_MAX_EPOCHS = 50` (cap, not target).

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across the 4
val splits (`val_single_in_dist`, `val_geom_camber_rc`, `val_geom_camber_cruise`,
`val_re_rand`). **Lower is better.** Paper-facing decision metric is
`test_avg/mae_surf_p`, evaluated from the best-val checkpoint.

## Current best metrics

| Metric | Value | PR |
|--------|-------|----|
| `val_avg/mae_surf_p` | **115.403** | [#1491](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/1491) |
| `test_avg/mae_surf_p` | NaN (cruise split NaN — see note) | #1491 |
| `test_avg/mae_surf_p` (3-split proxy, excl. cruise) | ~115.1 | #1491 |

> **⚠ test_geom_camber_cruise NaN (all current runs):** `data/scoring.py`
> (read-only) uses `err * surf_mask` where `NaN * 0 = NaN` in IEEE 754. The
> cruise test split has one sample with non-finite ground-truth pressure, causing
> the entire split's `mae_surf_p` accumulator to be poisoned. Ux/Uy on the same
> split remain finite. Until this is fixed at the data layer, `test_avg/mae_surf_p`
> will always be NaN. Use the 3-split proxy (single + rc + re_rand) as the
> paper-facing estimate.

---

## 2026-05-12 18:56 — PR #1491: Gradient clipping (max_norm=1.0) + weight_decay 1e-4→1e-3

Best round-1 result. Establishes the running baseline for charlie-pai2g-24h-r3.

**Per-split val (best checkpoint, epoch 12 / 14 run):**

| Split | `mae_surf_p` | `mae_surf_Ux` | `mae_surf_Uy` |
|-------|---:|---:|---:|
| val_single_in_dist | 133.094 | 2.142 | 0.762 |
| val_geom_camber_rc | 129.763 | 2.849 | 0.992 |
| val_geom_camber_cruise | 88.997 | 2.377 | 0.594 |
| val_re_rand | 109.758 | 2.481 | 0.776 |
| **avg** | **115.403** | **2.462** | **0.781** |

**Per-split test (best checkpoint):**

| Split | `mae_surf_p` |
|-------|---:|
| test_single_in_dist | 116.984 |
| test_geom_camber_rc | 119.259 |
| test_geom_camber_cruise | NaN (see note above) |
| test_re_rand | 109.150 |
| avg (3-split proxy) | ~115.1 |

**Config (merged into advisor branch train.py):**

| Param | Value |
|-------|-------|
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `lr` | 5e-4 |
| `weight_decay` | **1e-3** ← changed |
| `grad_clip` | **1.0** ← new field |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| epochs run | 14 / 50 configured |

Reproduce:
```
cd target/ && python train.py \
  --experiment_name grad-clip-wd1e-3 \
  --weight_decay 1e-3 \
  --grad_clip 1.0 \
  --epochs 50
```

Metrics: `models/model-grad-clip-wd1e-3-20260512-181000/metrics.jsonl`
