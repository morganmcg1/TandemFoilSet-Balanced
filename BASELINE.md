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
| `val_avg/mae_surf_p` | **112.546** | [#1520](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/1520) |
| `test_avg/mae_surf_p` | NaN (cruise split NaN — see note) | #1520 |
| `test_avg/mae_surf_p` (3-split proxy, excl. cruise) | **110.862** | #1520 |

> **⚠ test_geom_camber_cruise NaN (all current runs):** `data/scoring.py`
> (read-only) uses `err * surf_mask` where `Inf * 0 = NaN` in IEEE 754.
>
> **Root cause** (traced by tanjiro in PR #1494): hidden ground-truth file
> `splits_v2/.test_geom_camber_cruise_gt/000020.pt` contains **761 `+Inf` values
> in the pressure channel** of `y[:, 2]` (probably a divergent CFD solve written
> through unfiltered). The scoring code intends to skip samples with non-finite
> ground truth, but the subtraction `pred - y` happens before the sample-skip
> mask is applied, so `Inf` leaks into `err` and then `Inf * 0 = NaN` poisons
> the per-channel accumulator. Ux/Uy on the same split remain finite because
> they have no Inf in `y`.
>
> **Policy for this launch:** `data/scoring.py` is read-only per `program.md`,
> so we do not patch it. Two options for reporting `test_avg/mae_surf_p`:
>
> 1. **3-split proxy:** mean over `test_single_in_dist`, `test_geom_camber_rc`,
>    `test_re_rand` (skip cruise). Simple and well-defined.
> 2. **Safe re-eval side script (preferred):** load the saved checkpoint and
>    recompute MAE with `y` zero-filled wherever `y` is non-finite, before the
>    subtraction. Tanjiro's `models/model-re-film-conditioning-20260512-182128/test_safe_eval.log`
>    pattern: clone `y`, set `y[~isfinite(y).all(-1, keepdim=True)] = 0.0`,
>    then proceed normally. This preserves the existing skip-sample semantics
>    AND covers all 4 splits (199/200 cruise samples accumulated).
>
> When reporting paper-facing numbers, prefer the safe re-eval. Note both in
> the PR comment if they differ.

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

---

## 2026-05-12 19:53 — PR #1520: OneCycleLR + EMA(0.999)

New best result. Replaces PR #1491 as the running baseline.

**Per-split val (best checkpoint, epoch 14 / 14 run — still decreasing at cap):**

| Split | `mae_surf_p` | `mae_surf_Ux` | `mae_surf_Uy` |
|-------|---:|---:|---:|
| val_single_in_dist | 125.102 | 1.560 | 0.716 |
| val_geom_camber_rc | 136.044 | 2.948 | 1.029 |
| val_geom_camber_cruise | 86.305 | 1.182 | 0.491 |
| val_re_rand | 102.731 | 1.908 | 0.733 |
| **avg** | **112.546** | **1.900** | **0.742** |

**Per-split test (best checkpoint):**

| Split | `mae_surf_p` |
|-------|---:|
| test_single_in_dist | 113.886 |
| test_geom_camber_rc | 118.861 |
| test_geom_camber_cruise | NaN (see note above) |
| test_re_rand | 99.839 |
| avg (3-split proxy) | **110.862** |

**Config (merged into advisor branch train.py):**

| Param | Value |
|-------|-------|
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `lr` | 5e-4 |
| `weight_decay` | 1e-3 (from #1491) |
| `grad_clip` | 1.0 (from #1491) |
| `use_onecycle` | **True** ← new |
| `ema_decay` | **0.999** ← new |
| OneCycle peak LR | 5e-3 (10× base) |
| OneCycle pct_start | 0.05 (5% warmup) |
| OneCycle final min LR | 5e-6 |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| epochs run | 14 / 50 configured |

Reproduce:
```
cd target/ && python train.py \
  --experiment_name onecycle-ema-decay999 \
  --use_onecycle True \
  --ema_decay 0.999 \
  --epochs 50
```

Metrics: `models/model-charliepai2g24h3-fern-onecycle-ema-decay999-20260512-191518/metrics.yaml`
