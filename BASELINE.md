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
| `val_avg/mae_surf_p` | **99.879** | [#1484](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/1484) |
| `test_avg/mae_surf_p` (safe re-eval, 4-split) | **93.596** | #1484 |
| `test_avg/mae_surf_p` (3-split proxy, excl. cruise) | 99.616 | #1484 |

Previous best (superseded): val 103.100 / test 94.757 — PR #1495 (AoA+NACA augment).

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

---

## 2026-05-12 19:59 — PR #1495: AoA + NACA camber jitter augmentation

New best result. Replaces PR #1520 as the running baseline.

**Per-split val (best checkpoint, epoch 14 / 14 run):**

| Split | `mae_surf_p` | `mae_surf_Ux` | `mae_surf_Uy` |
|-------|---:|---:|---:|
| val_single_in_dist | 125.910 | 1.352 | 0.738 |
| val_geom_camber_rc | 114.346 | 2.221 | 0.980 |
| val_geom_camber_cruise | 77.995 | 0.895 | 0.520 |
| val_re_rand | 94.150 | 1.555 | 0.736 |
| **avg** | **103.100** | **1.506** | **0.744** |

**Per-split test (best checkpoint, safe re-eval — zero-fills non-finite y before subtraction):**

| Split | `mae_surf_p` |
|-------|---:|
| test_single_in_dist | 105.140 |
| test_geom_camber_rc | 100.580 |
| test_geom_camber_cruise | 83.481 (199/200 samples — see safe re-eval note above) |
| test_re_rand | 89.834 |
| **avg (4-split safe re-eval)** | **94.757** |
| avg (3-split proxy, excl. cruise) | 98.520 |

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
| `use_onecycle` | True (default, from #1520) |
| `ema_decay` | 0.999 (default, from #1520) |
| `augment` | **True** ← new default |
| `aoa_jitter_rad` | **0.00873** (±0.5°) ← new |
| `naca_jitter` | **0.002** ← new |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| epochs run | 14 / 14 configured (cosine T_max=14, full anneal) |

**Note on scheduler:** This baseline number 103.100 was achieved with cosine T_max=14 (NOT OneCycleLR). The merged train.py retains `use_onecycle=True` as default from #1520, so reproducing the **exact baseline number** requires `--use_onecycle False --epochs 14`. The composability of augmentation with OneCycleLR + EMA has NOT yet been measured directly.

Reproduce baseline (as run by thorfinn):
```
cd target/ && python train.py \
  --experiment_name geom-aoa-augment-r2 \
  --augment True \
  --aoa_jitter_rad 0.00873 \
  --naca_jitter 0.002 \
  --use_onecycle False \
  --epochs 14
```

Metrics: `models/model-geom-aoa-augment-r2-20260512-190924/metrics.yaml`

**Safe re-eval side script** for handling cruise NaN: `models/model-geom-aoa-augment-r2-20260512-190924/safe_re_eval.py`. Reusable across experiments — preserves the "skip non-finite samples" semantics by zero-filling `y` where non-finite before the subtraction. All future PRs should commit a similar safe re-eval log for paper-facing test reporting.

---

## 2026-05-13 00:25 — PR #1484: Huber loss (δ=0.5) on merged stack

**New best result. Replaces PR #1495 as running baseline.**

Arm A (δ=0.5) and Arm B (δ=1.0) rebased onto the merged stack
(`grad_clip=1.0`, `weight_decay=1e-3`, `use_onecycle=True`, `ema_decay=0.999`,
`augment=True`). Both ran 14 epochs in the 30-min cap (50 configured).
Both arms still descending at the cap → ceiling is timeout-limited.

**Per-split val (Arm A, huber_delta=0.5, EMA weights, best checkpoint):**

| Split | `mae_surf_p` | `mae_surf_Ux` | `mae_surf_Uy` |
|-------|---:|---:|---:|
| val_single_in_dist | 123.261 | 1.220 | 0.616 |
| val_geom_camber_rc | 118.368 | 2.144 | 0.876 |
| val_geom_camber_cruise | 69.270 | 0.799 | 0.435 |
| val_re_rand | 88.616 | 1.463 | 0.620 |
| **avg** | **99.879** | **1.406** | **0.637** |

**Per-split test (Arm A, safe 4-split re-eval via `safe_test_eval.py`):**

| Split | `mae_surf_p` |
|-------|---:|
| test_single_in_dist | 111.924 |
| test_geom_camber_rc | 101.713 |
| test_geom_camber_cruise | 75.537 (199/200 samples) |
| test_re_rand | 85.211 |
| **avg (4-split safe re-eval)** | **93.596** |
| avg (3-split proxy, excl. cruise) | 99.616 |

**Arm B (huber_delta=1.0)**: val 109.593 / test 102.399 — beats #1520 but
loses to #1495 baseline; clearly inferior to Arm A on every split.

**Config (merged into advisor branch train.py):**

| Param | Value |
|-------|-------|
| `huber_delta` | **0.5** ← new field (loss switches from MSE to Huber when set) |
| `n_hidden` / `n_layers` / `n_head` / `slice_num` / `mlp_ratio` | 128 / 5 / 4 / 64 / 2 |
| `lr` | 5e-4 |
| `weight_decay` | 1e-3 (from #1491) |
| `grad_clip` | 1.0 (from #1491) |
| `use_onecycle` | **True** (from #1520) — **note: this contradicts P4** |
| `ema_decay` | 0.999 (from #1520) |
| `augment` | True (from #1495) |
| `epochs` | 50 configured, 14 completed at 30-min cap |

**Note on schedule (P4 reconciliation):** Arm A used `--use_onecycle True --epochs 50` — the exact config flagged as "broken" in P4 (PR #1574). Yet it produced the new best result. Hypothesis: Huber loss is itself a gradient-clipping mechanism that interacts beneficially with the truncated OneCycleLR anneal. Recommend explicitly testing `--use_onecycle False --epochs 14` for Arm A in a follow-up to isolate the schedule contribution.

**Note on δ direction:** Round-1 result hinted that δ=0.5 hurts `val_single_in_dist`. On the merged stack the picture flips: δ=0.5 dominates δ=1.0 on every split *including* single_in_dist (123.26 vs 127.24). The round-1 finding was an artifact of the un-augmented, un-grad-clipped base.

Reproduce:
```
cd target/ && python train.py \
  --experiment_name huber-d0p5-onecycle-ema \
  --weight_decay 1e-3 --grad_clip 1.0 \
  --use_onecycle True --ema_decay 0.999 \
  --huber_delta 0.5 --epochs 50
```

Metrics: `models/model-huber-d0p5-onecycle-ema-20260512-225607/{metrics.yaml,test_safe_eval.json}`

**`safe_test_eval.py` is now an advisor-branch artifact** (committed in PR #1484) — every future PR should run it on the best checkpoint for paper-facing 4-split test reporting.
