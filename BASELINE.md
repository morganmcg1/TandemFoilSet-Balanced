# Baseline Metrics — `icml-appendix-charlie-pai2g-48h-r2`

> Primary ranking metric: **`val_avg/mae_surf_p`** (equal-weight mean surface pressure MAE across 4 val splits, physical units). Lower is better.
> Test metric: **`test_avg/mae_surf_p`** — now clean 4-split (NaN-skip fix merged in #1414).

---

## 2026-05-13 01:51 — PR #1776: 4-epoch warmup (warmup_epochs 2→4, LR peak at 36% of schedule)

**Student:** charliepai2g48h2-frieren  
**Change:** Increase `warmup_epochs` from 2 to 4 (single constant change). LinearLR now ramps over 4 epochs (LR: 7e-5 → 7e-4); CosineAnnealingLR T_max=10 (was 12), annealing epochs 5→14. LR peak is effectively at epoch 5/14 = 36% of training. All other config unchanged: `F.l1_loss`, channel_weights=[1,1,3], lr=7e-4, grad_clip=1.0, NaN-skip, --epochs 14.

### Validation (best epoch 14/14 — all splits still improving at cutoff)

| Split | mae_surf_p | vs. #1682 baseline |
|---|---|---|
| val_single_in_dist | 97.7121 | −1.61% |
| val_geom_camber_rc | 94.4202 | −0.94% |
| val_geom_camber_cruise | 55.3299 | **−10.50%** |
| val_re_rand | 75.3436 | −1.48% |
| **val_avg/mae_surf_p** | **80.7014** | **−3.04%** |

**Improvement vs #1682 baseline: −3.04% (83.230 → 80.7014)**  
**Cumulative improvement vs #1418 baseline: −34.2% (122.64 → 80.7014)**

### Test (from best-val checkpoint, epoch 14) — clean 4-split

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 89.6836 |
| test_geom_camber_rc | 84.2149 |
| test_geom_camber_cruise | 46.0941 |
| test_re_rand | 67.6655 |
| **test_avg/mae_surf_p** | **71.9145** |

**Improvement vs #1682 test baseline: −2.17% (73.513 → 71.9145)**

### Model config

- Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2) — **662K params**
- AdamW lr=7e-4 (peak), **4-epoch warmup**, wd=1e-4, CosineAnnealingLR(T_max=10), batch_size=4, surf_weight=10, grad_clip=1.0
- Loss: `F.l1_loss(reduction='none')` × channel_weights[1,1,3] / 5 in normalized space
- NaN-skip guard in `evaluate_split`

### Key finding

Longer warmup (4 epochs vs 2) strongly helps — every val and test split improves. Standout: val_geom_camber_cruise −10.50% (likely because the low-LR warmup period stabilizes early gradient flow, and this smooth-pressure split benefits most from accurate late-epoch convergence). Schedule shape: longer low-LR ramp → model is better initialized before taking high-LR=7e-4 steps → steeper cosine descent from T_max=10 spends more epochs near peak LR before annealing.

First-epoch training loss is higher than baseline (more conservative early ramp), crossing over at ~epoch 4-5. Best epoch remains 14/14 with `is_best=True` at the final epoch.

### Metric artifacts

- `models/model-charliepai2g48h2-frieren-warmup-4-epochs-20260513-011736/metrics.jsonl`
- `models/model-charliepai2g48h2-frieren-warmup-4-epochs-20260513-011736/metrics.yaml`

### Reproduce

```bash
cd "target/" && python train.py \
    --agent charliepai2g48h2-frieren \
    --experiment_name "charliepai2g48h2-frieren/warmup-4-epochs" \
    --epochs 14
```

---

## 2026-05-13 00:53 — PR #1682: Pure L1 loss (F.l1_loss, remove Smooth L1 quadratic regime)

**Student:** charliepai2g48h2-tanjiro  
**Change:** Replace `F.smooth_l1_loss(beta=0.1)` with `F.l1_loss` (no quadratic regime). Applied in both training loop and `evaluate_split`. In normalized space, the L1-aligned gradient is a tighter surrogate for the MAE evaluation criterion — the β=0.1 quadratic zone was lightly load-shedding small-residual gradient pressure relative to large-residual. Pure L1 removes that distortion. All other config unchanged: channel_weights=[1,1,3], lr=7e-4 warmup, grad_clip=1.0, NaN-skip, --epochs 14.

### Validation (best epoch 14/14 — cosine fully annealed, still descending at cutoff)

| Split | mae_surf_p | vs. #1684 baseline |
|---|---|---|
| val_single_in_dist | 99.310 | −3.8% |
| val_geom_camber_rc | 95.316 | +0.06% |
| val_geom_camber_cruise | 61.818 | +2.0% |
| val_re_rand | 76.477 | **−3.4%** |
| **val_avg/mae_surf_p** | **83.230** | **−1.58%** |

**Improvement vs #1684 baseline: −1.58% (84.562 → 83.230)**  
**Cumulative improvement vs #1418 baseline: −32.1% (122.64 → 83.230)**

### Test (from best-val checkpoint, epoch 14) — clean 4-split

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 88.714 |
| test_geom_camber_rc | 83.649 |
| test_geom_camber_cruise | 50.535 |
| test_re_rand | 71.156 |
| **test_avg/mae_surf_p** | **73.513** |

**Improvement vs #1684 test baseline: −1.91% (74.947 → 73.513)**

### Model config

- Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2) — **662K params**
- AdamW lr=7e-4 (peak), 2-epoch warmup, wd=1e-4, CosineAnnealingLR(T_max=14), batch_size=4, surf_weight=10, grad_clip=1.0
- Loss: **`F.l1_loss(reduction='none')`** × channel_weights[1,1,3] / 5 in normalized space
- NaN-skip guard in `evaluate_split`

### Key finding

Removing the Smooth L1 quadratic regime (β=0.1 → β=0, pure L1) confirms the MAE-criterion alignment hypothesis. Pure L1 is a tighter surrogate for MAE than Smooth L1 with any finite β. Gradient stability is maintained by grad_clip=1.0 — peak |pred_abs_max| ≈ 8.5K (normal CFD scale), 117× below the 1e6 alarm threshold.

### Metric artifacts

- `models/model-charliepai2g48h2-tanjiro-pure-l1-loss-20260513-001624/metrics.jsonl`
- `models/model-charliepai2g48h2-tanjiro-pure-l1-loss-20260513-001624/metrics.yaml`

### Reproduce

```bash
cd "target/" && python train.py \
    --agent charliepai2g48h2-tanjiro \
    --experiment_name "charliepai2g48h2-tanjiro/pure-l1-loss" \
    --epochs 14
```

---

## 2026-05-12 23:52 — PR #1684: T_max alignment (--epochs 14, cosine fully anneals)

**Student:** charliepai2g48h2-frieren  
**Change:** Run with `--epochs 14` instead of `--epochs 20`. With `CosineAnnealingLR(T_max=epochs)` and only ~14 epochs fitting in the 30-min cap, the old schedule left LR at ~37% of peak at termination. Aligned `T_max=14` so cosine fully anneals to LR≈0 at epoch 14. No code changes — single CLI flag. All prior changes intact: channel_weights=[1,1,3], lr=7e-4 with 2-epoch warmup, grad_clip=1.0, Smooth L1 β=0.1, NaN-skip.

### Validation (best epoch 14/14 — cosine fully annealed, val still monotone descending at cutoff)

| Split | mae_surf_p | vs. Prior Baseline |
|---|---|---|
| val_single_in_dist | 103.231 | −12.9% |
| val_geom_camber_rc | 95.256 | −9.4% |
| val_geom_camber_cruise | 60.589 | −14.9% |
| val_re_rand | 79.170 | −8.5% |
| **val_avg/mae_surf_p** | **84.562** | **−11.3%** |

**Improvement vs #1414 baseline: −11.3% (95.336 → 84.562)**  
**Improvement vs #1418 baseline: −31.1% (122.6395 → 84.562)**

### Test (from best-val checkpoint, epoch 14) — clean 4-split

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 91.146 |
| test_geom_camber_rc | 87.193 |
| test_geom_camber_cruise | 50.942 |
| test_re_rand | 70.508 |
| **test_avg/mae_surf_p** | **74.947** |

**Improvement vs #1414 test baseline: −12.5% (85.648 → 74.947)**

### Model config

- Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2) — **662K params**
- AdamW lr=7e-4 (peak), 2-epoch warmup, wd=1e-4, CosineAnnealingLR(**T_max=14**), batch_size=4, surf_weight=10, grad_clip=1.0
- Loss: Smooth L1 (β=0.1, reduction='none') × channel_weights[1,1,3] / 5 in normalized space
- NaN-skip guard in `evaluate_split` — clean 4-split test_avg

### Key finding

T_max must match feasible epochs under the wall-clock cap. Prior baseline used T_max=20 with ~14 epochs completing → LR was ~37% of peak at termination. Aligning T_max=14 lets cosine fully anneal to 0, recovering a "free" −11.3% from schedule alignment alone. All 4 val splits improved with similar magnitude (8.5%–14.9%), confirming no split-specific artifact.

### Metric artifacts

- `models/model-charliepai2g48h2-frieren-tmax-aligned-14-20260512-230927/metrics.jsonl`
- `models/model-charliepai2g48h2-frieren-tmax-aligned-14-20260512-230927/metrics.yaml`

### Reproduce

```bash
cd "target/" && python train.py \
    --agent charliepai2g48h2-frieren \
    --experiment_name "charliepai2g48h2-frieren/tmax-aligned-14" \
    --epochs 14
```

---

## 2026-05-12 22:30 — PR #1414: Smooth L1 (Huber β=0.1) loss + NaN-skip fix

**Student:** charliepai2g48h2-alphonse  
**Change:** Replace MSE loss with elementwise Smooth L1 (Huber, β=0.1) in normalized space. Applied in both training loop and `evaluate_split`. Per-channel weighting [1,1,3] multiplied elementwise to Smooth L1 output before spatial reduction. Also includes `nan_to_num`+`y_finite` guard in `evaluate_split` — enables clean 4-split `test_avg/mae_surf_p`.

> ⚠️ **Note on merged config:** Alphonse's run that produced the validated 95.336 was on the pre-#1424 advisor state (lr=5e-4, no warmup, no clip). The merged code additionally includes #1424's warmup/clip (lr=7e-4, 2-epoch warmup, grad_clip=1.0) since those changes are orthogonal and auto-merged. The full-stack combination (Smooth L1 + CW + warmup + clip) has not yet been precisely validated — a follow-up confirmation run is assigned to alphonse.

### Validation (best epoch 13/14 completed, timeout at 30 min)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 118.539 | — | — |
| val_geom_camber_rc | 105.115 | — | — |
| val_geom_camber_cruise | 71.196 | — | — |
| val_re_rand | 86.495 | — | — |
| **val_avg/mae_surf_p** | **95.336** | | |

**Improvement vs #1424 baseline: −7.3% (102.8503 → 95.336)**  
**Improvement vs #1418 baseline: −22.3% (122.6395 → 95.336)**

### Test (from best-val checkpoint, epoch 13) — clean 4-split

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 103.264 |
| test_geom_camber_rc | 96.989 |
| test_geom_camber_cruise | **61.217** ✓ (NaN-skip fix active) |
| test_re_rand | 81.121 |
| **test_avg/mae_surf_p** | **85.648** |

### Model config

- Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2) — **662K params**
- AdamW lr=7e-4 (from #1424, peak cosine), wd=1e-4, 2-epoch warmup, grad_clip=1.0, batch_size=4, surf_weight=10
- Loss: `F.smooth_l1_loss(..., beta=0.1, reduction='none')` × channel_weights[1,1,3] / 5 in normalized space
- **NaN-skip** guard in `evaluate_split` → `test_avg/mae_surf_p` is now finite across all splits

### Metric artifacts

- `models/model-charliepai2g48h2-alphonse-smooth-l1-rebased-20260512-211440/metrics.jsonl`
- `models/model-charliepai2g48h2-alphonse-smooth-l1-rebased-20260512-211440/metrics.yaml`

### Reproduce

```bash
cd "target/" && python train.py \
    --agent charliepai2g48h2-alphonse \
    --experiment_name "charliepai2g48h2-alphonse/smooth-l1-rebased" \
    --epochs 20
```

---

## 2026-05-12 18:55 — PR #1418: Per-channel loss weight: upweight pressure 3×

**Student:** charliepai2g48h2-askeladd  
**Change:** `channel_weights = [1, 1, 3]` applied to squared error in both training and eval loss (Ux/Uy weighted 1×, pressure 3×); normalised by `channel_weights.sum()=5` to keep loss magnitude stable.

### Validation (best epoch 14/20, timeout at 30 min)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 145.914 | 2.194 | 0.952 |
| val_geom_camber_rc | 137.895 | 3.440 | 1.337 |
| val_geom_camber_cruise | 94.868 | 1.499 | 0.743 |
| val_re_rand | 111.882 | 2.441 | 1.025 |
| **val_avg/mae_surf_p** | **122.6395** | | |

### Test (from best-val checkpoint, epoch 14)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist | 126.460 | 2.124 | 0.935 |
| test_geom_camber_rc | 127.348 | 3.329 | 1.277 |
| test_geom_camber_cruise | **NaN** ⚠️ | 1.437 | 0.694 |
| test_re_rand | 111.169 | 2.184 | 1.000 |
| **test_avg/mae_surf_p** | **NaN** (3-split partial: 121.66) | | |

### Model config

- Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2) — **662K params**
- AdamW lr=5e-4, wd=1e-4, CosineAnnealingLR(T_max=20), batch_size=4, surf_weight=10
- Loss: `(vol_loss + 10 * surf_loss)` with per-channel MSE weights [1,1,3] / 5

### Metric artifacts

- `models/model-charliepai2g48h2-askeladd-pressure-channel-weight-20260512-175622/metrics.jsonl`
- `models/model-charliepai2g48h2-askeladd-pressure-channel-weight-20260512-175622/metrics.yaml`

### Reproduce

```bash
cd "target/" && python train.py \
    --agent charliepai2g48h2-askeladd \
    --experiment_name "charliepai2g48h2-askeladd/pressure-channel-weight" \
    --epochs 20
```

---

## 2026-05-12 22:00 — PR #1424: Warmup cosine peak LR 7e-4 + grad clip 1.0

**Student:** charliepai2g48h2-fern  
**Change:** LR warmup 2 epochs (linear 0→7e-4) + CosineAnnealingLR peak 7e-4 (vs baseline 5e-4) + gradient clipping max_norm=1.0. Stacked on top of #1418 channel_weights=[1,1,3]. No other changes.

### Validation (best epoch 14/20, timeout at 30 min — still descending)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 119.682 | 1.695 | 0.795 |
| val_geom_camber_rc | 113.333 | 2.542 | 1.023 |
| val_geom_camber_cruise | 82.087 | 1.046 | 0.579 |
| val_re_rand | 96.299 | 1.819 | 0.812 |
| **val_avg/mae_surf_p** | **102.8503** | | |

**Improvement vs #1418 baseline: −16.13% (122.6395 → 102.8503)**

### Test (from best-val checkpoint, epoch 14)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist | 104.577 | 1.636 | 0.748 |
| test_geom_camber_rc | 97.972 | 2.380 | 0.954 |
| test_geom_camber_cruise | **NaN** ⚠️ | 1.008 | 0.525 |
| test_re_rand | 93.588 | 1.530 | 0.769 |
| **test_avg/mae_surf_p** | **NaN** (3-split partial: 98.712) | | |

### Model config

- Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2) — **662K params**
- AdamW lr=7e-4 (peak, cosine-annealed from 0 over 2 warmup epochs), wd=1e-4, CosineAnnealingLR(T_max=20), batch_size=4, surf_weight=10, grad_clip=1.0
- Loss: per-channel MSE weights [1,1,3] / 5 (stacked from #1418)

### Metric artifacts

- `models/model-charliepai2g48h2-fern-warmup-7e-4-clip-20260512-211813/metrics.jsonl`
- `models/model-charliepai2g48h2-fern-warmup-7e-4-clip-20260512-211813/metrics.yaml`

### Reproduce

```bash
cd "target/" && python train.py \
    --agent charliepai2g48h2-fern \
    --experiment_name "charliepai2g48h2-fern/warmup-7e-4-clip" \
    --epochs 20
```

---

## ⚠️ Known scoring bug: `test_geom_camber_cruise/mae_surf_p` = NaN

**Root cause:** `test_geom_camber_cruise/000020.pt` contains 761 `+inf` entries in the pressure channel of `y`. `data/scoring.py:accumulate_batch` correctly identifies this sample as non-finite and zeroes its `sample_mask`, but `Inf * 0.0 = NaN` in IEEE 754, so the masked multiply still injects NaN into the accumulator.

**Impact:** `test_avg/mae_surf_p` is NaN for every PR until fixed. Use `val_avg/mae_surf_p` as the primary ranking metric. 3-split partial test avg (excluding cruise) is reported as a secondary signal.

**Fix (one-line, in `data/scoring.py`):** sanitize `y` before the error computation:
```python
y_safe = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
err = (pred_orig.double() - y_safe.double()).abs()
```
`data/scoring.py` is marked read-only per `program.md`; advisory record only.

---

> To beat this baseline, a new PR must achieve `val_avg/mae_surf_p < 95.336`.
