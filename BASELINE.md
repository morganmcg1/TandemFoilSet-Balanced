# Baseline Metrics — `icml-appendix-charlie-pai2g-48h-r2`

> Primary ranking metric: **`val_avg/mae_surf_p`** (equal-weight mean surface pressure MAE across 4 val splits, physical units). Lower is better.
> Test metric: **`test_avg/mae_surf_p`** — now clean 4-split (NaN-skip fix merged in #1414).

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
