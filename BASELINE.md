# TandemFoilSet Baseline Metrics

Primary metric: `val_avg/mae_surf_p` — lower is better.
Paper metric: `test_avg/mae_surf_p` — lower is better.

---

## 2026-05-12 14:00 — PR #1502: Batch inverse-variance weighting for heteroscedastic Re

- **Branch:** `icml-appendix-willow-pai2g-48h-r4` (merged)
- **W&B run:** `e72nzxo5`
- **Best epoch:** 14 / 50 configured (hit 30-min wall-clock cap)
- **val_avg/mae_surf_p:** `126.0751` ← **current best**
- **test_avg/mae_surf_p:** `NaN` (pre-existing data/scoring bug: test_geom_camber_cruise sample 20 has 761 -inf in GT p-channel; `0×inf=NaN` in accumulate_batch poisons the split-average)

### Per-split val surface-p MAE (best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| `val_single_in_dist` | 160.7360 | 1.8779 | 0.8524 |
| `val_geom_camber_rc` | 133.2787 | 2.5736 | 1.0051 |
| `val_geom_camber_cruise` | 97.2075 | 1.5158 | 0.5869 |
| `val_re_rand` | 113.0781 | 1.9906 | 0.7708 |
| **val_avg** | **126.0751** | 1.9895 | 0.8038 |

### Per-split test surface-p MAE (best checkpoint — 3 of 4 clean)

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 145.4262 |
| `test_geom_camber_rc` | 117.4369 |
| `test_geom_camber_cruise` | **NaN** (data corruption) |
| `test_re_rand` | 109.2676 |
| **test_avg** | **NaN** (cruise split corrupts mean) |

Mean of 3 valid test splits: **~124.0** (indicative only).

### Reproduce

```bash
cd target && python train.py \
    --wandb_group per-sample-re-normalized-loss \
    --wandb_name bivw-mean1-clamp1e-4 \
    --agent willowpai2g48h4-tanjiro
```

### Notes

- BIVW weights each sample by `1 / var(y_norm_valid)`, normalized to mean=1.
  This re-balances gradient signal away from high-Re/high-variance samples.
- Test NaN is a known infrastructure issue, not a model quality issue.
  Until fixed: report all four individual test split numbers; compute manual
  3-split mean excluding cruise as a surrogate paper metric.

---

## 2026-05-12 20:30 — PR #1528: BIVW + zero-init surface correction head composition

- **Branch:** `willowpai2g48h4-thorfinn/surf-head-on-bivw` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **W&B run:** `an97gg8n`
- **Best epoch:** 13 / 14 run (hit 30-min wall-clock cap)
- **val_avg/mae_surf_p:** `119.2987` ← **current best** (−5.37% vs prior 126.0751)
- **test_avg/mae_surf_p:** `NaN` (same pre-existing cruise split bug)

### Per-split val surface-p MAE (best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| `val_single_in_dist` | 140.09 | — | — |
| `val_geom_camber_rc` | 142.40 | — | — |
| `val_geom_camber_cruise` | 85.98 | — | — |
| `val_re_rand` | 108.73 | — | — |
| **val_avg** | **119.2987** | — | — |

### Per-split test surface-p MAE (best checkpoint — 3 of 4 clean)

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 127.93 |
| `test_geom_camber_rc` | 127.18 |
| `test_geom_camber_cruise` | **NaN** (data corruption) |
| `test_re_rand` | 103.79 |
| **test_avg** | **NaN** (cruise split corrupts mean) |

Mean of 3 valid test splits: **~119.63** (indicative only).

### Reproduce

```bash
cd target && python train.py \
    --wandb_group surf-head-on-bivw \
    --wandb_name bivw-surf-head-zeroinit \
    --agent willowpai2g48h4-thorfinn
```

### Notes

- Composition: BIVW loss weighting (sample-level) + zero-init additive SurfaceCorrection MLP head (architectural specialisation).
- SurfaceCorrection head: `[3+24, 64, 64, 3]`, last layer zeroed at init, applied only at surface nodes.
- Regression on `val_geom_camber_rc` (+6.84%); three other splits improved substantially.
- Total params: 0.669M (Transolver 0.643M + SurfaceCorrection 0.026M).
- **All future PRs must beat `val_avg/mae_surf_p < 119.2987` to merge.**
- Test NaN infrastructure fixed in PR #1527 (merged) — `evaluate_split` now `nan_to_num`-guards both `pred_orig` and `y` before `accumulate_batch`, and passes an explicit `_y_ok` finite-sample mask. From PR #1527 forward, expect all four test split `mae_surf_p` values to be finite.
- Indicative test_avg from tanjiro's BIVW-only PR #1527 run (`dg5xbm6g`, no surf-head): `test_avg/mae_surf_p = 119.7792` with `test_geom_camber_cruise = 81.42`. Actual test_avg for BIVW+surf-head+fix combo pending next merged run.

---

## 2026-05-12 22:00 — PR #1558: Huber (SmoothL1) surface loss, delta=0.5

- **Branch:** `willowpai2g48h4-thorfinn/smooth-l1-surface-loss` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **W&B run:** `2w7nverc` (winning arm, delta=0.5); `3goyvktl` (delta=1.0 secondary)
- **Best epoch:** 14 / 14 completed (hit 30-min wall-clock cap; still improving)
- **val_avg/mae_surf_p:** `98.1642` ← **current best** (−17.72% vs prior 119.2987)
- **test_avg/mae_surf_p:** `NaN` (cruise split pre-existing bug); **test 3-split mean: 98.7537** (−17.45% vs ~119.63)

### Per-split val surface-p MAE (best checkpoint, delta=0.5)

| Split | mae_surf_p | vs prior baseline |
|-------|-----------|-------------------|
| `val_single_in_dist` | 123.14 | 140.09 → **−12.1%** ✓ |
| `val_geom_camber_rc` | 107.24 | 142.40 → **−24.7%** ✓ (OOD regression fully reversed) |
| `val_geom_camber_cruise` | 73.28 | 85.98 → **−14.8%** ✓ |
| `val_re_rand` | 88.99 | 108.73 → **−18.2%** ✓ |
| **val_avg** | **98.1642** | **−17.72%** |

### Per-split test surface-p MAE (3 of 4 clean, delta=0.5)

| Split | mae_surf_p | vs prior baseline |
|-------|-----------|-------------------|
| `test_single_in_dist` | 111.92 | 127.93 → **−12.5%** |
| `test_geom_camber_rc` | 98.91 | 127.18 → **−22.2%** |
| `test_geom_camber_cruise` | NaN | (pre-existing cruise bug) |
| `test_re_rand` | 85.43 | 103.79 → **−17.7%** |
| **test 3-split mean** | **98.7537** | **−17.45%** |

### Reproduce

```bash
cd target && python train.py \
    --huber_delta 0.5 \
    --wandb_group smooth-l1-surface-loss \
    --wandb_name huber-delta-0.5 \
    --agent willowpai2g48h4-thorfinn
```

### Notes

- Huber loss (SmoothL1, delta=0.5) on surface, MSE on volume. Applied to all 3 surface channels jointly.
- `delta=0.5` wins because most surface residuals in normalised space (~O(0.3–1.5)) fall in the L1 regime, giving constant-magnitude gradients that directly minimise MAE.
- `delta=1.0` gives only −1.3% (barely above noise floor) — too much residual in quadratic regime.
- Reverses val_geom_camber_rc OOD regression from PR #1528 (+6.84% → −24.7%): Huber suppresses the large-residual surf-head pull toward OOD outlier nodes.
- Synergy with BIVW: BIVW removes between-sample gradient inflation; Huber removes within-sample per-node gradient inflation — orthogonal channels that compound.
- **All future PRs must beat `val_avg/mae_surf_p < 98.1642` to merge.**
- Test cruise NaN is unchanged; use 3-split mean as surrogate paper metric.
