# Baseline — icml-appendix-charlie-pai2g-48h-r3

## 2026-05-12 19:30 — PR #1392: n_layers 5 → 6 (moderate depth increase)

**New best: `val_avg/mae_surf_p` = 128.127** (epoch 12, 30-min wall-clock cap)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | **6** ← changed |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | **4** (default from PR #1408) |
| `space_dim` | 2 |
| `unified_pos` | False |
| Loss | MSE in normalized space, surf_weight=10 |
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| `batch_size` | 4 |
| `epochs` | 50 |
| Run cap | 30 min wall clock per training execution |

> **Note:** The empirical run used `mlp_ratio=2` (branched before PR #1408). The merged train.py
> now defaults to `mlp_ratio=4, n_layers=6`. Future runs stack both improvements.

### Val metrics (best checkpoint, epoch 12)

| Split | `mae_surf_p` | `mae_surf_Ux` | `mae_surf_Uy` |
|---|---|---|---|
| val_single_in_dist | 159.746 | 1.890 | 0.915 |
| val_geom_camber_rc | 136.513 | 3.068 | 1.235 |
| val_geom_camber_cruise | 102.432 | 1.675 | 0.656 |
| val_re_rand | 113.819 | 2.338 | 0.914 |
| **val_avg/mae_surf_p** | **128.127** | — | — |

### Improvement vs PR #1408 baseline (141.356)

| Split | Old | New | Δ |
|---|---|---|---|
| val_single_in_dist | 171.424 | 159.746 | −6.8% |
| val_geom_camber_rc | 159.804 | 136.513 | −14.6% |
| val_geom_camber_cruise | 104.607 | 102.432 | −2.1% |
| val_re_rand | 129.589 | 113.819 | −12.2% |
| **val_avg** | **141.356** | **128.127** | **−9.4%** |

### Test metrics (best-val checkpoint, epoch 12)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 145.477 |
| test_geom_camber_rc | 122.697 |
| test_geom_camber_cruise | **NaN** (scoring bug — GT sample 20 has -inf pressure) |
| test_re_rand | 114.851 |
| test_avg (3 finite splits) | **~127.68** |
| **test_avg/mae_surf_p** | NaN (blocked by cruise bug) |

> **Note on test NaN:** Same scorer bug as PR #1408. Use `val_avg/mae_surf_p` as primary ranking
> metric for this branch.

### Metric artifacts

`models/model-charliepai2g48h3-nezuko-deeper-transolver-6layers-20260512-191742/metrics.jsonl`

### Reproduce

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <name> \
  --epochs 50 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10
```

Note: `mlp_ratio=4, n_layers=6` are now the defaults in `train.py` (merged from PRs #1408, #1392).
No extra flags needed.

---

## 2026-05-12 18:56 — PR #1408: MLP expansion ratio 2 → 4 (canonical transformer recipe)

**Previous best: `val_avg/mae_surf_p` = 141.356** (epoch 13, 30-min wall-clock cap)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | **4** ← changed |
| `space_dim` | 2 |
| `unified_pos` | False |
| Loss | MSE in normalized space, surf_weight=10 |
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| `batch_size` | 4 |
| `epochs` | 50 |
| Run cap | 30 min wall clock per training execution |

### Val metrics (best checkpoint, epoch 13)

| Split | `mae_surf_p` | `mae_surf_Ux` | `mae_surf_Uy` |
|---|---|---|---|
| val_single_in_dist | 171.424 | 2.560 | 1.119 |
| val_geom_camber_rc | 159.804 | 3.611 | 1.420 |
| val_geom_camber_cruise | 104.607 | 1.759 | 0.718 |
| val_re_rand | 129.589 | 2.258 | 0.940 |
| **val_avg/mae_surf_p** | **141.356** | — | — |

### Test metrics (best-val checkpoint)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 149.585 |
| test_geom_camber_rc | 142.249 |
| test_geom_camber_cruise | **NaN** (scoring bug — GT sample 20 has -inf pressure; `inf * 0 = NaN` in float64 accumulator) |
| test_re_rand | 126.704 |
| test_avg (3 finite splits) | **~139.51** |
| **test_avg/mae_surf_p** | NaN (blocked by cruise bug) |

> **Note on test NaN:** `data/scoring.py` (read-only) computes `err = (pred - y).abs()` before
> applying the per-sample finite-GT mask, so a single `-inf` in GT propagates to NaN even on
> masked (zero-weight) nodes. This affects all models on `test_geom_camber_cruise` until the
> scorer is patched. Use `val_avg/mae_surf_p` as the primary ranking metric for this branch.

### Metric artifacts

`models/model-charliepai2g48h3-thorfinn-mlp-ratio-4-20260512-175522/metrics.jsonl`

### Reproduce

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <name> \
  --epochs 50 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10
```

Note: `mlp_ratio=4` is now the default in `train.py` (merged from PR #1408). No extra flag needed.

---

## Benchmark to beat

**`val_avg/mae_surf_p` < 128.127** — lower is better.

All new student experiments should compare against this number. The per-split breakdown above
shows `val_single_in_dist` (159.7) and `val_geom_camber_rc` (136.5) are the hardest splits to
improve; `val_geom_camber_cruise` (102.4) and `val_re_rand` (113.8) are relatively stronger.
