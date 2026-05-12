# Baseline Metrics

## Current Baseline — PR #1479 (grad-clip-1)

**val_avg/mae_surf_p = 117.17** (epoch 13 / 14 completed in 30-min cap)

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: `AdamW(lr=5e-4, wd=1e-4)`, `CosineAnnealingLR(T_max=50)`
- `batch_size=4`, `surf_weight=10.0`, **`grad_clip=1.0`** ← key addition
- MSE loss in normalized space
- ~130s/epoch; 14 epochs in 30 min

**Per-split val at best epoch (13):**

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|-------|-----------|-------------|-------------|-----------|
| val_single_in_dist | 134.83 | 1.64 | 0.69 | 145.44 |
| val_geom_camber_rc | 134.17 | 2.34 | 1.01 | 144.74 |
| val_geom_camber_cruise | **87.04** | 1.21 | 0.54 | 85.52 |
| val_re_rand | 112.66 | 1.84 | 0.78 | 111.64 |
| **val_avg** | **117.17** | 1.76 | 0.76 | 121.84 |

**Test (3 of 4 splits; test_geom_camber_cruise NaN due to data bug in sample 20):**

| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 120.25 |
| test_geom_camber_rc | 122.24 |
| test_re_rand | 106.02 |
| test_geom_camber_cruise | NaN (data bug) |

**Artifact**: `models/model-charliepai2g24h1-thorfinn-grad-clip-1-20260512-180544/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# grad_clip=1.0 is now the default Config value on this branch
```

**Key diagnostic**: Pre-clip gradient norms range 50–800 (mean ~50, max ~800). Clipping fires on 100% of batches. The baseline Transolver is gradient-unstable without clipping — this is now mandatory.

---

## Update Log

| Date | PR | val_avg/mae_surf_p | test_avg (3-split) | Notes |
|------|----|--------------------|---------------------|-------|
| 2026-05-12 | #1479 | **117.17** | 116.17 | Round 1 winner. grad_clip=1.0; 14 epochs / 30 min |
