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
- **All future PRs must beat `val_avg/mae_surf_p < 126.0751` to merge.**
- Test NaN is a known infrastructure issue, not a model quality issue.
  Until fixed: report all four individual test split numbers; compute manual
  3-split mean excluding cruise as a surrogate paper metric.
