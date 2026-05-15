# Baseline Metrics — icml-appendix-charlie-pai2i-24h-r2

## Current best

### 2026-05-15 14:05 — PR #3208: Replace MSE with SmoothL1 (Huber) loss

- **val_avg/mae_surf_p:** 116.611 (best @ epoch 13; 14 epochs completed under 30-min cap)
- **test_avg/mae_surf_p:** NaN (pre-existing infra bug); 3 clean splits avg 114.59
- **Per-split val mae_surf_p:** single 161.69 | geom_rc 117.56 | geom_cruise 85.67 | re_rand 101.53
- **Per-split test mae_surf_p:** single 139.80 | geom_rc 104.38 | geom_cruise NaN | re_rand 99.60
- **Loss:** SmoothL1 (Huber, β=1.0) replacing MSE; everything else at reference config
- **Metric artifacts:** `models/model-charliepai2i24h2-fern-huber-loss-20260515-130151/metrics.{jsonl,yaml}`
- **Reproduce:** `cd target && python train.py --experiment_name huber-loss --agent fern --epochs 50`

## Reference model config
```python
model_config = dict(
    space_dim=2,
    fun_dim=22,      # X_DIM (24) - 2 position dims
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
)
```
~1M params.

## Reference training config
- AdamW: lr=5e-4, weight_decay=1e-4
- batch_size=4
- surf_weight=10.0 (additional surface loss weight in normalized MSE)
- epochs=50, cosine annealing schedule (T_max=epochs)
- Loss: MSE in normalized target space; vol_loss + surf_weight * surf_loss
- Balanced domain sampler (WeightedRandomSampler) over 1499 train samples

## Diagnostic targets (per split, surface MAE for p)
We track per-split surface pressure MAE separately so that geometry vs Re axes are visible:
- `val_single_in_dist/mae_surf_p`
- `val_geom_camber_rc/mae_surf_p`
- `val_geom_camber_cruise/mae_surf_p`
- `val_re_rand/mae_surf_p`

The same four exist for test splits. We report all four plus the equal-weight average.
