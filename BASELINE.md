# Baseline Metrics — icml-appendix-charlie-pai2i-24h-r2

## Current best

### 2026-05-15 15:30 — PR #3276: Gradient clip + AdamW selective decay (+ test NaN guard)

- **val_avg/mae_surf_p:** 109.681 (best @ epoch 14; 14 epochs completed under 30-min cap)
- **test_avg/mae_surf_p:** 97.315 (finite for first time — NaN guard fixed test_geom_camber_cruise)
- **Per-split val mae_surf_p:** single 148.09 | geom_rc 114.87 | geom_cruise 78.85 | re_rand 96.91
- **Per-split test mae_surf_p:** single 123.24 | geom_rc 104.76 | geom_cruise 68.48 | re_rand 92.79
- **Changes:** torch.nn.utils.clip_grad_norm\_(max_norm=1.0) + AdamW selective decay (LN/bias/1D no-decay) + NaN sample guard in evaluate_split
- **Optimizer groups:** decay=49 groups (0.655M params), no_decay=62 groups (0.008M params)
- **Loss:** SmoothL1 (Huber, β=1.0) — carried forward from PR #3208
- **Metric artifacts:** `models/model-grad-clip-selective-decay-20260515-142950/metrics.{jsonl,yaml}`
- **Reproduce:** `cd target && python train.py --experiment_name grad-clip-selective-decay --agent fern --epochs 50`
- **Delta vs PR #3208 baseline:** -5.94% val_avg/mae_surf_p (116.61 → 109.68)

### 2026-05-15 14:05 — PR #3208: Replace MSE with SmoothL1 (Huber) loss (superseded)

- **val_avg/mae_surf_p:** 116.611 (best @ epoch 13; 14 epochs completed under 30-min cap)
- **Metric artifacts:** `models/model-charliepai2i24h2-fern-huber-loss-20260515-130151/metrics.{jsonl,yaml}`

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
