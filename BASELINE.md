# Baseline Metrics — icml-appendix-charlie-pai2i-24h-r2

## Current best
- **Source:** initial Transolver baseline (`train.py` at branch HEAD `abc2559`)
- **Primary validation metric (lower is better):** `val_avg/mae_surf_p` — not yet measured on this branch.
- **Primary test metric:** `test_avg/mae_surf_p` — not yet measured on this branch.
- **Notes:** no PRs merged on this branch yet. The first round of experiments will both establish concrete baseline numbers and probe orthogonal improvement axes.

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
