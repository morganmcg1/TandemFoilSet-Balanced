# Baseline

The current research baseline on `icml-appendix-willow-pai2d-r2`.

## Default Transolver

Default `train.py` config — no overrides:

```
lr=5e-4  weight_decay=1e-4  batch_size=4  surf_weight=10.0  epochs=50
n_hidden=128  n_layers=5  n_head=4  slice_num=64  mlp_ratio=2
```

The primary ranking metric is `val_avg/mae_surf_p` — equal-weight mean surface
pressure MAE across the four validation tracks. Lower is better.

The current baseline is the unmodified `train.py` on this branch. No PR has yet
beat it; the first winning PR will populate this file with concrete metrics.

## Validation tracks

- `val_single_in_dist` — single-foil sanity
- `val_geom_camber_rc` — held-out raceCar tandem front-foil camber (M=6-8)
- `val_geom_camber_cruise` — held-out cruise tandem front-foil camber (M=2-4)
- `val_re_rand` — stratified Re holdout across all tandem domains

The paper-facing test counterpart is `test_avg/mae_surf_p`, computed at the end
of every run from the best validation checkpoint.
