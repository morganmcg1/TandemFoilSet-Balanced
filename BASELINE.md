# Baseline — TandemFoilSet — `icml-appendix-willow-pai2i-24h-r2`

## Status

Fresh research track. No PR has been merged into this advisor branch yet. The "baseline" is the configuration defined in `train.py` HEAD: a vanilla Transolver.

Once a PR beats the in-branch configured defaults on `val_avg/mae_surf_p` and is squash-merged, this file will be updated with the new PR number and metrics.

## Configured baseline (from `train.py` at HEAD)

| Component | Value |
|---|---|
| Model | Transolver (PhysicsAttention slice-token transformer) |
| `n_layers` | 5 |
| `n_hidden` | 128 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `space_dim` | 2 |
| `fun_dim` | `X_DIM - 2 = 22` |
| `out_dim` | 3 (`Ux, Uy, p` in normalized space) |
| Optimizer | AdamW |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| Scheduler | `CosineAnnealingLR(T_max=epochs)` — no warmup |
| Loss | `vol_loss + surf_weight * surf_loss`, MSE on normalized targets |
| `surf_weight` | 10.0 |
| `batch_size` | 4 |
| Sampler | `WeightedRandomSampler` over balanced domain weights |
| `epochs` | 50 (capped by `SENPAI_MAX_EPOCHS=50`) |
| Wall-clock cap | `SENPAI_TIMEOUT_MINUTES=30.0` |

## Primary ranking metric

- **Validation (checkpoint selection):** `val_avg/mae_surf_p` — equal-weight mean across the four val splits.
- **Test (paper-facing):** `test_avg/mae_surf_p` — same metric on the test splits, evaluated from the best-val checkpoint.

Lower is better. All metrics computed in original (denormalized) y-space, float64, surface-only nodes summed and divided by total surface-node count in the split.

## Per-split tracks

| Split | Tests |
|---|---|
| `val_single_in_dist` | Random holdout of single-foil samples |
| `val_geom_camber_rc` | Unseen front-foil camber M=6-8 (raceCar tandem P2) |
| `val_geom_camber_cruise` | Unseen front-foil camber M=2-4 (cruise tandem P2) |
| `val_re_rand` | Stratified Re holdout across tandem domains |

## Known baseline metrics

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | TBD (no training run on this branch yet — first round of PRs will establish) |
| `test_avg/mae_surf_p` | TBD |
