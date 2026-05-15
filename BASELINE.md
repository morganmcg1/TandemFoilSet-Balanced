# Baseline — TandemFoilSet — `icml-appendix-willow-pai2i-24h-r2`

## Current best

**PR #3200 (fern) — Fourier position features on (x, z), 8 frequency bands** — merged 2026-05-15 17:22

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | **121.4956** |
| `test_avg/mae_surf_p` | **112.4884** |
| Best val epoch | 14 (hit 30-min wall clock; val curve still descending) |
| W&B run | `t1ai7kzf` (8 bands, primary) / `oj5578rn` (12 bands, dropped) |
| Params | 0.67 M |
| Peak GPU memory | ~42.5 GB |

### Per-split surface-pressure MAE (8 bands, best-val checkpoint)

| Split | val | test |
|---|---|---|
| `single_in_dist` | 139.80 | 122.01 |
| `geom_camber_rc` | 138.71 | 133.37 |
| `geom_camber_cruise` | 93.55 | 83.11 |
| `re_rand` | 113.93 | 111.46 |

### Configuration

| Component | Value |
|---|---|
| Model | Transolver `n_layers=5, n_hidden=128, n_head=4, slice_num=64, mlp_ratio=2` |
| Input augmentation | 8 sinusoidal Fourier bands on normalized (x, z) → `fun_dim=54` (was 22) |
| Optimizer | AdamW `lr=5e-4, weight_decay=1e-4` |
| Scheduler | `CosineAnnealingLR(T_max=epochs)` (no warmup) |
| Loss | `vol_loss + 10.0 * surf_loss`, MSE on normalized targets |
| `batch_size` | 4 |
| Sampler | `WeightedRandomSampler` over balanced domain weights |
| Epochs | 50 (capped by `SENPAI_MAX_EPOCHS=50`; wall-clock cap typically hits ~ep 14) |
| Wall-clock cap | `SENPAI_TIMEOUT_MINUTES=30.0` |

### Defensive fix included in this PR

`evaluate_split` now zeros the row of `mask` for any sample with non-finite `y`, and replaces non-finite entries with `0.0`, before calling `accumulate_batch`. This works around the read-only `data/scoring.py:48` bug where `(pred - y).abs() * mask` produces `NaN` from `inf * 0` before the per-sample skip, poisoning the float64 accumulator (root cause: `splits_v2/.test_geom_camber_cruise_gt/000020.pt` has 761 non-finite `p` entries on volume nodes). Without this fix, `test_avg/mae_surf_p` reads `NaN` even for healthy training runs.

### Reproduce

```bash
cd target/ && python train.py --agent willowpai2i24h2-fern \
    --wandb_name "willowpai2i24h2-fern/fourier-features-8bands" \
    --wandb_group "willow-pai2i-24h-r2/fourier-features"
```

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

## Merge history

| Date | PR | Title | val_avg | test_avg | Δ val_avg vs prior |
|---|---|---|---|---|---|
| 2026-05-15 17:22 | #3200 | Fourier position encoding (8 bands) | 121.4956 | 112.4884 | first baseline |
