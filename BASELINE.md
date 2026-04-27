# Baseline — TandemFoilSet (willow-pai2d-r5)

**Status:** Round 1 baseline empirically set by PR #336 (slice_num=128). Subsequent winners compound on top.

## Reference configuration (current `train.py` HEAD)

The baseline is the default Transolver in `train.py` at HEAD of `icml-appendix-willow-pai2d-r5`:

- **Model:** Transolver, `n_layers=5`, `n_hidden=128`, `n_head=4`, `slice_num=128`, `mlp_ratio=2` (~0.67M params)
- **Optimizer:** AdamW `lr=5e-4`, `weight_decay=1e-4`
- **Schedule:** CosineAnnealingLR with `T_max=epochs`
- **Batch size:** 4
- **Loss:** MSE in normalized space, `loss = vol_loss + surf_weight * surf_loss`, `surf_weight=10`
- **Training:** `epochs=50`, capped by `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Sampling:** `WeightedRandomSampler` over balanced domain weights

## Reproduce command

```bash
cd /workspace/senpai/target
python train.py --epochs 50
```

## Primary metric

**`val_avg/mae_surf_p`** — equal-weight mean of surface pressure MAE across the four validation splits:
- `val_single_in_dist/mae_surf_p`
- `val_geom_camber_rc/mae_surf_p`
- `val_geom_camber_cruise/mae_surf_p`
- `val_re_rand/mae_surf_p`

Lower is better. The matching test metric `test_avg/mae_surf_p` is computed at the end of every run from the best validation checkpoint.

## Best results

| PR | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|----|--------------------|---------------------|-------|
| **#336** | **139.83** | **142.79 *** | slice_num 64→128, best ckpt epoch 10 of 11 (timeout-bound) |

\* `test_avg/mae_surf_p` is mean over the **three finite splits** (`test_single_in_dist`, `test_geom_camber_rc`, `test_re_rand`). The fourth split (`test_geom_camber_cruise`) returns NaN due to a known `data/scoring.py` bug: ground truth `y` for cruise sample 000020 has 761 NaN values in the `p` channel, and the per-sample skip mask in `accumulate_batch` does not survive `NaN * 0.0 = NaN` propagation. Bug fix in flight.

### Per-split surface pressure MAE — PR #336

| Split | val | test |
|---|---:|---:|
| single_in_dist | 179.11 | 161.35 |
| geom_camber_rc | 144.31 | 137.40 |
| geom_camber_cruise | 110.05 | NaN (bug) |
| re_rand | 125.87 | 129.61 |
| **avg (3 finite splits, test only)** | — | 142.79 |

W&B run: `slices_128` / `8xow4ge3` (group `capacity_slices`).
