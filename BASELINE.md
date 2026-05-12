# Baseline — `icml-appendix-charlie-pai2g-48h-r5`

This branch is the **Charlie no-W&B logging ablation, round 5 (charlie-pai2g-48h-r5)**.

Experiment metrics are written to local JSONL only (`models/<experiment>/metrics.jsonl`).
**Do not** add or query W&B / wandb experiment logging for this arm.

## Primary ranking metric

- **Validation:** `val_avg/mae_surf_p` — equal-weight mean of surface pressure MAE
  across the four val tracks (`val_single_in_dist`, `val_geom_camber_rc`,
  `val_geom_camber_cruise`, `val_re_rand`). Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` from the best-val checkpoint.

> ⚠️ **Round-5 scoring bug (affects all PRs):** `test_geom_camber_cruise/000020.pt`
> contains ±Inf values in the `p` channel (761 nodes of 225K). In `data/scoring.py`,
> the per-sample `y_finite` mask correctly identifies the bad sample, but the
> subsequent `err * surf_mask` sum computes `0 * Inf = NaN` (IEEE-754), which
> propagates into `test_avg/mae_surf_p`. **Round-5 merge decisions are therefore
> made on `val_avg/mae_surf_p` alone.** The test metric column below records the
> partial 3-split value (excluding `test_geom_camber_cruise`) until a data or
> scoring fix is in place.

## Reference configuration (train.py defaults)

```
lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50
model_config = dict(
    space_dim=2, fun_dim=22, out_dim=3,
    n_hidden=128, n_layers=5, n_head=4,
    slice_num=64, mlp_ratio=2,
)
optimizer = AdamW; scheduler = CosineAnnealingLR(T_max=epochs)
```

Each training execution is hard-capped by `SENPAI_TIMEOUT_MINUTES=30` (wall clock).
`--epochs 50` is an upper bound; runs typically reach 12-16 epochs under the
30-min cap at the default model size.

## Current best (val)

| Metric | Value | PR | Config | Notes |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **110.7608** | #1444 | Smooth-L1 (Huber β=1.0), else defaults | epoch 14 of 50; still improving at timeout |
| `test_avg/mae_surf_p` | NaN (bug) / 112.40 (3-split excl. cruise) | #1444 | — | see scoring bug note above |

All subsequent PRs must beat `val_avg/mae_surf_p < 110.7608` to be merged.

## 2026-05-12 — PR #1444: Swap MSE → Smooth-L1 (Huber, beta=1.0)

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 14 (wall-clock-bound at 30 min; model still improving)
- **Peak GPU memory:** 42.1 GB
- **Time per epoch:** ~131 s

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 135.16 | 1.719 | 0.769 |
| `val_geom_camber_rc` | 129.08 | 2.104 | 0.988 |
| `val_geom_camber_cruise` | 77.70 | 1.047 | 0.555 |
| `val_re_rand` | 101.10 | 1.607 | 0.740 |
| **val_avg** | **110.76** | — | — |

- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-smooth-l1-loss-20260512-180133/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-smooth-l1-loss-20260512-180133/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/smooth-l1-loss" \
      --epochs 50
  ```
  (plus the Smooth-L1 substitution in `train.py` — see PR #1444 diff)

## Reproduce command (reference defaults)

```bash
cd target && python train.py \
    --agent <student> \
    --experiment_name "<student>/<short-description>" \
    --epochs 50
```

Commit `models/<experiment>/metrics.jsonl` and `metrics.yaml` with the PR and
quote the key values in the PR results comment plus the
`SENPAI-RESULT` terminal marker.
