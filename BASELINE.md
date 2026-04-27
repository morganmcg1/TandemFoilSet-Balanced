# Baseline — `icml-appendix-charlie-pai2d-r1`

## Current best
- **`val_avg/mae_surf_p` = 132.276** (EMA, epoch 13/50, timeout-cut)
- **`test_avg/mae_surf_p` = 118.041**
- Set by **PR #356** (`charliepai2d1-tanjiro/ema-eval`), merged 2026-04-27 23:42 UTC.

## Configuration
- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Optimizer: AdamW (`lr=5e-4`, `weight_decay=1e-4`)
- Loss: `MSE_vol + 10.0 * MSE_surf` (normalized space)
- Schedule: cosine annealing over `epochs`
- Batch: `batch_size=4`, balanced-domain weighted sampling
- **EMA(0.999) shadow weights** drive validation, best-checkpoint selection, and final test eval. The companion raw-model val is logged for free-lunch attribution but does not feed checkpoint selection.
- **`evaluate_split` NaN-safe pre-pass:** samples with non-finite ground truth are dropped from `mask` and have `y` zeroed before loss/MAE computation. Required because `data/scoring.py:accumulate_batch` is read-only and IEEE 754 `NaN*0 = NaN` defeats its per-sample mask. Concretely affects `test_geom_camber_cruise` index 20, whose `y[p]` is non-finite.

## Primary ranking metric
- `val_avg/mae_surf_p` — mean surface-pressure MAE across the four val splits (lower is better)
- `test_avg/mae_surf_p` — final paper-facing number

## Per-split breakdown (PR #356 best-EMA-epoch checkpoint)

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 170.491 | 2.131 | 0.910 | 147.132 | 1.948 | 0.827 |
| geom_camber_rc | 144.104 | 3.014 | 1.099 | 127.917 | 2.988 | 1.076 |
| geom_camber_cruise | 100.492 | 1.612 | 0.633 | 84.026 | 1.332 | 0.553 |
| re_rand | 114.015 | 2.187 | 0.844 | 113.089 | 1.949 | 0.833 |
| **avg** | **132.276** | 2.236 | 0.872 | **118.041** | 2.054 | 0.822 |

## Reproduce (current baseline)
```
cd target/ && python train.py --agent <name> --experiment_name <name>/<slug>
```
(Trains with EMA shadow + NaN-safe pre-pass per current `train.py`. Hits `SENPAI_TIMEOUT_MINUTES=30` at ~ep13 of 50 configured.)

## History
- 2026-04-27 23:42 — **PR #356** (tanjiro/ema-eval): EMA(0.999) shadow + NaN-safe pre-pass. First measured baseline on this branch (val=132.276, test=118.041).
