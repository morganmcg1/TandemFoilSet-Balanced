# Baseline — `icml-appendix-charlie-pai2d-r1`

## Current best
- **`val_avg/mae_surf_p` = 110.822** (EMA, epoch 13/50, timeout-cut)
- **`test_avg/mae_surf_p` = 97.955**
- Set by **PR #402** (`charliepai2d1-tanjiro/grad-clip-0p5`), merged 2026-04-28 01:29 UTC.
- Beats prior baseline on all four val splits and all four test splits.

## Configuration
- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Optimizer: AdamW (`lr=5e-4`, `weight_decay=1e-4`)
- Loss: `MSE_vol + 10.0 * MSE_surf` (normalized space)
- Schedule: cosine annealing over `epochs`
- Batch: `batch_size=4`, balanced-domain weighted sampling
- **EMA(0.999) shadow weights** drive validation, best-checkpoint selection, and final test eval. Companion raw-model val is logged for free-lunch attribution.
- **`evaluate_split` NaN-safe pre-pass:** drops samples with non-finite ground truth from `mask` and zeros their `y` before loss/MAE computation. Required because `data/scoring.py:accumulate_batch` is read-only and IEEE 754 `NaN*0 = NaN` defeats its per-sample mask. Affects `test_geom_camber_cruise` index 20 (`y[:,2]` non-finite).
- **Gradient clipping at `max_norm=0.5`** between `loss.backward()` and `optimizer.step()`. Pre-clip mean grad norm 71 across training; clip is firing aggressively as an effective LR cap. Pre-clip mean-per-epoch norm logged as `train/grad_norm`.

## Primary ranking metric
- `val_avg/mae_surf_p` — mean surface-pressure MAE across the four val splits (lower is better)
- `test_avg/mae_surf_p` — final paper-facing number

## Per-split breakdown (PR #402 best-EMA-epoch checkpoint)

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 130.231 | 1.622 | 0.731 | 115.322 | 1.544 | 0.710 |
| geom_camber_rc | 129.768 | 2.635 | 1.025 | 111.362 | 2.463 | 0.947 |
| geom_camber_cruise | 82.706 | 1.018 | 0.520 | 68.760 | 0.961 | 0.466 |
| re_rand | 100.585 | 1.748 | 0.760 | 96.374 | 1.554 | 0.719 |
| **avg** | **110.822** | 1.756 | 0.759 | **97.955** | 1.631 | 0.711 |

## Reproduce (current baseline)
```
cd target/ && python train.py --agent <name> --experiment_name <name>/<slug>
```
(Trains with EMA(0.999) + NaN-safe pre-pass + grad-clip(0.5) per current `train.py`. Hits `SENPAI_TIMEOUT_MINUTES=30` at ~ep13 of 50 configured.)

## History
- 2026-04-27 23:42 — **PR #356** (tanjiro/ema-eval): EMA(0.999) shadow + NaN-safe pre-pass. First measured baseline (val=132.276, test=118.041).
- 2026-04-28 00:43 — **PR #374** (tanjiro/grad-clip-1p0): added `clip_grad_norm_(model.parameters(), max_norm=1.0)`. Baseline at val=113.157 (−14.45 % vs #356), test=99.322 (−15.86 %). Pre-clip grad norms 50–100× `max_norm` → effective LR cap.
- 2026-04-28 01:29 — **PR #402** (tanjiro/grad-clip-0p5): tightened `max_norm=1.0 → 0.5`. New baseline at val=110.822 (−2.07 % vs #374), test=97.955 (−1.38 %). Diminishing-returns shape on the clipping lever now mapped: no-clip → 1.0 = −14 %; 1.0 → 0.5 = −2 %.
