# Baseline — `icml-appendix-charlie-pai2d-r1`

## Current best
- **`val_avg/mae_surf_p` = 98.581** (EMA, epoch 13/50, timeout-cut)
- **`test_avg/mae_surf_p` = 87.881**
- Set by **PR #417** (`charliepai2d1-askeladd/ema-decay-0p99`), merged 2026-04-28 01:54 UTC.
- Askeladd's measured run was on `EMA(0.99) + max_norm=1.0 + lr=5e-4` (the post-#374 base). Squash-merge composed `ema_decay=0.99` (from #417) with `max_norm=0.5` (from #402) and `lr=1e-3` (from #408) → current baseline `train.py` is `EMA(0.99) + grad-clip(0.5) + lr=1e-3 + NaN-safe pre-pass`. The recorded numbers are the comparison floor; future re-runs of the unmodified baseline may land slightly different.

## Configuration
- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`)
- Loss: `MSE_vol + 10.0 * MSE_surf` (normalized space)
- Schedule: cosine annealing over `epochs`
- Batch: `batch_size=4`, balanced-domain weighted sampling
- **EMA(0.99) shadow weights** drive validation, best-checkpoint selection, and final test eval. Half-life ~70 steps ≈ 0.18 epochs at 375 batches/epoch — much shorter than the previous EMA(0.999), so the shadow tracks the live iterate within a fraction of an epoch.
- **`evaluate_split` NaN-safe pre-pass:** drops samples with non-finite ground truth from `mask` and zeros their `y` before loss/MAE computation.
- **Gradient clipping at `max_norm=0.5`** between `loss.backward()` and `optimizer.step()`. Pre-clip mean grad norm ~44 across training; clip envelope dominates per-step magnitude at lr=1e-3.

## Primary ranking metric
- `val_avg/mae_surf_p` — mean surface-pressure MAE across the four val splits (lower is better)
- `test_avg/mae_surf_p` — final paper-facing number

## Per-split breakdown (PR #417 best-EMA-epoch checkpoint, EMA(0.99) + max_norm=1.0 + lr=5e-4)

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 118.994 | 1.350 | 0.668 | 103.758 | 1.322 | 0.651 |
| geom_camber_rc | 107.257 | 2.185 | 0.928 | 97.234 | 2.064 | 0.868 |
| geom_camber_cruise | 75.099 | 0.830 | 0.508 | 63.068 | 0.789 | 0.451 |
| re_rand | 92.974 | 1.482 | 0.705 | 87.463 | 1.316 | 0.662 |
| **avg** | **98.581** | 1.462 | 0.702 | **87.881** | 1.373 | 0.658 |

## Reproduce (current baseline)
```
cd target/ && python train.py --agent <name> --experiment_name <name>/<slug>
```
(Trains with EMA(0.99) + NaN-safe pre-pass + grad-clip(0.5) + lr=1e-3 per current `train.py`. Hits `SENPAI_TIMEOUT_MINUTES=30` at ~ep13 of 50 configured.)

## History
- 2026-04-27 23:42 — **PR #356** (tanjiro/ema-eval): EMA(0.999) shadow + NaN-safe pre-pass. First measured baseline (val=132.276, test=118.041).
- 2026-04-28 00:43 — **PR #374** (tanjiro/grad-clip-1p0): added `clip_grad_norm_(model.parameters(), max_norm=1.0)`. val=113.157 (−14.45 % vs #356), test=99.322 (−15.86 %).
- 2026-04-28 01:29 — **PR #402** (tanjiro/grad-clip-0p5): tightened `max_norm=1.0 → 0.5`. val=110.822 (−2.07 %), test=97.955 (−1.38 %).
- 2026-04-28 01:41 — **PR #408** (fern/higher-lr-1e3): bumped `lr=5e-4 → 1e-3`. val=107.957 (−2.59 %), test=95.675 (−2.33 %).
- 2026-04-28 01:54 — **PR #417** (askeladd/ema-decay-0p99): tightened `ema_decay=0.999 → 0.99`. val=98.581 (−8.69 %), test=87.881 (−8.15 %). Mechanism: at 13-epoch under-converged budget, the live iterate is improving fast — shorter EMA window captures recent (better) iterate before old (worse) iterate drags the shadow back. EMA shadow consistently 20+ pts better than raw at every epoch except ep1.
