# Baseline — `icml-appendix-charlie-pai2d-r1`

## Current best
- **`val_avg/mae_surf_p` = 89.349** (EMA, epoch 12/50, timeout-cut)
- **`test_avg/mae_surf_p` = 79.191**
- Set by **PR #398** (`charliepai2d1-nezuko/swiglu-mlp-matched`), merged 2026-04-28 02:48 UTC.

## Configuration
- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- **MLP block**: SwiGLU `(W_g(x) ⊙ silu(W_v(x))) W_o` at `swiglu_inner=168` (matched param count vs GELU at the same `mlp_ratio=2 hidden=128`; 657,639 vs 662,359 = −0.71 %).
- Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`)
- Loss: `MSE_vol + 10.0 * MSE_surf` (normalized space)
- Schedule: cosine annealing over `epochs` (T_max=50)
- Batch: `batch_size=4`, balanced-domain weighted sampling
- **EMA(0.99) shadow weights** drive validation, best-checkpoint selection, and final test eval.
- **`evaluate_split` NaN-safe pre-pass:** drops samples with non-finite ground truth from `mask` and zeros their `y` before loss/MAE computation.
- **Gradient clipping at `max_norm=0.5`** between `loss.backward()` and `optimizer.step()`.

## Primary ranking metric
- `val_avg/mae_surf_p` — mean surface-pressure MAE across the four val splits (lower is better)
- `test_avg/mae_surf_p` — final paper-facing number

## Per-split breakdown (PR #398 best-EMA-epoch checkpoint)

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 100.379 | 1.186 | 0.589 | 90.874 | 1.161 | 0.564 |
| geom_camber_rc | 105.391 | 2.021 | 0.796 | 91.764 | 1.922 | 0.746 |
| geom_camber_cruise | 67.094 | 0.807 | 0.450 | 56.186 | 0.758 | 0.406 |
| re_rand | 84.532 | 1.356 | 0.637 | 77.940 | 1.212 | 0.593 |
| **avg** | **89.349** | 1.343 | 0.618 | **79.191** | 1.263 | 0.577 |

## Reproduce (current baseline)
```
cd target/ && python train.py --agent <name> --experiment_name <name>/<slug>
```
(Trains with SwiGLU(168) + EMA(0.99) + NaN-safe pre-pass + grad-clip(0.5) + lr=1e-3 per current `train.py`. Hits `SENPAI_TIMEOUT_MINUTES=30` at ~ep12 of 50 configured. Wall clock ~150 s/epoch — slightly slower than GELU baseline due to three matmul kernel-launches per block at small mlp_ratio=2 shapes.)

## History
- 2026-04-27 23:42 — **PR #356** (tanjiro/ema-eval): EMA(0.999) shadow + NaN-safe pre-pass. First measured baseline (val=132.276, test=118.041).
- 2026-04-28 00:43 — **PR #374** (tanjiro/grad-clip-1p0): added `clip_grad_norm_(model.parameters(), max_norm=1.0)`. val=113.157 (−14.45 %), test=99.322.
- 2026-04-28 01:29 — **PR #402** (tanjiro/grad-clip-0p5): tightened `max_norm=1.0 → 0.5`. val=110.822 (−2.07 %), test=97.955.
- 2026-04-28 01:41 — **PR #408** (fern/higher-lr-1e3): bumped `lr=5e-4 → 1e-3`. val=107.957 (−2.59 %), test=95.675.
- 2026-04-28 01:54 — **PR #417** (askeladd/ema-decay-0p99): tightened `ema_decay=0.999 → 0.99`. val=98.581 (−8.69 %), test=87.881.
- 2026-04-28 02:48 — **PR #398** (nezuko/swiglu-mlp-matched): replaced GELU MLP with SwiGLU at matched param count. val=89.349 (−9.36 %), test=79.191 (−9.89 %). First architectural merge after five variance-reduction-direction merges. Per-split: every val and test split improves, fixing the in-dist-vs-OOD trade-off that closed PR #355 (mlp_ratio=4 GELU).
