# Baseline — `icml-appendix-charlie-pai2d-r1`

## Current best
- **`val_avg/mae_surf_p` = 67.737** (EMA, epoch 12/50, timeout-cut)
- **`test_avg/mae_surf_p` = 59.447**
- Set by **PR #430** (`charliepai2d1-tanjiro/lion-optimizer`), merged 2026-04-28 03:46 UTC.
- Beats prior baseline (#398) by −24.19 % val / −24.94 % test — biggest single-PR delta on this branch.

## Configuration
- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- **MLP block**: SwiGLU `(W_g(x) ⊙ silu(W_v(x))) W_o` at `swiglu_inner=168` (matched param count, 657,639 params).
- **Optimizer: Lion** (sign-of-momentum) replacing AdamW. Inline implementation in `train.py`. `lr=1.7e-4`, `weight_decay=3e-4`, `betas=(0.9, 0.99)`. Default Lion-from-AdamW recipe (`lr_adamw / 3, wd_adamw × 3`).
- Loss: `MSE_vol + 10.0 * MSE_surf` (normalized space)
- Schedule: cosine annealing over `epochs` (T_max=50)
- Batch: `batch_size=4`, balanced-domain weighted sampling
- **EMA(0.99) shadow weights** drive validation, best-checkpoint selection, and final test eval.
- **`evaluate_split` NaN-safe pre-pass:** drops samples with non-finite ground truth from `mask` and zeros their `y` before loss/MAE computation.
- **Gradient clipping at `max_norm=0.5`** between `loss.backward()` and `optimizer.step()` — note: under Lion's sign-update, clipping the post-clipped gradient only smooths the momentum buffer, not the parameter update itself (sign-update is invariant to gradient magnitude). Kept in the code path for apples-to-apples lineage.

## Primary ranking metric
- `val_avg/mae_surf_p` — mean surface-pressure MAE across the four val splits (lower is better)
- `test_avg/mae_surf_p` — final paper-facing number

## Per-split breakdown (PR #430 best-EMA-epoch checkpoint)

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 78.016 | 0.782 | 0.442 | 68.654 | 0.768 | 0.422 |
| geom_camber_rc | 81.096 | 1.319 | 0.631 | 72.119 | 1.292 | 0.587 |
| geom_camber_cruise | 46.281 | 0.509 | 0.307 | 39.395 | 0.469 | 0.285 |
| re_rand | 65.556 | 0.901 | 0.469 | 57.621 | 0.786 | 0.427 |
| **avg** | **67.737** | 0.878 | 0.462 | **59.447** | 0.829 | 0.430 |

## Reproduce (current baseline)
```
cd target/ && python train.py --agent <name> --experiment_name <name>/<slug>
```
(Trains with Lion + SwiGLU(168) + EMA(0.99) + NaN-safe pre-pass + grad-clip(0.5) + lr=1.7e-4 per current `train.py`. Hits `SENPAI_TIMEOUT_MINUTES=30` at ~ep12 of 50 configured. Wall clock ~151 s/epoch.)

## History
- 2026-04-27 23:42 — **PR #356** (tanjiro/ema-eval): EMA(0.999) shadow + NaN-safe pre-pass. First measured baseline (val=132.276, test=118.041).
- 2026-04-28 00:43 — **PR #374** (tanjiro/grad-clip-1p0): added `clip_grad_norm_(1.0)`. val=113.157 (−14.45 %), test=99.322.
- 2026-04-28 01:29 — **PR #402** (tanjiro/grad-clip-0p5): tightened `max_norm=0.5`. val=110.822 (−2.07 %), test=97.955.
- 2026-04-28 01:41 — **PR #408** (fern/higher-lr-1e3): `lr=5e-4 → 1e-3`. val=107.957 (−2.59 %), test=95.675.
- 2026-04-28 01:54 — **PR #417** (askeladd/ema-decay-0p99): `ema_decay=0.999 → 0.99`. val=98.581 (−8.69 %), test=87.881.
- 2026-04-28 02:48 — **PR #398** (nezuko/swiglu-mlp-matched): GELU MLP → SwiGLU at matched params. val=89.349 (−9.36 %), test=79.191.
- 2026-04-28 03:46 — **PR #430** (tanjiro/lion-optimizer): AdamW → Lion (sign-of-momentum). val=**67.737** (−24.19 %), test=**59.447** (−24.94 %). Biggest single-PR delta on this branch. EMA-Lion interaction at decay 0.99 averages over Lion's substantial epoch-to-epoch raw variance, contributing to the gain.
