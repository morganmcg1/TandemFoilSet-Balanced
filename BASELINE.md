# Baseline — `icml-appendix-charlie-pai2d-r1`

## Current best
- **`val_avg/mae_surf_p` = 64.1585** (EMA, epoch 12/50, timeout-cut)
- **`test_avg/mae_surf_p` = 55.9296**
- Set by **PR #352** (`charliepai2d1-edward/smoothl1-surface`), merged 2026-04-28 04:33 UTC.
- Beats prior baseline (#430) by −5.28 % val / −5.92 % test.

## Configuration
- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- **MLP block**: SwiGLU `(W_g(x) ⊙ silu(W_v(x))) W_o` at `swiglu_inner=168` (matched param count).
- **Optimizer: Lion** (sign-of-momentum) `lr=1.7e-4`, `weight_decay=3e-4`, `betas=(0.9, 0.99)`.
- **Loss**: `MSE_vol + 10.0 * SmoothL1_surf` (β=1.0 in normalized space). Volume kept as MSE; surface uses Huber/SmoothL1 to route high-residual gradient through L1-regime.
- Schedule: cosine annealing over `epochs` (T_max=50)
- Batch: `batch_size=4`, balanced-domain weighted sampling
- **EMA(0.99) shadow weights** drive validation, best-checkpoint selection, and final test eval.
- **`evaluate_split` NaN-safe pre-pass**: drops samples with non-finite ground truth.
- **Gradient clipping at `max_norm=0.5`** between `loss.backward()` and `optimizer.step()` — under Lion this only smooths the momentum buffer (sign-update is invariant to gradient magnitude); kept for lineage.

## Primary ranking metric
- `val_avg/mae_surf_p` — mean surface-pressure MAE across the four val splits (lower is better)
- `test_avg/mae_surf_p` — final paper-facing number

## Per-split breakdown (PR #352 best-EMA-epoch checkpoint)

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 76.614 | 0.784 | 0.433 | 69.374 | 0.778 | 0.406 |
| geom_camber_rc | 75.185 | 1.237 | 0.590 | 66.429 | 1.198 | 0.559 |
| geom_camber_cruise | 42.373 | 0.562 | 0.296 | 34.888 | 0.478 | 0.266 |
| re_rand | 62.462 | 0.897 | 0.442 | 53.028 | 0.764 | 0.406 |
| **avg** | **64.158** | 0.870 | 0.440 | **55.930** | 0.804 | 0.409 |

## Reproduce (current baseline)
```
cd target/ && python train.py --agent <name> --experiment_name <name>/<slug>
```
(Trains with Lion + SwiGLU(168) + EMA(0.99) + NaN-safe pre-pass + grad-clip(0.5) + lr=1.7e-4 + SmoothL1(β=1.0) on surface per current `train.py`. Hits `SENPAI_TIMEOUT_MINUTES=30` at ~ep12 of 50 configured. Wall clock ~151 s/epoch.)

## History
- 2026-04-27 23:42 — **PR #356** (tanjiro/ema-eval): EMA(0.999) + NaN-safe pre-pass. val=132.276 / test=118.041.
- 2026-04-28 00:43 — **PR #374** (tanjiro/grad-clip-1p0): `clip_grad_norm_(1.0)`. val=113.157 (−14.45 %) / test=99.322.
- 2026-04-28 01:29 — **PR #402** (tanjiro/grad-clip-0p5): `max_norm=1.0 → 0.5`. val=110.822 (−2.07 %) / test=97.955.
- 2026-04-28 01:41 — **PR #408** (fern/higher-lr-1e3): `lr=5e-4 → 1e-3`. val=107.957 (−2.59 %) / test=95.675.
- 2026-04-28 01:54 — **PR #417** (askeladd/ema-decay-0p99): `ema_decay=0.999 → 0.99`. val=98.581 (−8.69 %) / test=87.881.
- 2026-04-28 02:48 — **PR #398** (nezuko/swiglu-mlp-matched): GELU MLP → SwiGLU at matched params. val=89.349 (−9.36 %) / test=79.191.
- 2026-04-28 03:46 — **PR #430** (tanjiro/lion-optimizer): AdamW → Lion. val=67.737 (−24.19 %) / test=59.447. Biggest single-PR delta on this branch.
- 2026-04-28 04:33 — **PR #352** (edward/smoothl1-surface): SmoothL1(β=1.0) on surface, MSE on volume. val=**64.1585** (−5.28 %) / test=**55.9296** (−5.92 %). Loss-form lever survives Lion's sign-update; per-split gain redistributes from `single_in_dist` (high-Re tail story under AdamW) to `geom_camber_cruise` (Lion's sign already absorbs the high-Re tail benefit).
