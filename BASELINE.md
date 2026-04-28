# Baseline — `icml-appendix-charlie-pai2d-r1`

## Current best
- **`val_avg/mae_surf_p` = 52.116** (EMA, epoch 14/50, timeout-cut)
- **`test_avg/mae_surf_p` = 45.413**
- Set by **PR #571** (`charliepai2d1-frieren/lion-beta2-0p999`), merged 2026-04-28 06:23 UTC.
- Beats prior baseline (#536) by −13.83 % val / −13.79 % test. All four splits gain ≥10 %; `single_in_dist` largest (−16.02 % val, −13.74 % test).
- **Squash-merge composed β2=0.999 (this PR) with lr=2.5e-4 (from #536) + β=0.5 (from #535)** — frieren's run was at lr=1.7e-4 + β2=0.999 + β=1.0 (pre-#535/#536 baseline); recorded baseline metrics are from that run. Future re-runs of the unmodified post-merge baseline (lr=2.5e-4 + β=0.5 + β2=0.999) may land slightly different — and likely better, since both lr and β were independently improved post-fork.

## Configuration
- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- **MLP block**: SwiGLU `(W_g(x) ⊙ silu(W_v(x))) W_o` at `swiglu_inner=168` (matched param count).
- **Optimizer: Lion** (sign-of-momentum) `lr=2.5e-4`, `weight_decay=3e-4`, `betas=(0.9, 0.999)`. Basin upper edge in [2.5e-4, 3.3e-4]; β2=0.999 (longer-history buffer) wins decisively over default 0.99.
- **Loss**: `MSE_vol + 10.0 * SmoothL1_surf(β=0.5)` in normalized space.
- Schedule: cosine annealing over `epochs` (T_max=50)
- Batch: `batch_size=4`, balanced-domain weighted sampling
- **TF32 matmul precision**: `torch.set_float32_matmul_precision('high')`. ~14 epochs in 30-min budget under TF32; lr=2.5e-4 cosine at ep14 ≈ 2.35e-4 (still ~94 % of peak).
- **EMA(0.99)** shadow weights drive validation, best-checkpoint selection, and final test eval.
- **`evaluate_split` NaN-safe pre-pass**: drops samples with non-finite ground truth.
- **Gradient clipping at `max_norm=0.5`** between `loss.backward()` and `optimizer.step()` — under Lion this only smooths the momentum buffer; kept for lineage.

## Primary ranking metric
- `val_avg/mae_surf_p` — mean surface-pressure MAE across the four val splits (lower is better)
- `test_avg/mae_surf_p` — final paper-facing number

## Per-split breakdown (PR #571 best-EMA-epoch checkpoint, lr=1.7e-4 + β=1.0 + β2=0.999)

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 56.653 | 0.633 | 0.355 | 51.361 | 0.633 | 0.343 |
| geom_camber_rc | 65.237 | 1.106 | 0.522 | 56.736 | 1.051 | 0.490 |
| geom_camber_cruise | 34.101 | 0.433 | 0.247 | 28.938 | 0.395 | 0.224 |
| re_rand | 52.472 | 0.745 | 0.381 | 44.618 | 0.660 | 0.343 |
| **avg** | **52.116** | 0.729 | 0.376 | **45.413** | 0.685 | 0.350 |

## Reproduce (current baseline)
```
cd target/ && python train.py --agent <name> --experiment_name <name>/<slug>
```
(Trains with TF32 + Lion(lr=2.5e-4) + SwiGLU(168) + EMA(0.99) + NaN-safe pre-pass + grad-clip(0.5) + SmoothL1(β=0.5)/MSE-vol per current `train.py`. Hits `SENPAI_TIMEOUT_MINUTES=30` at ~ep14 of 50 configured. Wall clock ~131 s/epoch.)

## History
- 2026-04-27 23:42 — **PR #356** (tanjiro/ema-eval): EMA(0.999) + NaN-safe pre-pass. val=132.276 / test=118.041.
- 2026-04-28 00:43 — **PR #374** (tanjiro/grad-clip-1p0): `clip_grad_norm_(1.0)`. val=113.157 (−14.45 %).
- 2026-04-28 01:29 — **PR #402** (tanjiro/grad-clip-0p5): `max_norm=1.0 → 0.5`. val=110.822 (−2.07 %).
- 2026-04-28 01:41 — **PR #408** (fern/higher-lr-1e3): `lr=5e-4 → 1e-3`. val=107.957 (−2.59 %).
- 2026-04-28 01:54 — **PR #417** (askeladd/ema-decay-0p99): `ema_decay=0.999 → 0.99`. val=98.581 (−8.69 %).
- 2026-04-28 02:48 — **PR #398** (nezuko/swiglu-mlp-matched): GELU MLP → SwiGLU at matched params. val=89.349 (−9.36 %).
- 2026-04-28 03:46 — **PR #430** (tanjiro/lion-optimizer): AdamW → Lion. val=67.737 (−24.19 %). Biggest single-PR delta.
- 2026-04-28 04:33 — **PR #352** (edward/smoothl1-surface): SmoothL1(β=1.0)/MSE-vol. val=64.158 (−5.28 %).
- 2026-04-28 05:17 — **PR #491** (fern/tf32-matmul-precision): TF32 fp32-matmul. val=63.218 (−1.47 %). Throughput multiplier (14 epochs vs 12).
- 2026-04-28 05:27 — **PR #535** (edward/smoothl1-beta-0p5): SmoothL1 β=1.0 → 0.5. val=61.508 (−2.70 %).
- 2026-04-28 06:12 — **PR #536** (tanjiro/lion-lr-2p5e-4): Lion `lr=1.7e-4 → 2.5e-4`. val=**60.478** (−1.67 %) / test=**52.676** (+0.65 %, within noise). All val splits improve uniformly; `single_in_dist` largest gain (−11.95 % val, −14.17 % test). Lion basin upper edge confirmed in [2.5e-4, 3.3e-4].
- 2026-04-28 06:23 — **PR #571** (frieren/lion-beta2-0p999): Lion `β2=0.99 → 0.999` (longer-history buffer for sign-update direction). val=**52.116** (−13.83 %) / test=**45.413** (−13.79 %). All four splits gain ≥10 %; `single_in_dist` largest (−16.02 % val, −13.74 % test). Establishes β1-vs-β2 mechanism distinction: β1 trades responsiveness for direction smoothness (lose case in #545); β2 trades a few warm-up batches for persistent direction smoothness while retaining full responsiveness (clean win here). Largest single-PR delta after #430 (Lion adoption).
