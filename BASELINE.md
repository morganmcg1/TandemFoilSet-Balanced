# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-16 17:35 UTC

## Current best — PR #3854: slice_num=16 + Huber δ=0.5 — fern

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **57.6953** | run `bg8etivu` (best epoch 17) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **56.8613** | run `bg8etivu` |

Per-split validation (run `bg8etivu` vs prev #3924 SGDR T_0=8 baseline, 60.8893):

| Split | mae_surf_p | Δ vs #3924 |
|---|---|---|
| val_single_in_dist | 65.990 | **−4.95%** |
| val_geom_camber_rc | 71.815 | **−3.24%** |
| val_geom_camber_cruise | 38.006 | **−6.17%** |
| val_re_rand | 54.970 | **−7.45%** |

Per-split test (run `bg8etivu`):

| Split | mae_surf_p | Δ vs #3924 |
|---|---|---|
| test_single_in_dist | 57.502 | **−6.24%** |
| test_geom_camber_rc | 64.529 | **−3.17%** |
| test_geom_camber_cruise | NaN (data/scoring.py bug — known fleet-wide) | — |
| test_re_rand | 48.552 | **−2.22%** |

Key mechanistic finding: **Slice_num=16 + Huber δ=0.5 compound delivers the biggest single-experiment win since SwiGLU.** With dim_head=64 (from n_head=2), each slice can absorb ~50 nodes (~800 surface nodes / 16 slices) without losing per-head signal. δ=0.5 sharpens loss on small residuals; coarser slicing localizes attention pooling. The two mechanisms push the model toward fitting fine-grained pressure structure without fighting each other. Improvement is uniform across all 4 val splits (−3.24% to −7.45%) and 3 valid test splits (−2.22% to −6.24%) — not a lucky single-split win. Each 2× reduction in slice_num so far has helped (64→32: −3.02%; 32→16: −5.16%), suggesting the optimum may be even coarser. SGDR was NOT used in this run.

Merged from PR #3854, student `willowpai2i48h2-fern`.

---

## Current best configuration

slice_num=16 + Huber δ=0.5 + vel-asinh s=0.5 + n_head=2 + SwiGLU MLP + Asinh pressure compression + EMA (fast decay) + gradient clipping:
- **`--slice_num 16`** ← NEW (PR #3854): 4× coarser than baseline 64; ~50 nodes per slice
- **`--huber_delta 0.5`** ← NEW (PR #3854 stack): tighter quadratic transition for small residuals
- **`--asinh_vel_scale 0.5`** (PR #3789): applies `asinh(vel / 0.5)` to velocity channels (Ux, Uy); pressure unchanged
- **`--n_head 2`** (PR #3794): wider per-head attention dim (64 vs 32); also 14% faster per epoch
- **`--use_swiglu --mlp_ratio 1.333`** (PR #3723): SwiGLU in all TransolverBlock MLPs; param-count matched
- **`asinh_p_scale = 1.0`** (PR #3475)
- **`ema_decay = 0.99`** (PR #3474)
- **`grad_clip = 5.0`**
- Validation, checkpoint selection, and test eval all use EMA shadow weights
- **NO SGDR** in this run; SGDR + slice=16 + δ=0.5 super-compound is untested

## Reproduce

```bash
cd target/ && python train.py \
  --grad_clip 5.0 \
  --huber_delta 0.5 \
  --ema_decay 0.99 \
  --asinh_p_scale 1.0 \
  --use_swiglu \
  --mlp_ratio 1.333 \
  --n_head 2 \
  --asinh_vel_scale 0.5 \
  --slice_num 16 \
  --agent <student>
```

## Baseline configuration

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=2, slice_num=16, mlp_ratio=1.333, use_swiglu=True, asinh_vel_scale=0.5`
- Optimizer: AdamW — `lr=5e-4, weight_decay=1e-4`
- Schedule: `CosineAnnealingLR(T_max=epochs)` (wall clock binds at ~15 epochs with n_head=2 at ~107 s/epoch)
- Loss: `F.huber_loss(delta=0.5)` → `vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`
- Gradient clip: `clip_grad_norm_(model.parameters(), 5.0)` before optimizer step
- EMA: **`ema_decay=0.99`**, shadow model updated after every optimizer step
- Sampler: balanced 3-domain `WeightedRandomSampler`
- Batch size: 4
- Hard caps: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30`

---
