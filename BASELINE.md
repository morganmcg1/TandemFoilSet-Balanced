# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-16 21:30 UTC

## Current best — PR #4067: AdamW β2=0.95 — alphonse

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **56.4260** | run `3pc74k8f` (best epoch 17, slice=16 stack) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **55.3387** | run `3pc74k8f` |

Per-split validation (run `3pc74k8f` vs prior #3854 slice=16+default β2 baseline, 57.6953):

| Split | mae_surf_p | Δ vs #3854 |
|---|---|---|
| val_single_in_dist | 65.188 | −1.21% |
| val_geom_camber_rc | 67.131 | **−6.52%** ← dominant residual gain |
| val_geom_camber_cruise | 37.922 | −0.22% |
| val_re_rand | 55.464 | +0.90% |
| **val_avg** | **56.426** | **−2.20%** |

Per-split test (run `3pc74k8f`):

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 58.0236 |
| test_geom_camber_rc | 60.6063 |
| test_geom_camber_cruise | NaN (data/scoring.py bug — known fleet-wide) |
| test_re_rand | 47.3864 |

Key mechanistic finding: **AdamW β2 = 0.95 (instead of default 0.999) halves the second-moment EMA half-life from ~693 steps to ~13 steps**. With only ~6000 total steps in our 30-min budget, β2=0.999 cannot adapt the per-parameter second-moment estimate fast enough — the optimizer effectively uses epoch-1 gradient statistics throughout training. β2=0.95 lets per-parameter step sizes track late-training gradient statistics within each epoch. The win concentrates on val_geom_camber_rc (−6.52%, the hardest OOD-camber split). best_epoch=17 confirms snappier adaptation didn't land in a worse local minimum.

**Note on baseline**: this win was measured on **slice=16**, not slice=8 (the prior best stack at val=56.8954). Result still beats slice=8 baseline by −0.47 val and −0.64 test, so merging is correct. The slice=8 + β2=0.95 compounding is **untested** and is the next experiment (re-validation assigned to alphonse).

Merged from PR #4067, student `willowpai2i48h2-alphonse`.

---

## Previous baseline — PR #4062: slice_num=8 — fern (superseded 21:30 UTC)

- `val_avg/mae_surf_p`: 56.8954 (run `vzpgr8us`)
- `test_3split/mae_surf_p`: 55.9817 (run `vzpgr8us`)
- Stack: slice=8 + Huber δ=0.5 + default β2=0.999

---

## Current best configuration

slice_num=16 + AdamW β2=0.95 + Huber δ=0.5 + vel-asinh s=0.5 + n_head=2 + SwiGLU MLP + Asinh pressure compression + EMA (fast decay) + gradient clipping:
- **`--adamw_beta2 0.95`** ← NEW (PR #4067): fast second-moment EMA adaptation
- **`--slice_num 16`** ← measured baseline (the merged code default may be slice=8; the winning RUN was on slice=16)
- **`--huber_delta 0.5`** (PR #3854 stack): tighter quadratic transition for small residuals
- **`--asinh_vel_scale 0.5`** (PR #3789): applies `asinh(vel / 0.5)` to velocity channels (Ux, Uy); pressure unchanged
- **`--n_head 2`** (PR #3794): wider per-head attention dim (64 vs 32); also 14% faster per epoch
- **`--use_swiglu --mlp_ratio 1.333`** (PR #3723): SwiGLU in all TransolverBlock MLPs; param-count matched
- **`asinh_p_scale = 1.0`** (PR #3475)
- **`ema_decay = 0.99`** (PR #3474)
- **`grad_clip = 5.0`**
- Validation, checkpoint selection, and test eval all use EMA shadow weights
- **NO SGDR**

## Reproduce (winning run alphonse #4067)

```bash
cd target/ && python train.py \
  --grad_clip 5.0 \
  --huber_delta 0.5 \
  --ema_decay 0.99 \
  --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 \
  --n_head 2 \
  --asinh_vel_scale 0.5 \
  --slice_num 16 \
  --adamw_beta2 0.95 \
  --agent <student>
```

**Compounding check (next experiment)**: replace `--slice_num 16` with `--slice_num 8` to test whether the β2=0.95 win compounds with the previously merged slice=8 stack.

## Baseline configuration

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=2, slice_num=8, mlp_ratio=1.333, use_swiglu=True, asinh_vel_scale=0.5`
- Optimizer: AdamW — `lr=5e-4, weight_decay=1e-4`
- Schedule: `CosineAnnealingLR(T_max=epochs)` (wall clock binds at ~15 epochs with n_head=2 at ~107 s/epoch)
- Loss: `F.huber_loss(delta=0.5)` → `vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`
- Gradient clip: `clip_grad_norm_(model.parameters(), 5.0)` before optimizer step
- EMA: **`ema_decay=0.99`**, shadow model updated after every optimizer step
- Sampler: balanced 3-domain `WeightedRandomSampler`
- Batch size: 4
- Hard caps: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30`

---
