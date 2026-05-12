# SENPAI Research Results

---

## 2026-05-12 20:41 — PR #1462: [warmup-cosine-v2] 1-epoch LinearLR warmup + CosineAnnealingLR

- **Branch**: charliepai2g24h1-frieren/warmup-cosine
- **Hypothesis**: Linear LR warmup prevents overly aggressive early steps, improving convergence stability.
- **Status**: CLOSED (dead end — within-noise tie with baseline; warmup is redundant with grad_clip=1.0)

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | 97.0766 |
| Baseline | 96.5587 |
| Delta | +0.5% (WORSE, within noise) |
| test_avg/mae_surf_p (4-split) | 85.5327 |

**Conclusion**: Warmup is redundant with grad_clip=1.0 at this budget. grad_clip bounds the effective step size on 100% of batches, eliminating the "too-aggressive first step" regime that warmup is designed to prevent. The two mechanisms are redundant. Split-level results trade off (rc improved, single_in_dist worsened), consistent with noise. Frieren's mechanistic analysis is correct and conclusive. **Warmup at this budget is exhausted.**

---

## 2026-05-12 19:52 — PR #1518: [higher-lr-cosine-14] lr=1e-3 + CosineAnnealingLR(T_max=14)

- **Branch**: charliepai2g24h1-thorfinn/higher-lr-cosine-14
- **Hypothesis**: With `grad_clip=1.0` bounding the effective step size, raising lr from 5e-4 to 1e-3 yields faster convergence; reducing T_max from 50 to 14 ensures the cosine schedule reaches its low-LR fine-tuning phase within the actual 14-epoch budget.
- **Status**: MERGED — new baseline. Also includes y-sanitization fix in `train.py:evaluate_split`.

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | **96.5587** |
| val_single_in_dist | 108.58 |
| val_geom_camber_rc | 110.59 |
| val_geom_camber_cruise | 74.35 |
| val_re_rand | 92.71 |
| test_avg/mae_surf_p (4-split, NaN-free) | **85.87** |
| test_single_in_dist | 94.97 |
| test_geom_camber_rc | 99.77 |
| test_geom_camber_cruise | 61.86 |
| test_re_rand | 86.88 |
| Peak VRAM | 42.11 GB |
| Epochs | 14 (~131s/epoch, 30-min cap) |

**Convergence trace**: crossed old 117.17 baseline at epoch 10 (110.19); val still falling at epoch 14 (100.34 → 98.66 → 96.56). Pre-clip norms: mean 23–66, max 288–740. Clipping fires ~100% of batches.

**Analysis**: Dominant mechanism: T_max=14 let the cosine schedule actually reach its low-LR fine-tuning phase, which T_max=50 never achieved in 14 epochs. The higher LR (1e-3 vs 5e-4) accelerated early convergence and compound with the schedule effect. The y-sanitization fix made the cruise test split computable for the first time. -17.6% improvement over previous baseline.

**Key insight**: Val was still falling at epoch 14 — the model has not fully converged. A slightly higher LR or different schedule tail may extract more.

**Artifacts**: `models/model-charliepai2g24h1-thorfinn-higher-lr-cosine-14-20260512-191045/metrics.jsonl`

---

## 2026-05-12 19:30 — PR #1473: [huber-loss] Switch MSE → Huber loss (delta=0.5)

- **Branch**: charliepai2g24h1-tanjiro/huber-loss
- **Hypothesis**: Huber loss caps outlier-residual gradients on extreme-value mesh nodes, improving stability and final val MAE.
- **Status**: Winner pending rebase — sent back to resolve merge conflicts and re-verify on grad-clip baseline

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 13) | **111.296** ⭐ |
| val_single_in_dist | 133.55 |
| val_geom_camber_rc | 122.56 |
| val_geom_camber_cruise | 89.51 |
| val_re_rand | 99.57 |
| test_avg/mae_surf_p (3 splits, excl cruise) | 112.51 |
| Peak VRAM | 42.1 GB |
| Epochs | 14 (~130s/epoch, 30-min cap) |
| huber_l2_frac (epoch 1 → 13) | 0.749 → 0.931 |
| huber_delta | 0.5 |

**Analysis**: Clean training trajectory, no instabilities. L2-fraction climbed from 75% → 93% across training, exactly the "outlier-capping early, MSE-like late" pattern the PR predicted. **111.296 beats the 117.17 baseline by ~5%** — clear winner.

**Caveat**: This run forked from pre-grad-clip-1 base. The advisor branch now has grad_clip=1.0 as default. Student must rebase, re-run with grad-clip active, and apply the y-sanitization fix in train.py:evaluate_split. The 117.17 grad-clip baseline becomes the head-to-head comparison target.

**Bonus**: Tanjiro diagnosed the test_geom_camber_cruise NaN bug independently — `Inf*0=NaN` in scoring.py accumulator from a single sample with -inf pressure values. Their proposed fix is correct in spirit but must be applied in train.py:evaluate_split (since `data/` is read-only per program.md).

**Artifacts**: `models/model-charliepai2g24h1-tanjiro-huber-loss-20260512-180430/metrics.jsonl`

---

## 2026-05-12 18:57 — PR #1479: [grad-clip-1] Add gradient clipping (clip_norm=1.0) for training stability

- **Branch**: charliepai2g24h1-thorfinn/grad-clip-1
- **Hypothesis**: Gradient clipping (clip_norm=1.0) prevents gradient explosions from extreme-value mesh nodes and stabilizes early training.
- **Status**: MERGED — new baseline

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 13) | **117.17** |
| val_single_in_dist/mae_surf_p | 134.83 |
| val_geom_camber_rc/mae_surf_p | 134.17 |
| val_geom_camber_cruise/mae_surf_p | 87.04 |
| val_re_rand/mae_surf_p | 112.66 |
| test_avg/mae_surf_p | NaN (cruise GT bug) |
| test 3-split avg (excl cruise) | 116.17 |
| Peak VRAM | 42.1 GB |
| Epochs completed / 30 min | 14 (~130s/epoch) |
| Pre-clip gradient norms | mean 50–100, max 300–800 |

**Key finding**: Pre-clip gradient norms are 50–800, clipping fires on **100% of batches** every epoch. The baseline Transolver is gradient-unstable without clipping. grad_clip=1.0 is now mandatory baseline infrastructure.

**NaN bug discovered**: test_geom_camber_cruise sample 20 has ±Inf in GT pressure channel; `Inf*0=NaN` poisons the accumulator in scoring.py. Fix: sanitize y before calling accumulate_batch in train.py.

**Artifacts**: `models/model-charliepai2g24h1-thorfinn-grad-clip-1-20260512-180544/metrics.jsonl`

---

## 2026-05-12 18:56 — PR #1457: [surf-weight-50] Raise surf_weight 10→50

- **Branch**: charliepai2g24h1-askeladd/surf-weight-50
- **Status**: Sent back for round 2

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | 135.36 |
| val_geom_camber_cruise | 104.81 |
| val_geom_camber_rc | 157.27 |
| val_re_rand | 124.40 |
| val_single_in_dist | 154.98 |
| Epochs | 14 |

**Analysis**: Healthy surf/vol balance (no volume collapse). Did not beat grad-clip-1 (117.17). Sent back to combine surf_weight=30 with the new grad-clip baseline.

---

## 2026-05-12 18:55 — PR #1458: [wider-deeper] Scale Transolver n_hidden=256, n_layers=6, n_head=8

- **Branch**: charliepai2g24h1-edward/wider-deeper
- **Status**: Sent back for round 2 with batch_size=4

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 7) | 126.99 |
| val_geom_camber_cruise | 95.27 |
| val_geom_camber_rc | 141.93 |
| val_re_rand | 113.94 |
| val_single_in_dist | 156.82 |
| n_params | 3.0M (not 8M as estimated) |
| Peak VRAM | 49.43 GB |
| Epochs | 7 (batch_size=2 → ~295s/epoch) |

**Analysis**: Val still falling at epoch 7; model not converged. With batch_size=4 (safe given 49GB VRAM) should complete 14 epochs. Trajectory (204→127 in 7 eps) very promising — extrapolates well below 117.

---

## 2026-05-12 18:53 — PR #1462: [warmup-cosine] Add 2-epoch linear LR warmup

- **Branch**: charliepai2g24h1-frieren/warmup-cosine
- **Status**: Sent back for round 2

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 13) | 131.83 |
| val_geom_camber_cruise | 103.34 |
| val_geom_camber_rc | 131.83 |
| val_re_rand | 119.68 |
| val_single_in_dist | 172.48 |
| Epochs | 14 |

**Analysis**: Schedule worked mechanically. With only 14 epochs, 2-epoch warmup consumes 14% of budget. With grad-clip now merged (more stable training), sent back with shorter 1-epoch warmup + start_factor=0.1.

---

## Round 1 WIP — Still Running

| PR | Student | Hypothesis |
|----|---------|------------|
| #1456 | alphonse | bf16-amp |
| #1460 | fern | relative-l2-loss |
| #1467 | nezuko | more-slices-128 |
| #1473 | tanjiro | huber-loss |
