# SENPAI Research Results

---

## 2026-05-12 22:xx — PR #1473: [huber-loss v3] Huber(δ=0.1) on relative-L2 normalized residuals

- **Branch**: charliepai2g24h1-tanjiro/huber-loss
- **Hypothesis**: Applying Huber(δ=0.1) to the per-sample energy-normalized residuals (relative-L2 space) provides intra-sample outlier capping on top of inter-sample scale normalization.
- **Status**: MERGED — new baseline 89.3940

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | **89.3940** |
| val_single_in_dist | 109.01 |
| val_geom_camber_rc | 101.19 |
| val_geom_camber_cruise | **66.36** |
| val_re_rand | 81.02 |
| test_avg/mae_surf_p (4-split) | **79.5993** |
| test_single_in_dist | 98.51 |
| test_geom_camber_rc | 88.12 |
| test_geom_camber_cruise | 54.80 |
| test_re_rand | 76.97 |
| Baseline | 89.6121 |
| Delta | **-0.24%** |
| huber_delta | 0.1 (normalized space) |
| L2-fraction (ep 1 → 14) | 33% → 63% |
| grad clip_frac (ep 14) | 0.075 (vs 0.984 on rel-L2-only) |

**Analysis**: The compound works: Huber(δ=0.1) in normalized space is genuinely active throughout (L2-fraction 33%→63%, vs 93%→94% with the raw-space δ=0.5 that largely collapsed to MSE). The most striking diagnostic is clip_frac dropping from ~1.0 (MSE baseline) to 0.075 by epoch 14 — the loss surface is dramatically smoother. val_re_rand showed the biggest improvement (84.29→81.02). val_geom_camber_rc regressed slightly (97.99→101.19) — this is the hardest OOD split and may need the re-conditioned-scaling architecture fix. Win is narrow (-0.24%) but clean, monotone, and confirmed in committed JSONL.

**Key mechanism**: relative-L2 handles inter-sample scale variation (across Re regimes); Huber handles intra-sample node outliers (within each sample). Complementary mechanisms.

**Artifacts**: `models/model-charliepai2g24h1-tanjiro-huber-loss-20260512-211810/metrics.jsonl`

---

## 2026-05-12 21:xx — PR #1458: [wider-deeper v2] Scale Transolver n_hidden=256, n_layers=6, n_head=8

- **Branch**: charliepai2g24h1-edward/wider-deeper
- **Hypothesis**: 3M-param Transolver (4.5× baseline) converges to a better solution within 30-min cap.
- **Status**: CLOSED — not competitive in 30-min epoch budget

**v1 results (batch_size=2, old LR, pre-grad-clip base):**

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 7) | 126.9875 |
| Epochs | 7 (~311s/epoch, still falling at timeout) |

**v2 results (batch_size=4, lr=1e-3, T_max=14, grad-clip base):**

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 6) | 158.0610 |
| test_avg/mae_surf_p | 145.9203 |
| Peak VRAM | 98.83 GB |
| Epochs | 6 (near-OOM, 311s/epoch) |

**Analysis**: At 3M params, the model gets only 6-7 epochs in 30 min vs 14 for the 662K baseline. The v2 regression (158 vs v1's 127) was caused by applying lr=1e-3 (calibrated for the small model) to the large model — 100% clip rate throughout. Even v1's promising trajectory (204→127 in 7 epochs) cannot plausibly converge below 89.61 given the epoch budget. Closed; edward reassigned to per-channel-loss-weights.

**Artifacts**: `models/model-charliepai2g24h1-edward-wider-deeper-20260512-180258/metrics.jsonl`, `models/model-charliepai2g24h1-edward-wider-deeper-v2-20260512-201243/metrics.jsonl`

---

## 2026-05-12 21:xx — PR #1539: [lr-1.5e-3-cosine-14] Test lr ceiling: 1.5e-3 + T_max=14

- **Branch**: charliepai2g24h1-thorfinn/lr-1.5e-3-cosine-14
- **Hypothesis**: lr=1.5e-3 (vs 1e-3 baseline) with T_max=14; tests whether the LR ceiling is higher than 1e-3.
- **Status**: CLOSED — LR ceiling confirmed, regression

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | 100.2425 |
| val_single_in_dist | 119.01 |
| val_geom_camber_rc | 110.47 |
| val_geom_camber_cruise | 76.51 |
| val_re_rand | 94.98 |
| test_avg/mae_surf_p | 89.1065 |
| Baseline (new) | 89.6121 |
| Delta | +11.8% (WORSE) |
| Clip frac | 99–100% throughout |

**Analysis**: lr=1.5e-3 is confirmed above the LR ceiling. 100% clip rate throughout — pushing nominal LR higher just increases the wasted component beyond the clip threshold. single_in_dist degraded most (+10.43 vs val). The ceiling experiment is complete: lr=1e-3 is the right AdamW LR for this dataset/architecture at grad_clip=1.0. To push past this ceiling, need a different optimizer (SOAP, quasi-Newton) rather than higher LR. Thorfinn reassigned to soap-optimizer (H1, HIGH priority).

**Artifacts**: `models/model-charliepai2g24h1-thorfinn-lr-1.5e-3-cosine-14-20260512-200502/metrics.jsonl`

---

## 2026-05-12 21:xx — PR #1456: [bf16-amp] bf16 automatic mixed precision

- **Branch**: charliepai2g24h1-alphonse/bf16-amp
- **Hypothesis**: bf16-amp reduces memory pressure and increases throughput, enabling more epochs in the 30-min wall-clock budget.
- **Status**: SENT BACK (v2) — regression due to schedule misalignment; rebase + T_max=18 required

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p | 114.21 (REGRESSION) |
| Baseline | 96.5587 |
| Delta | +18.2% (WORSE) |
| Epochs completed | 18 (vs 14 baseline — +30% throughput confirmed) |

**Analysis**: bf16-amp confirmed 30% throughput gain (18 vs 14 epochs). Regression was entirely due to running with old `T_max=50`: at 18 epochs, only 36% of the cosine schedule was used. Sent back to rebase onto new baseline (rel-L2 loss), set `T_max=18` to match new epoch budget, and re-run. If throughput advantage holds, the compound (18 epochs × aligned schedule × relative-L2 base) should beat 89.61.

**Artifacts**: Not yet committed (stale branch, not merged)

---

## 2026-05-12 21:xx — PR #1473: [huber-loss-v2] Huber loss (δ=0.5) rebased onto grad-clip baseline

- **Branch**: charliepai2g24h1-tanjiro/huber-loss
- **Hypothesis**: Huber loss caps outlier-residual gradients on extreme-value mesh nodes; on top of grad_clip + lr=1e-3 baseline, should improve convergence stability and final val MAE.
- **Status**: SENT BACK (v3) — beat old baseline (96.56) but not new baseline (89.61 from relative-l2); next step is Huber on top of relative-L2

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | **90.0929** |
| val_single_in_dist | 104.29 |
| val_geom_camber_rc | 101.05 |
| val_geom_camber_cruise | 70.12 |
| val_re_rand | 84.92 |
| test_avg/mae_surf_p | 78.97 |
| huber_delta | 0.5 |
| huber_l2_frac (ep 1 → 14) | 76% → 94% |
| Baseline (old) | 96.5587 |
| Delta vs old baseline | -6.69% |
| New baseline | 89.6121 |
| Delta vs new baseline | +0.54% (just missed) |

**Analysis**: Clean training, no instabilities, L2-fraction trajectory perfect (capping early, MSE-like late). Beat the old MSE baseline by 6.69% but lost to fern's relative-L2 by a narrow margin (90.09 vs 89.61). Sent back to compound: apply Huber to normalized residuals in relative-L2 space. New delta should be tuned to the normalized scale (~0.05–0.1 rather than 0.5). The mechanisms are complementary: relative-L2 handles inter-sample scale variation, Huber handles intra-sample node outliers.

**Artifacts**: `models/` path TBD after v3 re-run

---

## 2026-05-12 21:xx — PR #1460: [relative-l2-loss] Per-sample relative L2 loss

- **Branch**: charliepai2g24h1-fern/relative-l2-loss
- **Hypothesis**: Relative L2 loss (`||pred-y||²/||y||²`) normalizes by sample energy, automatically down-weighting high-energy (high-Re) samples and up-weighting low-energy ones — a better inductive bias for the multi-Re dataset.
- **Status**: MERGED — new baseline

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | **89.6121** |
| val_single_in_dist | 109.07 |
| val_geom_camber_rc | 97.99 |
| val_geom_camber_cruise | **67.09** |
| val_re_rand | 84.29 |
| test_avg/mae_surf_p (4-split) | **78.14** |
| test_single_in_dist | 91.14 |
| test_geom_camber_rc | 85.89 |
| test_geom_camber_cruise | 56.35 |
| test_re_rand | 79.18 |
| Peak VRAM | 42.11 GB |
| Epochs | 14 (~131s/epoch, 30-min cap) |
| Baseline | 96.5587 |
| Delta | **-7.20%** |

**Analysis**: Relative-L2 loss's per-sample energy normalization creates a flatter cross-split loss landscape. The gradient clip fraction dropped to 0.984 at ep 14 (vs 1.0 throughout on MSE) — the loss surface is genuinely smoother. Val was still falling at ep 14 (95.94 → 93.35 → 89.61 in last 3 epochs), indicating more headroom with more epochs. Cruise split improved dramatically (67.09 vs 74.35 baseline). RC and single_in_dist improved but remain the hardest splits — both span the full Re range and benefit most from architecture-level scale separation (H2 re-conditioned-scaling).

**Artifacts**: `models/model-charliepai2g24h1-fern-relative-l2-loss-20260512-200551/metrics.jsonl`

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
