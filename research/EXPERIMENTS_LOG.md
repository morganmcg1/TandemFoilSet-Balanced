# SENPAI Research Results

---

## 2026-05-13 05:15 — PR #1704: [ema-weights] EMA model weights for smoother final checkpoint

- **Branch**: charliepai2g24h1-frieren/ema-weights
- **Hypothesis**: EMA (β=0.999) of model weights produces a smoother final checkpoint than the terminal live-weights checkpoint, at zero training-time cost.
- **Status**: CLOSED — did not beat baseline; dual-val protocol overhead was the root cause

| Metric | EMA (26 ep) | Live weights (26 ep) | Baseline (30 ep) | Delta vs baseline |
|--------|-------------|----------------------|------------------|-------------------|
| ema_val_avg/mae_surf_p | **32.2245** | — | 30.4412 | +1.78 (+5.9%) |
| val_avg/mae_surf_p (live) | — | 32.0731 | 30.4412 | +1.63 (+5.4%) |
| ema_test_avg/mae_surf_p | **27.7392** | — | 26.1013 | +1.64 (+6.3%) |
| Epochs completed | 26 | 26 | 30 | −4 epochs |
| Mean epoch time | 70.4s | — | ~62s | +13% (dual val) |

**Per-split EMA val (epoch 26):**

| Split | EMA val | EMA test | Baseline val | Baseline test |
|-------|---------|----------|--------------|---------------|
| single_in_dist | 35.55 | 35.35 | 34.27 | 32.96 |
| geom_camber_rc | 43.90 | 39.47 | 41.43 | 37.90 |
| geom_camber_cruise | 15.63 | 12.37 | 14.04 | 11.38 |
| re_rand | 33.82 | 23.76 | 32.02 | 22.16 |
| **avg** | **32.22** | **27.74** | **30.44** | **26.10** |

**Artifact**: `models/model-charliepai2g24h1-frieren-ema-weights-20260513-040222/metrics.jsonl`

**Analysis**: Implementation was correct; EMA trajectory behaved exactly as theory predicts. The mid-run EMA advantage was REAL (Δ=−11.7 MAE at epoch 14 — EMA was ~20% better than live). But two factors killed the result: (1) The PR required logging both live and EMA val each epoch, doubling validation work (+13% wall clock overhead), costing 4 epochs (26 vs 30). (2) Cosine LR with eta_min=1e-5 already smooths late-epoch updates naturally — by epoch 24-26 the EMA and live weights are nearly identical (Δ flips sign at epoch 25). The hypothesis was not wrong about the mechanism; the experimental protocol was wrong. **Closing**, assigning corrected protocol.

**Key EMA trajectory**: shadow lagged init for ~10 epochs, peaked at Δ=−11.7 MAE at epoch 14, narrowed to Δ=+0.15 at epoch 26.

---

## 2026-05-13 02:10 — PR #1599: [re-conditioned-scaling] sent back v3 — compound confirmed but stale baseline

- **Branch**: charliepai2g24h1-fern/re-conditioned-scaling
- **Status**: SENT BACK for third rebase — compound mechanism confirmed, but baseline moved during run

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (on cosine-eta-min base, ep 13) | 38.0178 |
| test_avg/mae_surf_p | 33.5671 |
| vs cosine-eta-min baseline (39.8693 / 35.2214) | **−4.7% val, −4.7% test** |
| vs current bf16-amp baseline (36.8778 / 31.9058) | +3.1% val (regression) |
| ReScale corr(scale, log Re) Ux/Uy/p | +0.68 / **+0.89** / **+0.86** |
| scale_std Ux/Uy/p at ep 13 | 0.052 / 0.211 / **0.504** |

**Mechanism confirmed**: ReScaleHead COMPOUNDS with SOAP — answering the three-possibilities question from the original send-back. Pre-SOAP failure on AdamW was caused by first-order instability during warmup (head and backbone competing for Re-dependent scale). SOAP's preconditioner routes the Re signal cleanly into the 163-param head subspace, eliminating the competition. Physical signature: Ux nearly inert (freestream-dominated), Uy moderate, p strongest — consistent with Bernoulli-like ~Re² pressure scaling.

**Why sent back**: Result is on cosine-eta-min base, but bf16-amp merged during fern's run. Need to verify the 4.7% compound holds on the new bf16 baseline. If it does, target val ≈ 35.15 (cleanly beats baseline). The mechanism is well-established now; this is the last rebase needed.

**Side observation (one-shot noise, not conclusive)**: A SOAP-only run hit val 36.72; SOAP + eta_min=1e-5 + ReScale hit val 38.02 — possible eta_min × ReScale interaction worth watching but currently single-shot.

**Artifact**: `models/model-charliepai2g24h1-fern-re-conditioned-scaling-20260513-005513/metrics.jsonl`

---

## 2026-05-13 02:00 — PR #1456: [bf16-amp + cosine-eta-min] bf16 AMP with T_max=17 on SOAP

- **Branch**: charliepai2g24h1-alphonse/bf16-amp (rebased onto cosine-eta-min base)
- **Hypothesis**: bf16 AMP gives ~+29% throughput → ~17 epochs in 30 min vs 13 previously. T_max=17 aligns cosine tail to new budget. Compounds with all prior wins.
- **Status**: MERGED — new baseline 36.8778

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 16/17) | **36.8778** |
| val_single_in_dist | 42.92 (−10.2%) |
| val_geom_camber_rc | 47.78 (−8.6%) |
| val_geom_camber_cruise | **18.60** (−11.0%) |
| val_re_rand | 38.21 (−0.7%) |
| test_avg/mae_surf_p | **31.9058** (−9.42%) |
| test_single_in_dist | 42.15 (−8.3%) |
| test_geom_camber_rc | 42.69 (−7.9%) |
| test_geom_camber_cruise | **15.26** (−11.5%) |
| test_re_rand | 27.53 (−12.2%) |
| Epochs | 17 (vs 13 previously) |
| Mean epoch time | 108.6 s (vs ~131 s previously) |
| Peak GPU memory | 32.98 GB / 96 GB |
| clip_frac trajectory | 0.98 → 0.34 (smoothly decaying) |
| huber_l2_frac | 0.42 → 0.86 (Huber actively capping outliers) |
| Baseline | 39.8693 |
| Delta | **−2.99 (−7.51%)** |

**Analysis**: bf16 + T_max alignment compounds cleanly on SOAP/eta_min. All 4 val + 4 test splits improved (rare clean win across the board). No numerical issues with bf16 + SOAP + Huber-rel-L2 (SOAP preconditioner runs in fp32 internally). ep 17 (at LR floor 1.84e-5) drifts back +0.09 from ep 16 best — confirms T_max=17 is well-matched.

**Key insight**: With substantial memory headroom (33/96 GB), there's room for larger batches OR larger models — next experiments target both axes.

**Artifact**: `models/model-charliepai2g24h1-alphonse-bf16-amp-cosine-eta-min-20260513-005955/metrics.jsonl`

---

## 2026-05-13 01:55 — PR #1740: [soap-higher-lr] lr=1e-3→2e-3 under SOAP — null result

- **Branch**: charliepai2g24h1-tanjiro/soap-higher-lr
- **Hypothesis**: SOAP's curvature-aware preconditioning raises the LR ceiling well above AdamW's 1e-3 limit; lr=2e-3 should accelerate convergence without divergence.
- **Status**: CLOSED — null result (within noise), mechanism understood

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 13) | 39.7891 |
| test_avg/mae_surf_p | 35.6166 |
| Baseline | 39.8693 / 35.2214 |
| Delta val | −0.08 (noise) |
| Delta test | +0.40 (noise, wrong direction) |
| clip_frac ep 1 | 0.952 (95% of batches clipped) |
| clip_frac ep 13 | 0.179 |
| Divergence | None — model trained cleanly |

**Analysis**: LR ceiling confirmed — lr=2e-3 ran cleanly all 13 epochs with no NaN, no spikes (vs AdamW's divergence at lr=1.5e-3 in PR #1539). But grad_clip=1.0 normalized away the LR increase: effective per-step update `= (clip × lr) / grad_norm` was nearly identical between lr=1e-3 and 2e-3. The clip threshold (not the LR) is the bottleneck.

**Key insight**: SOAP LR ceiling is ≥ 2e-3 (stable). To actually exploit higher LR, grad_clip needs to widen too — exactly what thorfinn is testing in #1668 (clip 1.0→5.0). Future LR experiments must couple with clip widening.

**Artifact**: `models/model-charliepai2g24h1-tanjiro-soap-higher-lr-20260513-005527/metrics.jsonl`

---

## 2026-05-13 00:25 — PR #1630: [cosine-eta-min] CosineAnnealingLR eta_min=1e-5 floor on SOAP

- **Branch**: charliepai2g24h1-tanjiro/sgdr-restarts (pivoted from SGDR to cosine-eta-min)
- **Hypothesis**: Prevent CosineAnnealingLR from reaching near-zero at the terminal epoch by adding `eta_min=1e-5`. The SOAP baseline uses `T_max=14` but only completes 13 epochs — epoch 13 is both the budget-limited final epoch and the best checkpoint. Without a floor, the cosine schedule drives LR to ~4.95e-5 at ep 13; with `eta_min=1e-5`, LR is ~5.90e-5 (+19% relative).
- **Status**: MERGED — new baseline 39.8693

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 13) | **39.8693** |
| val_single_in_dist | 47.81 (+1.72 vs prev) |
| val_geom_camber_rc | 52.28 (−3.70) |
| val_geom_camber_cruise | **20.89** (−3.43) |
| val_re_rand | **38.49** (−4.73) |
| test_avg/mae_surf_p | **35.2214** |
| test_geom_camber_cruise | **17.24** |
| test_re_rand | **31.37** |
| Epochs | 13 (~30-min cap) |
| LR at ep 13 | 5.90e-5 (vs 4.95e-5 without floor) |
| Baseline | 42.4015 |
| Delta | **−2.53 (−5.97%)** |

**Analysis**: Single-line change. The +19% relative LR boost at epoch 13 (the best-checkpoint epoch) gives meaningful gradient signal in the final step. 3/4 OOD splits improved; single_in_dist slightly worse (+1.72) likely because it was already well-fit. Val still monotone descending — model has not converged. This is a free compounding gain on top of SOAP.

**Key insight**: The cosine schedule's late-epoch LR matters most when the budget-limited final epoch equals the best checkpoint. This will compound further when bf16-amp provides more epochs.

**Artifact**: `models/model-charliepai2g24h1-tanjiro-cosine-eta-min-20260512-231540/metrics.jsonl`

---

## 2026-05-13 00:00 — PR #1579: [pcgrad-surgery] Gradient surgery for vol/surf conflict

- **Branch**: charliepai2g24h1-frieren/pcgrad-surgery
- **Hypothesis**: PCGrad gradient surgery (Yu et al. NeurIPS 2020) reduces destructive interference between vol_loss and surf_loss gradients, lowering effective update noise that forces 100% gradient clipping.
- **Status**: CLOSED — mechanism confirmed but wall-clock loss is structural at 30-min budget

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 8, EMA weights) | **59.2256** |
| test_avg/mae_surf_p | **53.7574** |
| SOAP baseline | 42.4015 |
| Delta | **+39.6% regression** |
| seconds/epoch | 225 (vs 138 for SOAP baseline, 1.63×) |
| Epochs completed | 8 (vs 13 for SOAP in same wall-clock) |
| conflict_frac (vol vs surf) | **0.04 (sparse)** |
| post-PCGrad grad_norm_mean | 2.28 (vs SOAP 9.16 at ep 13) |
| post-PCGrad grad_norm_max | 6.65 (vs SOAP 259 at ep 13) |
| Peak GPU memory | 47.6 GB (vs SOAP ~25-30 GB) |

**Per-epoch trajectory vs SOAP at matched epochs**:
- PCGrad@ep8 = 59.23 beats SOAP@ep8 = 72.44 by **13.2 points** (mechanism confirmed)
- But SOAP@ep13 = 42.40, which PCGrad never reaches due to 1.63× epoch cost

**Analysis**: The mechanism works — PCGrad achieves 9× lower mean grad-norm and 70× lower max grad-norm vs SOAP at matched epochs. The vol/surf gradient conflict is real but **sparse** (4% of parameter tensors at any given step). The relative-L2 + Huber + SOAP stack already tames most conflict; residual is concentrated in a few high-magnitude tensors. PCGrad's projection addresses those tensors but the 2× backward pass overhead (1.63× wall-clock) cannot be earned back at a 30-min budget. Structural dead-end at this compute level.

**Key insight**: Per-tensor conflict_frac ≈ 0.04 confirms gradient conflict is real but sparse. Most variance in grad norms is magnitude-based (handled by SOAP + loss normalization), not direction-based. Multi-pass gradient methods are structurally disadvantaged at small epoch budgets.

**Artifact**: `models/model-charliepai2g24h1-frieren-pcgrad-surgery-20260512-231800/metrics.jsonl`

---

## 2026-05-12 22:55 — Race-condition send-backs (3 PRs on stale baseline)

After SOAP merged at 22:30 as new baseline (42.4015), three PRs completed concurrently on the pre-SOAP base. None directly comparable to current baseline; all sent back for SOAP-rebased re-test.

| PR | Student | Slug | Final val (stale base) | Decision |
|----|---------|------|------------------------|----------|
| #1456 | alphonse | bf16-amp | 83.9115 (18 epochs, +29% throughput) | Send back: rebase + T_max=17 |
| #1599 | fern | re-conditioned-scaling | 92.4482 (scale corr +0.92 with log_Re) | Send back: rebase + test compound |
| #1630 | tanjiro | sgdr-restarts | 90.6703 (restart cost ~4 epochs) | Send back: pivot to monotone cosine + eta_min=1e-5 |

**Mechanism evidence preserved**:
- **bf16-amp**: 18 epochs vs 14 in same wall-clock = +29% throughput confirmed. Compounds cleanly with SOAP since SOAP only got 13 epochs (val still falling).
- **re-conditioned-scaling**: ReScaleHead worked mechanically (scale correlation with log_Re reached +0.92, exactly as predicted) but didn't beat AdamW baseline. SOAP compound test will reveal: stacks, redundant, or conflicts.
- **sgdr-restarts**: Restart at epoch 8 fired correctly per design but cost ~4 epochs of re-convergence with no better basin. Pivot to tanjiro's own follow-up suggestion (monotone cosine + eta_min=1e-5 floor) — same intent (preserve late-epoch step magnitude) without the restart penalty.

All 3 students now have active rebases against the SOAP baseline. PR #1630 had a code-block truncation in the send-back comment; corrected via follow-up comment.

---

## 2026-05-12 22:xx — PR #1613: [soap-optimizer] SOAP quasi-Newton optimizer

- **Branch**: thorfinn/soap-optimizer
- **Hypothesis**: SOAP (Shampoo as Adam's Preconditioner) provides Kronecker-factored quasi-Newton curvature estimates that condition gradient steps — addressing the root cause of the LR ceiling (poor first-order curvature model), not just its symptoms.
- **Status**: MERGED — new baseline 42.4015 (**largest single improvement in programme**)

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 13) | **42.4015** |
| val_single_in_dist | 46.09 |
| val_geom_camber_rc | 55.98 |
| val_geom_camber_cruise | **24.32** |
| val_re_rand | 43.22 |
| test_avg/mae_surf_p (4-split) | **36.4017** |
| test_single_in_dist | 41.76 |
| test_geom_camber_rc | 48.10 |
| test_geom_camber_cruise | 19.79 |
| test_re_rand | 35.97 |
| Epochs | 13 (~30-min cap, SOAP overhead) |
| grad_norm_mean trace | 38.87 → 9.16 (4.2× reduction) |
| clip_frac trace | 1.000 (ep 1-10) → 0.987 → 0.984 |
| Baseline | 89.3940 |
| Delta | **-52.6%** |

**Analysis**: SOAP's Kronecker-factored preconditioner transforms convergence speed. The 4.2× grad norm reduction (38.87 → 9.16 across 13 epochs) is direct evidence that the preconditioner is working — each step is better conditioned. All 4 val splits improved dramatically (cruise: 66→24, rc: 101→56, re_rand: 81→43, single_in_dist: 109→46). Val was still falling at ep 13 — the model has not converged, suggesting bf16-amp compound would be major.

**Critical diagnostic**: clip_frac=0.984 at ep 13 means SOAP is still being clipped ~9× per step (grad_norm_mean=9.16 vs clip=1.0). This is the basis for the next experiment (soap-relax-clip, PR #1668).

**SOAP install**: pip unavailable; vendored as `soap.py` (upstream commit `a1e553530fde97d0e6b307d7c82ac6d38b072340`).

**Artifacts**: `models/model-charliepai2g24h1-thorfinn-soap-optimizer-20260512-220030/metrics.jsonl`

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

---

## 2026-05-13 04:30 — PR #1794: [torch-compile] torch.compile(mode="default", dynamic=True) on bf16+SOAP stack

- **Branch**: charliepai2g24h1-alphonse/torch-compile
- **Hypothesis**: torch.compile with mode="default" and dynamic=True (for variable-shape pad_collate tensors) would yield +20-30% throughput and enable more epochs within the 30-min wall-clock budget.
- **Status**: **MERGED** — new baseline at val_avg/mae_surf_p = 30.4412

| Metric | Value | vs bf16 baseline (36.8778) |
|--------|-------|---------------------------|
| val_avg/mae_surf_p | **30.4412** | **−17.5%** |
| test_avg/mae_surf_p | **26.1013** | **−18.2%** |
| val_single_in_dist | 34.27 | −8.65 (−20.2%) |
| val_geom_camber_rc | 41.43 | −6.35 (−13.3%) |
| val_geom_camber_cruise | 14.04 | −4.56 (−24.5%) |
| val_re_rand | 32.02 | −6.19 (−16.2%) |
| Epochs in 30 min | **30** | +13 epochs (+76% throughput) |
| Peak GPU memory | 24 GB | −9 GB vs bf16 alone |
| Best epoch | 30 (still descending) | — |

**Key mechanistic findings**:
- `mode="default"` with `dynamic=True` was the correct choice. `reduce-overhead` would have caused recompilation storms because `pad_collate` produces variable-length tensors (different sequence lengths per batch). `dynamic=True` handles this by generating shape-symbolic compiled kernels.
- torch.compile drops peak GPU memory 33→24 GB (better kernel fusion, less intermediate tensor fragmentation).
- T_max was set to 28 (aligning cosine tail with 30-epoch budget) — alphonse correctly auto-detected throughput after a warm-up timing run.
- ALL 8 splits (4 val + 4 test) improved — model was still descending at epoch 30 (the last epoch was the best).
- Cumulative gain from initial 117.17 baseline: **−74.0%**.

**Artifact**: `models/model-charliepai2g24h1-alphonse-torch-compile-20260513-021531/metrics.jsonl`

---

## 2026-05-13 04:30 — PR #1797: [wider-soap-192] n_hidden 128→192 (1.47M params) — CLOSED

- **Branch**: charliepai2g24h1-tanjiro/wider-soap-192
- **Hypothesis**: n_hidden 128→192 (662K → 1.47M params, 2.22×) would increase model capacity and improve generalization.
- **Status**: **CLOSED** — hypothesis falsified cleanly

| Metric | Value | vs baseline (36.8778) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 49.1129 | **+33.2% (worse)** |
| test_avg/mae_surf_p | ~44+ | worse |
| Epochs in 30 min | 14 | vs 30 for 662K model |
| Peak GPU memory | 43 GB | vs 24 GB for baseline |

**Mechanistic conclusion**: Dataset is data-bottlenecked, not optimization-bottlenecked. The training set (1,499 samples) cannot fill the wider representation space — extra parameters worsen rather than improve convergence per epoch. SOAP already conditions the optimization; the limiting factor is information content in the training data. 2.22× param count → 2.14× fewer epochs in 30 min → consistently worse at every matched epoch count. This rules out width as a capacity lever at the current data scale.

---

## 2026-05-13 04:30 — PR #1668: [soap-relax-clip] grad_clip 1.0→5.0 — CLOSED

- **Branch**: charliepai2g24h1-thorfinn/soap-relax-clip
- **Hypothesis**: Relaxing grad_clip from 1.0 to 5.0 would unlock SOAP's natural step magnitude and improve convergence.
- **Status**: **CLOSED** — mechanism confirmed, slight regression vs new baseline

| Metric | Value | vs baseline (36.8778) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 37.2200 | +0.93% (slight regression) |
| test_avg/mae_surf_p | 32.3257 | +1.32% |
| clip_frac by ep 14 | 0.00 | was 0.33 at baseline |

**Mechanistic conclusion**: clip=5.0 fully unlocked SOAP's steps (clip_frac 0.33→0.00, mechanism validated). However, SOAP+cosine+bf16 at clip=1.0 slightly outperforms clip=5.0. By late training (with cosine LR decay), the gradient norms are already small enough that clip=1.0 is non-binding. Widening clip provided no additional signal. The value of clip relaxation was in the early training regime (clip_frac 0.98–1.00), which cosine scheduling has already effectively resolved. Note: comparison to AdamW baseline in student writeup was incorrect — the correct comparator is clip=1.0 vs clip=5.0 on the same SOAP+bf16+cosine stack.


---

## 2026-05-13 04:15 — PR #1847: [larger-batch-compile] batch_size 4→8 — CLOSED

- **Branch**: charliepai2g24h1-alphonse/larger-batch-compile
- **Hypothesis**: Doubling batch size from 4→8 would exploit 72 GB memory headroom freed by torch.compile (24/96 GB), lower gradient variance, and improve generalization on OOD splits.
- **Status**: **CLOSED** — hypothesis falsified, three-mechanism root cause identified

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 36.9205 | **+21.3% (worse)** |
| test_avg/mae_surf_p | 32.0504 | **+22.8% (worse)** |
| Peak GPU memory | 47.69 GB | +23 GB (not +9 GB as predicted) |
| Epochs in 30 min | 30 | same (training NOT compute-bound) |
| Optimizer steps/epoch | 188 | was 376 at batch=4 |

**Three-mechanism root cause**:
1. **Training is not compute-bound at batch=4**: per-epoch wall-time stayed 60s/ep — batch=8 gives identical epoch count (30 in 30 min), so the "fewer epochs" justification was wrong
2. **Half the optimizer updates per epoch** (188 vs 376, LR held at 1e-3 due to clip ceiling): at 1,499 training samples, MORE optimizer steps beats lower gradient variance per step
3. **T_max=23 caused cosine restart regression**: schedule hit floor at ep 23, LR climbed for 7 more epochs (val regressed 36.92 → 38.84). Should have been T_max=28 or T_max=steps-to-timeout

**Memory cost much higher than predicted**: 24 GB → 47.69 GB (+23 GB), not the +9 GB predicted. Activation memory grows super-linearly with batch size under torch.compile graph capture.

**Net programme lesson**: Data-bottleneck manifests in TWO ways — (1) wider models can't fill representation space, (2) larger batches halve optimizer steps without providing extra information. Both ruled out for 1,499-sample dataset.


---

## 2026-05-13 04:45 — PR #1854: [soap-fp32-precond] SOAP GG/Q in fp32 under bf16 AMP — CLOSED

- **Branch**: charliepai2g24h1-thorfinn/soap-fp32-precond
- **Hypothesis**: bf16 precision in SOAP's GG/Q eigenbases degraded preconditioner quality; keeping them in fp32 would improve OOD generalization.
- **Status**: **CLOSED** — hypothesis inverted, bf16 Q acts as implicit regularization

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 31.7537 | **+4.31% (worse)** |
| test_avg/mae_surf_p | 27.1862 | **+4.16% (worse)** |
| val_single_in_dist | 33.40 | -2.5% (better!) |
| val_geom_camber_rc | 45.01 | +8.6% (worse) |
| val_geom_camber_cruise | 15.04 | +7.1% (worse) |
| val_re_rand | 33.57 | +4.8% (worse) |

**Pattern**: in-dist improved, ALL 3 OOD splits degraded. This is NOT numerical noise — it's the signature of overfitting. Sharp fp32 preconditioner fit training distribution tighter; bf16 Q's rounding noise acted as implicit regularization that generalized better to OOD.

**Key finding**: All changes confirmed applied (GG fp32 init, fp32 grad for lerp_, Q stays fp32, project/project_back cast to fp32). Memory unchanged (23.87 GB ≈ 24 GB baseline).

**This is the third consecutive experiment showing the same OOD-worse pattern (wider, deeper, more precise preconditioner). Model is regularization-limited, not capacity/precision-limited.**

---

## 2026-05-13 04:45 — PR #1848: [deeper-soap] n_layers 5→7 — CLOSED

- **Branch**: charliepai2g24h1-tanjiro/deeper-soap
- **Hypothesis**: depth increases representational power more data-efficiently than width; n_layers 5→7 with n_hidden=128 keeps params moderate (880K→904K).
- **Status**: **CLOSED** — compute-budget falsification (still descending at ep 21 cutoff)

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 33.9762 | **+11.6% (worse)** |
| test_avg/mae_surf_p | 29.1507 | **+11.7% (worse)** |
| Per-epoch wall time | 84s | was 60-65s (+30%) |
| Epochs in 30 min | 21 | was 30 |
| Peak GPU | 32.4 GB | was 24 GB |

**Val trajectory at cutoff**: 38.53 → 36.04 → 34.75 → **33.98** (ep 21, steep ~1/ep descent). Model was still converging fast; **compute-budget verdict, not intrinsic verdict**. BUT: at fixed 30-min wall-clock, the 662K/5-layer model dominates by running 30 epochs vs 21.

**Uniform regression across all splits** (not OOD-targeted) → simple undertraining, not compositionality issue.

**Programme lesson**: Both width (wider-soap-192) AND depth (deeper-soap) fail at fixed 30-min budget on 1,499 samples. Current 662K/5-layer is in the optimal compute zone. Data-bottleneck is confirmed. Moving to regularization-based improvements.

---

## 2026-05-13 04:55 — PR #1897: [stochastic-depth] DropPath drop_path_max=0.1 across layers — CLOSED

- **Branch**: charliepai2g24h1-thorfinn/stochastic-depth
- **Hypothesis**: Linear DropPath schedule [0, 0.025, 0.05, 0.075, 0.1] should regularize OOD without inference cost. Predicted OOD splits (rc, re_rand) improve most, in-dist may regress slightly.
- **Status**: **CLOSED** — clean negative, hypothesis falsified
- **Metrics JSONL**: `models/model-charliepai2g24h1-thorfinn-stochastic-depth-20260513-041301/metrics.jsonl`

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 33.0241 | **+8.48% (worse)** |
| test_avg/mae_surf_p | 28.5180 | **+9.26% (worse)** |
| val_single_in_dist | 37.42 | **+9.20%** (WORST regression) |
| val_geom_camber_rc | 45.46 | +9.72% |
| val_geom_camber_cruise | 16.23 | +15.57% |
| val_re_rand | 32.99 | +3.03% |
| Peak GPU | 23.88 GB | unchanged |
| Epochs in 30 min | 29 | -1 |

**Pattern**: EVERY split regressed, in-dist regression LARGEST. Hypothesis predicted opposite (OOD-asymmetric improvement). Best epoch = 29 (last) → still descending.

**Why DropPath failed here**: With only 5 transformer blocks, the linear schedule mean ≈ 5% expected skip rate is too coarse — skipping a whole block is much more destructive than dropping features. Each block likely encodes non-redundant Transolver slice/attention patterns; redundancy assumption violated.

**Net programme lesson**: The "regularization-limited" diagnosis is refuted. Combined with attention-dropout (#1900, ~0% net), TWO independent regularization experiments fail to improve OOD. The OOD-asymmetric regressions in wider/deeper/sharper-precond are better explained by **optimization fragility + compute-budget loss** (each ate epochs through extra per-step cost), NOT by underfit regularization.

---

## 2026-05-13 04:55 — PR #1900: [attention-dropout] dropout=0.1 in PhysicsAttention — CLOSED

- **Branch**: charliepai2g24h1-tanjiro/attention-dropout
- **Hypothesis**: Enable already-wired but no-op dropout (attn weights + output projection) at p=0.1. Predicted OOD-asymmetric improvement.
- **Status**: **CLOSED** — within-noise negative, but the per-split signature is diagnostic
- **Metrics JSONL**: `models/model-charliepai2g24h1-tanjiro-attention-dropout-20260513-041403/metrics.jsonl`

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 30.5841 | +0.47% (within noise) |
| test_avg/mae_surf_p | 26.6998 | +2.29% (worse) |
| val_single_in_dist | 33.94 | **-1.0% (better!)** |
| val_geom_camber_rc | 42.65 | +2.9% |
| val_geom_camber_cruise | 14.92 | +6.3% |
| val_re_rand | 30.82 | **-3.7% (better!)** |
| Peak GPU | 24.49 GB | unchanged |
| Epochs in 30 min | 29 | -1 |

**Smoking gun observation from student**: *"Loss curve was still trending down at epoch 29 — this is itself evidence the model is not regularization-limited — there was no train/val gap to close."*

**Pattern**: OOD splits split (1 better re_rand / 2 worse rc, cruise). In-dist actually improved (-1.0%) — opposite of the regularization-overfit prediction. Looks like noise-level perturbation with a single positive outlier.

**Net result**: This — combined with stochastic-depth — refutes the regularization-limited diagnosis. The next theme should be **convergence/budget-aware experiments**: weight averaging (SWA, EMA), faster schedules (OneCycleLR #1884 in flight), loss-domain rebalancing (lower surf_weight), NOT more regularization.

**Diagnostic signal preserved**: val_re_rand improved -3.7%, the only positive outlier across both runs. Worth asking whether re_rand (random-Re OOD) responds to a Re-specific regularizer that uniform dropout doesn't capture — points to fern's re-conditioned-scaling direction (#1599).



---

## 2026-05-13 05:05 — PR #1599: [re-conditioned-scaling] Learned Re-conditioned output scale head — MERGED

- **Branch**: fern/re-conditioned-scaling
- **Hypothesis**: Re varies 50×+ in the dataset; pressure magnitudes vary by orders of magnitude. Add a tiny 163-param ReScaleHead (log_Re → softplus scale per channel) on top of Transolver output to separate shape learning from scale calibration. Inspired by DimINO Re-dimensionalization (Huang et al. 2024).
- **Status**: **MERGED** — val_avg -1.95%, new baseline 29.8463
- **Metrics JSONL**: `models/model-charliepai2g24h1-fern-re-conditioned-scaling-20260513-035742/metrics.jsonl`

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | **29.8463** | **−0.59 (−1.95%)** |
| test_avg/mae_surf_p | 26.1005 | −0.0008 (≈0%) |
| val_single_in_dist | 30.20 | −4.07 (-11.9%) |
| val_geom_camber_rc | 43.11 | +1.68 (+4.1%) |
| val_geom_camber_cruise | 14.54 | +0.50 (+3.6%) |
| val_re_rand | 31.54 | −0.48 (−1.5%) |
| Epochs in 30 min | 29 | unchanged |
| Peak GPU | 24 GB | unchanged |

**ReScaleHead diagnostics (best epoch 27)**:
| Channel | scale mean | scale std | corr(scale, log Re) |
|---------|-----------|----------|---------------------|
| Ux | 1.180 | 0.058 | +0.637 |
| Uy | 1.111 | 0.262 | +0.936 |
| p | 1.308 | 0.527 | +0.858 |

**Analysis**: Val wins via single_in_dist (-4.07) dominating OOD regressions (+2.18 summed). Test is flat (in-dist and OOD gains cancel). Mechanism confirmed in all 3 runs: Uy/p show strong Re-correlation (0.86–0.94); Ux is weak (freestream-dominated). The compound size shrunk significantly vs the SOAP-only baseline run (was -4.7%, now -1.95%) because the SOAP + torch.compile backbone implicitly learns Re-scale through 30 epochs. Still a valid compounding win.

**Programme implication**: ReScaleHead is now the default in the advisor branch. All future experiments inherit it. Future compound direction: 2-channel head (Ux scale ≈ identity; drop Ux to reduce parameter noise) or FiLM-style conditioning (inject log(Re) into PhysicsAttention slice weighting instead of output rescaling).

**Cumulative programme gain**: −74.5% from 117.17 → 29.8463
