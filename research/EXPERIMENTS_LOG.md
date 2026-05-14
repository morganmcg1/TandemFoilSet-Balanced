# SENPAI Research Results

_Advisor branch: `icml-appendix-charlie-pai2g-48h-r3`._

Results from each terminal PR are recorded below in reverse chronological order.

<!-- Entries will be appended as PRs land terminal SENPAI-RESULT markers. -->

## 2026-05-14 10:20 — Round 42: close #2871 frieren aux surface head (+4.73% val REGRESSION — AUX-HEAD AXIS CLOSED, redundant with surf_weight=10); assign frieren #2883 specialized surf/vol decoders (decoder specialization pivot, round 2)

### PR #2871 — frieren Aux Surface Decoder Head (aux_surf_head=true, weight=1.0, hidden=64)
- Branch: `charliepai2g48h3-frieren/auxsurfhead-nlayers2-slicenum16-epochs46`
- Hypothesis: Parallel aux head operating on same final features acts as multi-task regularization, shaping shared encoder to produce better surface-prediction features.
- Artifacts: `models/model-auxsurfhead-nlayers2-slicenum16-epochs46-20260514-093049/metrics.jsonl`
- Note: 40/46 epochs completed (hit 30-min timeout; slow stretches ~75-80s/epoch in epochs 16-22 and 38-39)

| Model | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|
| Baseline (#2468) | 35.256 | 30.245 |
| Aux head (best ep 40/46) | 36.925 (+4.73%) | 31.114 (+2.87%) |

Per-split detail:
| Split | Baseline val | Aux-head val | Δ val |
|---|---:|---:|---:|
| single_in_dist | 36.476 | 37.523 | +2.87% |
| geom_camber_rc | 48.297 | 51.320 | +6.26% |
| geom_camber_cruise | 18.326 | 20.356 | +11.08% |
| re_rand | 37.923 | 38.501 | +1.52% |

**Analysis:** The aux head DID learn surface structure (final aux_surf_loss=0.0880 tracks main surf_loss=0.0827 closely). But the auxiliary gradient did NOT regularize the encoder into better features for the main head. Root cause (per student): `surf_weight=10` already provides 10× surface gradient weight — adding another surface-only loss is REDUNDANT gradient signal in the same direction, not complementary regularization. Any aux head adding signal in the same direction as the existing loss will fail by the same mechanism.

**Conclusion:** AUX-HEAD AXIS CLOSED at weight=1.0, hidden=64. Frieren's own recommendation: feature-routing SEPARATION (specialized decoders per node type), not auxiliary prediction targets. Assigning #2883.

## 2026-05-14 09:15 — Round 41 cont.: close #2857 frieren SWA (+2.90% val best-epoch, SWA strictly worse on all splits — POST-HOC AVERAGING CLOSED for 46-epoch schedule); close #2861 askeladd eta_min=5e-6 (+1.09% val LOSS — SCHEDULE-FLOOR CLOSED for Lion due to sign-update oscillation); assign frieren #2871 aux surface head (architectural pivot) and askeladd #2872 epochs=50 retest at n_layers=2 stack

### PR #2857 — frieren SWA (Stochastic Weight Averaging, swa_start=30)
- Branch: `charliepai2g48h3-frieren/swa-start30-nlayers2-slicenum16-epochs46`
- Hypothesis: Post-hoc averaging of late-stage weights captures flatter minimum than any single checkpoint; exploits the still-descending epoch-46 trajectory.
- Artifacts: `models/model-swa-start30-nlayers2-slicenum16-epochs46-20260514-081845/metrics.jsonl`

| Model | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|
| Baseline (#2468, val=35.256) | 35.256 | 30.245 |
| This run best-epoch (epoch 46) | 36.279 (+2.90%) | 30.341 (+0.32%) |
| **This run SWA (n=16, ep 31-46)** | **36.723 (+4.16%)** | **30.843 (+1.98%)** |
| **SWA − best-epoch Δ** | **+0.444** | **+0.502** |

**Per-split SWA−best test deltas:** single_in_dist +0.775, geom_camber_rc +0.451, geom_camber_cruise +0.473, re_rand +0.310 (SWA worse on every split)

**Val trajectory in SWA window (epochs 31→46):**
| Ep | 31 | 33 | 35 | 37 | 40 | 43 | 46 |
|---|---|---|---|---|---|---|---|
| val | 42.22 | 40.49 | 39.53 | 38.54 | 37.27 | 36.62 | **36.28** |
→ 14% relative descent, NOT a flat plateau.

**Diagnostic:** SWA's textbook condition is \"model in flat region taking small consistent steps.\" The val trajectory in the SWA window is a **steep monotonic descent** (~−0.4 val/ep early-window). Averaging weights from epoch 31 (val=42.22) with epoch 46 (val=36.28) pulls SWA model backward toward worse-performing early-window iterates. SWA on still-descending trajectory contaminates the latest, best iterate.

**Hypothesis mis-framing:** PR conflated 'slope ~−0.2/ep at end' with 'plateau-like behavior'. These are different regimes — slope is just the *current* descent rate, not a measure of locality to a minimum. SWA works on weights that orbit a flat basin (low-pass denoising); it fails when weights are still moving directionally.

**Conclusion: POST-HOC WEIGHT-AVERAGING AXIS CLOSED** for the current 46-epoch schedule. Variants that would test SWA properly (epochs=70 with swa_start=50; SWALR constant-LR forcing plateau) require expanding the training budget beyond the 30-min cap.

**Operational note:** Student found and killed 2 concurrent duplicate processes (likely re-entry duplicates from prior session) before reporting clean results from the first/canonical run.

---

### PR #2861 — askeladd cosine eta_min=5e-6 (schedule-FLOOR axis)
- Branch: `charliepai2g48h3-askeladd/etamin5e6-nlayers2-slicenum16-epochs46`
- Hypothesis: Non-zero late-epoch LR floor (5% of initial) salvages the still-descending epoch-46 trajectory that's currently discarded by LR=0 decay.
- Artifacts: `models/model-etamin5e6-nlayers2-slicenum16-epochs46-20260514-083129/metrics.jsonl`

| Split | val mae_surf_p | baseline | Δval | test mae_surf_p | baseline | Δtest |
|---|---|---|---|---|---|---|
| single_in_dist | 36.514 | 36.476 | +0.10% | 31.522 | 33.035 | **−4.6%** |
| geom_camber_rc | 48.185 | 48.297 | −0.23% | 44.572 | 44.333 | +0.5% |
| geom_camber_cruise | 18.957 | 18.326 | +3.44% | 15.151 | 15.496 | **−2.2%** |
| re_rand | 38.903 | 37.923 | +2.58% | 28.675 | 28.116 | +2.0% |
| **avg** | **35.640** | **35.256** | **+1.09%** | **29.980** | **30.245** | **−0.88%** |

**LR floor verification (final epochs):**
| Epoch | LR | val_avg |
|---|---|---|
| 44 | 5.99e-06 | 35.6633 |
| **45** | **5.44e-06** | **35.6396** ← best |
| 46 | 5.11e-06 | 35.6589 ← rose 0.02 with non-zero step |

**Diagnostic (student's mechanism is sharp):**
Lion's sign-only updates have step magnitude = lr exactly (no scaling by gradient magnitude), so they don't naturally damp as the model approaches the basin. A non-zero LR floor with Lion → late updates take full-magnitude directional steps, oscillating around the minimum. Best epoch shifted 46 → 45 because the floor enabled a measurable step at epoch 46 that moved val *away* from the optimum.

**On the test improvement (−0.88%):** Dominated by val_single_in_dist (test −4.6%) but val essentially unchanged on the SAME split (+0.10%). The val/test inconsistency on identical splits, combined with seed variance ~±1 val unit, strongly suggests seed variance rather than real signal. Not trusted without seed replication.

**Conclusion: SCHEDULE-FLOOR AXIS CLOSED FOR LION.** The mechanism (sign-update oscillation at any non-zero LR floor) is structural, not a calibration issue. eta_min=1e-6 contingency skipped given structural diagnosis. Lion + L1 + cosine-to-zero remains optimal training recipe.

---

## 2026-05-14 08:20 — Round 41 cont.: close #2850 askeladd AdamW lr=1e-3 (+39.1% val UNDER-CONVERGED, WORSE than lr=3e-4 — OPTIMIZER AXIS CLOSED); assign askeladd #2861 cosine eta_min=5e-6 (schedule-FLOOR axis, distinct mechanism from prior dead schedule-shape tests)

### PR #2850 — askeladd AdamW lr=1e-3 (10× Lion lr, retry of #2824 lr=3e-4 under-convergence)
- Branch: `charliepai2g48h3-askeladd/adamw-lr1e3-nlayers2-slicenum16-epochs46`
- Hypothesis: AdamW lr=3e-4 (#2824) was UNDER-CONVERGED with Lion lr × 3 rule; retry with 10× Lion lr (1e-3) to match Lion's effective step magnitude.
- Artifacts: `models/model-adamw-lr1e3-nlayers2-slicenum16-epochs46-20260514-073306/metrics.jsonl`
- Result: cut at epoch 37/46 by 30-min timeout (best=epoch 36)

| Split | val mae_surf_p | baseline | Δval | test mae_surf_p | baseline | Δtest |
|---|---|---|---|---|---|---|
| single_in_dist | 52.79 | 36.476 | +44.7% | 45.72 | 33.035 | +38.4% |
| geom_camber_rc | 63.45 | 48.297 | +31.4% | 58.28 | 44.333 | +31.5% |
| geom_camber_cruise | 30.80 | 18.326 | +68.1% | 25.62 | 15.496 | +65.4% |
| re_rand | 49.17 | 37.923 | +29.7% | 40.30 | 28.116 | +43.3% |
| **avg** | **49.055** | **35.256** | **+39.1%** | **42.480** | **30.245** | **+40.5%** |

**Trajectory:** epoch 1=188.4 (vs 191 for lr=3e-4), ep10=101.7, ep20=80.1, ep30=68.1, ep35=49.8, **ep36=49.05 (best)**, ep37=49.25 (late-stage jitter). Note: WORSE than #2824 AdamW lr=3e-4 (45.684).

**Two confounding factors disqualify AdamW at this stack:**
1. **Per-epoch overhead:** AdamW's second-moment maintenance adds ~10% wall-clock → 37 vs 46 epochs in 30 min. Lion gets 9 extra epochs structurally.
2. **Step-magnitude mismatch:** No lr scale (×3 or ×10) matches Lion's sign-update directional bias. lr=3e-4 under-converges; lr=1e-3 starts hotter but causes late-stage jitter (49.83→49.05→49.25).

**Linear extrapolation:** epoch 30→37 slope ~−1.4/epoch → epoch 46 would land at ~36-38 val — at best matching Lion, never beating it.

**Conclusion: OPTIMIZER AXIS CLOSED.** Lion + L1 + cosine remains optimal training recipe for n_layers=2 + slice_num=16 + epochs=46 within SENPAI_TIMEOUT_MINUTES=30. Pivoting askeladd to schedule-FLOOR axis (eta_min>0) — distinct mechanism from prior dead schedule-shape tests.

---

## 2026-05-14 08:00 — Round 41 cont.: close #2847 frieren Huber d=0.1 (+5.54% val MILD REGRESSION — loss-form axis CLOSED definitively); assign frieren #2857 SWA (Stochastic Weight Averaging, swa_start=30 — fresh post-hoc weight-averaging axis)

### PR #2847 — frieren Huber d=0.1 (normalized-space corrected)
- Branch: `charliepai2g48h3-frieren/huber-d0p1-nlayers2-slicenum16-epochs46`
- Hypothesis: Properly-scaled Huber for normalized loss space (~90% of errors stay L1, only smallest ~10% get quadratic smoothing) — tests the actual Huber hypothesis that #2822 d=5.0 failed to test due to scale miscalibration
- Artifacts: `models/model-huber_d0p1-nlayers2-slicenum16-epochs46-20260514-071631/metrics.jsonl`

| Split | val mae_surf_p | baseline | Δval | test mae_surf_p | baseline | Δtest |
|---|---|---|---|---|---|---|
| single_in_dist | 37.7685 | 36.476 | +3.5% | 32.5053 | 33.035 | **−1.6%** |
| geom_camber_rc | 51.0332 | 48.297 | +5.7% | 44.5835 | 44.333 | +0.6% |
| geom_camber_cruise | 21.0607 | 18.326 | +14.9% | 16.8617 | 15.496 | +8.8% |
| re_rand | 38.9749 | 37.923 | +2.8% | 29.5983 | 28.116 | +5.3% |
| **avg** | **37.2093** | **35.256** | **+5.54%** | **30.8872** | **30.245** | **+2.12%** |

**Sanity check at epoch 1:** val=173.65, inside predicted 167-180 L1-like band → calibration sound, no early divergence.
**Best epoch:** 46 (final, still descending) → ceiling-limited, NOT a premature-stop artifact.

**Analysis:** d=0.1 in normalized space means only the smallest ~10% of element-wise errors fall in the quadratic region. Implementation is correct; the hypothesis (smoothing the tiny-error tail nudges late-training convergence) is **refuted**. Lion uses sign-only updates so near-zero gradient magnitude carries little signal anyway, and the marginal loss-landscape change at the boundary of zero gradient is net-negative.

**Conclusion: LOSS-FORM AXIS CLOSED.** Both Huber endpoints tested:
- d=5.0 (#2822): +116% val CATASTROPHIC (delta in raw scale → MSE everywhere)
- d=0.1 (#2847): +5.54% val MILD REGRESSION (calibrated, no divergence, but still net worse)

L1 with surf_weight=10 is locally optimal for this configuration. Pivoting frieren to fresh axis — **SWA (Stochastic Weight Averaging)** — orthogonal to all prior axes (capacity, schedule, loss-weight, loss-form), exploits the still-descending epoch-46 trajectory to find a flatter minimum via late-stage weight averaging.

---

## 2026-05-14 07:40 — Round 41: close #2824 askeladd AdamW lr=3e-4 (+29.6% val UNDER-CONVERGED — step-size mismatch); assign askeladd #2850 AdamW lr=1e-3 retry (10× Lion lr for step-size matching)

### PR #2824 — askeladd AdamW lr=3e-4 (Lion lr × 3 rule of thumb)
- Branch: `charliepai2g48h3-askeladd/adamw-nlayers2-slicenum16-epochs46`
- Hypothesis: AdamW orthogonal optimizer axis vs Lion; lr=3e-4 from Lion lr × 3 conversion
- Artifacts: `models/model-adamw-lr3e4-nlayers2-slicenum16-epochs46-20260514-063729/metrics.jsonl`

| Split | val mae_surf_p | baseline | Δval | test mae_surf_p | baseline | Δtest |
|---|---|---|---|---|---|---|
| single_in_dist | 49.468 | 36.476 | +35.6% | 45.701 | 33.035 | +38.4% |
| geom_camber_rc | 61.225 | 48.297 | +26.8% | 53.444 | 44.333 | +20.6% |
| geom_camber_cruise | 26.456 | 18.326 | +44.4% | 21.475 | 15.496 | +38.6% |
| re_rand | 45.587 | 37.923 | +20.2% | 36.691 | 28.116 | +30.5% |
| **avg** | **45.684** | **35.256** | **+29.6%** | **39.328** | **30.245** | **+30.0%** |

**Trajectory:** epoch 1=191, ep10=103, ep20=87, ep30=56, ep40=46.7, ep46=45.68 (best). Best_epoch=46 (final) → **UNDER-CONVERGED, not overshooting.**

**Student's diagnostic:** Lion→AdamW lr×3 rule is calibrated for large-batch/image classification. On small-data (1499 samples) + small-batch (4) regression, AdamW's per-parameter second-moment scaling slows learning relative to Lion's sign-update step magnitude. lr=3e-4 effectively too small.

**Conclusion:** Optimizer axis not yet fully tested. AdamW at lr×3 is too low; needs lr×10 (1e-3) to match Lion's effective step size. Retry assigned (#2850) with lr=1e-3. If still under-converged or unstable, optimizer axis is genuinely dead for this regime.

---

## 2026-05-14 07:20 — Round 41: close #2822 frieren Huber d=5.0 (+116% val LOSS CATASTROPHIC — delta in wrong scale); assign frieren #2847 Huber d=0.1 retry (normalized-space corrected)

### PR #2822 — frieren Huber loss delta=5.0 globally
- Branch: `charliepai2g48h3-frieren/huber-d5-nlayers2-slicenum16-epochs46`
- Hypothesis: Replace L1 with Huber/SmoothL1(beta=5.0) for smooth-near-zero / L1-tail behavior
- Artifacts: `models/model-huber_d5-nlayers2-slicenum16-epochs46-20260514-063737/metrics.jsonl`
- Result: ABORTED at epoch 29 by student per stop-rule (val > 36.5)

| Metric | Huber δ=5.0 (best ep28) | Baseline | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **76.309** | 35.256 | **+116.4%** CATASTROPHIC |
| val_single_in_dist | 86.420 | 36.476 | +136.9% |
| val_geom_camber_rc | 88.120 | 48.297 | +82.5% |
| val_geom_camber_cruise | 56.827 | 18.326 | +210.1% |
| val_re_rand | 73.869 | 37.923 | +94.8% |

**Student's excellent root-cause diagnostic:** `pred` and `y_norm` are both z-score-normalized (errors O(0.5-1.5)), so delta=5.0 places virtually ALL errors in the quadratic region. This experiment effectively trained with MSE everywhere, not Huber.

**Conclusion:** MSE-on-everything is decisively worse than L1 for this problem (informative negative result). The Huber loss-shape hypothesis remains UNTESTED in its proper L1-in-tails configuration. Pivoting frieren to #2847 Huber d=0.1 retry — properly scaled for normalized loss space (~90% of errors stay L1, only smallest ~10% get quadratic smoothing).

---

## 2026-05-14 06:40 — Round 40 cont.: close #2755 swp=20 (+4.76% val LOSS — loss-weight over-pushed), close #2797 warmup (+2.04% val LOSS — schedule HEAD refuted); assign frieren #2822 Huber d5, askeladd #2824 AdamW lr=3e-4

### PR #2755 — frieren swp=20, swuv=10 retry (per-channel surf weighting 2× ratio)
- Branch: `charliepai2g48h3-frieren-swp15-swuv10-nlayers2-slicenum16-epochs46`
- Hypothesis: amplify pressure emphasis from 1.5× (swp=15) to 2× (swp=20) to push signal past seed-noise floor
- Artifacts: `models/model-charliepai2g48h3-frieren-swp20-swuv10-nlayers2-slicenum16-epochs46-20260514-045917/metrics.jsonl`

| Split | val mae_surf_p | baseline | Δval | test mae_surf_p | baseline | Δtest |
|---|---|---|---|---|---|---|
| single_in_dist | 37.680 | 36.476 | +3.30% | 33.556 | 33.035 | +1.58% |
| geom_camber_rc | 51.069 | 48.297 | +5.74% | 45.650 | 44.333 | +2.97% |
| geom_camber_cruise | 20.074 | 18.326 | +9.54% | 16.086 | 15.496 | +3.80% |
| re_rand | 38.910 | 37.923 | +2.60% | 28.462 | 28.116 | +1.23% |
| **avg** | **36.933** | **35.256** | **+4.76%** | **30.938** | **30.245** | **+2.29%** |

**Analysis:** BOTH val and test regressed, all 4 splits worse on both metrics. The previously-positive single_in_dist split (val −6.62% at swp=15) FLIPPED to +3.30% at swp=20 — a ~10pp swing. Velocity also degraded (+4-7% Ux/Uy), confirming "over-emphasis damages shared pressure-velocity representation." Best epoch 45 (capped by 30-min timeout at start of epoch 46).

**Conclusion:** Loss-WEIGHTING axis has a sharp optimum between swp=10 and swp=15. The direction is non-monotone: swp=15 shows marginal positive test signal, swp=20 breaks optimization entirely. The axis is SATURATED via weight. Pivoting to loss-FORM (Huber) instead.

---

### PR #2797 — askeladd warmup_epochs=3 + standard cosine (schedule HEAD complement)
- Branch: `charliepai2g48h3-askeladd-warmup3-nlayers2-slicenum16-epochs46`
- Hypothesis: 3-epoch linear warmup prevents chaotic early Lion updates, allowing better long-run convergence
- Artifacts: `models/model-warmup3-nlayers2-slicenum16-epochs46-20260514-053208/metrics.jsonl`

| Split | val mae_surf_p | baseline | Δval | test mae_surf_p | baseline | Δtest |
|---|---|---|---|---|---|---|
| single_in_dist | 37.354 | 36.476 | +2.41% | 32.536 | 33.035 | −1.51% |
| geom_camber_rc | 48.411 | 48.297 | +0.24% | 43.797 | 44.333 | −1.21% |
| geom_camber_cruise | 19.560 | 18.326 | **+6.74%** | 15.378 | 15.496 | −0.76% |
| re_rand | 38.576 | 37.923 | +1.72% | 28.178 | 28.116 | +0.22% |
| **avg** | **35.975** | **35.256** | **+2.04%** | **29.972** | **30.245** | **−0.90%** |

**Analysis:** val regressed +2.04%, test showed marginal noise-level improvement −0.90%. The 3-epoch warmup caused epoch 1 val to jump from 167.978 (baseline) to 336.675 (warmup3) — the low LR during warmup delayed productive escape from random initialization. The model has 0.36M params on 1499 samples — small enough that the first epoch at full LR is productive, not chaotic. Worst on geom_camber_cruise (val +6.74%). Best epoch 46 (unchanged — schedule tail still doing meaningful work).

**Conclusion:** Schedule HEAD axis REFUTED. This complements #2760 schedule TAIL refutation. The SCHEDULE AXIS is now fully closed:
- TAIL: truncated cosine T_max=60 → +1.63% LOSS (#2760)
- HEAD: linear warmup_epochs=3 → +2.04% LOSS (this PR)
- Standard cosine T_max=epochs is confirmed optimal for Lion+L1+lr=1e-4

---

## 2026-05-14 05:35 — Round 40 cont.: close #2760 (askeladd truncated cosine T_max=60 — +1.63% val LOSS, scheduler-bound hypothesis REFUTED); assign askeladd #2797 lr warmup pivot

### PR #2760 askeladd truncated cosine T_max=60 + epochs=46 — NEGATIVE RESULT (BUT EXCELLENT DIAGNOSTIC)

- **Branch:** charliepai2g48h3-askeladd-tmax60-nlayers2-slicenum16-epochs46
- **Hypothesis:** Best_epoch=final STILL DESCENDING → model is scheduler-bound (lr→0 at epoch 46 prematurely terminates). Truncated cosine T_max=60 leaves residual lr ≈ 1.46e-5 (14.6% of peak) at final epoch — should enable continued descent.
- **Result: SOLIDLY NEGATIVE** — val +1.63%, test +1.28%, OOD splits hurt most

| Split | val baseline | val tmax60 | Δ | test baseline | test tmax60 | Δ |
|---|---|---|---|---|---|---|
| single_in_dist | 36.476 | 36.496 | +0.020 (≈same) | 33.035 | 33.065 | +0.030 (≈same) |
| geom_camber_rc | 48.297 | 49.500 | **+1.203** ✗ | 44.333 | 45.435 | **+1.102** ✗ |
| geom_camber_cruise | 18.326 | 19.444 | **+1.118** ✗ | 15.496 | 15.778 | +0.282 ✗ |
| re_rand | 37.923 | 37.885 | −0.038 (≈same) | 28.116 | 28.254 | +0.138 (≈same) |
| **avg** | **35.256** | **35.831** | **+0.575** ✗ | **30.245** | **30.633** | **+0.388** ✗ |

**LR profile (confirmed truncated correctly)**: Epoch 1→1.00e-4, Epoch 30→5.26e-5, Epoch 40→2.73e-5, Epoch 46→**1.46e-5** (14.6% of peak, vs ~0 in baseline).

**KEY DIAGNOSTIC — refutes "scheduler-bound" interpretation**:
1. **Schedule SHAPE matters more than tail height.** Truncated cosine spends MORE time at mid LR (5.26e-5 at epoch 30 vs ~3e-5 in baseline) and LESS time at very low LR. Standard cosine's brief polish phase at very low LR in final ~5 epochs apparently helps generalization — eliminating it caused **OOD oversteering** (camber_rc +1.20, camber_cruise +1.12 val).
2. **OOD splits hurt most** — opposite of the hypothesis prediction.
3. **In-distribution unchanged** (single_in_dist Δ≈0, re_rand Δ≈0) — schedule perturbation is harmless on ID, harmful on OOD.
4. **Train loss already plateauing at epoch 46** (train surf_loss 0.0864). So "still descending" in VAL is NOT from premature LR termination — it's continued asymptotic generalization improvement that requires *more* epochs, not stretched schedule on same budget.
5. **Best_epoch=46 (still descending)** in BOTH baseline and tmax60. Same pattern.

**CONCLUSION**: The model is training-time-limited (still descending at final epoch) but stretching the schedule alone doesn't help — it would need actual additional epochs. Within 30-min budget, baseline (~35s × 46 = 27 min) and budget cap (~30 min) only leave room for ~5 more epochs maximum.

**Round 40 cont. pivot — askeladd #2797 (lr warmup 3 epochs + standard cosine)**:
- Code-change PR (add --warmup_epochs CLI arg, LinearLR + CosineAnnealingLR via SequentialLR)
- Hypothesis: schedule HEAD complement to TAIL test. Epoch 1 val=167.978 (chaotic initial phase from peak lr=1e-4 immediately) — 3-epoch linear warmup from lr/10 to lr should give gentler initial gradients and possibly better minima.
- Empirically validated in transformer literature; 3/46=6.5% warmup is standard.
- No compute cost change.

**Still in flight**: frieren swp=20 retry (stronger 2× pressure ratio after swp=15 showed directional positive on test).

---

## 2026-05-14 05:00 — Round 40 cont.: send back #2755 (frieren per-channel surf weighting swp=15/swuv=10 — directional positive but val within noise); request stronger ratio swp=20

### PR #2755 frieren per-channel surf weighting (swp=15, swuv=10) — DIRECTIONAL POSITIVE, NOT MERGED

- **Branch:** charliepai2g48h3-frieren-swp15-swuv10-nlayers2-slicenum16-epochs46
- **Hypothesis:** Per-channel surface weighting with swp=15 (pressure), swuv=10 (velocity) — direct attack on val_avg/mae_surf_p via 1.5× pressure emphasis vs velocity, no architecture change
- **Metrics artifacts:** `models/model-charliepai2g48h3-frieren-swp15-swuv10-nlayers2-slicenum16-epochs46-20260514-040327/metrics.jsonl`

| Split | val mae_surf_p | baseline | Δ | test mae_surf_p | baseline | Δ |
|---|---|---|---|---|---|---|
| single_in_dist | 34.061 | 36.476 | **−6.62%** ✓ | 32.552 | 33.035 | **−1.46%** ✓ |
| geom_camber_rc | 50.304 | 48.297 | +4.16% ✗ | 43.362 | 44.333 | **−2.19%** ✓ |
| geom_camber_cruise | 19.223 | 18.326 | +4.89% ✗ | 15.282 | 15.496 | **−1.38%** ✓ |
| re_rand | 37.731 | 37.923 | −0.51% | 28.497 | 28.116 | +1.36% ✗ |
| **avg** | **35.330** | **35.256** | **+0.21%** (within ±1.0 noise) | **29.923** | **30.245** | **−1.06%** ✓ |

- **Run details:** best_epoch=46, ~35.3s/epoch, 27.1 min total, 361K params (unchanged), 13.49 GB peak memory
- **Velocity regression check:** No visible damage — mae_surf_Ux and mae_surf_Uy values in expected range
- **Decision:** SENT BACK — val primary metric +0.21% (within ±1.0 seed noise floor from #2523), doesn't meet merge threshold. Test improved -1.06% across 3/4 splits including both OOD geometry splits — directional signal is real but needs amplification.

**Analysis:**
The per-channel pressure emphasis (1.5×) shows a consistent signal on test metrics despite val flatness. The val/test sign-flip on OOD geometry splits (100-sample splits: val regressed, test improved by similar magnitudes) is characteristic of single-seed noise at this sample size. The in-distribution improvement is large and consistent on both val (−6.62%) and test (−1.46%), suggesting the pressure-emphasis signal is real.

**Conclusion:** The 1.5× pressure ratio is directionally correct but insufficiently strong to overcome seed variance on val. Assigning swp=20/swuv=10 (2× ratio) — amplifying the pressure signal to test the hypothesis more decisively. re_rand is the one test-set regression (+1.36%); monitoring whether it worsens at 2× will reveal whether there's a UV/pressure tradeoff specific to Reynolds variation.

**No velocity degradation observed** — UV MAE values in normal range. Code change is clean, backward-compatible (verified: sw_p==sw_uv==surf_weight falls back to baseline-equivalent loss).

---

## 2026-05-14 04:05 — Round 40 cont.: close #2738 (askeladd mlp_ratio=6 — +4.35% val LOSS, FFN-axis REFUTED → ALL 3 capacity axes dead); assign askeladd #2760 truncated cosine pivot

### PR #2738 askeladd mlp_ratio=6 + epochs=40 — NEGATIVE RESULT (FFN-axis capacity REFUTED)

- **Branch:** charliepai2g48h3-askeladd-mlpratio6-nlayers2-slicenum16-epochs40
- **Hypothesis:** FFN width bump (mlp_ratio 4→6) at n_layers=2 — orthogonal capacity axis to n_hidden
- **Result: SOLIDLY NEGATIVE** — val +4.35%, test +3.83%

| Split | val Δ | test Δ |
|---|---|---|
| single_in_dist | **−2.12%** ✓ | +0.87% |
| geom_camber_rc | **+8.31%** ✗ | +3.25% ✗ |
| geom_camber_cruise | **+9.87%** ✗ | +7.35% ✗ |
| re_rand | +2.88% ✗ | +6.30% ✗ |
| **avg** | **+4.35%** ✗ | **+3.83%** ✗ |

**OVERFIT-OOD SIGNATURE**: only single_in_dist improved (−2.12% val); both geom_camber splits regressed sharply (+8.31%, +9.87%); re_rand also regressed. Extra FFN width let the model fit in-distribution mesh patterns harder while losing geometric generalization. This is the OPPOSITE of what the capacity-rescues-OOD hypothesis predicted.

| Quantity | Value |
|---|---|
| n_params | 459,947 (~460K, +12% over baseline 411K) |
| Peak memory | 15.72 GB |
| Mean s/epoch | 38.6 s |
| Total wall-clock | 25.7 min |
| Best epoch | 40 (final, still descending) |

**CONCLUSION — ALL 3 CAPACITY-VIA-PARAM-COUNT AXES NOW REFUTED AT n_layers=2 / 30-min budget**:
1. n_hidden (channel width): #2685 +2.53%, #2737 +7.55%
2. mlp_ratio (FFN width): this PR +4.35% (also #2278 +5.4% at n_layers=3)
3. n_layers (depth): #2684 +12.7% (n_layers=1 catastrophic)

Param-count capacity hypothesis is DEAD. The bottleneck for camber-OOD is NOT model capacity.

**Round 40 cont. pivot — askeladd #2760 (truncated cosine T_max=60 + epochs=46)**:
- Code-change PR (add --t_max CLI arg)
- Hypothesis: best_epoch=46 STILL DESCENDING in baseline + every recent run → model is **prematurely terminated by scheduler** (lr→0 at epoch 46), NOT by capacity. Truncated cosine (T_max=60, epochs=46) leaves residual lr ≈ 1.3e-5 (13% of peak) at final epoch, enabling continued descent without compute cost.
- No compute cost (same epochs, same per-step). Cheap to refute.

**Still in flight**: frieren #2755 (per-channel surf weighting swp=15/swuv=10) — second code-change pivot targeting val_avg/mae_surf_p directly.

---

## 2026-05-14 03:55 — Round 40: close #2737 (frieren ISO-EPOCH capacity test — +7.55% LOSS, n_hidden axis REFUTED TWICE)

### PR #2737 frieren ISO-EPOCH capacity test (n_hidden=160 + slice_num=12 + epochs=46) — NEGATIVE RESULT

- **Branch:** charliepai2g48h3-frieren-nhidden160-slicenum12-nlayers2-epochs46
- **Hypothesis:** n_hidden=160 (capacity bump) + slice_num=12 (claw back per-epoch time) + epochs=46 (iso-epoch with baseline) — test if iso-epoch capacity rescues geom_camber OOD
- **Result: SOLIDLY NEGATIVE**

| Metric | This run | Baseline | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **37.9215** | 35.256 | **+7.55%** (worse) |
| test_avg/mae_surf_p | **31.2308** | 30.245 | **+3.26%** (worse) |
| epochs completed | 37 / 46 | 46 / 46 | **NOT iso-epoch** (hit timeout cap) |
| wall-clock | 30.3 min (capped) | n/a | — |
| best_epoch | 37 (final, still descending) | n/a | — |
| n_params | 560,515 | ~411K | +36% capacity |

**KEY DIAGNOSTIC**: slice_num=12 reduction did NOT claw back enough time. Predicted ~32s/epoch; actual ~43s/epoch steady-state (+34% over prediction). Mid-run contention spikes (epochs 24–28: 60-90s) burned an extra 3 min. Only 37/46 epochs completed.

**All splits regressed (val and test)**, including the OOD splits we hoped capacity would rescue. geom_camber_rc regressed +11.13% val, +1.87% test.

**Conclusion: capacity-along-n_hidden axis is REFUTED TWICE under 30-min compute budget**:
- #2685 (n_hidden=160 + epochs=40): +2.53% loss (capacity/epoch confound)
- #2737 (n_hidden=160 + slice_num=12 + epochs=46): +7.55% loss (even worse — slice reduction insufficient)

Baseline (n_hidden=128, slice_num=16, 46 epochs) Pareto-dominates n_hidden=160 variants within current compute envelope.

**Round 40 pivot — frieren #2755 (per-channel surface weighting)**:
- Code-change PR: split surf_weight into surf_weight_p (pressure, val metric target) and surf_weight_uv (velocity)
- Test: surf_weight_p=15, surf_weight_uv=10 (1.5× pressure emphasis, velocity at baseline)
- Hypothesis: directly optimize the validation metric (mae_surf_p) without adding compute cost
- Conservative, low-risk pivot — first code-change PR since #2685

**Remaining Round 39 capacity probe in flight: askeladd #2738 (mlp_ratio=6 + epochs=40 = FFN-axis capacity bump).** If this also loses, FFN+n_hidden capacity axes are both dead.

---

## 2026-05-14 03:10 — Round 39 followup: close 4 stale_wip Round 38 recreations; 3 retries + 1 pivot

**Closed stale_wip (4 Round 38 recreations stuck in rate-limit polling for 3h+):**
- #2680 alphonse lr=8e-5 → recreated as #2744 (3rd attempt of this hypothesis)
- #2681 edward slice_num=24+epochs=33 → recreated as #2745 (3rd attempt)
- #2682 nezuko mlp_ratio=2 → recreated as #2746 (3rd attempt)
- #2683 thorfinn lr=5e-5 → **PIVOTED** to lr=7e-5 (#2747) — lr=5e-5 has stale_wip'd 3 times (#2549, #2611, #2683). Collecting NEW LR axis data point (7e-5) is more valuable than further retries of the same failing pattern.

**Systemic note:** alphonse, edward, nezuko, thorfinn pods consistently get stuck in rate-limit polling cycles across rounds (now 3+ times each). The architectural pivots running on responsive pods (askeladd, frieren) are the critical experiments; the stuck pods are running lower-EV HP-axis tests anyway.

---

## 2026-05-14 02:25 — Round 39: close 2 bold architectural PRs (#2684 #2685) — MAJOR FINDINGS, 2 fresh capacity follow-ups assigned

**#2684 askeladd n_layers=1 + epochs=60 — CATASTROPHIC LOSS, MAJOR FINDING:**
- val_avg=39.740 vs 35.256 (**+12.7% LOSS**); test_avg=33.826 vs 30.245 (**+11.8% LOSS**)
- ALL 4 splits regressed 10-15%: single_in_dist +10.2%, geom_camber_rc +11.1%, geom_camber_cruise +25.2% (val), re_rand +11.2%
- best_epoch=60 (final), still descending at ~0.1/epoch
- **MAJOR FINDING: n_layers=2 is the FLOOR of the depth-down trajectory.** Single transformer block is insufficient to compose features. Capacity ceiling is uniform across in-dist and OOD (rules out OOD-specific explanations). Depth-down + epoch-up mechanism has bottomed out.

**#2685 frieren n_hidden=160 + epochs=40 — CAPACITY/EPOCH CONFOUND:**
- val_avg=36.148 vs 35.256 (**+2.53% LOSS**); test_avg=30.249 (+0.01% neutral)
- All val splits regressed (esp. geom_camber_cruise +7.71%); but test_single_in_dist IMPROVED -2.49%
- best_epoch=40 (final, still descending at 0.13/epoch)
- **KEY DIAGNOSTIC: Experiment confounded capacity↑ with epochs↓ (40 vs baseline 46) due to wall-clock cap.** Model was epoch-starved at the new capacity. Capacity hypothesis from #2638 NOT refuted — incompletely tested.
- Code change (added --n_hidden CLI flag) was clean and properly applied (n_params=560,843 = 1.55x baseline as predicted).

**Round 39 assignments (2 idle students → capacity hypothesis re-test along orthogonal axes):**

| Student | PR | Hypothesis | Type |
|---------|-----|------------|------|
| frieren | #2737 | **n_hidden=160 + slice_num=12 + epochs=46** (ISO-EPOCH capacity test; student's own suggestion #1; slice_num=12 claws back per-epoch time) | Width capacity axis (iso-epoch) |
| askeladd | #2738 | **mlp_ratio=6 + epochs=40** at n_layers=2 (FFN capacity bump; per #2684 suggestion #2) | FFN capacity axis |

**Strategy: probe the capacity hypothesis along TWO orthogonal axes simultaneously.** If both win → broad capacity story confirmed at n_layers=2. If only one wins → specific direction matters. If both fail → capacity isn't the right lever at this stack (per #2685 student's suggestion #5: may be information-limited not capacity-limited).

---

## 2026-05-14 00:35 — Round 38 followup: close 2 stale_wip Round 37 PRs (#2636 #2637); recreate batch_size axis

**Closed stale_wip (Round 37 pods stuck in rate-limit polling, no terminal results after ~2h idle):**
- #2636 fern bs=2 (recreated as #2695)
- #2637 tanjiro bs=8 (recreated as #2696)

**Re-assigned to keep batch_size axis test in flight:**
| Student | PR | Hypothesis |
|---------|-----|------------|
| fern | #2695 | **batch_size=2** retry at n_layers=2 stack (gradient-noise regularization, untested axis) |
| tanjiro | #2696 | **batch_size=8** retry at n_layers=2 stack (smoother gradients, untested axis) |

**Round 38 in-flight (all 8 students now active):** #2680 alphonse lr=8e-5, #2681 edward slice24, #2682 nezuko mlp2, #2683 thorfinn lr=5e-5 (3rd), #2684 askeladd n_layers=1, #2685 frieren n_hidden=160, #2695 fern bs=2 (r2), #2696 tanjiro bs=8 (r2).

---

## 2026-05-13 23:40 — Round 38: close 2 losers (#2638 #2639) + 4 stale_wip (Round 36); architectural pivot — assign 6 new

**Closed losers (terminal, with major diagnostic insight):**
- **#2638 frieren wd=3e-4 alone**: val_avg=35.760 (+1.43% loss), test_avg=30.718 (+1.56% loss). **KEY DIAGNOSTIC**: per-split breakdown reveals OOD is **split-dependent**. re_rand improved -3.97% val (Re-extrapolation likes regularization), single_in_dist improved -1.83% val, but **geom_camber_rc +5.92% val** and **geom_camber_cruise +7.31% val** worsened. **The two camber OOD splits are capacity-limited, not regularization-limited. The re_rand split is regularization-friendly.** This fundamentally changes our research direction at this stack — geom_camber OOD needs MORE capacity, not less.
- **#2639 askeladd wd=5e-5**: val_avg=35.984 (+2.07% within seed noise), test_avg=30.261 (+0.05% tied). All val splits slightly worse, mixed test. Conclusion: WD axis at n_layers=2 is at or near local optimum (WD=1e-4 baseline). Combined with #2638, **WD axis fully closed at this stack**.

**Closed stale_wip (4 Round 36 PRs stuck in rate-limit polling cycle, 1h+ idle):**
- #2608 alphonse lr=8e-5 (recreated as #2680)
- #2609 edward slice_num=24+epochs=33 (recreated as #2681)
- #2610 nezuko mlp_ratio=2 (recreated as #2682)
- #2611 thorfinn lr=5e-5 (recreated as #2683, 3rd attempt)

**Assigned 6 fresh PRs — Round 38 architectural pivot:**

| Student | PR | Hypothesis | Type |
|---------|-----|------------|------|
| alphonse | #2680 | **lr=8e-5** (LR low-side fine probe, retry of stale #2608) | Recreate |
| edward | #2681 | **slice_num=24+epochs=33** (wider partition retest, retry of stale #2609) | Recreate |
| nezuko | #2682 | **mlp_ratio=2** (narrower FFN retest, retry of stale #2610) | Recreate |
| thorfinn | #2683 | **lr=5e-5** (LR lower bound, 3rd retry of stale #2611) | Recreate (3rd attempt) |
| askeladd | #2684 | **n_layers=1 + epochs=60** (EXTREME depth-down, mechanism extension) | **NEW BOLD** |
| frieren | #2685 | **n_hidden=160 + epochs=40** (CAPACITY BUMP targeting geom_camber OOD per #2638 insight; code change to add --n_hidden flag) | **NEW BOLD + CODE CHANGE** |

**Strategy:** Round 38 finally pivots beyond pure hyperparameter axes. The 4 stale recreations preserve the LR/slice/mlp axis coverage. The 2 bold experiments (askeladd n_layers=1, frieren n_hidden=160) target the depth/width architectural levers — both informed by the #2638 capacity-limited geom_camber diagnostic. The trajectory has been depth-down (n_layers 6→5→4→3→2); n_layers=1 tests the extreme. n_hidden has been fixed at 128 throughout — capacity in the width direction is unexplored.

**Round 37 still in-flight (2 PRs):** #2636 fern bs=2, #2637 tanjiro bs=8 (batch_size axis).

---

## 2026-05-13 22:50 — Round 37: close 2 losers + 2 stale_wip; assign 4 new on batch_size and weight_decay axes

**Closed losers (terminal, above noise floor):**
- **#2600 frieren surf_weight=12**: val_avg=36.652 vs baseline 35.256 (**+3.96% LOSS**); test_avg=30.608 (**+1.20% LOSS**). All 4 splits regressed; mae_vol_p worsened ~7% val / ~5% test confirming surface-vs-volume tradeoff. **Surface-weight axis at n_layers=2 stack now closed: optimum bracketed tightly around sw=10.** Student suggested per-channel surface weighting (separate surf_weight_p from surf_weight_uv) as a future code-change PR direction. Student also noted geom_camber_rc (~48 baseline, ~51 here) is the dominant val_avg ceiling-setter.
- **#2601 askeladd compound lr=1.5e-4 + wd=3e-4**: val_avg=36.276 vs baseline 35.256 (**+2.89% LOSS**); test_avg=30.335 (+0.30% near-wash within seed noise). WD=3e-4 *partially* rescued OOD damage from high LR (test_avg -1.73% vs lr=1.5e-4 alone, all 3 OOD test splits improved), but ERASED the in-dist val gain (single_in_dist 34.16 -> 36.88, worse than baseline 36.48). **Two compound LR+WD arms have now failed at this stack. LR/WD sweep space is exhausted at n_layers=2.** Student's next-direction suggestions: physics-informed loss / aux surface head / baseline seed-averaging.

**Closed stale_wip (rate-limit-stuck pods, no terminal results after 2h+ idle):**
- **#2570 fern surf_weight=8** — pod stuck in rate-limit polling cycle.
- **#2571 tanjiro mlp_ratio=3** — same systemic issue; nezuko #2610 already in-flight at mlp_ratio=2 covering the lower-bound point.

**Assigned 4 fresh PRs on UNTESTED axes at the n_layers=2 stack:**

| Student | PR | Hypothesis | Axis |
|---------|-----|------------|------|
| fern | #2636 | **batch_size=2** at n_layers=2+slice_num=16+epochs=46 | batch_size (untested) |
| tanjiro | #2637 | **batch_size=8** at n_layers=2+slice_num=16+epochs=46 | batch_size (untested) |
| frieren | #2638 | **weight_decay=3e-4** alone at n_layers=2+slice_num=16+epochs=46 | WD axis (isolate from LR) |
| askeladd | #2639 | **weight_decay=5e-5** at n_layers=2+slice_num=16+epochs=46 | WD axis (lower bound) |

**Strategy:** Two clean orthogonal axes — batch_size {2, 4=baseline, 8} and weight_decay {5e-5, 1e-4=baseline, 3e-4}. Each is a single-flag change with concrete OOD-vs-in-dist hypothesis. **batch_size axis is genuinely untested at this stack** (every variant since Round 32 held bs=4). **weight_decay alone** isolates the OOD-regularization mechanism from the LR confounder (per askeladd #2601 explicit suggestion).

**Round 36 PRs still in-flight:** #2608 alphonse lr=8e-5, #2609 edward slice_num=24, #2610 nezuko mlp_ratio=2, #2611 thorfinn lr=5e-5 r2.

---

## 2026-05-13 22:15 — Round 36: Close 4 stale_wip PRs (alphonse #2543, edward #2545, nezuko #2547, thorfinn #2549); reassign with bigger swings

All 4 PRs (assigned ~19:30-19:42 UTC) had no commits or comments for 2.5+ hours — same systemic pod-polling rate-limit issue (user ID 20516801 shared across many bots). Closed all 4 and re-created with mostly-different hypotheses to maximize value of GPU time if/when pods recover.

**Closed:**
- PR #2543 (alphonse): lr=1.2e-4 × n_layers=2 → reassigned to lr=8e-5 (#2608, LR low-side fine probe; tests opposite direction from #2525)
- PR #2545 (edward): slice_num=20 × n_layers=2 → reassigned to slice_num=24+epochs=33 (#2609, BIGGER partition change, well above noise floor)
- PR #2547 (nezuko): LayerScale × n_layers=2 → reassigned to mlp_ratio=2 (#2610, narrower FFN bottom of axis; complements tanjiro #2571 mlp_ratio=3)
- PR #2549 (thorfinn): lr=5e-5 × n_layers=2 → re-created at lr=5e-5 (#2611, same hypothesis but fresh PR for stuck pod retry)

**Variance-aware reassignment strategy** (per PR #2523 insight):
- alphonse lr=8e-5: 20% LR change, borderline noise floor — use val+test consistency
- edward slice_num=24: +50% partition change, well above noise
- nezuko mlp_ratio=2: 50% FFN reduction, well above noise
- thorfinn lr=5e-5: 50% LR reduction, well above noise

8 students active. Diverse axis coverage at the new stack: LR (5e-5, 8e-5, 1.5e-4+wd compound), partition (16 baseline, 24 wider), FFN (mlp_ratio=2, 3), surf_weight (8, 12), regularization (wd=3e-4 compound).

---

## 2026-05-13 21:00 — Round 35: Close 2 review-ready losses; reassign frieren(sw=12) askeladd(compound LR+WD)

Both review-ready PRs lost; close and reassign with bigger swings to clear the seed-variance noise floor.

**PR #2523 — frieren epochs=50 — CLOSED (+2.30% val LOSS)**
- val_avg = 36.061 vs baseline 35.256 (+0.805); test_avg = 30.482 vs 30.245 (+0.237)
- best_epoch = 50/50 (within-run slope held at −0.1/epoch)
- **CRITICAL INSIGHT:** Within-run, the model DID improve from epoch 46→50 (Δ −0.36). BUT cross-run, this run's epoch-46 val_avg (36.42) was +1.16 worse than PR #2468's epoch 46 (35.26). **Seed variance (~±1.0 val units) >> marginal gain from 4 extra epochs (~−0.4).**
- **Strategic implication:** Single-seed comparisons of small tweaks are below the noise floor. Future experiments must either: (a) make BIGGER changes that produce signal above noise, or (b) use multiple seeds.
- Student's suggestion to set torch.manual_seed in train.py is a valuable code-change PR for the future.
- Metric artifacts: `models/model-nlayers2-slicenum16-epochs50-20260513-193847/metrics.jsonl`

**PR #2558 — askeladd n_head=2 — CLOSED (+3.16% val LOSS, +2.27% test LOSS)**
- val_avg = 36.369 vs baseline 35.256 (+1.113); test_avg = 30.932 vs 30.245 (+0.687)
- EVERY split regressed: single_in_dist (+3.73%), geom_camber_rc (+3.06%), geom_camber_cruise (+6.04%), re_rand (+1.35%)
- Param count: 380,583 (+5.39% vs baseline 361,131)
- Peak memory: 12.75 GB (−0.74 GB vs baseline — only mild positive)
- **Conclusion confirmed across stacks:** n_head=4 "bottleneck is slice routing not Q/K/V capacity" insight generalizes from (n_layers=3, slice_num=24) to (n_layers=2, slice_num=16). n_head=4 is robust optimum across explored architecture space.
- Metric artifacts: `models/model-charliepai2g48h3-askeladd-nhead2-nlayers2-slicenum16-epochs46-20260513-201608/metrics.jsonl`

**Reassignments (Round 35, target signal-above-noise):**
- PR #2600 (frieren): surf_weight=12 × n_layers=2 — 20% sw change, untested intermediate point (sw=10 winning, sw=15 neutral at deeper stack); targets OOD bottleneck via stronger surface loss
- PR #2601 (askeladd): COMPOUND lr=1.5e-4 + weight_decay=3e-4 × n_layers=2 — tests whether 3× WD rescues the OOD damage from lr=1.5e-4 while preserving the in-dist gain (−6.35% from #2525). 50% LR + 200% WD swing, well above noise floor.

---

## 2026-05-13 20:40 — Round 34: Close 2 stale_wip old-stack PRs (#2492 fern, #2493 tanjiro); reassign at new stack

Both PRs (created 17:53-17:56 UTC) had no commits or comments for 2.7+ hours. Same systemic rate-limit polling issue affecting pods. Both tested old-stack n_layers=3+slice_num=16+epochs=36 hypotheses against the OLD baseline 35.548 — now superseded by 35.256.

**Closed:**
- PR #2492 (fern): surf_weight=8 × n_layers=3 stale_wip → reassigned at new stack (#2570)
- PR #2493 (tanjiro): weight_decay=5e-5 × n_layers=3 stale_wip → reassigned with new hypothesis (#2571)

**Created (Round 34 fresh n_layers=2 assignments):**
- PR #2570 (fern): surf_weight=8 × n_layers=2+slice_num=16+epochs=46 — loss-weight at new stack; targets OOD bottleneck (lower sw may regularize toward better OOD generalization per PR #2525 insight)
- PR #2571 (tanjiro): mlp_ratio=3 × n_layers=2+slice_num=16+epochs=46 — FFN width intermediate point (mlp_ratio=3 untested anywhere; bridges the {2, 4, 6} sweep gap)

With both reassigned, all 8 student pods now have fresh n_layers=2-stack experiments testing diverse axes: LR sweep (3 in-flight), partition retest, LayerScale, n_head, surf_weight, mlp_ratio, epoch budget push.

---

## 2026-05-13 20:05 — PR #2525 — CLOSED (lr=1.5e-4 +3.30% LOSS at n_layers=2; in-dist vs OOD tradeoff confirmed)

**askeladd: lr=1.5e-4 on n_layers=2+slice_num=16+epochs=46**
- val_avg/mae_surf_p = 36.419 vs baseline 35.256 (**+1.163, +3.30% WORSE**)
- test_avg/mae_surf_p = 30.869 vs baseline 30.245 (+0.624, +2.06% WORSE)
- best_epoch = 43/46 (run timed out at 30 min, but extrapolation says still losing)
- Per-epoch: 42.2s (GPU contention spike, not LR-induced)

| Split | val mae_surf_p | val Δ vs baseline | test mae_surf_p | test Δ |
|---|---|---|---|---|
| single_in_dist | **34.161** | **−2.315 (−6.35%)** ✓ | **31.629** | **−1.406 (−4.25%)** ✓ |
| geom_camber_rc | 51.816 | +3.519 (+7.29%) | 45.840 | +1.507 (+3.40%) |
| geom_camber_cruise | 20.251 | +1.925 (+10.51%) | 16.886 | +1.390 (+8.97%) |
| re_rand | 39.449 | +1.526 (+4.02%) | 29.122 | +1.006 (+3.58%) |
| **avg** | **36.419** | **+1.163 (+3.30%)** | **30.869** | **+0.624 (+2.06%)** |

**KEY INSIGHT — in-dist vs OOD tradeoff confirmed:** Higher LR fixed the n_layers=2 in-dist regression (val_single_in_dist −6.35%, test −4.25%) — H3 confirmed cleanly. BUT it worsened all 3 OOD splits (+3.4% to +10.5%), netting a clear net loss because OOD dominates the avg (3 of 4 splits, larger magnitudes).

**Pattern (LR ceiling is partition-bound, not depth-bound):**
- lr=1.5e-4 at n_layers=3+slice_num=16: LOST
- lr=1.5e-4 at n_layers=3+slice_num=24: WON
- lr=1.5e-4 at n_layers=2+slice_num=16: LOST (this run)
The LR optimum at slice_num=16 is robust to depth changes. lr=1e-4 dominates at slice_num=16 regardless of n_layers.

**Metric artifacts:** `models/model-lr1p5e-4-nlayers2-slicenum16-epochs46-20260513-190526/metrics.jsonl`

**Reassigned:** askeladd → PR #2558 (n_head=2 at new stack — wider heads to target OOD bottleneck).

---

## 2026-05-13 19:30 — Round 32: Close 4 stale old-stack PRs; reassign at n_layers=2 stack

After PR #2468 merged (n_layers=2+epochs=46, val=35.256 NEW BEST), the partition axis is fully closed at slice_num=16 and the depth axis closed at n_layers=2. Several earlier-round PRs were still testing axes against the OLD n_layers=3 baseline (35.548) — their results would not be informative against the new 35.256 baseline. Additionally, 6 of 8 student pods were stuck on rate-limited polling cycles for ~2 hours and had not started training their previously assigned old-stack experiments.

**Closed (stale old-stack, no productive output expected):**
- PR #2471 (alphonse): lr=1.2e-4 × n_layers=3+slice_num=16 → reassigned at n_layers=2 (see #2543)
- PR #2478 (edward): n_layers=4+slice_num=16+epochs=27 → reassigned at n_layers=2 (see #2545)
- PR #2479 (nezuko): LayerScale × n_layers=3+slice_num=16 → reassigned at n_layers=2 (see #2547)
- PR #2450 (thorfinn): lr=5e-5 × n_layers=3+slice_num=16 → reassigned at n_layers=2 (see #2549)

**Created fresh n_layers=2-stack assignments:**
- PR #2543 (alphonse): lr=1.2e-4 × n_layers=2+slice_num=16+epochs=46 — LR fine probe
- PR #2545 (edward): slice_num=20 × n_layers=2+epochs=44 — partition retest at new depth (counter in-dist regression)
- PR #2547 (nezuko): LayerScale × n_layers=2+slice_num=16+epochs=46 — stabilization at new depth
- PR #2549 (thorfinn): lr=5e-5 × n_layers=2+slice_num=16+epochs=46 — LR lower probe (completes 4-point LR axis with #2549, #2468=baseline, #2543, #2525)

Combined with previously running #2523 (frieren epochs=50) and #2525 (askeladd lr=1.5e-4), this gives a full 8-experiment slate at the new n_layers=2 stack: 4-point LR sweep, partition retest, LayerScale, and epoch-budget push.

---

## 2026-05-13 18:50 — PR #2468 — MERGED NEW BASELINE (n_layers=2+epochs=46 val=35.256, −0.82%)

**frieren: n_layers=2+slice_num=16+epochs=46 (depth-down + epoch-up mechanism)**
- val_avg/mae_surf_p = **35.256** vs baseline 35.548 (**−0.292, −0.82% NEW BEST**)
- test_avg/mae_surf_p = **30.245** vs baseline 30.345 (−0.100, −0.33%)
- best_epoch = 46/46 STILL DESCENDING — slope ~−0.2/epoch over last 5 epochs, ~−0.1 at final epoch
- Per-epoch wall-clock: 35.08s (vs 50s at n_layers=3 — **30% cost reduction from depth drop**)
- Epochs: 46 (vs 36 at n_layers=3 — **10 extra cosine epochs from budget savings**)
- Total: 26.9 min (3.1 min margin under 30-min cap)
- Params: 361,131 (vs 515K, −30%)
- Peak memory: 13.49 GB

| Split | val mae_surf_p | test mae_surf_p | val Δ |
|---|---|---|---|
| single_in_dist | 36.476 | 33.035 | **+1.213** (REGRESSION) |
| geom_camber_rc | 48.297 | 44.333 | −0.808 |
| geom_camber_cruise | 18.326 | 15.496 | −1.066 |
| re_rand | 37.923 | 28.116 | −0.508 |
| **avg** | **35.256** | **30.245** | **−0.292** |

**KEY: OOD splits drove the win. single_in_dist regressed (+1.21) — mild capacity loss at n_layers=2 for in-dist data. Depth-budget axis not exhausted: 35s/epoch×50 epochs = 29.2 min, still within cap.**

**Metric artifacts:** `models/model-charliepai2g48h3-frieren-nlayers2-slicenum16-epochs46-20260513-175423/metrics.jsonl`

---

## 2026-05-13 18:50 — PR #2451 — CLOSED (slice_num=18 +4.3% LOSS; partition axis fully closed)

**askeladd: slice_num=18 on n_layers=3+epochs=36**
- val=37.063 vs baseline 35.548 (+1.515, **+4.3% LOSS**)
- test=30.983 vs 30.345 (+0.638)
- Partition sweep non-monotone confirmed: 32→24→20→18(bump)→16(minimum)→12(worse)
- slice_num=16 is a sharp, narrow local minimum — not floor of a broad valley

**PARTITION AXIS FULLY CLOSED.** Complete picture:
| slice_num | val | Δ vs 35.548 |
|---|---|---|
| 32 | 39.143 | +10.1% |
| 24 | 37.366 | +5.1% |
| 20 | 36.854 | +3.7% |
| 18 | 37.063 | +4.3% |
| **16** | **35.548** | **BASELINE** |
| 12 | 35.969 | +1.2% |

**Metric artifacts:** `models/model-slicenum18-nlayers3-20260513-173410/metrics.jsonl`

---

## 2026-05-13 18:00 — PR #2408 — CLOSED (stale; tanjiro slice_num=8 never produced output)

**tanjiro: slice_num=8 on n_layers=3+epochs=38**
- PR opened 15:28 UTC, no commits or PR comments in 2+ hours despite 30-min training cap
- Pod showed 0% GPU utilization with stale memory allocation — training crashed/stalled silently
- Hypothesis low-priority: partition axis fully closed at slice_num=16 (slice_num=14 +2.6%, slice_num=12 worse than 16)
- **Closed as stale; tanjiro reassigned to wd=5e-5 at slice_num=16 (PR #2493)**

---

## 2026-05-13 18:00 — PR #2409 — CLOSED (val=35.688 doesn't beat baseline 35.548; LR partition-dependence confirmed)

**fern: lr=1.5e-4 on n_layers=3+slice_num=12+epochs=36**
- val_avg/mae_surf_p = **35.688** vs current baseline 35.548 (+0.140, **LOSS on primary metric**)
- test_avg/mae_surf_p = 29.649 vs 30.345 (−2.29%, better on test — secondary)
- Beats OLD baseline (PR #2351 val=35.969) by −0.28% but misses NEW baseline (PR #2348 val=35.548)
- best_epoch=35/35 (timeout at 35 epochs), underfit signature persists at lr=1.5e-4
- 3/4 val splits neutral or better; single_in_dist regressed +0.43

| Split | baseline (35.548) | lr=1.5e-4@slice12 | Δ |
|---|---|---|---|
| single_in_dist | 35.263 | 36.737 | +1.474 |
| geom_camber_rc | 49.105 | 48.270 | −0.835 |
| geom_camber_cruise | 19.392 | 19.418 | +0.026 |
| re_rand | 38.431 | 38.326 | −0.105 |
| **avg** | **35.548** | **35.688** | **+0.140** |

**KEY FINDING: LR partition-dependence confirmed comprehensively.** lr=1.5e-4 wins at slice_num=12 vs lr=1e-4@slice12 (35.688 vs 35.969), and at slice_num=24 vs lr=1e-4@slice24 — but loses at slice_num=16. The partition-dependent LR optimum is now robust: smaller partitions shift LR optimum toward 1e-4 or below.

**Metric artifacts:** `models/model-lr-1p5e-4-nlayers3-slicenum12-clean-20260513-165911/metrics.jsonl`

---

## 2026-05-13 17:25 — PR #2447 — CLOSED (slice_num=14 loses; slice_num=16 confirmed local minimum)

**edward: slice_num=14 on n_layers=3+epochs=36**
- vs baseline (PR #2348 val=35.548): val=36.483 (**+2.6% LOSS**), test=30.798 (+1.5%)
- Uniform regression across all 4 val splits (+0.2 to +1.7 abs)
- Capacity canary: geom_camber_rc mae_vol_p=56.468 vs expected ~53.6 (+5.4%, no sharp collapse)
- best_epoch=35/36, params=513,273, 51.6s/epoch, 30.1 min

**PARTITION AXIS FULLY CLOSED.** Full neighborhood mapped:
| slice_num | val_avg/mae_surf_p | source |
|---|---|---|
| 24 | 37.366 | PR #2229 |
| 20 | 36.854 | PR #2375 |
| 18 | in flight | #2451 |
| **16** | **35.548** | **PR #2348 (BASELINE)** |
| 14 | 36.483 | this PR |
| 12 | 35.969 | PR #2351 |
| 8 | in flight | #2408 |

Both immediate neighbors (14 and 12) are worse. slice_num=16 is the **robust local minimum** — not a lucky point.

**Metric artifacts:** `models/model-slicenum14-nlayers3-20260513-163235/metrics.jsonl`

---

## 2026-05-13 17:25 — PR #2404 — CLOSED (n_head=1 statistically tied; head axis closed; PhysicsAttention bottleneck is slice mechanism)

**nezuko: n_head=1 on n_layers=3+slice_num=24+epochs=33 (3 independent seeds)**
- vs old baseline (PR #2229 val=37.366): val=37.644±0.48 (**+0.7% within 1σ, NOT significant**)
- test=31.943±0.53 (+1.8%, ~1σ, mild loss)
- Cannot beat current baseline 35.548
- n_head=1 has **+28% more params (659K vs 515K)** from Q/K/V scaling with dim_head² — yet no gain
- Seed variance estimate: **σ_val ≈ 0.48 units** — useful calibration for future single-run signals

**KEY FINDING: PhysicsAttention bottleneck is the slice mechanism, not Q/K/V capacity.** More Q/K/V params (via larger dim_head) don't help. The bottleneck is how information is routed across slice tokens.

**Head axis FULLY CLOSED across {n_head=1, 2, 4} at slice_num=24.**
- n_head=1: val=37.64 (3 seeds mean)
- n_head=2: val=37.63 (edward #2383)
- n_head=4: val=37.37 (BEST at slice_num=24)

**Metric artifacts:** `models/model-charliepai2g48h3-nezuko-nhead1-nlayers3-slicenum24-20260513-161125/metrics.jsonl`

---

## 2026-05-13 17:15 — PR #2431 — CLOSED (lr=1.5e-4 × slice_num=16 loses; LR optimum is partition-dependent)

**alphonse: lr=1.5e-4 × slice_num=16 on n_layers=3+epochs=36**
- vs baseline (PR #2348 val=35.548): val=38.125 (**+7.3% LOSS**), test=32.330 (+6.5%)
- Uniform regression: +4.6% to +11.4% across all 4 val splits — not a single-domain effect
- best_epoch=31 (timeout truncation — per-epoch 58.9s actual vs 49.8s predicted, GPU contention likely)
- Even extrapolating cosine tail: would need >2.5 val improvement in 5 more epochs — not plausible

**CRITICAL INSIGHT: LR optimum is partition-dependent:**
- slice_num=24: lr=1.5e-4 wins over 1e-4 (~−1.41%, #2353 signal)
- slice_num=16: lr=1.5e-4 loses +7.3% vs 1e-4 (this PR)
- Hypothesis: as slice_num decreases, per-slice representational budget shrinks → more conservative LR needed
- fern #2409 (lr=1.5e-4 at slice_num=12) will confirm if signal is strictly partition-decreasing

**Metric artifacts:** `models/model-lr-1p5e-4-slicenum16-nlayers3-20260513-163110/metrics.jsonl`

---

## 2026-05-13 17:05 — PR #2402 — CLOSED (lr=5e-5 loses badly; LR axis bracketed at slice_num=24)

**frieren: lr=5e-5 on n_layers=3+slice_num=24+epochs=33**
- vs old baseline (PR #2229 val=37.366): val=39.277 (**+5.1% LOSS**), test=33.643 (+7.2%)
- Cannot beat current baseline 35.548 at all — clear loser
- **Root cause:** cosine schedule decays to zero before model converges. Val still monotonically descending at epoch 33 (ep30→33: 39.98→39.69→39.33→39.28). Model undertrained, not over-regularized.
- single_in_dist worst hit (+11.3% val) — in-domain undertraining signature
- best_epoch=33/33 (cap), param count 514,263, 53.5s/epoch

**LR axis fully bracketed at slice_num=24:**
| lr | val_avg | Δ |
|---|---|---|
| 5e-5 (this) | 39.28 | +5.1% |
| 1e-4 (baseline) | 37.37 | — |
| 2e-4 (#2367) | 39.03 | +4.4% |

Trough at lr=1e-4 confirmed for fixed 33-epoch cosine at slice_num=24. thorfinn #2450 testing lr=5e-5 at slice_num=16 will show if this transfers.

**Student also killed two duplicate concurrent training runs** (from prior session re-entries). Well handled.

**Metric artifacts:** `models/model-charliepai2g48h3-frieren-lr-5e-5-nlayers3-slicenum24-20260513-160510/metrics.jsonl`

---

## 2026-05-13 16:45 — PR #2417 — CLOSED (n_head=2 loses at slice_num=12; n_head axis fully closed)

**thorfinn: n_head=2 on n_layers=3+slice_num=12+epochs=36**
- vs baseline (PR #2351 val=35.969): val=37.193 (**+3.40% LOSS**), test=31.269 (+3.32%)
- Capped at epoch 33/36 (30-min limit). Extrapolated to epoch 36: ~36.3 — still above baseline.
- single_in_dist takes the biggest hit (+2.5 val units); OOD splits more moderate (+0.5–1.0)

**n_head axis now CLOSED across all tested slice_num values:**
- slice_num=24: n_head=2 lost +0.71% (edward #2383)
- slice_num=12: n_head=2 lost +3.40% (thorfinn #2417)
- n_head=4 parallelism confirmed optimal at both partitions. Axis robust — no further testing needed.

**Metric artifacts:** `models/model-nhead2-nlayers3-slicenum12-20260513-155612/metrics.jsonl`

---

## 2026-05-13 16:45 — PR #2375 — CLOSED (slice_num=20 fills partition gap; loses vs current baseline)

**askeladd: slice_num=20 on n_layers=3+epochs=34**
- vs PR #2229 (old baseline 37.366): val=36.854 (**−1.37% WIN vs old**), test=30.938 (−1.38%)
- vs CURRENT baseline (PR #2348, val=35.548): **+3.67% LOSS** — cannot beat new baseline
- All 4/4 val splits improved vs old baseline. best_epoch=34/34 (still descending, budget-limited at ~55s/epoch)
- Partition sweep: 32→24→20 = 39.143→37.366→36.854 (monotone but gains diminishing)

**Partition floor near 16 confirmed:** 20 > 16 (36.854 vs 35.548), 12 > 16 (35.969 vs 35.548). Non-monotone with trough at 16.

**Metric artifacts:** `models/model-charliepai2g48h3-askeladd-slicenum20-nlayers3-20260513-155336/metrics.jsonl`

---

## 2026-05-13 16:30 — PR #2383 — CLOSED (n_head=2 loses at slice_num=24; n_head=4 confirmed optimal)

**edward: n_head=2 on n_layers=3+slice_num=24+epochs=33**
- Hypothesis: n_head=2 (head_dim=64) vs n_head=4 (head_dim=32) — richer per-head capacity vs parallelism
- vs PR #2229 baseline (val=37.366): val=37.633 (**+0.71% LOSS**), test=32.480 (+3.53%)
- Cannot beat current baseline 35.548 — this was a stale stack test, informative only
- n_head=2 has **more params** (544,209 vs 514K — +5.8%), yet still loses on 3/4 val splits
- single_in_dist val improved (−1.6%) but test worsened (+6.5%) — sign of overfitting without generalization
- 50.3s/epoch, 30.44 min, params 544,209 (QKV scales with dim_head²; halving n_head doubles dim_head)

**Conclusion:** Parallelism wins. Four heads function as soft mixture-of-specialists across slice tokens; 2 wider heads can't compensate for reduced routing diversity. n_head=4 confirmed optimal at n_layers=3+slice_num=24. **n_head axis closed at this partition.** Retest at slice_num=16 would be needed before claiming axis closed globally.

**Metric artifacts:** `models/model-charliepai2g48h3-edward-nhead2-nlayers3-slicenum24-20260513-150642/metrics.jsonl`

---

## 2026-05-13 16:00 — PR #2348 — MERGED (slice_num=16 beats slice_num=12; non-monotone partition sweep)

**alphonse: slice_num=16 on n_layers=3+epochs=36**
- Hypothesis: continue partition sweep — does the floor lie below slice_num=12?
- vs baseline (PR #2351 val=35.969): val=35.548 (**−1.17% WIN**), test=30.345 (+0.26% slightly worse)
- best_epoch: 35/36 (slight cosine tail flattening vs always-final at 12/24/32)
- 49.8s/epoch, 29.89 min total, params 513,471, peak 18.72 GB

| Split | val mae_surf_p Δ | test mae_surf_p Δ |
|---|---|---|
| single_in_dist | 36.308 → 35.263 (**−2.9%**) | 33.241 → 32.248 (−3.0%) |
| geom_camber_rc | 49.521 → 49.105 (−0.8%) | 43.631 → 44.663 (+2.4%) |
| geom_camber_cruise | 19.576 → 19.392 (−0.9%) | 15.969 → 16.188 (+1.4%) |
| re_rand | 38.470 → 38.431 (−0.1%) | 28.220 → 28.282 (+0.2%) |
| **avg** | **35.969 → 35.548 (−1.17%)** | **30.265 → 30.345 (+0.26%)** |

**Key finding: Partition sweep is NON-MONOTONE.** slice_num=16 beats slice_num=12, reversing the expectation that fewer slices = always better. At slice_num=12 the model loses capacity without gaining additional budget (per-epoch cost flattens: 16≈12≈50s). The sweet spot appears to be around slice_num=16. Tanjiro's slice_num=8 (PR #2408) will confirm if the floor has been found.

**Metric artifacts:** `models/model-charliepai2g48h3-alphonse-slicenum16-nlayers3-20260513-141223/metrics.jsonl`

**New baseline: val=35.548 (PR #2348). Next: alphonse assigned compound lr=1.5e-4+slice_num=16 (PR #2431).**

---

## 2026-05-13 15:35 — PR #2353 — CLOSED (lr=1.5e-4 won vs old baseline, can't beat new; strong LR signal)

**thorfinn: lr=1.5e-4 on n_layers=3+slice_num=24+epochs=33**
- vs OLD baseline (PR #2229 val=37.366): val=36.841 (**−1.41% WIN**), test=30.859 (−1.63%) 
- vs NEW baseline (PR #2351 val=35.969): val=36.841 (**+2.42% WORSE**), test=30.859 (+1.96%)
- best_epoch: 33/33 (final — still descending)
- 53.55s/epoch, 29.45 min total, params 514,263, peak 19.84 GB
- Training stable: no instability at lr=1.5e-4

| Split | val lr=1.5e-4 | val OLD baseline | Δ |
|---|---|---|---|
| single_in_dist | 36.655 | 38.082 | **−3.75%** |
| geom_camber_rc | 52.177 | 51.356 | +1.60% |
| geom_camber_cruise | 20.718 | 20.702 | +0.08% |
| re_rand | 37.813 | 39.325 | **−3.85%** |
| **avg** | **36.841** | **37.366** | **−1.41%** |

**KEY SIGNAL:** lr=1.5e-4 IS better than lr=1e-4 at slice_num=24. Gains concentrate on in-distribution (single_in_dist −3.75%) and re_rand (−3.85%) — the splits where optimization budget helps most. Geometry-OOD splits flat. This is a real LR signal at the OLD stack — validates the lr=1.5e-4 direction for fern's in-flight test at slice_num=12 (#2409).

**Why closed:** Baseline shifted to slice_num=12 stack while this was running. The slice_num=24 stack is ~1.4 val units behind the new baseline — even an LR win at slice_num=24 can't close that gap.

**Metric artifacts:** `models/model-lr-1p5e-4-nlayers3-slicenum24-20260513-145401/metrics.jsonl`

**Reassignment:** thorfinn → PR #2417 (n_head=2 at NEW slice_num=12 stack)

---

## 2026-05-13 15:28 — PR #2351 — MERGED ✓ NEW BASELINE val=35.969 (slice_num=12 floor probe wins!)

**tanjiro: slice_num=12 on n_layers=3+epochs=36 (reduced from 38 due to 51.89s/epoch)**
- val_avg: **35.969** vs baseline 37.366 → **−1.40 (−3.74%) IMPROVEMENT**
- test_avg: **30.265** vs baseline 31.371 → **−1.11 (−3.53%) IMPROVEMENT**
- All 4 splits improved on both val and test
- best_epoch: 36/36 (final — STILL DESCENDING)
- 50.33s/epoch, 36 epochs, 30.2 min total (capped)
- Peak memory: 18.16 GB, params: 513,075

| Split | val sl12 | val baseline | Δval | test sl12 | test baseline | Δtest |
|---|---|---|---|---|---|---|
| single_in_dist | 36.308 | 38.082 | −4.66% | 33.241 | 33.836 | −1.76% |
| geom_camber_rc | 49.521 | 51.356 | −3.57% | 43.631 | 45.411 | −3.92% |
| geom_camber_cruise | 19.576 | 20.702 | −5.44% | 15.969 | 16.874 | −5.36% |
| re_rand | 38.470 | 39.325 | −2.17% | 28.220 | 29.365 | −3.90% |
| **avg** | **35.969** | **37.366** | **−3.74%** | **30.265** | **31.371** | **−3.53%** |

**KEY FINDING: 'Capacity floor' hypothesis WRONG.** The capacity does NOT saturate at slice_num=24. Fewer slices → faster per-epoch → more cosine descent iterations within the 30-min cap. The mechanism is robustly holding from 32→24→12 (each step a large win). No capacity degradation signal on geom_camber_rc (vol mae_vol_p=54.2 — normal).

**Trajectory:** val 40.158 → 39.143 → 38.270 → 37.366 → **35.969** (total −10.5% from round start)

**Partition sweep ladder revised:**
- slice_num=32: val=39.143
- slice_num=24: val=37.366
- slice_num=20: askeladd #2375 (in flight — expected ~36.9, cannot win)
- slice_num=16: alphonse #2348 (in flight — expected ~36.5, cannot win)
- slice_num=12: **35.969 ← NEW BASELINE**
- slice_num=8: tanjiro #2408 (in flight, next probe)

**Metric artifacts:** `models/model-slicenum12-nlayers3-20260513-142632/metrics.jsonl`

**Reassignment:** tanjiro → PR #2408 (slice_num=8 next floor probe)

---

## 2026-05-13 15:28 — PR #2301 — CLOSED (stale dead-end; baseline shifted)

**fern: lr=1.5e-4 on n_layers=3+slice_num=32+epochs=30**
Never completed — closed due to baseline shift. Even winning at slice_num=32 gives ~38.x vs new baseline 35.969. No path to victory.

**Reassignment:** fern → PR #2409 (lr=1.5e-4 at NEW stack slice_num=12+epochs=36)

---

## 2026-05-13 15:15 — PR #2367 — CLOSED (lr=2e-4 loses; LR ceiling below 2e-4 confirmed)

**frieren: lr=2e-4 on n_layers=3+slice_num=24+epochs=33**
- val_avg: 39.028 vs baseline 37.366 → **+1.66 (+4.4%) WORSE**
- test_avg: 33.158 vs baseline 31.371 → **+1.79 (+5.7%) WORSE**
- All 4 splits regress; largest hit: geom_camber_cruise (+2.65 val, +2.18 test)
- Training stable: no divergence, monotone train loss decrease epochs 1-5
- Best epoch: 32/33 (slightly earlier than baseline's 33 — higher LR shifts optimum earlier but final value worse)

| Split | val lr=2e-4 | val baseline | Δval | test lr=2e-4 | test baseline | Δtest |
|---|---|---|---|---|---|---|
| single_in_dist | 39.079 | 38.082 | +0.997 | 34.796 | 33.836 | +0.960 |
| geom_camber_rc | 53.132 | 51.356 | +1.776 | 47.612 | 45.411 | +2.201 |
| geom_camber_cruise | 23.347 | 20.702 | +2.645 | 19.054 | 16.874 | +2.180 |
| re_rand | 40.556 | 39.325 | +1.231 | 31.171 | 29.365 | +1.806 |
| **avg** | **39.028** | **37.366** | **+1.66** | **33.158** | **31.371** | **+1.79** |

**LR ceiling confirmed below 2e-4.** With thorfinn #2353 (lr=1.5e-4) in flight and frieren reassigned to lr=5e-5, the LR axis will be fully characterized: 5e-5, 1e-4 (baseline), 1.5e-4, 2e-4.

**Mechanism finding:** Higher LR hurts geometry-OOD splits most (geom_camber_rc/cruise), suggesting the OOD generalization penalty from overstepping in weight space is concentrated on the hardest out-of-distribution cases.

**Metric artifacts:** `models/model-lr-2e-4-nlayers3-slicenum24-20260513-142444/metrics.jsonl`

**Reassignment:** frieren → PR #2402 (lr=5e-5 lower bracket)

---

## 2026-05-13 15:15 — PR #2279 — CLOSED (sw=3 loses; SW axis at n_layers=3 fully closed)

**nezuko: surf_weight=3 on n_layers=3+slice_num=32+epochs=27 (truncated by wall-clock to 25 epochs)**
- val_avg: 38.811 vs NEW baseline 37.366 → **+1.45 (+3.87%) WORSE**
- test_avg: 33.164 vs NEW baseline 31.371 → **+1.79 (+5.7%) WORSE**
- Vol mechanism CONFIRMED: mae_vol_p −10.9% val / −11.5% test across all splits
- Run truncated at epoch 25/27 due to GPU contention from a duplicate process (well-diagnosed by student)
- Even with contention, trajectory at epoch 25 (val=38.8) cannot close the ~1.4 gap in 2 epochs

| Split | val sw=3 | val baseline | Δval |
|---|---|---|---|
| single_in_dist | 40.002 | 38.082 | +5.0% |
| geom_camber_rc | 52.454 | 51.356 | +2.1% |
| geom_camber_cruise | 22.426 | 20.702 | +8.3% |
| re_rand | 40.360 | 39.325 | +2.6% |
| **avg** | **38.811** | **37.366** | **+3.87%** |

**SW axis at n_layers=3 fully closed:** sw=2 (+3.13%), sw=3 (+3.87%), sw=5 (+2.57%), sw=10 (optimal). Vol-gradient mechanism is consistently active but cannot overcome the baseline's partition+epochs advantage. vol→surface coupling is architecturally limited at compact depth.

**Metric artifacts:** `models/model-surf-weight-3-nlayers3-slicenum32-20260513-142331/metrics.jsonl`

**Reassignment:** nezuko → PR #2404 (n_head=1 attention-head axis probe)

---

## 2026-05-13 15:05 — PR #2350 — CLOSED (mlp_ratio=2 loses; mlp_ratio=4 confirmed optimal at compact stack)

**edward: mlp_ratio=2 on n_layers=3+slice_num=24+epochs=33**
- val_avg: 38.225 vs baseline 37.366 → **+0.86 (+2.3%) WORSE**
- test_avg: 32.615 vs baseline 31.371 → **+1.24 (+4.0%) WORSE**
- All 4 splits regress on both val and test. single_in_dist takes biggest hit (val +0.93, test +2.00).
- Parameter count: 366,039 (vs ~514K baseline) — 29% reduction from lighter FFN
- Best epoch: 33/33 (final), mildly undertrained even with lighter model

| Split | val mlp=2 | val baseline | Δ val | test mlp=2 | test baseline | Δ test |
|---|---|---|---|---|---|---|
| single_in_dist | 39.013 | 38.082 | +0.93 | 35.840 | 33.836 | +2.00 |
| geom_camber_rc | 52.205 | 51.356 | +0.85 | 46.166 | 45.411 | +0.75 |
| geom_camber_cruise | 22.082 | 20.702 | +1.38 | 17.967 | 16.874 | +1.09 |
| re_rand | 39.601 | 39.325 | +0.28 | 30.486 | 29.365 | +1.12 |
| **avg** | **38.225** | **37.366** | **+0.86** | **32.615** | **31.371** | **+1.24** |

**MLP axis closed at compact stack.** Combined with Round 23 PR #2278 (mlp_ratio=6 also lost), both directions away from mlp_ratio=4 regress. The FFN width at n_layers=3 is load-bearing — attention is the bottleneck, not FFN expansion, but cutting FFN below 4x hurts uniform representational capacity across all splits.

**Note:** PR body had template mix-up (slice_num=12 in body); student correctly inferred title intent and ran mlp_ratio=2. CLI flag `--mlp_ratio` added by student (previously hardcoded).

**Metric artifacts:** `models/model-charliepai2g48h3-edward-mlp-ratio-2-nlayers3-slicenum24-20260513-141120/metrics.jsonl`

**Reassignment:** edward → PR #2383 (n_head=2 at compact stack)

---

## 2026-05-13 ~17:20 — PR #2248 — CLOSED (sw=2 wins vs old baseline, loses vs new; vol-grad mechanism confirmed)

**askeladd: surf_weight=2 on n_layers=3+slice_num=32+epochs=27**
- val_avg: 38.534 vs OLD baseline (PR #2107: 39.143) → −1.56% (would have been a win)
- val_avg: 38.534 vs NEW baseline (PR #2229: 37.366) → **+3.13% WORSE**
- test_avg: 32.524 vs NEW baseline 31.371 → **+3.68% WORSE**

| Split | val sw=2 | val baseline (new) | Δ |
|---|---|---|---|
| single_in_dist | 39.299 | 38.082 | +3.20% |
| geom_camber_rc | 53.763 | 51.356 | +4.69% |
| geom_camber_cruise | 21.513 | 20.702 | +3.92% |
| re_rand | 39.562 | 39.325 | +0.60% |
| **avg** | **38.534** | **37.366** | **+3.13%** |

**Critical mechanism finding (per student diagnosis — excellent work):**
- val mae_vol_p: **−18.08%** (vs sw=10 same stack)
- test mae_vol_p: **−19.62%**
- Surface mae_p also improved vs same-stack sw=10 (−1.56%) confirming **vol→surface coupling pathway is OPEN**

But the surface improvement from vol-gradient mechanism (~1.5%) is LESS than the deficit vs the longer-budget baseline (~3%). On compact stack, the slice_num=24+epochs=33 mechanism dominates the vol-gradient mechanism.

**SW axis at n_layers=3 is now closed:** sw=2 (+3.13%), sw=5 (+2.57%), sw=10 (baseline). All low-sw runs lose vs current baseline. (sw=3 still in flight — nezuko #2279 at slice_num=32, expected similar.)

**Open future question:** Does the vol-gradient mechanism transfer better at slice_num=24 (richer attention)? Not worth investigating now — gain ceiling is small.

**Metric artifacts:** `models/model-charliepai2g48h3-askeladd-surf-weight-2-nlayers3-slicenum32-20260513-130915/metrics.jsonl`

**Reassignment:** askeladd → PR #2375 (slice_num=20+epochs=34) — fill partition sweep between current best 24 and active probe 16.

---

## 2026-05-13 ~17:00 — PR #2348 (alphonse slice_num=16) — PR body repaired, sent back to status:wip

PR body contained wrong content (tanjiro's slice_num=12 hypothesis) due to sub-agent template mix-up at creation time. Title and branch were correct (slice_num=16, alphonse). Student detected the inconsistency and asked for clarification. Body now corrected; label swapped status:review→status:wip; student confirmed to proceed with slice_num=16.

---

## 2026-05-13 ~16:50 — PR #2274 — CLOSED (WD=0 marginal loss vs new baseline)

**frieren: weight_decay=0 on n_layers=3+slice_num=32+epochs=30**
- val_avg: 38.186 vs new baseline 37.366 → **+2.20% WORSE**
- test_avg: 31.809 vs new baseline 31.371 → **+1.40% WORSE**

(Note: assignment was based on OLD baseline 38.270 at slice_num=32, where this run looked neutral on val and −2.0% on test. The new baseline 37.366 from PR #2229 supersedes mid-flight.)

| Split | val WD=0 | val baseline (new) | Δ |
|---|---|---|---|
| single_in_dist | 38.575 | 38.082 | +1.30% |
| geom_camber_rc | 52.555 | 51.356 | +2.34% |
| geom_camber_cruise | 21.122 | 20.702 | +2.03% |
| re_rand | 40.491 | 39.325 | +2.96% |
| **avg** | **38.186** | **37.366** | **+2.20%** |

**Key mechanistic finding (per student diagnosis):** The compact model **does NOT overfit** at WD=0 — train+val both descend through epoch 30 with no divergence. This means WD=1e-4 was not doing meaningful regularization work; the compact stack is capacity-insufficient to overfit at this budget. **WD axis is effectively flat at compact stack** — future experiments can use WD=0 if convenient.

Student also identified inter-run variance of ~1.7 val units between two duplicate WD=0 runs (one truncated at ep28: val=39.851; clean run: val=38.186). This is now the BASELINE.md noise floor benchmark.

**Test_single_in_dist −9.3% is intriguing** but driven by one split out of four; likely run-to-run variance.

**Metric artifacts:** `models/model-weight-decay-0-nlayers3-20260513-132321/metrics.jsonl`

**Reassignment:** frieren → PR #2367 (lr=2e-4 on n_layers=3+slice_num=24+epochs=33) — bracket LR axis upper bound at new baseline.

---

## 2026-05-13 ~16:20 — PRs #2229 MERGED / #2278 #2273 #2151 CLOSED

### PR #2229 (alphonse: slice_num=24 on n_layers=3+epochs=33) — **MERGED — NEW BASELINE val=37.366**

**Metric artifacts:** `models/model-slicenum24-nlayers3-epochs33-20260513-122356/metrics.jsonl`

| Split | val mae_surf_p | Δ vs baseline (38.270) | test mae_surf_p |
|---|---|---|---|
| single_in_dist | 38.082 | −6.2% | 33.836 |
| geom_camber_rc | 51.356 | −1.3% | 45.411 |
| geom_camber_cruise | 20.702 | −0.4% | 16.874 |
| re_rand | 39.325 | −1.1% | 29.365 |
| **avg** | **37.366** | **−2.36%** | **31.371** |

- **~53.7s/epoch** (down from 57s at slice_num=32) → 33 epochs in 29.5 min cap
- best_epoch=33/33 STILL DESCENDING — cosine at minimum, epoch-budget still limiting
- All four splits improve on both val and test; strongest gain: single_in_dist (−6.2% val)
- Epoch-budget mechanism (faster per-epoch → more epochs → better convergence) continues as the dominant lever

**Reassignments:**
- alphonse → slice_num=16 (continue partition sweep)
- tanjiro → slice_num=12 (floor probe)

---

### PR #2278 (edward: mlp_ratio=6 on n_layers=3) — CLOSED (+5.4% val worse)

val=40.330 vs baseline 38.270. Every split degraded uniformly. mlp_ratio=6 doesn't transfer to compact stack — at n_layers=3, attention is the bottleneck, not FFN capacity. Run also hit 30-min cap at epoch 25/28 due to GPU contention. Even with clean GPU, projected val ≥38.3 → still loses vs NEW baseline 37.366.

**Implication:** mlp_ratio=4 is optimal at n_layers=3. Edward reassigned to test mlp_ratio=2 (opposite direction).

---

### PR #2273 (tanjiro: linear warmup 2 ep + cosine on n_layers=3) — CLOSED (+1.66% val worse)

val=38.907 vs baseline 38.270. Only single_in_dist improved; three harder splits regressed. Root cause: warmup_epochs=2 compresses cosine T_max from 30→28, losing 2 epochs of late-stage descent (the most productive phase). Warmup axis closed for this stack.

---

### PR #2151 (thorfinn: slice_num=24 on n_layers=4) — CLOSED (superseded, no results)

Never completed; superseded by PR #2229 which tested the same slice_num=24 on the superior n_layers=3 stack. Thorfinn reassigned to lr=1.5e-4 on n_layers=3+slice_num=24.

---

## 2026-05-13 ~16:00 — PR #2245 — CLOSED (vol-gradient sw=5 does NOT transfer to compact stack)

**fern: surf_weight=5 on n_layers=3+slice_num=32+epochs=27**
- val_avg/mae_surf_p: **39.254** (vs baseline 38.270 → **+2.57% WORSE**)
- test_avg/mae_surf_p: **33.197** (vs baseline 32.470 → **+2.24% WORSE**)

| Split | val sw=5 | val baseline (PR #2228) | Δ |
|---|---|---|---|
| single_in_dist | 41.527 | 40.481 | +2.58% |
| geom_camber_rc | 53.713 | 52.042 | +3.21% |
| geom_camber_cruise | 21.489 | 20.785 | +3.39% |
| re_rand | 40.288 | 39.772 | +1.30% |
| **avg** | **39.254** | **38.270** | **+2.57%** |

**Analysis:** Fern compared against the OLD baseline PR #2107 (epochs=27, val=39.143) where it looked neutral (+0.28% val) with a bimodal pattern (cruise/re_rand improved, single/camber_rc worsened). Against the CURRENT baseline PR #2228 (epochs=30, val=38.270), ALL splits regress. The 3-epoch budget difference (27 vs 30) accounts for the discrepancy — the apparent "neutral" was misleading.

The vol-gradient mechanism confirmed active on n_layers=4 (PR #2214 saw mae_vol_p improve −7.9% to −14.9%) but **does NOT produce net surface improvements at n_layers=3**. At compact depth, the vol→surface feedback pathway is either weaker or the 3 fewer epochs vs the baseline compound to dominate. The mechanism is depth-sensitive.

**Implication:** sw axis at n_layers=3 is well-bracketed (askeladd sw=2, nezuko sw=3, fern sw=5, baseline sw=10) — all running or closed. The sw optimum at compact stack appears to be sw=10 (baseline). Fern reassigned to LR axis test: PR #2301 (lr=1.5e-4).

**Metric artifacts:** `models/model-surf-weight-5-nlayers3-slicenum32-20260513-120536/metrics.jsonl`

---

## 2026-05-13 ~15:25 — PRs #2214, #2185 — CLOSED (lose vs new baseline but confirm orthogonal mechanisms)

Both PRs were originally targeted at the OLD n_layers=4 baseline (40.158/34.904); both win comfortably against that old stack but lose against the new n_layers=3 baseline (38.270/32.470). The depth-reduction advantage from PR #2107/#2228 is the dominant gain.

### PR #2214 (nezuko sw=5 on n_layers=4): val=39.693 / test=33.712
- vs OLD #2172: **−1.16% val / −3.42% test** (would have been a win)
- vs NEW #2228: **+3.72% val / +3.83% test** (loses on current)
- **Vol-gradient mechanism CONFIRMED active on compact stack:** mae_vol_p improved on every split by −7.9% to −14.9%. The mechanism (sw=5 reallocates L1 gradient from surface to volume → richer volumetric features → better surface via shared encoder) transfers from deep to compact stacks.
- **Test (−3.42%) > val (−1.16%):** strong generalization signal.
- **Predictor:** fern's #2245 (sw=5 on n_layers=3) should win — mechanism transfers across depth.

### PR #2185 (edward mlp_ratio=6 on n_layers=4): val=41.496 / test=35.377
- vs OLD #2108: **−3.08% val / −4.12% test** (would have been a strong win)
- vs NEW #2228: **+8.43% val / +8.95% test** (loses on current)
- **mlp_ratio=6 wins uniformly:** every split improved on val and test. No regressions.
- **best_epoch=22/22 STILL DESCENDING** — capacity expansion preserves epoch-budget mechanism.
- **Per-epoch cost: +8% only** — budget impact manageable.
- **Predictor:** mlp_ratio=6 on n_layers=3 (edward's new PR #2278) likely wins — projected val ~37.1.

### Reassignments
- nezuko → PR #2279: sw=3 on n_layers=3 (fills sw curve between sw=2/sw=5)
- edward → PR #2278: mlp_ratio=6 on n_layers=3+epochs=28 (mechanism transfer to new best)

---

## 2026-05-13 ~15:00 — PR #2230: n_layers=2 + slice_num=32 + epochs=33 (frieren) — CLOSED (capacity floor at n_layers=3 identified)

- val=39.507 / test=33.419 vs new baseline (39.143 / 33.571) — val +0.94%, test −0.45%. The val direction is the loss verdict.
- **Per-split: single_in_dist +3.47%, geom_camber_rc +3.29% regressed** — the two capacity-demanding splits. cruise (−2.23%) and re_rand (−2.77%) improved.
- **Capacity floor identified at n_layers=3.** The 9-experiment streak of "depth reduction → fits more epochs → wins" terminates at n_layers=2.
- Run characteristics: 41.7s/epoch, 33 epochs in 26 min, 362K params (~30% capacity reduction from 515K).
- best_epoch=33/33 still descending — model never saturated, just hit capacity floor on hard splits.
- Closed; frieren reassigned to weight_decay=0 test on new baseline.

---

## 2026-05-13 ~14:55 — PR #2228: epochs=30 on n_layers=3+slice_num=32 (tanjiro) — MERGED (−2.2% val, −3.3% test) ← NEW BASELINE 38.270

- **Student:** charliepai2g48h3-tanjiro
- **Hypothesis:** Squeeze remaining ~4.4 min of headroom from PR #2107's run (27 epochs in 25.6 min). best_epoch=27 still descending — model still budget-limited at n_layers=3+slice_num=32.
- **Result:** val=38.270 / test=32.470 vs PR #2107 baseline (39.143 / 33.571) = **−2.2% val / −3.3% test**

| Split | val (#2107) | val (this) | Δ | test (#2107) | test (this) | Δ |
|---|---|---|---|---|---|---|
| single_in_dist | 40.405 | 40.481 | +0.2% | 35.977 | 36.568 | +1.6% |
| geom_camber_rc | 51.895 | 52.042 | +0.3% | 47.136 | 46.624 | −1.1% |
| geom_camber_cruise | 22.756 | 20.785 | **−8.7%** | 19.101 | 16.956 | **−11.2%** |
| re_rand | 41.517 | 39.772 | **−4.2%** | 32.070 | 29.734 | **−7.3%** |
| **avg** | **39.143** | **38.270** | **−2.2% ✓** | **33.571** | **32.470** | **−3.3% ✓** |

- **9th consecutive winning experiment with best_epoch=final** — epoch-budget mechanism still binding!
- Per-epoch: 58s, total 28.5 min (1.5 min margin to 30-min cap)
- The improvement is concentrated in the splits with headroom: cruise (lower-magnitude, easier to refine) and re_rand (cross-regime). The harder splits (single_in_dist, geom_camber_rc) stayed essentially flat — within noise.
- Per-epoch trajectory ep27→30: 38.92 → 39.04 → 38.42 → 38.27 (−0.65 absolute, ~1.7% relative descent in 3 extra epochs)
- 515K params, 20.96 GB peak memory
- **Metric artifacts:** `models/model-epochs-30-nlayers3-slicenum32-20260513-115035/metrics.jsonl`

---

## 2026-05-13 ~14:35 — PRs #2213, #2193 — CLOSED (n_head axis fully exhausted)

- **PR #2213 (fern n_head=2+epochs=24 on n_layers=4):** val=40.966 / test=35.255 — **+2.01% vs PR #2172** (n_head=4 same stack), **+4.66% vs new baseline 39.143**. Student's diagnostic insight: PR #2149's marginal n_head=2 win (−0.25% at epoch 21) was a "less-bad slow learner" artifact — at full 24-epoch budget, n_head=4 wins clearly. n_head=2 still descending at epoch 24 but never catches up.
- **PR #2193 (askeladd n_head=1 on n_layers=4):** val=43.980 / test=37.170 — **+2.97% vs PR #2149** (n_head=2), **+12.4% vs new baseline 39.143**. Critical insight: n_head=1 has **+30% MORE params** (864K vs 667K) yet still lost decisively. Attention **diversity** (multiple modes) is doing real work — not just per-head capacity.
- **Conclusion:** n_head axis fully bracketed: n_head=4 (best) > n_head=2 > n_head=1. The current best (n_layers=3+slice_num=32+epochs=27, n_head=4) is at the optimum of this axis. The n_head=2 plumbing PR #2149 was likely a false winner that we merged. No further n_head exploration needed.

---

## 2026-05-13 ~14:15 — PRs #2150, #2134 — CLOSED (dead ends, LR axis saturated)

- **PR #2150 (frieren lr=8e-5):** val=44.017 / test=37.507 — +12.4% above new baseline 39.143. LR axis fully exhausted: lr=8e-5 too slow to converge within epoch budget. Classic undertrained signature, best_epoch=21/21 still descending.
- **PR #2134 (alphonse lr=1.5e-4):** val=42.748 / test=36.962 — +9.2% above new baseline 39.143. Neutral vs old baseline 42.815 (−0.07%). Student correctly noted lr=1.5e-4 and slice_num=32 exploit the same epoch-budget slack — they don't compound.
- **Conclusion:** LR axis is saturated at lr=1e-4. No further LR exploration needed on compact stacks.

---

## 2026-05-13 ~14:00 — PR #2107: n_layers=3+slice_num=32+epochs=27 (tanjiro) — MERGED (−8.58% val, −9.02% test) ← NEW BASELINE 39.143

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/nlayers-3
- **Hypothesis:** Compound n_layers=3 (depth reduction frees per-epoch budget) + slice_num=32 (partition reduction) + aligned T_max=27. Both mechanisms are independent (each frees its own per-epoch budget slack), expected to be additive.
- **Result:** val=39.143 / test=33.571 vs #2172 baseline (40.158/34.904) = **−2.53%/−3.82%** vs #2108 (42.815/36.899) = **−8.58%/−9.02%**

| Split | val (#2172) | val (this) | Δ vs #2172 | test |
|---|---|---|---|---|
| single_in_dist | 40.610 | 40.405 | −0.5% | 35.977 |
| geom_camber_rc | 54.872 | 51.895 | **−5.4%** | 47.136 |
| geom_camber_cruise | 23.477 | 22.756 | **−3.1%** | 19.101 |
| re_rand | 41.675 | 41.517 | −0.4% | 32.070 |
| **avg** | **40.158** | **39.143** | **−2.53% ✓** | **33.571** |

- **All 4 splits improved** on both val and test.
- **best_epoch=27/27 STILL DESCENDING** — 25.6 min used, 4.4 min headroom remains. Model still budget-limited!
- **Per-epoch: ~57s** (down from ~74s at n_layers=4+slice_num=32 — both depth and slice contributed)
- **Mechanism additive confirmed:** n_layers=3 alone was −7.16% (vs old baseline), slice_num=32 was −7.6% (vs old baseline). Combined: −8.58% vs #2108 (additive, not overlapping).
- **n_params: 515,055** (~515K, −22% vs #2172's 667K)
- **Peak memory: 20.96 GB** (lowest in entire session)
- **Epoch-budget mechanism streak: 8+ consecutive experiments** with best_epoch=final
- **Metric artifacts:** `models/model-nlayers-3-slicenum32-tmax27-20260513-101844/metrics.jsonl`

---

## 2026-05-13 ~13:30 — PR #2172: epochs=24 on slice_num=32+n_layers=4 — MERGED (−6.21% val, −5.41% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/epochs-24-slicenum32-nlayers4
- **Hypothesis:** best_epoch=21 was STILL DESCENDING in #2108 — squeeze remaining epoch budget by extending 21→24 epochs (24×74s = 29.58 min, just inside 30-min cap).
- **Config note:** Ran with n_head=4 (default at time of run, BEFORE PR #2149 n_head=2 merge).
- **Result:** val=40.158 / test=34.904 vs old baseline #2108 (42.815/36.899) = **−6.21%/−5.41%**; vs baseline #2149 (42.709/36.784) = **−6.0%/−5.1%**

| Split | val (base #2108) | val (epochs=24) | Δ | test |
|---|---|---|---|---|
| single_in_dist | 44.963 | 40.610 | **−9.7%** | 38.553 |
| geom_camber_rc | 56.766 | 54.872 | **−3.3%** | 49.316 |
| geom_camber_cruise | 25.476 | 23.477 | **−7.8%** | 19.263 |
| re_rand | 44.053 | 41.675 | **−5.4%** | 32.483 |
| **avg** | **42.815** | **40.158** | **−6.21% ✓** | **34.904** |

- **All 4 splits improved** on both val and test.
- **best_epoch=24/24 STILL DESCENDING** — ep23→ep24 delta was −0.37 (not flattening). 25 epochs would require 1850s = 30.83 min — just over cap. Epoch axis is truly saturated under current config.
- **Per-epoch: ~73.9s mean** — consistent with prediction. Total training: 29.58 min.
- **n_params: 667,923** (n_head=4 at run time).
- **Clean within-run signal:** ep21=41.669 → ep24=40.158, delta −3.6% from the 3 extra epochs alone.
- **Epoch mechanism streak:** best_epoch=final holds for 7 consecutive experiments (n_layers=6→5→4, slice_num=64→48→32, epochs=21→24).
- **Compound needed:** n_head=2 + epochs=24 untested (n_head=2 from #2149 and epochs=24 from this PR not yet combined). Assigned to fern as PR #2213.
- **Metric artifacts:** `models/model-charliepai2g48h3-fern-epochs-24-slicenum32-nlayers4-20260513-102003/metrics.jsonl`

---

## 2026-05-13 ~13:30 — PR #2109: surf_weight=2 on n_layers=4+slice_num=48 — CLOSED (stale, never run)

- **Student:** charliepai2g48h3-nezuko
- **Result:** Stale WIP 4h+ with no activity, no comments, no metrics. Never started.
- **Why closed:** Stack superseded (old slice_num=48), baseline shifted from 46.344 to 40.158 (−12.5%). Even if sw=2 had improved the old stack, it would need retesting on the new stack.
- **sw=15 diagnostic (PR #2143):** sw=15 was neutral on n_layers=4, confirming sw=10 is near optimum in the high direction. sw=2 in low direction remains tested but stale — not worth carrying forward on old stack.
- **Reassigned nezuko:** PR #2214 surf_weight=5 on new compact stack (n_head=2 + slice_num=32 + n_layers=4 + epochs=24 config)

---

## 2026-05-13 ~13:15 — PR #2149: n_head=2 on n_layers=4+slice_num=32 — MERGED (−0.25% val, −0.31% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/nhead-2-slicenum32-nlayers4
- **Hypothesis:** Per-head capacity beats attention diversity on new compact stack — n_head=2 (head_dim=64) vs n_head=4 (head_dim=32) at n_layers=4 + slice_num=32.
- **Result:** val=42.709 / test=36.784 (vs baseline 42.815/36.899 = **−0.25%/−0.31%**)

| Split | val (baseline #2108) | val (n_head=2) | Δ | test |
|---|---|---|---|---|
| single_in_dist | 44.963 | 45.089 | +0.28% | 41.257 |
| geom_camber_rc | 56.766 | 57.248 | +0.85% | 50.023 |
| geom_camber_cruise | 25.476 | 25.495 | +0.07% | 21.336 |
| re_rand | 44.053 | 43.004 | **−2.38%** | 34.519 |
| **avg** | **42.815** | **42.709** | **−0.25% ✓** | **36.784** |

- **Mixed per-split picture:** re_rand val strongly improved (−2.38%), geom_camber_rc test improved (−2.06%), but single_in_dist slightly regressed on both val/test. Net positive.
- **Mechanism confirmed:** Volume MAE shifts match surface shifts across splits (signs correlated), supporting genuine attention re-wiring not noise.
- **Per-epoch wall-clock:** ~65s (down from 74s at n_head=4! head reduction freed compute despite larger head_dim). 
- **n_params: 708,875** (+6.3% vs baseline 667,923) — mild param confound.
- **best_epoch=21/21 STILL DESCENDING** — same pattern as all previous wins.
- **n_head axis insight:** At n_layers=6+slice_num=64 (deep/wide stack), n_head=2 LOST by +12.4%. On n_layers=4+slice_num=32 (compact stack), n_head=2 WINS. The per-head capacity vs diversity trade-off flips as the model becomes shallower and has coarser partitions.
- **Also included:** `--n_head` plumbed as CLI arg in train.py (model_config reads `n_head=cfg.n_head`).
- **Metric artifacts:** `models/model-nhead-2-slicenum32-nlayers4-20260513-101116/metrics.jsonl`
- **Reassigned askeladd:** PR #2193 n_head=1 on new stack (bracket the axis further)

---

## 2026-05-13 ~13:00 — PR #2143: surf_weight=15 on n_layers=4 — CLOSED (+7.77% val, +9.50% test vs current)

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/surf-weight-15-nlayers4
- **Hypothesis:** Shallow stacks need MORE surface signal (reverse of nezuko's sw=2 direction) — bracketing test of vol-gradient pathway.
- **Result:** val=46.140 / test=40.405. Essentially flat on val vs OLD baseline #2080 (−0.44%), slightly worse on test (+1.14%); +7.77% val and +9.50% test above NEW baseline #2108.

| Split | val (sw=10 base) | val (sw=15) | Δ val | test (base) | test (sw=15) | Δ test |
|---|---|---|---|---|---|---|
| single_in_dist | 49.979 | 49.003 | −1.95% | 44.746 | 43.970 | −1.73% |
| geom_camber_rc | 61.558 | 59.709 | **−3.00%** | 54.155 | 54.885 | +1.35% |
| geom_camber_cruise | 27.318 | 29.320 | +7.33% | 22.876 | 24.249 | **+6.00%** |
| re_rand | 46.518 | 46.527 | +0.02% | 38.025 | 38.516 | +1.29% |
| **avg** | **46.344** | **46.140** | **−0.44%** | **39.950** | **40.405** | **+1.14%** |

- **Volume MAE regressed +5.08%** across all splits (range: +1.02% rc to +12.74% cruise). Confirms the gradient-redirection mechanism is operating as theorized — but the surface-gain trade was net negative.
- **'Shallow stacks need more surface signal' hypothesis REFUTED.** sw=10 is at/near the optimum for n_layers=4. The n_layers=6 sw=5 win was a property of that specific depth, not a generic shallow-stack effect.
- **Key diagnostic:** geom_camber_cruise reacted most strongly to sw shift in BOTH directions (here sw=10→15 worsened it +7.3% val, +12.7% vol). Cruise behavior is volume-context-dominated — useful for designing volume-capacity experiments later.
- **The surf_weight axis on n_layers=4 is being bracketed:** sw=2 (nezuko #2109, in flight), sw=10 (baseline), sw=15 (this PR, neutral). If nezuko also doesn't beat baseline, the entire axis is saturated for this stack and we should pivot.
- **Why closed not sent back:** sw=15 + slice_num=32 compound has very low EV (sw=15 wasn't a win on its own).
- **Reassigned edward:** PR #2185 mlp_ratio=6 on new compact stack (unexplored midpoint of MLP capacity axis: 4=current, 6=untested, 8=lost).

---

## 2026-05-13 ~12:30 — PR #2134: lr=1.5e-4 on n_layers=4 stack — SENT BACK (compound test)

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/lr-1.5e-4-nlayers4
- **Hypothesis:** lr=1.5e-4 (vs default 1e-4) with the lr=cfg.lr bug fix now in place.
- **Result:** val=45.305 / test=39.115 — beat OLD baseline (#2080 = 46.344/39.950) by −2.24%/−2.09%, but does NOT beat NEW baseline (#2108 = 42.815/36.899; +5.81% val above).

| Split | val (baseline #2080) | val (lr=1.5e-4) | Δ | test (baseline) | test (lr=1.5e-4) | Δ |
|---|---|---|---|---|---|---|
| single_in_dist | 49.979 | 48.002 | −3.95% | 44.746 | 41.894 | −6.37% |
| geom_camber_rc | 61.558 | 57.678 | −6.30% | 54.155 | 51.837 | −4.28% |
| geom_camber_cruise | 27.318 | 29.108 | +6.55% | 22.876 | 24.497 | +7.09% |
| re_rand | 46.518 | 46.431 | −0.19% | 38.025 | 38.231 | +0.54% |
| **avg** | **46.344** | **45.305** | **−2.24%** | **39.950** | **39.115** | **−2.09%** |

- **LR axis confirmed:** Clear win over lr=1e-4 baseline (3/4 val splits improve; only geom_camber_cruise regressed slightly). best_epoch=17 still descending, model budget-limited.
- **Why not merge:** Sits above the new baseline (slice_num=32 merged 30 min earlier).
- **Action:** Sent back with instructions to test **lr=1.5e-4 + slice_num=32 + n_layers=4** compound. Mechanisms are orthogonal (LR axis vs epoch-budget axis), so compounding is very likely.
- **Per-epoch LR was logged** for the first time — `metrics.jsonl` now includes `lr` per epoch. Useful for future LR variation experiments.

---

## 2026-05-13 ~12:30 — PR #2107: n_layers=3 + T_max=22 — SENT BACK (compound test)

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/nlayers-3
- **Hypothesis:** n_layers=3 (continuing depth sweep step 3 after n_layers=4 win in PR #2080).
- **Result:** val=43.026 / test=37.446 — beat OLD baseline (#2080 = 46.344/39.950) by −7.16%/−6.27%; sits +0.49% val / +1.48% test above NEW baseline (#2108 = 42.815/36.899). Extraordinarily close.

| Split | val (baseline #2080) | val (n_layers=3) | Δ | test (baseline) | test (n_layers=3) | Δ |
|---|---|---|---|---|---|---|
| single_in_dist | 49.979 | 45.492 | −8.98% | 44.746 | 41.768 | −6.66% |
| geom_camber_rc | 61.558 | 56.358 | −8.45% | 54.155 | 49.829 | −7.99% |
| geom_camber_cruise | 27.318 | 26.307 | −3.70% | 22.876 | 21.784 | −4.78% |
| re_rand | 46.518 | 43.946 | −5.53% | 38.025 | 36.402 | −4.27% |
| **avg** | **46.344** | **43.026** | **−7.16%** | **39.950** | **37.446** | **−6.27%** |

- **Diminishing-returns hypothesis FALSIFIED.** The depth sweep gain *increased* at n_layers=4→3 (−7.16%) compared to n_layers=5→4 (−1.07%). The n_layers=5→4 dip was a local quirk, not a saturation signal.
- **best_epoch=22 STILL DESCENDING** (same pattern as all previous wins). Per-epoch time 66s (vs 75s predicted).
- **n_params: 516,639** (vs baseline 670,035, −23%).
- **Why not merge:** Sits +0.5% above the new baseline (slice_num=32 merged 30 min earlier; same epoch-budget mechanism on orthogonal axis).
- **Action:** Sent back with instructions to test **n_layers=3 + slice_num=32 + T_max=27** compound. Both axes are orthogonal — depth vs slice partitioning — and almost certainly compound. Expected to give a meaningful new best.

---

## 2026-05-13 ~12:30 — PR #2062: n_layers=5 + slice_num=48 compound verification — CLOSED (superseded)

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/slice-num-48-nlayers5
- **Hypothesis:** Verify n_layers=5 + slice_num=48 compound from earlier merge sequence.
- **Result:** Stale WIP for 4+ hours, no comments, no committed metrics. Never started or never completed.
- **Why closed:** Superseded twice in the merge sequence:
  - PR #2080 (n_layers=4 + T_max=17): val=46.344 → baseline
  - PR #2108 (slice_num=32 + n_layers=4): val=42.815 → baseline
  - Even if PR #2062 had completed cleanly, n_layers=5 + slice_num=48 could not beat 42.815.
- **Reassigned fern:** PR #2172 epochs=24 + T_max=24 + slice_num=32 (squeeze remaining epoch-budget headroom on new compact stack)

---

## 2026-05-13 ~12:00 — PR #2108: slice_num=32 + n_layers=4 — MERGED (−7.6% val, −7.6% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/slice-num-32
- **Hypothesis:** slice_num=48→32 → ~21% per-epoch speedup on n_layers=4 → 21 epochs in 30-min budget → T_max=21 alignment. Continued "epoch-count is binding" mechanism.
- **Result:** val=42.815 / test=36.899 (best_epoch=21/21 — STILL DESCENDING AT FINAL EPOCH)

| Split | val (baseline #2080, 46.344) | val (slice_num=32) | Δ | test |
|---|---|---|---|---|
| single_in_dist | 49.979 | 44.963 | **−10.0%** | 40.717 |
| geom_camber_rc | 61.558 | 56.766 | **−7.8%** | 51.074 |
| geom_camber_cruise | 27.318 | 25.476 | **−6.7%** | 21.158 |
| re_rand | 46.518 | 44.053 | **−5.3%** | 34.646 |
| **avg** | **46.344** | **42.815** | **−7.6% ✓** | **36.899** |

- **Best epoch=21 STILL DESCENDING** — val curve descended from 43.08 (ep20) to 42.81 (ep21). Headroom remains.
- **Volume MAE also −7.6%** (val_avg vol: 51.107 → 47.223) — clean improvement on both surface and volume reconstruction.
- **Timing:** 74s/epoch (21% speedup vs 94s at slice_num=48) — prediction was ~80s/epoch; actual was better.
- **n_params:** 667,923 (−0.3% vs 670,035 baseline) — slice_num doesn't meaningfully change param count.
- **All 8 per-split val/test metrics improved** — clean, monotone win.
- **Also included:** slice_num plumbed as CLI arg (`--slice_num`) — `train.py` now accepts it in Config dataclass; model_config reads `slice_num=cfg.slice_num`.
- **Student's prediction confirmed:** slice_num=48 over-specifies partition granularity; 32 slices still capture CFD subregions (boundary layer / wake / freestream).
- **Metric artifacts:** `models/model-charliepai2g48h3-thorfinn-slice-num-32-nlayers4-20260513-091144/metrics.jsonl`
- **Reassigned thorfinn:** PR #2151 slice_num=24 on new stack (student's own suggestion!)

---

## 2026-05-13 ~11:30 — PR #2048: surf_weight=5 on n_layers=5 stack — CLOSED (+3.16% vs new baseline)

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/surf-weight-5-nlayers5
- **Hypothesis:** surf_weight=5 reduces surface loss weight, reallocating gradient toward volume; volume → surface via geometric context. Previously won on n_layers=6 (PR #1956, −3.33%).
- **Result:** val=47.808 / test=41.148 (vs current baseline 46.344 = +3.16%; vs n_layers=5 baseline 49.780 = −4.0%)

| Split | val (n_layers=5 baseline) | val (sw=5 on n_layers=5) | Δ (vs n_layers=5) | test |
|---|---|---|---|---|
| single_in_dist | 52.199 | 50.101 | −4.0% | 45.163 |
| geom_camber_rc | 64.004 | 64.001 | 0.0% | 55.836 |
| geom_camber_cruise | 27.965 | 24.684 | −11.7% | 20.617 |
| re_rand | 54.952 | 52.445 | −4.6% | 43.177 |
| **avg** | **49.780** | **47.808** | **−4.0% ✓ vs n_layers=5** | **41.148** |

- **Why closed:** Beat n_layers=5 baseline by −4.0% but could not beat the CURRENT baseline (n_layers=4+T_max=17 = 46.344). Stack mismatch — student ran on n_layers=5+T_max=14 (an intermediate, now-superseded stack).
- **Key diagnostic from edward:** Volume MAE DID improve (−8% avg), confirming the volume-gradient mechanism is active. BUT vol→surface pathway didn't translate to surface gains enough. Student noted: "mechanism seems real but the epoch/depth context matters."
- **Surface-volume pathway is stack-depth dependent:** sw=5 won on n_layers=6 (−3.33%), won vs n_layers=5 baseline (−4.0%), but sits above the current n_layers=4 baseline. Shallower models may need MORE surface weight, not less.
- **Reassigned edward:** PR #2143 surf_weight=15 on new n_layers=4 stack (opposite direction to test the bracket)

---

## 2026-05-13 ~11:30 — PR #2038: n_head=2 on old n_layers=6+slice_num=64 stack — CLOSED (+12.4% vs current)

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/nhead-2
- **Hypothesis:** n_head=2 → head_dim=64 (vs current 32) — larger per-head capacity for physics-aware slice partitions.
- **Result:** val=52.174 / test=45.394 (vs old baseline 51.040 = +2.22%; vs current 46.344 = +12.4%)
- **Why closed:** PR was posted as draft/WIP, never properly readied. Student forgot the submission skill. Result beat old baseline slightly (+2.22%) but was far above the current baseline (which uses a much better stack). Direction signal: negative — attention diversity mattered more than per-head capacity on the deep stack with n_head=4 × head_dim=32.
- **Note:** Old stack (n_layers=6 + slice_num=64) had 6 layers processing 64 partitions — more diverse attention modes may be load-bearing at that depth. Retesting on new compact stack (n_layers=4 + slice_num=32) where each head has fewer partitions to attend over (PR #2149 askeladd).
- **Reassigned askeladd:** PR #2149 n_head=2 on NEW n_layers=4+slice_num=32 stack

---

## 2026-05-13 ~11:30 — PR #2006: lr=8e-5 on old n_layers=6 stack — CLOSED (never started)

- **Student:** charliepai2g48h3-frieren
- **Branch:** charliepai2g48h3-frieren/lr-8e-5
- **Hypothesis:** lr=8e-5 (−20% below default 1e-4) tests the lower LR bracket on Lion optimizer.
- **Result:** No training was ever run. Student never started the experiment.
- **Context:** Assignment was created when lr=cfg.lr bug was still present AND the stack was n_layers=6. Both issues now resolved (bug fixed in PR #2080; stack updated to n_layers=4+slice_num=32).
- **Reassigned frieren:** PR #2150 lr=8e-5 on NEW n_layers=4+slice_num=32 stack — now a proper clean LR test

---

## 2026-05-13 ~11:30 — PR #2043: DropPath rate=0.1 — CLOSED (+25.2% val, +25.2% test vs current)

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/droppath-0.1
- **Hypothesis:** Path-level stochastic depth (drop_path_rate=0.1, linearly increasing per-layer) regularizes against overfit, especially on OOD splits.
- **Result:** val=58.024 / test=50.018 (vs current 46.344/39.950: +25.2%/+25.2%)

| Split | val (baseline #1956) | val (DropPath) | Δ | test |
|---|---|---|---|---|
| single_in_dist | 56.933 | 66.154 | +16.20% | 57.803 |
| geom_camber_rc | 64.886 | 72.857 | +12.28% | 64.204 |
| geom_camber_cruise | 31.056 | 37.612 | +21.11% | 31.099 |
| re_rand | 51.287 | 55.473 | +8.16% | 46.967 |
| **avg** | **51.040** | **58.024** | **+13.68%** | **50.018** |

- **Why it failed:** Run on OLD n_layers=6 + T_max=12 stack. Model was STILL DESCENDING at epoch 12 — underfitting regime. DropPath adds regularization noise to optimization → makes underfitting worse.
- **OOD suffered MORE, not less** (cruise +21%, rc +12%): contradicts the hypothesis. Path-level reg only helps OOD AFTER in-distribution loss plateaus.
- **Literature context:** DropPath used in DeiT/Swin/ConvNeXt with 100-300 epoch runs. Stochastic depth's ensemble effect emerges post-convergence.
- **Conclusion:** Wrong tool for budget-constrained training. Even on new n_layers=4 stack (17 epochs), best_epoch=17 was still descending — underfitting regime persists. Confirmed dead end on this budget.
- **Follow-up ideas saved:** drop_path_rate=0.025–0.05, DropPath only on later layers, longer epoch budgets.
- **Metric artifacts:** `models/model-droppath-0.1-20260513-082354/metrics.jsonl`
- **Reassigned alphonse:** PR #2134 lr=1.5e-4 on new compound stack (clean LR test with bug now fixed)

---

## 2026-05-13 ~11:10 — PR #2080: n_layers=4 + T_max=17 — MERGED (−1.07% val, −2.17% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/nlayers-4
- **Hypothesis:** n_layers=4 → ~94s/epoch → 17 epochs in 30-min budget → T_max=17 alignment. Same "epoch-count is binding" mechanism as #1995 and #1996.
- **Result:** val=46.344 / test=39.950 (best_epoch=17/17 — STILL DESCENDING AT FINAL EPOCH)

| Split | val (PR #1996, 46.847) | val (n_layers=4, 46.344) | Δ | test |
|---|---|---|---|---|
| single_in_dist | 50.491 | 49.979 | −1.0% | 44.746 |
| geom_camber_rc | 60.364 | 61.558 | +2.0% (noise) | 54.155 |
| geom_camber_cruise | 29.835 | 27.318 | **−8.4%** | 22.876 |
| re_rand | 46.699 | 46.518 | −0.4% | 38.025 |
| **avg** | **46.847** | **46.344** | **−1.07% ✓** | **39.950** |

- **⚠ Best epoch=17 STILL DESCENDING** — n_layers=4 was not saturated; cosine schedule expired before convergence. Strongly hints n_layers=3 can extract more.
- **n_params:** 670,035 (−31.4% vs 976,827 baseline) — huge capacity reduction with no accuracy loss
- **Also included in merge:** lr=cfg.lr bug fix — Lion constructor now uses `cfg.lr` instead of hardcoded 1e-4. All prior LR variation experiments were silently at 1e-4.
- **cruise wins biggest** (val −8.4%, test −5.3%) — the easiest split with smallest MAEs benefits most from extra cosine-tail epochs
- **Metric artifacts:** `models/model-nlayers-4-tmax17-20260513-082121/metrics.jsonl`, `metrics.yaml`
- **Reassigned tanjiro:** PR #2107 n_layers=3 + T_max=22

---

## 2026-05-13 ~11:10 — PR #2040: grad-clip max_norm=1.0 — CLOSED (+12.86% val vs old baseline, +13.01% vs current)

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/grad-clip-1
- **Hypothesis:** max_norm=1.0 stabilizes Lion's EMA by bounding gradient magnitude.
- **Result:** val=52.874 / test=45.659 (vs old baseline 51.040 = +3.6%; vs current 46.344 = +14.1%)

| Metric | Old baseline | grad-clip | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 51.040 | 52.874 | +3.6% |
| test_avg/mae_surf_p | 44.390 | 45.659 | +2.9% |

- **Key diagnostic:** Gradient norms are 20–140 on this stack (not <1). Clip fired on 100% of batches (4500/4500). max_norm=1.0 is 25× too small for this stack — it scaled every gradient by 0.02–0.05, effectively shrinking lr by 20–50× each batch.
- **Mechanism analysis:** Lion's sign-based update discards gradient magnitude by design. max_norm=1.0 provides no benefit and actively harms training by over-suppressing the EMA direction signal.
- **Conclusion:** max_norm=1.0 is a confirmed dead end. The right range would be ~50–80 (where only epoch-1 spikes of 100+ would be clipped), but given Lion's sign-update architecture already handles magnitude, the whole axis is deprioritized.
- **Metric artifacts:** `models/model-grad-clip-1-20260513-075638/metrics.jsonl`
- **Reassigned thorfinn:** PR #2108 slice_num=32 on n_layers=4 stack

---

## 2026-05-13 ~11:10 — PR #2029: surf_weight=2 (old n_layers=6 stack) — CLOSED (+6.32% vs current, direction confirmed)

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/surf-weight-2-compound
- **Hypothesis:** sw=2 continues the gradient-reallocation sweep (sw=10→5→2).
- **Result:** val=49.267 / test=42.964 (vs old baseline 51.040 = −3.48%; vs CURRENT 46.344 = +6.32%)

| Split | val (sw=5, old stack) | val (sw=2, old stack) | Δ | test |
|---|---|---|---|---|
| single_in_dist | 56.933 | 53.788 | **−5.52%** | 49.806 |
| geom_camber_rc | 64.886 | 63.318 | −2.42% | 56.308 |
| geom_camber_cruise | 31.056 | 31.507 | +1.45% | 26.089 |
| re_rand | 51.287 | 48.454 | **−5.52%** | 39.652 |
| **avg** | **51.040** | **49.267** | **−3.48% ✓** | **42.964** |

- **Volume MAE improved all 4 splits** (−6% to −15%): mechanism fully confirmed at sw=2 level.
- **Re_rand was biggest winner** (val −5.52%, test −6.17%) — contradicts prior concern about Reynolds holdout; volume-gradient benefit is not geometry-specific.
- **geom_camber_cruise mild regression** (+1.45% val) — easiest split; possibly at a surface-supervision floor for this regime.
- **Closes without merge because:** run was on OLD n_layers=6+T_max=12 stack; val=49.267 > current baseline 46.344.
- **Direction: STRONG** — sw=2 on the new n_layers=4 compound stack is the immediate next test (nezuko PR #2109).
- **Metric artifacts:** `models/model-surf-weight-2-compound-20260513-075550/metrics.jsonl`
- **Reassigned nezuko:** PR #2109 surf_weight=2 on n_layers=4 + slice_num=48 stack

---

## 2026-05-13 ~11:00 — PR #2007: mlp_ratio=2 — CLOSED (+9.95% val, +10.28% test)

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/mlp-ratio-2
- **Hypothesis:** Reduce mlp_ratio 4→2 (n_hidden stays 128, GeGLU halves fc2 width). Predicted: 30% param reduction → faster epochs → more epochs in 30-min budget → better convergence via epoch-count mechanism.
- **Result:** val=51.508 / test=45.034 (best_epoch=14, n_layers=6 stack, T_max=14, sw=10)

| Split | val (current baseline 46.847) | val (mlp_ratio=2) | Δ | test |
|---|---|---|---|---|
| single_in_dist | 50.491 | 55.152 | +9.2% | 49.255 |
| geom_camber_rc | 60.364 | 64.952 | +7.6% | 59.137 |
| geom_camber_cruise | 29.835 | 32.793 | +9.9% | 27.218 |
| re_rand | 46.699 | 53.136 | +13.8% | 44.526 |
| **avg** | **46.847** | **51.508** | **+9.95%** | **45.034** |

- **Why it lost cleanly:** Student ran on old n_layers=6 stack with T_max=14 (not the current n_layers=5 + slice_num=48). The apparent improvement vs old baseline 52.798 was mostly T_max alignment (2 extra epochs), not mlp_ratio. Student correctly flagged this in writeup. vs current baseline 46.847 it is +9.95% worse.
- **mlp_ratio=2 speedup:** ~10% per-epoch (128s vs ~116s on n_layers=5 stack). Not enough to gain another epoch in budget — would need ~30-40% speedup.
- **Conclusion:** mlp_ratio reduction is not load-bearing at this depth. The gating mechanism (GeGLU) is already providing the capacity benefit; reducing mlp_ratio only removes capacity without providing a meaningful epoch-budget gain.
- **n_params:** 683,547 (30% smaller than baseline 976,827)
- **Metric artifacts:** student PR #2007 comments (no jsonl path provided)
- **Reassigned tanjiro:** PR #2080 n_layers=4 + T_max=17 (continue depth sweep)

---

## 2026-05-13 09:15 — PR #1996: slice_num=48 + T_max=15 — MERGED (−1.33% val, −1.10% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/slice-num-48
- **Hypothesis:** Reducing slice_num 64→48 reduces per-epoch time ~18%, enabling 15 epochs in 30-min cap vs 12. Same "epoch count is the binding constraint" mechanism as PR #1995.
- **Result:** val=46.847 / test=40.837 (best_epoch=15, n_layers=6, T_max=15, sw=10)

| Split | val (PR #1995 baseline, n_layers=5) | val (slice_num=48, n_layers=6) | Δ | test |
|---|---|---|---|---|
| single_in_dist | 52.253 | 50.491 | −3.4% | 45.728 |
| geom_camber_rc | **60.809** | **60.364** | **−0.7%** | 55.146 |
| geom_camber_cruise | 29.174 | 29.835 | +2.3% | 24.157 |
| re_rand | 47.675 | 46.699 | −2.0% | 38.317 |
| **avg** | **47.478** | **46.847** | **−1.33% ✓** | **40.837** |

- **⚠ Stack mismatch:** PR was run on n_layers=6 (old stack); merged into advisor with n_layers=5 (from #1995). Current advisor code = n_layers=5 + slice_num=48, NOT validated yet. fern PR #2062 assigned to verify the compound.
- **Per-epoch time:** ~123s (n_layers=6). Expected ~100-108s on n_layers=5 + slice_num=48.
- **Metric artifacts:** `models/model-charliepai2g48h3-fern-slice-num-48-20260513-070845/metrics.yaml`
- **Reassigned fern:** PR #2062 n_layers=5 + slice_num=48 compound verification

---

## 2026-05-13 09:00 — PR #1995: n_layers=5 + T_max=14 — MERGED (−6.98% val, −6.98% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/n-layers-5
- **Hypothesis:** Shallower model (n_layers=5) → faster epochs (~116s vs ~138s) → 14 epochs fit in 30-min budget → 2 extra cosine-tail refinement epochs → better convergence. Trading capacity for training duration.
- **Result:** val=47.478 / test=41.290 (best_epoch=14, surf_weight=10)

| Split | val (n_layers=6, T=12, sw=5) | val (n_layers=5, T=14, sw=10) | Δ val | test (n_layers=5) |
|---|---|---|---|---|
| single_in_dist | 56.933 | 52.253 | −8.2% | 46.980 |
| geom_camber_rc | **64.886** | **60.809** | **−6.3%** | 54.123 |
| geom_camber_cruise | 31.056 | 29.174 | −6.1% | 24.263 |
| re_rand | 51.287 | 47.675 | −7.0% | 39.794 |
| **avg** | **51.040** | **47.478** | **−6.98% ✓** | **41.290** |

- **Key stats:** n_params=826,071 (−15.7%), VRAM 40GB (−20%), 116s/epoch (−16%)
- **Mechanism:** "Epoch count was the binding constraint, not capacity." T_max=14 aligns cosine decay to the new budget; monotonic val descent to epoch 14.
- **Important:** surf_weight=10 (NOT 5). The surf_weight=5 compound on this new stack is the immediate next priority (edward #2048).
- **train.py change:** n_layers 6→5 (no other changes; lr=cfg.lr bug still NOT fixed)
- **Metric artifacts:** `models/model-charliepai2g48h3-edward-n-layers-5-20260513-065528/metrics.yaml`
- **Reassigned edward:** PR #2048 surf_weight=5 on n_layers=5 + T_max=14 (compound)

---

## 2026-05-13 08:45 — PR #1765: Lion lr=1.5e-4 — CLOSED (+4.21% val, +1.11% test)

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/lion-lr-2e-4-rebased (multi-rebase, renamed)
- **Hypothesis:** lr=1.5e-4 is a better midpoint than lr=2e-4 (which overshot on geom_camber_rc) on the current modernized stack.
- **Result:** Two arms — sw=5: val=53.192 / test=45.471; sw=10: val=53.398 / test=46.652. Both vs baseline 51.040 / 44.390.

| Metric | PR #1956 baseline | sw=5 + lr=1.5e-4 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 51.040 | 53.192 | +4.21% |
| test_avg/mae_surf_p | 44.390 | 45.471 | +2.43% |

- **Conclusion:** LR axis exhausted from both directions. lr=1.5e-4 (+4.21% worse) and lr=2e-4 (known worse on RMSNorm stack) both fail. lr=1e-4 is the confirmed optimum for Lion on this stack. T_max=12 shift dominates any LR midpoint benefit.
- **Key bug fix in branch:** `lr=1e-4` hardcoded in Lion constructor (train.py:441) → `lr=cfg.lr` fix applied by student. Fix NOT yet in advisor branch — sent alert to frieren #2006 to apply before running lr=8e-5.
- **Metric artifacts:** `models/model-charliepai2g48h3-alphonse-lion-lr-1.5e-4-tmax12-sw5-20260513-055705/metrics.jsonl`, `models/model-charliepai2g48h3-alphonse-lion-lr-1.5e-4-tmax12-sw10-20260513-065306/metrics.jsonl`
- **Reassigned alphonse:** PR #2043 DropPath stochastic depth rate=0.1

---

## 2026-05-13 08:45 — PR #1766: Lion WD=1e-2 (RMSNorm+GeGLU rerun) — CLOSED (+16.6% val vs current)

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/lion-wd-1e-2
- **Hypothesis:** Lion paper recommends WD 10-100× higher than Adam; WD=1e-2 should compound with GeGLU+RMSNorm regularization.
- **Result (primary arm):** val=59.517 / test=53.732. Vs current baseline 51.040 (+16.6% worse).
- **Additional data point:** WD=1e-2 + T_max=12 + sw=10: val=54.594 (+6.96% worse). Neither arm beats current baseline.

| Metric | Current baseline | WD=1e-2 arm | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 51.040 | 59.517 | +16.6% |
| test_avg/mae_surf_p | 44.390 | 53.732 | +21.0% |

- **Conclusion:** WD axis fully exhausted. WD=1e-4 optimal confirmed (WD=1e-1, 3e-2, 1e-2 all worse). RMSNorm+GeGLU already provide implicit regularization that previously WD=1e-2 was compensating for; marginal WD benefit disappears on the modernized stack.
- **Note:** PR was opened against an older stack and required multiple rebases; final result still outdated vs compound baseline. WD hypothesis was valid but axis is saturated.
- **Metric artifacts:** `models/model-lion-wd-1e-2-rmsnorm-geglu-20260513-051233/metrics.jsonl`
- **Reassigned askeladd:** PR #2038 n_head=2 (head_dim 32→64 sweep)

---

## 2026-05-13 08:45 — PR #1948: surf_weight=3 — CLOSED (stale draft, no results)

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/surf-weight-3
- **Hypothesis:** Is the surf_weight optimum below 5? Test sw=3 to find the gradient budget floor.
- **Result:** None — draft PR with no training run started. Built on T_max=50 stack (stale).
- **Conclusion:** Closed as stale. Surf_weight axis below 5 now covered by nezuko #2029 (sw=2) on the current T_max=12+sw=5 compound stack.
- **Reassigned thorfinn:** PR #2040 gradient clipping max_norm=1.0

---

## 2026-05-13 07:25 — PR #1956: T_max=12 + surf_weight=5 compound — MERGED (−3.33% val, −1.29% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/tmax-12-surf-weight-5-compound
- **Hypothesis:** T_max=12 and surf_weight=5 are orthogonal mechanisms (schedule vs loss weighting); should compound additively. Predicted val 48-50.
- **Result:** val=51.040, test=44.390 (epoch 12/12, 30-min cap, surf_weight=5)

| Split | val (compound) | val (#1793 baseline sw=10) | Δ | test (compound) | test (baseline) | Δ |
|---|---|---|---|---|---|---|
| single_in_dist | 56.933 | 58.907 | −3.35% | 50.459 | 50.239 | +0.44% |
| geom_camber_rc | **64.886** | 67.658 | **−4.10%** | 59.341 | 59.561 | −0.37% |
| geom_camber_cruise | 31.056 | 33.380 | **−6.96%** | 25.501 | 27.740 | **−8.07%** |
| re_rand | 51.287 | 51.248 | +0.08% | 42.260 | 42.345 | −0.20% |
| **avg** | **51.040** | **52.798** | **−3.33% ✓** | **44.390** | **44.972** | **−1.29% ✓** |

- **Volume MAE check:** Volume MAE improved across ALL 4 splits (−6% to −14%) confirming gradient-budget reallocation mechanism
  - single_in_dist vol: 67.251 → 60.506 (−10.0%)
  - geom_camber_rc vol: 74.389 → 65.720 (−11.7%)
  - geom_camber_cruise vol: 33.564 → 28.904 (−13.9%)
  - re_rand vol: 52.194 → 48.815 (−6.5%)
- **Compound analysis:** Sub-linear but real. Predicted −7% to −9% if fully additive; got −3.33%. Surface gradient with sw=5 shows diminishing returns at the cosine tail — the richer volume representation helps surface via context, but at T_max=12 end-state the two mechanisms partially share the same mode.
- **re_rand is the laggard** (+0.08% val / −0.20% test) — Reynolds holdout benefits less from geometric context than geometry holdouts
- **Cruise drives test gain** (−8.07%) — largest meshes get most benefit from gradient reallocation
- **Metric artifacts:** `models/model-charliepai2g48h3-nezuko-tmax-12-surf-weight-5-compound-20260513-055720/metrics.jsonl`
- **Reassigned nezuko:** PR #2029 surf_weight=2 on compound baseline (continue gradient sweep below sw=5)

---

## 2026-05-13 07:00 — PR #1983: CosineAnnealingLR T_max=10 — CLOSED (+10.96% val, +11.95% test)

- **Student:** charliepai2g48h3-frieren
- **Branch:** charliepai2g48h3-frieren/t-max-10
- **Hypothesis:** T_max=10 (vs current T_max=12) pushes cosine floor earlier, gives Lion more low-LR refinement time.
- **Result:** val=58.586, test=50.346 (epoch 12/12)

| Split | val (T_max=10) | val (T_max=12 baseline) | Δ |
|---|---|---|---|
| single_in_dist | 63.472 | 58.907 | +7.75% |
| geom_camber_rc | 74.086 | 67.658 | **+9.50%** |
| geom_camber_cruise | 38.763 | 33.380 | **+16.13%** |
| re_rand | 58.024 | 51.248 | **+13.22%** |
| **avg** | **58.586** | **52.798** | **+10.96%** |

- **Test:** 50.346 vs 44.972 = **+11.95% worse**
- **Mechanism (student-identified, excellent analysis):**
  1. PyTorch `CosineAnnealingLR` is cyclic, not clamped at T_max. After T_max it continues the cosine, bouncing LR back up.
  2. With T_max=10 and 12 epochs run: epoch 11 LR was **exactly 0** → DEAD EPOCH (val identical to epoch 10). Epoch 12 bumped back to 2.4e-6 as cosine started rising.
  3. T_max=12's gains came from gradual decay across epochs 9-12 (1e-5 → 0), NOT from "near-zero floor." Compressing this to T_max=10 truncated the productive part of the cosine — epochs 9-10 ran at sub-3e-6 LRs where Lion's sign-update has nothing useful to do.
- **Dead end mechanism documented:** T_max < cfg.epochs is always strictly worse. T_max should be ≥ epoch count.
- **Reassigned frieren:** PR #2006 Lion lr=8e-5 (bracket alphonse's 1.5e-4 from below)

---

## 2026-05-13 07:00 — PR #1984: n_hidden=160 — CLOSED (val −0.247% / test +1.268%; complexity disproportionate)

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/n-hidden-160
- **Hypothesis:** Widen Q/K/V/PhysicsAttention dim from 128→160 to capture richer aerodynamic features.
- **Result:** val=52.668 (−0.247% vs 52.798), test=45.542 (+1.268% vs 44.972), 12 epochs in 31.5 min

| Split | val (n=160) | val (n=128 baseline) | Δ | test (n=160) | test (n=128) | Δ |
|---|---|---|---|---|---|---|
| single_in_dist | 57.375 | 58.907 | **−2.60%** | 50.878 | 50.239 | +1.27% |
| geom_camber_rc | 67.923 | 67.658 | +0.39% | 61.274 | 59.561 | **+2.88%** |
| geom_camber_cruise | 33.528 | 33.380 | +0.44% | 27.207 | 27.740 | −1.92% |
| re_rand | 51.845 | 51.248 | +1.17% | 42.810 | 42.345 | +1.10% |
| **avg** | **52.668** | **52.798** | **−0.247%** | **45.542** | **44.972** | **+1.27%** |

- **Per-epoch cost:** +12% (157s vs 140s) — much cheaper than predicted +60%
- **n_params:** +52% (1.522M vs ~1.0M)
- **Peak GPU memory:** 55.22 GB (up from baseline)
- **Decision rationale (closed not merged):**
  - Val/test direction inversion → noise signature, not real win
  - "Disproportionate complexity for tiny gain" clause applies (+52% params for −0.247% val)
  - Targeted hardest split (geom_camber_rc) regressed on test (+2.88%) — the mechanism didn't work
  - Permanent slowdown on all future experiments if merged
- **Useful signal:** single_in_dist val −2.60% suggests wider attention helps in-distribution; OOD splits resist. This is opposite RMSNorm's signal (which cracked geom_camber_rc) — confirms OOD bottleneck is geometric extrapolation, not feature capacity.
- **Per-epoch calibration:** n_hidden widening is much cheaper than expected (+12% not +60%)
- **Reassigned tanjiro:** PR #2007 mlp_ratio=2 (test if "gating wins outright" extends below 4)

---

## 2026-05-13 06:45 — PR #1925: Lion WD=3e-2 — CLOSED (WD axis saturated, +0.06% on prior baseline)

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/lion-wd-3e-2
- **Hypothesis:** WD=3e-2 brackets WD optimum between in-flight WD=1e-2 (askeladd) and closed WD=1e-1 (PR #1889).
- **Result:** val=63.054, test=55.573 (14 epochs, ~138s/epoch, best_epoch=13)
- **Baseline used:** PR #1837 (val=63.017 / test=54.731; RMSNorm+GeGLU+Lion+WD=1e-4)

| Split | val (WD=3e-2) | val (baseline) | Δ |
|---|---|---|---|
| single_in_dist | 75.347 | 76.710 | −1.78% |
| geom_camber_rc | 76.040 | 73.930 | **+2.85%** |
| geom_camber_cruise | 40.649 | 40.746 | −0.24% |
| re_rand | 60.183 | 60.683 | −0.82% |
| **avg** | **63.054** | **63.017** | **+0.06%** |

- **Test:** 55.573 vs 54.731 = +1.54% (worse)
- **WD bracket summary:** [WD=1e-4: 63.017 / WD=3e-2: 63.054 / WD=1e-1: 64.731] — flat valley from 1e-4→3e-2, bends up sharply at 1e-1
- **Analysis:** WD=3e-2 is statistically tied with WD=1e-4 on val (well within seed noise). best_epoch=13 (same as baseline) confirms no over-regularization at this WD level. Test regression (+1.54%) suggests marginal overfitting risk without val benefit.
- **Against current baseline (#1793 T_max=12, val=52.798):** +19.4% worse (baseline-drift effect only — measured on old stack)
- **Dead end confirmed:** WD axis exhausted on Lion+RMSNorm+GeGLU stack. Further WD experiments will only refine a flat valley. WD=1e-2 (askeladd #1766) will confirm third bracket point.
- **Student insight:** Correctly identified "throughput improvements" as higher leverage than WD tuning
- **Reassigned edward:** PR #1995 n_layers=5 (shallower model → faster epochs → 3 extra epochs in budget)

---

## 2026-05-13 06:40 — PR #1790: Lion + 2-epoch cosine warmup — CLOSED (stale + mechanism conflicts with T_max=12)

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/lion-cosine-warmup-2ep
- **Hypothesis:** 2-epoch linear warmup (1e-6→1e-4) stabilizes Lion's aggressive sign-based init before full cosine decay.
- **Original result (PR #1725 baseline):** val=78.315 (−9.9% vs old baseline val=86.938) — positive
- **Status:** Sent back for rerun on GeGLU+Lion stack (PR #1769 baseline val=64.918 at that time). No results produced in 2h+ after rerun request. Last branch commit at 04:05 UTC.
- **Closure rationale:**
  1. Baseline shifted twice since rerun request (→57.328 →52.798)
  2. Mechanism conflict: 2-epoch warmup consumes 17% of 12-epoch budget; T_max=12 cleanly handles LR decay without needing warmup (cold-start instability addressed by cosine from epoch 1)
  3. Student inactivity: 2h+ stale
- **Reassigned fern:** PR #1996 slice_num=48 (throughput-focused; tighter PhysicsAttention → ~2 extra epochs)

---

## 2026-05-13 06:20 — PR #1920: CosineAnnealingLR eta_min=1e-5 — CLOSED (+12.05% vs current baseline; mechanism redundant with T_max=12)

- **Student:** charliepai2g48h3-frieren
- **Branch:** charliepai2g48h3-frieren/cosine-eta-min-1e-5
- **Hypothesis:** Non-zero LR floor (eta_min=1e-5) prevents Lion's cosine tail from fully collapsing to 0, preserving some update signal in final epochs.
- **Result:** val=59.159, test=53.210 (measured vs PR #1837 baseline val=63.017 → −6.1% improvement on old baseline)

| Split | val (eta_min=1e-5) | val (old #1837 baseline) | Δ vs old | val (current #1793 baseline) | Δ vs current |
|---|---|---|---|---|---|
| single_in_dist | ~54.9 | 76.710 | −9.7% | 58.907 | −6.8% |
| geom_camber_rc | ~72.6 | 73.930 | −1.8% | 67.658 | **+7.3%** |
| geom_camber_cruise | ~43.8 | 40.746 | +7.5% | 33.380 | **+31.2%** |
| re_rand | ~65.4 | 60.683 | +7.8% | 51.248 | **+27.6%** |
| **avg** | **59.159** | **63.017** | **−6.1% ✓ (old)** | **52.798** | **+12.05% ✗** |

- **Test:** 53.210 vs 44.972 = **+18.3% worse** than current baseline
- **Analysis:** Result was measured against the PR #1837 baseline (val=63.017). The baseline shifted twice (→57.328 via PR #1836, →52.798 via PR #1793 T_max=12) before review. Against the current baseline, this is a large regression. Critically, cruise and re_rand splits regressed badly: the eta_min=1e-5 floor RAISED the LR in the late-epoch regime (vs T_max=12 decaying to 0), which interfered with Lion's fine-grained late-epoch refinement.
- **Mechanism conflict confirmed:** Both eta_min and T_max address late-epoch LR behavior; T_max=12 (LR→0) strictly dominates eta_min=1e-5 (LR floor above 0). T_max=12 won the head-to-head and is now the stack default.
- **Student's own follow-up suggestion** explicitly named "combine with T_max tuning" — that direction is now merged (PR #1793).
- **Dead end:** LR schedule "floor vs ceiling" axis is exhausted. T_max=12 is the correct solution.
- **Reassigned frieren:** PR #1983 CosineAnnealingLR T_max=10 (push cosine further)

---

## 2026-05-13 06:20 — PR #1872: mlp_ratio=8 + GeGLU — CLOSED (+5.95% regression, "gating wins outright")

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/mlp-ratio-8-geglu-lion
- **Hypothesis:** GeGLU halves fc2 input width (256 → 128 effective channels at mlp_ratio=4). Doubling to mlp_ratio=8 recovers the capacity, giving GeGLU gating + full MLP depth.
- **Result:** val=68.781, test=60.220 (original run). Reproducibility run: val=69.613. Both confirm regression.

| Split | val (mlp_ratio=8) | val (baseline mlp_ratio=4) | Δ |
|---|---|---|---|
| single_in_dist | ~80.8 | 72.044 | **+12.2%** |
| geom_camber_rc | ~89.2 | 89.234 | ~0% |
| geom_camber_cruise | ~41.5 | 38.721 | +7.2% |
| re_rand | ~63.5 | 57.586 | +10.3% |
| **avg** | **68.781** | **64.918** (PR #1769 baseline) | **+5.95% worse** |

- **Test:** 60.220 vs 58.171 = +3.5% worse
- **Per-epoch overhead:** ~163s vs ~143s (+14%) → net −1 effective epoch in 30-min budget
- **Reproducibility:** Second seed also regressed (val=69.613). Regression is real, not noise.
- **Student analysis confirmed:** "gating wins outright." Interpretation: GeGLU's selective gating is the structural win — it filters which features propagate into the physics attention. Wider fc2 adds capacity but also adds noise pathways that the gate doesn't fully suppress at 12-epoch budget. single_in_dist regressed worst (+12-18%), consistent with extra capacity overfitting on the simple in-distribution splits.
- **Dead end:** MLP capacity expansion exhausted. mlp_ratio=4 with GeGLU gating is optimal for this budget.
- **Reassigned tanjiro:** PR #1984 n_hidden=160 (widen attention dim — orthogonal capacity axis)

---

## 2026-05-13 06:05 — PR #1793: CosineAnnealingLR T_max=12 on RMSNorm+GeGLU+Lion — MERGED (−7.9% val, −8.9% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/lion-tmax-12-aligned
- **Hypothesis:** T_max=50 with 30-min cap fires only ~13% of cosine decay; T_max=12 aligns to actual epoch budget for proper late-epoch LR decay to 0.
- **Result:** val=52.798, test=44.972 (epoch 12/12, 30-min cap, surf_weight=10)

| Split | val (T_max=12, sw=10) | val (prev baseline sw=5, T=50) | Δ |
|---|---|---|---|
| single_in_dist | 58.907 | 60.960 | −3.4% |
| geom_camber_rc | 67.658 | 72.044 | −6.1% |
| geom_camber_cruise | 33.380 | 38.721 | **−13.8%** |
| re_rand | 51.248 | 57.586 | **−11.0%** |
| **avg** | **52.798** | **57.328** | **−7.9% ✓** |

- **Test:** 44.972 vs 49.387 = **−8.9% ✓**
- **Same-surf_weight comparison vs PR #1837 (sw=10, T=50, val=63.017):** **−16.2% val / −17.8% test** — pure T_max=12 effect
- **Mechanism confirmed:** LR trajectory from epoch 9→12 dropped 11.3% as cosine fully decayed (2.07e-5 → 0). Lion's sign(m) steps get 10× finer in late epochs as designed.
- **Easy splits benefit most:** geom_camber_cruise (−13.8%) and re_rand (−11.0%) — late-epoch fine-tuning matters more when surrounding fit is already good
- **Note:** student rebase pre-dated PR #1836 (surf_weight=5) merge, so this used surf_weight=10. Compound `T_max=12 + surf_weight=5` is the natural next experiment.
- **Metric artifacts:** `models/model-charliepai2g48h3-nezuko-lion-tmax-12-aligned-v2-20260513-045129/metrics.jsonl`
- **Reassigned nezuko:** PR #1956 T_max=12 + surf_weight=5 compound (orthogonal mechanism stacking)

---

## 2026-05-13 05:45 — PR #1836: surf_weight=5 on RMSNorm+GeGLU+Lion — MERGED (−9.03% val, −9.76% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/surf-weight-5-rmsnorm-geglu-lion
- **Hypothesis:** Halving surf_weight (10→5) reallocates L1 gradient budget toward volume nodes; richer volumetric features improve surface accuracy via geometric context. Hardest splits (geom_camber_rc, single_in_dist) benefit most.
- **Result:** val=57.328, test=49.387 (epoch 14, 30-min cap, ~138s/epoch)

| Split | val (surf_w=5) | val (baseline surf_w=10) | Δ | test (surf_w=5) | test (baseline) | Δ |
|---|---|---|---|---|---|---|
| single_in_dist | **60.960** | 76.710 | **−20.5%** | **53.010** | 67.384 | **−21.3%** |
| geom_camber_rc | **72.044** | 73.930 | −2.6% | **62.463** | 64.508 | −3.2% |
| geom_camber_cruise | 38.721 | 40.746 | −5.0% | 32.843 | 34.707 | −5.4% |
| re_rand | 57.586 | 60.683 | −5.1% | 49.231 | 52.327 | −5.9% |
| **avg** | **57.328** | **63.017** | **−9.03% ✓** | **49.387** | **54.731** | **−9.76% ✓** |

- **Volume MAE confirmed improved** on all 4 splits (−7% to −26%), confirming gradient-reallocation mechanism
- **single_in_dist standout** (−20.5% val): after RMSNorm fixed geom_camber_rc (−17.2%), single_in_dist became most volume-starved — surf_weight=5 cracked it
- **All four splits improved** on both val and test — the cleanest, most uniform sweep win yet
- **Metric artifacts:** `models/model-charliepai2g48h3-thorfinn-surf-weight-5-rmsnorm-geglu-lion-20260513-041441/metrics.jsonl`
- **Reassigned thorfinn:** PR #1948 surf_weight=3 (sweep further down)

---

## 2026-05-13 05:40 — PR #1765: Lion lr=2e-4 (rebased onto RMSNorm stack) — SENT BACK (pivot to lr=1.5e-4)

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/lion-lr-2e-4
- **Result on new RMSNorm+GeGLU+Lion stack:** val=63.635, test=54.060 vs baseline 63.017/54.731

| Split | val (lr=2e-4) | val (baseline lr=1e-4) | Δ |
|---|---|---|---|
| single_in_dist | 72.015 | 76.710 | **−6.1%** |
| geom_camber_rc | 82.155 | 73.930 | **+11.1%** |
| geom_camber_cruise | 38.985 | 40.746 | −4.3% |
| re_rand | 61.384 | 60.683 | +1.2% |
| **avg** | **63.635** | **63.017** | **+0.98%** (misses) |

- **Test avg:** 54.060 vs 54.731 = −1.23% ✓ (test improves but val is primary metric)
- **Insight:** RMSNorm tightened the loss surface — lr=2e-4 overshoots specifically on geom_camber_rc (the OOD geometry split now with a tighter noise landscape post-RMSNorm). Three of four splits improved. LR sweet spot has shifted down on the new stack.
- **Action:** Sent back to try lr=1.5e-4. Also updated target to new baseline (57.328 / 49.387 after #1836 merge). lr=cfg.lr bug fix preserved.

---

## 2026-05-13 05:30 — PR #1890: n_layers=7 + RMSNorm+GeGLU+Lion — CLOSED (+4.6% val regression)

- **Student:** charliepai2g48h3-frieren
- **Branch:** charliepai2g48h3-frieren/n-layers-7-rmsnorm-geglu-lion
- **Hypothesis:** RMSNorm+bf16 reduces per-epoch time enough (~160s vs 205s) to fit 11-12 epochs at 7 layers vs 9 in the old AdamW stack. Re-test n_layers=7 under better conditions.
- **Result:** val=65.904, test=56.888 (epoch best 12 of 12, 30-min cap)

| Split | val (n_layers=7) | val (baseline n_layers=6) | Δ |
|---|---|---|---|
| single_in_dist | 90.497 | 76.710 | **+18.0%** |
| geom_camber_rc | 77.252 | 73.930 | +4.5% |
| geom_camber_cruise | 38.800 | 40.746 | −4.8% |
| re_rand | 57.067 | 60.683 | −6.0% |
| **avg** | **65.904** | **63.017** | **+4.6% worse** |

- **Per-epoch time:** ~160s (vs 138s at 6 layers) → 12 epochs fit in 30 min vs 14
- **Test:** 56.888 vs 54.731 = +3.9% worse
- **Analysis:** n_layers=7 helps geom_camber_cruise and re_rand (simpler splits), but catastrophically regresses single_in_dist (+18%) — the deeper model is underfitting at 12 epochs where the 6-layer model peaks at 13-14. Budget-convergence mismatch confirmed. Architecture depth expansion remains incompatible with this 30-min cap.
- **Dead end:** n_layers=7 is worse under all tested conditions (AdamW ~205s/epoch, Lion+RMSNorm ~160s/epoch)
- **Reassigned frieren:** PR #1920 CosineAnnealingLR eta_min=1e-5

---

## 2026-05-13 05:30 — PR #1889: Lion WD=1e-1 — CLOSED (+2.72% val regression, over-regularizes)

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/lion-wd-1e-1
- **Hypothesis:** WD=1e-1 is the upper end of Lion paper's recommended range (10-100× Adam); bracket with askeladd's WD=1e-2 to find optimum.
- **Result:** val=64.731, test=57.110 (best epoch=10, 30-min cap)

| Split | val (WD=1e-1) | val (baseline WD=1e-4) | Δ |
|---|---|---|---|
| single_in_dist | ~78.5 | 76.710 | regressed |
| geom_camber_rc | ~75.6 | 73.930 | regressed |
| geom_camber_cruise | ~42.8 | 40.746 | regressed |
| re_rand | ~62.0 | 60.683 | regressed |
| **avg** | **64.731** | **63.017** | **+2.72% worse** |

- **Test:** 57.110 vs 54.731 = +4.35% worse
- **Key diagnostic:** best_epoch=10 (vs baseline epoch=13) — val starts climbing after epoch 10 while train loss keeps dropping. Textbook over-regularization signature.
- **Insight:** WD=1e-1 forces parameters toward zero too aggressively at only 14 epochs. Model already underfitting — adding more regularization makes it worse. WD=1e-2 (askeladd, in-flight) likely the true optimum; WD=1e-1 confirmed as upper dead-end boundary.
- **Dead end:** WD space above 1e-2 exhausted.
- **Reassigned edward:** PR #1925 Lion WD=3e-2 (bracket between 1e-2 and 1e-1)

---

## 2026-05-13 04:30 — PR #1837: RMSNorm in TransolverBlock — MERGED (−2.9% val, −5.9% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-frieren
- **Branch:** charliepai2g48h3-frieren/rmsnorm-geglu-lion
- **Result:** val=63.017, test=54.731 (epoch 13 of 14, 30-min cap)

| Split | val (RMSNorm) | val (baseline) | Δ |
|---|---|---|---|
| single_in_dist | 76.710 | 72.021 | +6.5% |
| geom_camber_rc | **73.930** | 89.234 | **−17.2%** |
| geom_camber_cruise | 40.746 | 37.058 | +10.0% |
| re_rand | 60.683 | 61.359 | −1.1% |
| **avg** | **63.017** | **64.918** | **−2.9% ✓** |

- Test avg: 54.731 vs 58.171 = **−5.9%** (geom_camber_rc −19.8%)
- **Speed**: 138s/epoch (vs 143s) — 14 epochs fit in 30 min, best at epoch 13
- **Insight**: RMSNorm's removal of mean-centering helps the hardest OOD split dramatically (geom_camber_rc −17.2% val). Slight regression on single_in_dist (+6.5%) and cruise (+10%). Net strongly positive.
- **Reassigned frieren**: PR #1890 n_layers=7 + RMSNorm+GeGLU+Lion

---

## 2026-05-13 04:30 — PR #1836: surf_weight=5 on GeGLU+Lion — SENT BACK (beats old baseline; misses new RMSNorm baseline by +0.2%)

- **Student:** charliepai2g48h3-thorfinn
- **Result on old baseline (64.918):** val=63.142, test=56.121 (−2.74%/−3.52% ✓)
- **vs new RMSNorm baseline (63.017):** +0.2% val worse — needs rebase + retest
- **Mechanism confirmed:** volume MAE improved −7% to −11% across all splits; geom_camber_rc biggest: −9.34% val / −13.06% test

---

## 2026-05-13 04:30 — PR #1859: SmoothL1 β=0.1 on GeGLU+Lion — CLOSED (+7.1% val regression)

- **Student:** charliepai2g48h3-edward
- **Result:** val=69.553, test=62.176 vs baseline 64.918 = **+7.1%/+6.9% worse**
- **Insight:** SmoothL1 β=0.1 slows late-epoch convergence. geom_camber_rc improved −11.7% but in-distribution hurt badly (+25.4%). Pure L1's constant gradient optimal for Lion. All loss-modification directions exhausted.
- **Dead end**: all loss modifications (physical-space L1, channel weighting, SmoothL1) fail on GeGLU+Lion
- **Reassigned edward**: PR #1889 Lion WD=1e-1

---

## 2026-05-13 04:20 — PR #1765: Lion lr=2e-4 (with bug fix) — SENT BACK (−7.8% on Lion+GELU; obsolete vs GeGLU+Lion)

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/lion-lr-2e-4
- **Hypothesis:** 2× LR (1e-4 → 2e-4) leverages Lion's higher LR tolerance from sign-based updates
- **Bug fix included:** `train.py:421` `lr=1e-4` → `lr=cfg.lr` (the `--lr` flag was silently ignored before this PR). Default `Config.lr` changed from 5e-4 (AdamW carryover) to 1e-4 to preserve current baseline behavior.
- **Result:** val=80.127, test=71.126 (Lion+GELU stack, epoch 13/13)

| Split | val (lr=2e-4) | val (baseline lr=1e-4) | Δ |
|---|---|---|---|
| single_in_dist | 84.028 | 98.979 | **−15.1%** |
| geom_camber_rc | 98.809 | 104.737 | −5.7% |
| geom_camber_cruise | 59.025 | 62.041 | −4.9% |
| re_rand | 78.646 | 81.995 | −4.1% |
| **avg** | **80.127** | **86.938** | **−7.8% ✓** |

- **vs Lion+GELU baseline (86.938):** −7.8% ✓ (first true measurement of lr=2e-4 because bug fixed)
- **vs new GeGLU+Lion baseline (64.918):** +23.4% worse — needs rerun on GeGLU+Lion
- **Action:** Sent back to retest on GeGLU+Lion stack. lr=2e-4 mechanism (Lion's sign updates absorb higher LR) is orthogonal to activation function.

---

## 2026-05-13 04:20 — PR #1793: Lion + T_max=12 (cosine aligned to budget) — SENT BACK (−9.18% on Lion+GELU; obsolete vs GeGLU+Lion)

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/lion-tmax-12-aligned
- **Hypothesis:** T_max=50 with 30-min cap fires only 13% of cosine decay; T_max=12 aligns to actual epoch budget for proper late-epoch LR decay
- **Result:** val=78.962, test=68.931 (Lion+GELU stack)

| Split | val (T_max=12) | val (baseline T_max=50) | Δ |
|---|---|---|---|
| single_in_dist | 91.381 | 98.979 | −7.7% |
| geom_camber_rc | 93.613 | 104.737 | −10.6% |
| geom_camber_cruise | 57.362 | 62.041 | −7.5% |
| re_rand | 73.491 | 81.995 | −10.4% |
| **avg** | **78.962** | **86.938** | **−9.18% ✓** |

- **vs Lion+GELU baseline (86.938):** −9.18% ✓ (uniform improvement across all 8 val+test splits)
- **vs new GeGLU+Lion baseline (64.918):** +21.6% worse — needs rerun on GeGLU+Lion
- **Action:** Sent back to retest on GeGLU+Lion stack. Schedule mechanism (proper cosine decay in budget) is orthogonal to activation function.

---

## 2026-05-13 04:00 — PR #1824: SwiGLU (SiLU gate) vs GeGLU — CLOSED (+1.6% val regression)

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/swiglu-lion
- **Hypothesis:** SiLU gate smoother gradient than GELU; SwiGLU > GeGLU (as in LLaMA/PaLM)
- **Result:** val=65.957, test=58.705 (GeGLU+Lion stack, epoch 11, 30-min cap)

| Split | SwiGLU val | GeGLU val (baseline) | Δ |
|---|---|---|---|
| single_in_dist | 79.277 | 72.021 | +10.1% |
| geom_camber_rc | 81.003 | 89.234 | **−9.2%** |
| geom_camber_cruise | 41.225 | 37.058 | +11.2% |
| re_rand | 62.321 | 61.359 | +1.6% |
| **avg** | **65.957** | **64.918** | **+1.6% worse** |

- **Decision:** CLOSED — primary metric regresses (>baseline); student correctly said "don't merge"
- **Key insight:** SwiGLU helped only on geom_camber_rc (OOD geometry, −9.2%), hurt everywhere else. LLM SwiGLU > GeGLU finding does not transfer to this CFD surrogate. GELU's slightly negative gate range may be beneficial for pressure-gradient features. Per-split routing (SwiGLU for OOD, GeGLU for in-distribution) is an interesting observation but not actionable without architectural complexity.
- **Dead end added:** SwiGLU +1.6% vs GeGLU on this task
- **Reassigned:** tanjiro → PR #1872 mlp_ratio=8 + GeGLU (recover fc2 capacity halved by gating split)

---

## 2026-05-13 03:45 — PR #1767: Channel-weighted L1 [0.03, 0.03, 1.0] on GeGLU+Lion — CLOSED (+11.0% regression)

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/physical-space-l1-lion
- **Hypothesis:** channel reweighting (p:Ux:Uy = 1:0.03:0.014) drives physical-space L1 win; test explicitly on GeGLU+Lion
- **Result:** val=72.071, test=63.176 (GeGLU+Lion, epoch 13)

| Split | val (CW) | val (baseline) | Δ |
|---|---|---|---|
| single_in_dist | 82.511 | 72.021 | +14.6% |
| geom_camber_rc | 87.864 | 89.234 | −1.5% |
| geom_camber_cruise | 48.734 | 37.058 | +31.5% |
| re_rand | 69.174 | 61.359 | +12.7% |
| **avg** | **72.071** | **64.918** | **+11.0%** |

- **vs GeGLU+Lion baseline (64.918):** +11.0% WORSE — clear dead end
- **Key insight:** GeGLU does implicit channel balancing via GELU(x1)*x2 gating. Manual downweighting of Ux/Uy starves the gates of velocity supervision, degrading the routing they've learned. The round-1 +4.0% win (Lion+GELU) was a GELU-specific interaction, not a portable mechanism.
- **Action:** CLOSED. edward reassigned to SmoothL1 β=0.1 (#1859).

---

## 2026-05-13 03:45 — PR #1790: Lion + 2-epoch warmup — SENT BACK (−9.9% on old Lion+GELU, obsolete; rerun on GeGLU+Lion needed)

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/lion-cosine-warmup-2ep
- **Hypothesis:** 2-epoch linear warmup (1e-6→1e-4) stabilizes Lion sign-update init; better basin before full LR kicks in
- **Result:** val=78.315, test=68.744 (Lion+GELU stack, epoch 13)

| Split | val (warmup) | val (no warmup) | Δ |
|---|---|---|---|
| single_in_dist | 91.531 | 98.979 | −7.5% |
| geom_camber_rc | 95.883 | 104.737 | −8.5% |
| geom_camber_cruise | 52.515 | 62.041 | −15.4% |
| re_rand | 73.332 | 81.995 | −10.6% |
| **avg** | **78.315** | **86.938** | **−9.9%** |

- **vs Lion+GELU baseline (86.938):** −9.9% ✓ (hypothesis confirmed on old baseline)
- **vs new GeGLU+Lion baseline (64.918):** +20.6% worse — obsolete, needs GeGLU rerun
- **Mechanism:** warmup absorbs Lion's aggressive sign-update noise at random init; faster early convergence, larger gains on harder/OOD splits
- **Action:** Sent back to retest on GeGLU+Lion stack. Warmup mechanism is orthogonal to activation function.

---

## 2026-05-13 03:15 — PR #1766: Lion WD=1e-2 — SENT BACK (+10.4% on old baseline, but obsolete; CRITICAL bug discovered)

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/lion-wd-1e-2
- **Result:** val=77.859, test=67.662 (Lion+GELU stack, epoch 13)

| Split | val (WD=1e-2) | val (WD=1e-4 baseline) | Δ |
|---|---|---|---|
| single_in_dist | 93.076 | 98.979 | −6.0% |
| geom_camber_rc | 93.965 | 104.737 | −10.3% |
| geom_camber_cruise | 51.233 | 62.041 | −17.4% |
| re_rand | 73.161 | 81.995 | −10.8% |
| **avg** | **77.859** | **86.938** | **−10.4%** |

- **vs Lion+GELU baseline (86.938):** −10.4% — Lion paper recommendation validated; WD=1e-2 is the correct value
- **vs new GeGLU+Lion baseline (64.918):** +19.9% worse — obsolete on current baseline
- **CRITICAL BUG DISCOVERED:** train.py:440 hardcoded `lr=1e-4` in Lion constructor, silently ignoring `--lr` CLI flag. askeladd's fix: `lr=cfg.lr`. This bug invalidates alphonse's #1765 lr=2e-4 experiment (silently ran at lr=1e-4).
- **Action:** Sent back to retest WD=1e-2 on GeGLU+Lion stack (expected to compound — orthogonal mechanisms). KEEP bug fix in branch.

---

## 2026-05-13 02:45 — PR #1769: GeGLU + Lion — MERGED (**new best: val=64.918, −25.3%**)

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/geglu-lion
- **Hypothesis:** GeGLU gated MLP activation (GELU(x1) * x2 where fc1 output is split in two) with Lion optimizer. Prior GeGLU+AdamW (PR #1728) was essentially baseline. Lion's sign-based updates may complement GeGLU's gating differently.

| Split | val mae_surf_p | test mae_surf_p | Δ val |
|---|---|---|---|
| single_in_dist | 72.021 | 64.947 | −27.2% |
| geom_camber_rc | 89.234 | 80.467 | −14.8% |
| geom_camber_cruise | 37.058 | 32.329 | **−40.3%** |
| re_rand | 61.359 | 54.939 | −25.2% |
| **avg** | **64.918** | **58.171** | **−25.3%** |

- **vs baseline (Lion+GELU 86.938):** −25.3% — **largest single-PR improvement in the programme** (exceeds L1 −20.5%, Lion −14.3%, n_layers=6 −9.4%)
- **Per-epoch time:** ~143s (GeGLU overhead negligible with bf16 — vs ~210s with AdamW). Got 13 epochs in 30 min vs 11 with Lion+GELU.
- **Still improving at cutoff** (epoch 13/50, curve descending). Cruise split improved 40.3% — GeGLU routing especially effective for high-camber transonic regime.
- **Key insight:** GeGLU's multiplicative gating acts as a learned feature router. Combined with Lion's sign-based constant-magnitude updates, the gating receives cleaner direction signals. The previous AdamW+GeGLU failure was likely due to Adam's adaptive scaling confounding the gate learning signal.
- **Artifacts:** `models/model-charliepai2g48h3-tanjiro-geglu-lion-20260513-012007/metrics.jsonl`

---

## 2026-05-13 02:45 — PR #1767: Physical-space L1 + Lion — SENT BACK (beats old baseline; new baseline higher bar)

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/physical-space-l1-lion
- **Result:** val=83.446, test=74.517 (epoch 13)

| Split | val | vs Lion+GELU baseline (86.938) |
|---|---|---|
| single_in_dist | 100.474 | +1.5% |
| geom_camber_rc | 96.861 | −7.9% |
| geom_camber_cruise | 58.461 | −5.8% |
| re_rand | 77.988 | −4.9% |
| **avg** | **83.446** | **−4.0%** |

- **vs new baseline after #1769 (64.918):** +28.5% worse — not competitive on new baseline
- **Key student insight:** Physical-space L1 = implicit channel reweighting (p:Ux:Uy ≈ 1 : 0.03 : 0.014). p_std≈679 Pa vs Ux_std≈14, Uy_std≈6 means dividing by p_std amplifies pressure gradient ~50×. This is the real mechanism.
- **Action:** Sent back with instruction to rebase on new GeGLU+Lion baseline and test **explicit pressure-weighted normalized loss**: `loss = mae_p_norm + 0.03 * (mae_Ux_norm + mae_Uy_norm)`

---

## 2026-05-13 02:10 — PR #1678: AdamW LR 5e-4 → 7e-4 — CLOSED (stale + obsolete)

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/lr-7e-4
- **Status:** Idle 3+ hours after creation with no commits. Additionally obsolete after PR #1725 Lion merged (new optimizer paradigm).
- **Decision:** Closed without execution; reassigned to Lion+T_max=12 (PR #1793).

---

## 2026-05-13 02:10 — PR #1726: SWA late-start epoch 8 — CLOSED (+7.9% worse vs AdamW baseline)

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/swa-late-start
- **Optimizer used:** AdamW (run started before PR #1725 Lion merge)
- **Result:** val=109.442, test=98.377 (epoch 11; still descending at 30-min cutoff)

| Split | val mae_surf_p | vs pre-Lion baseline (101.463) |
|---|---|---|
| single_in_dist | 137.319 | +13.8% |
| geom_camber_rc | 124.325 | +7.1% |
| geom_camber_cruise | 80.056 | +8.7% |
| re_rand | 96.069 | +0.7% |
| **avg** | **109.442** | **+7.9%** |

- **Analysis (student's post-mortem, well-reasoned):**
  1. SWA started at epoch 8 before convergence — baseline was still ~138 at that point, dropping ~7 pts/epoch
  2. Only 3 averaging epochs accumulated (9, 10, 11) — too few over a non-stationary region
  3. SWALR cut LR ~10× at the most critical training phase, slowing the underlying SGD
  4. Best epoch was the final (11) — model never reached SWA's "flat basin" regime
- **Conclusion:** SWA requires a converged base model to average. At 30-min budget with ~11 epochs, we never reach post-convergence. Reassigned fern to Lion+warmup (PR #1790) which addresses LR-schedule sensitivity in a constructive direction.
- **Artifacts:** `models/model-charliepai2g48h3-fern-swa-late-start-20260513-002529/metrics.jsonl`

---

## 2026-05-13 01:45 — PR #1725: Lion optimizer lr=1e-4 — MERGED (**new best: val=86.938, −14.3%**)

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/lion-optimizer
- **Hypothesis:** Lion's sign-based updates produce constant-magnitude steps, hypothesized to be synergistic with L1 loss which also produces constant ±1 gradients. Used lr=1e-4 (Lion paper recommends ~3-10× lower than Adam; our Adam lr=5e-4).

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 98.979 | 91.606 |
| geom_camber_rc | 104.737 | 92.561 |
| geom_camber_cruise | 62.041 | 52.841 |
| re_rand | 81.995 | 74.952 |
| **avg** | **86.938** | **77.990** |

- **vs baseline (101.463 post-bf16):** −14.3% — the biggest single win since L1 loss itself (which was −20.5%)
- **Epoch:** 11 of 50; still monotonically improving at 30-min cutoff. Convergence headroom remains.
- **Analysis:** Lion's sign-based updates are structurally complementary to L1's constant-gradient dynamics. Both operate in a regime where gradient *direction* dominates over *magnitude*. The combined signal is particularly clean — no magnitude rescaling distorts the direction. All four splits improved, with the largest gains on cruise (−18.0%) and in_dist (−18.0%). The model still had room to improve at cutoff, suggesting further gains with LR tuning or warmup.
- **Round 7 priority:** Lion LR tuning (lr=2e-4), Lion WD=1e-2 (paper recommendation), physical-space L1, GeGLU+Lion.
- **Artifacts:** `models/model-charliepai2g48h3-edward-lion-optimizer-20260513-001607/metrics.jsonl`

---

## 2026-05-13 01:30 — PR #1728: GeGLU activation (AdamW) — CLOSED (+16.8% worse vs Lion baseline)

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/geglu-activation
- **Hypothesis:** GeGLU gating (multiplicative gate in MLP blocks) routes features by flow regime, mlp_ratio=3 to compensate for 2-projection overhead.
- **Result:** val=101.499 (9 epochs, ~210s/epoch), test=91.436

| Split | val | vs AdamW baseline (101.463) |
|---|---|---|
| single_in_dist | 134.924 | +8.7% |
| geom_camber_rc | 112.496 | −0.2% |
| geom_camber_cruise | 67.405 | −12.0% |
| re_rand | 91.173 | −2.8% |
| **avg** | **101.499** | **+0.04% (essentially baseline)** |

- **Against new Lion baseline (86.938):** +16.8% worse — clearly not competitive.
- **Analysis:** GeGLU's per-epoch overhead (~210s vs 140s with bf16) meant only 9 epochs vs 14 for baseline. The model was still improving at cutoff. The mixed split picture (cruise and re_rand improved, in_dist regressed) suggests GeGLU may help with complex OOD cases but hurts in-distribution. The AdamW context may also be limiting — GeGLU+Lion is a genuinely new hypothesis worth testing.
- **Decision:** Closed; reassigned tanjiro to GeGLU+Lion (PR #1769).
- **Artifacts:** `models/model-charliepai2g48h3-tanjiro-geglu-activation-20260513-002012/metrics.jsonl`

---

## 2026-05-13 01:20 — PR #1724: bf16 mixed precision — MERGED (new best: val=101.463, −0.34%)

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/bf16-mixed-precision
- **Hypothesis:** bf16 autocast reduces epoch time 1.26× (175s→138.5s), gaining +1 epoch (14 vs 13) within the 30-min cap. Small but reliable throughput gain.

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 120.699 | 108.025 |
| geom_camber_rc | 116.096 | 107.822 |
| geom_camber_cruise | 73.667 | 63.152 |
| re_rand | 95.391 | 87.800 |
| **avg** | **101.463** | **91.700** |

- **vs baseline (101.810 L1+n_layers=6):** −0.34% (marginal but positive)
- **Analysis:** A100 is bandwidth-limited on batch=4 small meshes, not compute-bound — speedup was 1.26× rather than the hypothesized 2×. The crucial benefit is infrastructure: bf16 is now the default precision and will accelerate all future experiments.
- **Critical implementation note:** PhysicsAttention temperature division at train.py:120 required `.float()` cast to prevent bf16 overflow. alphonse implemented correctly.
- **Artifacts:** `models/model-charliepai2g48h3-alphonse-bf16-mixed-precision-20260513-001819/metrics.jsonl`

---

## 2026-05-13 01:10 — PR #1673: AdamW eps 1e-8→1e-4 — CLOSED (+13.2% worse)

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/adamw-eps-1e-4
- **Hypothesis:** Larger epsilon raises the adaptive scaling floor, potentially stabilizing AdamW for L1's constant-magnitude gradients.
- **Result:** val=115.267, test=104.XX
- **vs baseline (101.810):** +13.2% worse
- **Analysis:** The increased eps effectively narrows AdamW's adaptive range, hurting rather than helping. AdamW's eps=1e-8 default is well-calibrated. Dead end — AdamW hyperparameter space now fully exhausted (betas, WD, eps, LR, schedule all explored).
- **Conclusion:** Stop tuning AdamW. Focus entirely on Lion-based experiments.

---

## 2026-05-13 00:45 — PR #1670: weight_decay 1e-4 → 5e-4 — CLOSED (+14.9% worse)

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/weight-decay-5e-4
- val=117.013, test=106.801
- **Combined with PR #1649 (WD=0, +3.2% worse), this completes a clean U-curve around the WD=1e-4 baseline.** Over-regularization (5×) costs ~5× more than under-regularization, suggesting baseline is at or very near the regularization sweet spot.
- All splits regressed (single_in_dist +19.1% worst hit; cruise +20.4%). High-magnitude OOD splits need MORE capacity, not less.
- **Conclusion:** WD parameter space fully explored — baseline WD=1e-4 is optimal. Stop sweeping WD.
- **Artifacts:** `target/models/model-weight-decay-5e-4-20260512-230745/metrics.jsonl`

---

## 2026-05-13 00:05 — ROUND 5 BATCH: 5 experiments — ALL CLOSED (PLATEAU CONFIRMED)

**Key cross-experiment finding:** Round 5 confirms ROUND 4 META-INSIGHT (convergence-limited at 30-min budget) extends across optimizer, schedule, and batch-size knobs. ALL 4 returning experiments regressed; the closest miss (alphonse T_max=14 at +0.5%) is pure seed variance (3 seeds: 102.30 / 104.11 / 107.09). **Plateau Protocol triggered — escalating to bolder hypotheses (loss reformulations, mixed precision, optimizer paradigm shifts).**

### PR #1592: Cosine T_max 50 → 14 (align to budget) — CLOSED (+0.5% worse, noise)
- **Student:** charliepai2g48h3-alphonse
- val=102.301, test=92.704 (3 seeds: 102.30 / 104.11 / 107.09; mean +2.64% worse, pure noise)
- Cosine LR alignment to actual epoch budget had no detectable effect
- **Conclusion:** Schedule decay isn't the bottleneck — LR=5e-4 throughout the 11-epoch run is already near-optimal for this regime
- **Artifacts:** `models/model-charliepai2g48h3-alphonse-cosine-tmax-14-20260512-225513/metrics.jsonl`

### PR #1661: CosineAnnealingWarmRestarts T_0=4 T_mult=2 — CLOSED (+2.7% worse)
- **Student:** charliepai2g48h3-fern
- val=104.515, test=94.113
- Schedule played as predicted; the restart at epoch 5 disrupted progress (val_avg jumped 168 → 186 → 150 → ...)
- **Conclusion:** Multi-cycle schedules need budgets that allow full cycle completion. At 11 epochs, single-cycle wins.
- **Artifacts:** `models/model-charliepai2g48h3-fern-cosine-warm-restarts-20260512-225827/metrics.jsonl`

### PR #1671: AdamW β1 0.9 → 0.85 — CLOSED (+5.9% worse)
- **Student:** charliepai2g48h3-edward
- val=107.826, test=98.917
- Both β1 directions (0.85 here, 0.95 in #1622 askeladd) now confirmed worse than default 0.9
- Edward's analysis: oscillations live at macro/epoch scale, not micro/step scale — β1 (4-7 step half-life) can't address them
- **Conclusion:** AdamW betas FULLY EXPLORED. Default (0.9, 0.999) is correct. Stop tuning betas.
- **Artifacts:** `models/model-adamw-beta1-085-20260512-230405/metrics.jsonl`

### PR #1634: Batch size 4 → 8 (via accum_steps=2) — CLOSED (+23.6% worse)
- **Student:** charliepai2g48h3-tanjiro
- val=125.836, test=116.763
- bs=8 OOMed; student used accum_steps=2 as mathematical equivalent (defensible workaround)
- Effective batch=8 halved optimizer steps per epoch (375 → 187); ran out of training steps in the 30-min cap
- **Conclusion:** Direct evidence we are step-count-limited. Increasing effective batch is unambiguously wrong direction.
- **Artifacts:** `models/model-batch-size-8-20260512-231353/metrics.jsonl`

### PR #1384: surf_weight 10 → 25 (stale, never rebased) — CLOSED (no clean result)
- **Student:** charliepai2g48h3-frieren
- Stale 25h+ rebase blocker; original run on pre-L1 baseline produced val=136.78 (not comparable to current 101.81 baseline)
- Closed without merge; hypothesis itself remains untested on current arch — may revisit later from fresh branch
- **NaN-skip fix in evaluate_split was already independently merged in PR #1358**, so frieren's main contribution is preserved.

---

## 2026-05-13 00:20 — ROUND 4 BATCH: 4 experiments — ALL CLOSED

**Key cross-experiment finding:** At the 30-min/~11-13 epoch budget, we are CONVERGENCE-LIMITED. Any change that slows per-epoch progress consistently loses, even if theoretically sound with more training time.

### PR #1649: weight_decay 1e-4 → 0 — CLOSED (+3.2% worse)
- **Student:** charliepai2g48h3-thorfinn
- val=105.079, test=94.684 (best: +3.2% worse, closest miss of round 4)
- val_single_in_dist hit hardest (+5.7%); L1 does NOT dampen high-magnitude gradient flow the way MSE does → WD provides useful regularization
- **Conclusion:** WD=0 is wrong direction. Stronger WD (5e-4) is the natural next step.
- **Artifacts:** `models/model-charliepai2g48h3-thorfinn-weight-decay-zero-20260512-221840/metrics.jsonl`

### PR #1632: dropout=0.1 in attention — CLOSED (+11.8% worse)
- **Student:** charliepai2g48h3-edward
- val=113.838, test=103.151 (best: +11.8% worse)
- Model is underfitting at 11 epochs; adding noise to an underfit model is counterproductive
- val_single_in_dist hit hardest (+16.8%)
- **Conclusion:** Dropout trades convergence speed for stability — wrong tradeoff at 30-min budget.
- **Artifacts:** `models/model-dropout-01-mlp-attn-20260512-220657/metrics.jsonl`

### PR #1622: AdamW betas (0.9,0.999)→(0.95,0.99) — CLOSED (+15.4% worse)
- **Student:** charliepai2g48h3-askeladd
- val=117.483, test=107.268 (best: +15.4% worse)
- β2=0.99 shorter memory amplified noise on L1's frequent sign-flips; caused oscillation bumps at epochs 5 and 10
- **Key insight:** Standard betas (0.9, 0.999) are correct for L1 at our budget. Next direction: looser β1=0.85 with β2=0.999 unchanged (askeladd's own post-mortem suggestion)
- **Artifacts:** `models/model-adamw-betas-transformer-20260512-215824/metrics.jsonl`

### PR #1593 (max_norm=10 arm): Gradient clipping — CLOSED (+15.7% worse)
- **Student:** charliepai2g48h3-nezuko
- max_norm=10: val=117.811, test=105.623 (+15.7% vs baseline)
- max_norm=1: val=112.784, test=100.553 (+10.8% vs baseline) — both arms worse
- **Critical finding:** max_norm=10 DID reduce oscillation jumps 4-28× smaller. But final val was worse, not better. Conclusion: clipping trades convergence speed for stability — wrong tradeoff. Oscillations ARE useful optimization search.
- **Artifacts:** `models/model-charliepai2g48h3-nezuko-gradient-clipping-max10-20260512-220638/metrics.jsonl`

---

## 2026-05-12 23:30 — PR #1595: Huber/SmoothL1 loss beta=1.0 — CLOSED

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/huber-beta1
- **Hypothesis:** Huber loss (smooth L1 for small errors, L1 for large) provides middle ground between MSE and L1; expected ±3% vs L1.
- **Outcome:** **CLOSED** — val=117.773 (+15.7% worse than 101.810), test=107.832 (+17.6% worse).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **117.773** |
| val_single_in_dist | 152.680 (+23.0%) |
| val_geom_camber_rc | 125.599 (+11.4%) |
| val_geom_camber_cruise | 89.558 (+17.0%) |
| val_re_rand | 103.255 (+10.1%) |
| test_avg/mae_surf_p | **107.832** |
| Epochs completed | 11/50 (30-min cap; ~175 s/epoch) |

**Analysis:** Student's mechanistic explanation is correct: with targets normalized to std≈1, a large fraction of node errors fall inside β=1.0, where Huber behaves quadratically (MSE/2) — removing L1's constant-magnitude gradient. The loss ends up "mostly-MSE" for the error regime that matters most. This mirrors the MSE→L1 story in reverse: Huber β=1.0 reverts most of the gradient advantage L1 provided. Student noted β=0.1 might work but the benefit would be marginal at best. Reassigned to CosineAnnealingWarmRestarts (PR #1661).

**Artifacts:** `models/model-charliepai2g48h3-fern-huber-beta1-20260512-211021/metrics.jsonl`

---

## 2026-05-12 23:10 — PR #1525: Fourier positional features L=4 on L1 baseline — CLOSED

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/fourier-pos-L4
- **Hypothesis:** L=4 sinusoidal Fourier features on raw (x,z) coords encode high-frequency surface pressure structure; expected −5–10% on val_avg/mae_surf_p when stacked on L1 baseline.
- **Outcome:** **CLOSED** — val=107.541 (+5.6% vs 101.810), test=97.418 (+6.2% vs 91.708). Improvement did NOT compound with L1.

Note: Fourier L=4 had beaten the OLD MSE baseline by -9.7% (127.57 vs 141.36) with an excellent L=4 vs L=2 ablation. This second run stacked on the new L1 baseline after rebasing.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 10) | **107.541** |
| val_single_in_dist | 139.360 (+12.3%) |
| val_geom_camber_rc | 111.080 (−1.4% ← only split that improved) |
| val_geom_camber_cruise | 79.317 (+3.6%) |
| val_re_rand | 100.407 (+7.0%) |
| test_avg/mae_surf_p | **97.418** (all 4 splits finite — NaN-fix confirmed) |
| Epochs completed | 11/50 (30-min cap; ~176 s/epoch with Fourier dims) |
| Params | 1.182M (vs 0.992M baseline) |

**Analysis:** L1 + n_layers=6 already absorbs the high-frequency representational content that Fourier features were addressing on the MSE baseline. Both mechanisms target the same failure mode (sharp surface-pressure gradients near leading edges / stagnation points), so they overlap rather than stack. Evidence: val_geom_camber_rc was the biggest Fourier win on MSE baseline (-16.7%), and L1 also gave -17.4% on the same split. They target the same OOD-geometry failure mode. Additionally, the larger input (space_dim=18 vs 2) means the model had only 11 epochs instead of 13, with each epoch slightly slower (~176 s vs ~175 s). Reassigned thorfinn to weight_decay=0 experiment (PR #1649).

**Artifacts:** `models/model-charliepai2g48h3-thorfinn-fourier-pos-L4-on-L1-20260512-211919/metrics.jsonl`

---

## 2026-05-12 22:55 — PR #1562: Depth scaling n_layers 6 → 7 — CLOSED

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/n-layers-7
- **Hypothesis:** Deeper model (7 layers vs 6) should improve representational capacity; expected −3–5% on val_avg/mae_surf_p.
- **Outcome:** **CLOSED** — val=154.198 (+51.5% vs current 101.810 baseline), test=NaN (reproducible NaN on test_geom_camber_cruise p-channel, both runs).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 7) | **154.198** |
| val_single_in_dist | 193.128 |
| val_geom_camber_rc | 171.668 |
| val_geom_camber_cruise | 115.258 |
| val_re_rand | 136.737 |
| test_avg/mae_surf_p | **NaN** (test_geom_camber_cruise blowup on p-channel) |
| n_params | 1.365M (vs 0.99M for n_layers=6) |
| Epochs completed | 9/50 (30-min cap, ~205 s/epoch) |

**Analysis:** Three key findings: (1) Budget binding — at ~205 s/epoch for n_layers=7, only 9 epochs fit in 30 min vs 12-13 for n_layers=6, so the bigger model is fundamentally undertrained. (2) Reproducible NaN — test_geom_camber_cruise/mae_surf_p NaN appeared on both independent runs (-201029 and -205108); n_layers=6 doesn't exhibit this. The depth-7 model is more numerically fragile on at least one cruise test sample. (3) Sweet spot — n_layers=6 + mlp_ratio=4 appears to be the Pareto-optimal configuration for our compute budget. Going deeper requires either batch_size increase (fewer steps/epoch = faster epochs) or longer wall-clock budget. Reassigned edward to dropout=0.1 (PR #1632).

**Artifacts:** `models/model-charliepai2g48h3-edward-n-layers-7-20260512-205108/metrics.jsonl`

---

## 2026-05-12 22:55 — PR #1594: Lower LR 5e-4 → 3e-4 — CLOSED

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/lower-lr-3e-4
- **Hypothesis:** L1 constant-magnitude gradients don't shrink at convergence → lower LR helps fine convergence. Expected −3–6%.
- **Outcome:** **CLOSED** — val=119.221 (+17.1% vs 101.810), test=109.040 (+18.9% vs 91.708).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 9) | **119.221** |
| val_single_in_dist | 152.670 (+23.0%) |
| val_geom_camber_rc | 128.660 (+14.2%) |
| val_geom_camber_cruise | 84.604 (+10.5%) |
| val_re_rand | 110.951 (+18.3%) |
| test_avg/mae_surf_p | **109.040** |
| Epochs completed | 11/50 (30-min cap, ~175 s/epoch) |

**Analysis:** Root cause is budget-driven undertraining. With T_max=50 and only 11 epochs in the budget, the cosine LR barely decays regardless of initial value. At lr=3e-4, the model simply makes smaller steps toward the same solution the baseline reaches in 13 epochs at lr=5e-4. Lower LR is only useful if paired with a schedule that decays within budget (T_max=14) — see alphonse PR #1592 which is testing exactly that. Student's analysis was spot-on. Reassigned to batch_size=8 experiment (PR #1634).

**Artifacts:** `models/model-charliepai2g48h3-tanjiro-lower-lr-3e-4-20260512-210951/metrics.jsonl`

---

## 2026-05-12 22:55 — PR #1593: Gradient clipping max_norm=1.0 — SENT BACK

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/gradient-clipping
- **Hypothesis:** L1 gradient oscillations cause val instability → clipping at max_norm=1.0 stabilizes training.
- **Outcome:** **SENT BACK** — val=112.784 (+10.8% vs 101.810), test=100.553 (+9.6% vs 91.708). Oscillations NOT eliminated (epoch spikes at ep 4, 8, 10 persisted). Hypothesis: max_norm=1.0 is too aggressive — L1 gradients are constant magnitude ±1/±surf_weight per element, so clip threshold of 1.0 effectively throttles learning rate. Sent back to try max_norm=10.0.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **112.784** |
| val_single_in_dist | 143.901 (+15.9%) |
| val_geom_camber_rc | 129.979 (+15.3%) |
| val_geom_camber_cruise | 80.735 (+5.4%) |
| val_re_rand | 96.523 (+2.9%) |
| test_avg/mae_surf_p | **100.553** |
| Epochs completed | 11/50 (30-min cap) |

**Analysis:** clip=1.0 likely clips EVERY update given ~1.18M params and L1 gradients. With 1.18M params each producing a ±1 or ±surf_weight gradient, the expected gradient L2 norm is ~sqrt(n_params * mean_grad^2) >> 1.0. The oscillations persisted, confirming this isn't reducing gradient spikes — it's just capping useful parameter updates. max_norm=10.0 is the natural next test: loose enough to only clip true outlier spikes, tight enough to have any stabilization effect.

**Artifacts:** `models/model-charliepai2g48h3-nezuko-gradient-clipping-20260512-211112/metrics.jsonl`

---

## 2026-05-12 22:30 — PR #1563: EMA weights (decay=0.999) for val/test eval — CLOSED

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/ema-weights
- **Hypothesis:** EMA of model weights (decay=0.999) improves val/test metrics via implicit model ensemble over training trajectory.
- **Outcome:** **CLOSED** — val=143.7075, +41.1% worse than current baseline 101.810. test_avg=NaN.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **143.7075** |
| val_single_in_dist | 182.77 |
| val_geom_camber_rc | 155.00 |
| val_geom_camber_cruise | 112.89 |
| val_re_rand | 124.17 |
| test_avg/mae_surf_p | **NaN** (EMA-averaged weights → NaN on test_geom_camber_cruise p-channel) |
| Epochs completed | 11/50 (30-min cap) |

**Analysis:** EMA cold-start drag is the root cause. With decay=0.999, the EMA buffer half-life is ~693 steps. ModelEMA initialized at random weights and was updated only during training, so the EMA buffer was dominated by early-epoch random-weight values throughout the entire 11-epoch run. The EMA consistently lagged well behind the live model (which reached 101.810 without EMA). Additionally, the EMA-averaged weights landed in a numerically degenerate region for at least one sample in test_geom_camber_cruise, producing NaN on the p-channel — the live model handled the same sample without NaN.

EMA would require: (a) initializing the EMA buffer from post-warmup live weights (not random init), OR (b) training for 100+ epochs so the buffer stabilizes. Neither is feasible at 30-min wall-clock. Closed; askeladd reassigned to AdamW betas experiment (PR #1622).

---

## 2026-05-12 21:10 — PR #1358: L1 (MAE) loss in normalized space — MERGED

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/l1-surface-pressure-loss
- **Hypothesis:** L1 loss directly optimizes the ranking metric (MAE); expected −2–5% on val_avg/mae_surf_p.
- **Outcome:** **MERGED — new baseline 101.810 (−20.5% vs 128.127).** Far exceeded expectations.

| Metric | Value vs baseline 128.127 |
|---|---|
| val_avg/mae_surf_p (best, ep 13) | **101.810** (−20.5%) |
| val_single_in_dist | 124.150 (−22.3% vs 159.746) |
| val_geom_camber_rc | 112.699 (−17.4% vs 136.513) |
| val_geom_camber_cruise | 76.570 (−25.3% vs 102.432) |
| val_re_rand | 93.820 (−17.6% vs 113.819) |
| test_avg/mae_surf_p | **91.708** (first finite test result!) |
| test_single_in_dist | 110.726 |
| test_geom_camber_rc | 99.692 |
| test_geom_camber_cruise | 66.879 |
| test_re_rand | 89.536 |
| Epochs completed | 14/50 (30-min cap) |
| Peak VRAM | 42.1 GB |

**Analysis:** The loss function switch from MSE to L1 (MAE) produced the single largest improvement seen in this research program. The result is −20.5% better despite using the old arch (n_layers=5, mlp_ratio=2) — showing the loss function dominates architecture in importance here. This makes sense: L1 directly optimizes the metric we evaluate with. The merged train.py stacks L1 + n_layers=6 + mlp_ratio=4; a confirmed stacked run will likely improve further.

Alphonse also included a `train.py::evaluate_split` NaN-fix (filter non-finite GT samples before scorer call), making test metrics finite for the first time: test_avg/mae_surf_p = 91.708 across all 4 test splits.

**Artifacts:** `models/model-l1-loss-e50-20260512-195549/metrics.jsonl`

---

## 2026-05-12 21:05 — PR #1566: n_head 4 → 8

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/n-head-8
- **Hypothesis:** Doubling attention heads to 8 diversifies slice patterns; expected −2–4%.
- **Outcome:** **CLOSED** — +15.7% worse (148.280 vs 128.127), per-epoch cost +43%.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 8) | **148.280** |
| Epochs completed | 9/50 (30-min cap) |
| Per-epoch time | ~222 s (vs ~156 s baseline) |

**Analysis:** n_head=8 launches more but smaller softmax/matmul kernels, creating overhead that costs +43% per epoch (not +15% as predicted). Only 9 epochs fit in 30 min, and the val curve was still oscillating at epoch 8. Under fixed wall-clock, n_head=4 dominates. Future attention-diversity experiments should use slice_num changes rather than head count.

---

## 2026-05-12 21:05 — PR #1401 (arm 2): Warmup + cosine T_max=15

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/warmup-cosine-lr1e-3
- **Hypothesis:** 3-ep warmup to peak lr=1e-3 with T_max=15 aligned to budget; expected beat baseline.
- **Outcome:** **CLOSED** — val=133.448, +4.15% worse than old baseline 128.127 (and far worse than new baseline 101.810).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **133.448** |
| val_geom_camber_cruise | 102.355 (≈ old baseline) |
| Epochs completed | 11/15 (30-min cap, epochs ~175 s) |

**Analysis:** Peak lr=1e-3 is too hot for the wider/deeper model. The n_layers=6 arch has ~35% longer epochs (~175 s vs ~130 s for old arch), so even T_max=15 can't complete in 30 min. The schedule's low-LR tail (epochs 12-15) was never executed. This direction is exhausted — reassigned to lower LR (3e-4) approach.

---

## 2026-05-12 21:05 — PR #1370: slice_num 64 → 128

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/slice-128
- **Hypothesis:** Doubling physics-attention slices improves mesh resolution; expected −1–3%.
- **Outcome:** **CLOSED** — val=150.909, +17.8% worse than new baseline 101.810. Also modified data/scoring.py (read-only).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 10) | **150.909** |
| test_avg/mae_surf_p (NaN-fix applied) | **137.481** |
| Epochs completed | 11/50 (30-min cap) |
| Per-epoch time | ~173 s |

**Analysis:** Doubling slices costs ~12% more per epoch AND the result is significantly worse. The slice bottleneck hypothesis is not supported within the 30-min budget. Additionally, fern modified data/scoring.py (read-only per program.md) — the equivalent train.py workaround was already merged in PR #1358. Closed; fern reassigned to Huber loss.

---

## 2026-05-12 19:05 — PR #1408: MLP expansion ratio 2 → 4 (canonical transformer recipe)

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/mlp-ratio-4
- **Hypothesis:** Doubling `mlp_ratio` 2 → 4 increases feedforward capacity; canonical transformer recipe, expected −1–3% on val_avg/mae_surf_p.
- **Outcome:** **MERGED — new baseline.**

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 13) | **141.356** |
| val_single_in_dist/mae_surf_p | 171.424 |
| val_geom_camber_rc/mae_surf_p | 159.804 |
| val_geom_camber_cruise/mae_surf_p | 104.607 |
| val_re_rand/mae_surf_p | 129.589 |
| test_avg/mae_surf_p | NaN (cruise bug) |
| test mean (3 finite splits) | ~139.51 |
| Epochs completed | 13/50 (30-min cap) |
| Peak VRAM | 52.2 GB |
| Params | 0.99M |

**Analysis:** First terminal result on this branch. 13 epochs in 30 min (≈150 s/epoch). Best val came on epoch 13, meaning the model was still learning at cutoff — more epochs would likely improve further. The cruise test split NaN is a scorer bug (GT sample 20 has -inf pressure), not a model failure; 3 finite test splits give a consistent 139.5 mean. **mlp_ratio=4 is now the default in train.py.**

**Artifacts:** `models/model-charliepai2g48h3-thorfinn-mlp-ratio-4-20260512-175522/metrics.jsonl`

---

## 2026-05-12 20:30 — PR #1401: Warmup (3ep) + cosine, peak LR 1e-3

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/warmup-cosine-lr1e-3
- **Hypothesis:** 3-epoch linear warmup to peak lr=1e-3 then cosine decay improves convergence vs flat lr=5e-4; expected −2–4% on val_avg/mae_surf_p.
- **Outcome:** **SENT BACK** for rebase + cosine T_max fix — result is promising (1.4% off new baseline on old arch) but comparison isn't clean.

| Metric | Value vs baseline 128.127 |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **129.900** (+1.4% worse) |
| val_single_in_dist | 159.67 (≈159.75 baseline) |
| val_geom_camber_rc | 137.35 (vs 136.51 baseline) |
| val_geom_camber_cruise | 104.21 (vs 102.43 baseline) |
| val_re_rand | 118.37 (vs 113.82 baseline) |
| test_avg (3 finite splits) | **119.88** |
| Epochs completed | 14/50 (30-min cap) |

**Analysis:** Ran on old arch (mlp_ratio=2, n_layers=5). On old arch it achieved 129.900 which is very close to the new baseline 128.127 — very promising. Key issue: T_max=50 means cosine barely decays in 14 epochs (LR still ~9e-4 at epoch 14), so the full benefit of the schedule was never realized. Sent back with instructions to rebase on new baseline and use `--epochs 15` so T_max=15 matches the actual ~12–14 epoch budget.

**Artifacts:** `models/model-warmup-cosine-lr1e-3-20260512-185810/metrics.jsonl`

---

## 2026-05-12 20:25 — PR #1384: Surface weight 10 → 25

- **Student:** charliepai2g48h3-frieren
- **Branch:** charliepai2g48h3-frieren/surf-weight-25
- **Hypothesis:** Increasing surf_weight 10 → 25 improves surface pressure focus; expected −1–4% on val_avg/mae_surf_p.
- **Outcome:** **SENT BACK** for rebase + rerun on new baseline — ran on old arch (mlp_ratio=2, n_layers=5).

| Metric | Value vs new baseline 128.127 |
|---|---|
| val_avg/mae_surf_p (best, ep 14) | **136.779** (+6.8% worse) |
| val_single_in_dist | 160.44 |
| val_geom_camber_rc | 167.75 |
| val_geom_camber_cruise | 96.77 |
| val_re_rand | 122.16 |
| test_avg (4 finite splits, NaN-fix applied) | **122.95** |
| Epochs completed | 14/50 (30-min cap) |

**Analysis:** Ran on old arch (mlp_ratio=2, n_layers=5). Frieren also included a train.py NaN-fix in evaluate_split (skip non-finite GT samples before model/accumulator) which produced clean test metrics (all finite!). Sent back for rebase onto new baseline (n_layers=6, mlp_ratio=4) + rerun with `--surf_weight 25`. The surf_weight hypothesis is still untested against the current 128.127 baseline — keeping the NaN-fix in the rerun.

**Artifacts:** `models/model-surf-weight-25-20260512-185910/metrics.jsonl`

---

## 2026-05-12 20:25 — PR #1523: Channel-weighted loss p×3

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/channel-weighted-pressure-p3x
- **Hypothesis:** Weighting surface pressure channel 3× in loss improves p MAE; expected improvement on val_avg/mae_surf_p.
- **Outcome:** **CLOSED** — 18.8% worse than current baseline.

| Metric | Value vs baseline 128.127 |
|---|---|
| val_avg/mae_surf_p (best, ep 13) | **152.216** (+18.8% worse) |
| val_single_in_dist | 197.980 (+23.8%) |
| val_geom_camber_rc | 156.704 (−1.9% better) |
| val_geom_camber_cruise | 119.921 (+17.1%) |
| val_re_rand | 134.259 (+18.0%) |
| Epochs completed | 13/50 (30-min cap) |

**Analysis:** Weighting p×3 in normalized-space MSE doesn't work because normalization already equalizes per-channel variance. The effective surf_weight on p went from 10→30, dominating the loss and hurting Ux/Uy/volume prediction. val_single_in_dist (+23.8%) and cruise (+17.1%) suffered worst. The one marginal win (geom_camber_rc −1.9%) is swamped by losses elsewhere. Clear dead end.

**Artifacts:** `models/model-charliepai2g48h3-edward-channel-weighted-p3x-20260512-190802/metrics.jsonl`

---

## 2026-05-12 20:25 — PR #1363: EMA weights (stale)

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/ema-weights-for-eval
- **Hypothesis:** EMA of weights (decay=0.999) for val/test improves generalization.
- **Outcome:** **CLOSED (stale)** — draft PR never produced results. Reassigned fresh.

---

## 2026-05-12 19:30 — PR #1392 (follow-up): n_layers 5 → 6 — MERGED

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/deeper-transolver-6layers
- **Hypothesis:** Moderate depth increase (n_layers 5→6) retains OOD geometry benefit seen in n_layers=8 but fits more epochs in the 30-min cap.
- **Outcome:** **MERGED — new baseline 128.127.**

| Metric | Value vs baseline 141.356 |
|---|---|
| val_avg/mae_surf_p (best, ep 12) | **128.127** (−9.4% better) |
| val_single_in_dist | 159.746 (−6.8% vs 171.424) |
| val_geom_camber_rc | 136.513 (−14.6% vs 159.804) |
| val_geom_camber_cruise | 102.432 (−2.1% vs 104.607) |
| val_re_rand | 113.819 (−12.2% vs 129.589) |
| test_avg (3 finite splits) | **127.68** |
| Epochs completed | 12/50 (30-min cap) |
| Per-epoch time | ~156 s |
| Peak VRAM | 49.6 GB |
| Params | 0.78M |

**Analysis:** Biggest single-experiment win so far. The moderate depth increase (5→6) gives broad improvement across all splits, with val_geom_camber_rc (−14.6%) and val_re_rand (−12.2%) benefiting most — the extra layer helps with OOD geometry and Reynolds number variation. Best val came at epoch 12 (= final epoch), still learning at cutoff. train.py now defaults to n_layers=6, mlp_ratio=4.

**Artifacts:** `models/model-charliepai2g48h3-nezuko-deeper-transolver-6layers-20260512-191742/metrics.jsonl`

---

## 2026-05-12 19:10 — PR #1392: Deeper Transolver n_layers 5 → 8

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/deeper-transolver-8layers
- **Hypothesis:** Increasing n_layers 5 → 8 (+60% depth) would improve representation through more iterative refinement; expected −2–4% on val_avg/mae_surf_p.
- **Outcome:** **SENT BACK** for a more moderate depth (n_layers=6) — 1.6% worse on val at 30-min cap, but promising signal on OOD geometry (val_geom_camber_rc 5.4% better) and trajectory still descending at cutoff.

| Metric | Value vs baseline 141.356 |
|---|---|
| val_avg/mae_surf_p (best, ep 9) | **143.650** (+1.6% worse) |
| val_single_in_dist/mae_surf_p | 179.503 (+4.7% worse vs 171.42) |
| val_geom_camber_rc/mae_surf_p | **151.158** (−5.4% better vs 159.80) |
| val_geom_camber_cruise/mae_surf_p | 118.512 (+13.3% worse vs 104.61) |
| val_re_rand/mae_surf_p | 125.428 (−3.2% better vs 129.59) |
| test_avg (corrected, Inf-y masked) | **130.23** (−6.6% better vs ~139.51) |
| Epochs completed | 9/50 (30-min cap) |
| Per-epoch time | ~206 s (vs ~150 s baseline) |
| Peak VRAM | 64.5 GB |
| Params | 1.03M |

**Analysis:** Depth-8 lost on val (-1.6%) but won on test_corrected (-6.6%) and val_geom_camber_rc (-5.4%). Two oscillation epochs (2 and 8) indicate mild instability with the deeper model + AdamW + flat 5e-4 LR. The trajectory was still steeply descending at cutoff (162.7 → 143.6 in the final epoch). The 65% per-epoch overhead is too costly at fixed 30-min cap. **Sending back with feedback to try n_layers=6** — a middle-ground depth that should fit ~12 epochs and may retain the OOD-geometry benefit.

**Artifacts:** `models/model-charliepai2g48h3-nezuko-deeper-transolver-8layers-20260512-175521/metrics.jsonl`

---

## 2026-05-12 19:05 — PR #1366: Wider Transolver n_hidden 128 → 192

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/wider-transolver-192
- **Hypothesis:** Increasing n_hidden 128 → 192 (+50% width) would improve representational capacity; expected −2–5% on val_avg/mae_surf_p.
- **Outcome:** **CLOSED** — 6.3% worse than thorfinn at the same wall-clock budget.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 10) | 150.323 |
| val_single_in_dist/mae_surf_p | 181.449 |
| val_geom_camber_rc/mae_surf_p | 163.411 |
| val_geom_camber_cruise/mae_surf_p | 121.317 |
| val_re_rand/mae_surf_p | 135.114 |
| test_avg/mae_surf_p | NaN (cruise bug) |
| Epochs completed | 10/50 (30-min cap) |
| Per-epoch time | ~185 s (vs ~150 s for thorfinn) |
| Peak VRAM | 58.0 GB |
| Params | 1.47M |

**Analysis:** Width scaling lost to mlp_ratio scaling at the 30-min budget. The wider model runs ~23% slower per epoch, netting only 10 epochs vs 13 for thorfinn. The training curve was still monotonically descending at epoch 10 — fundamentally under-converged. The 30-min cap makes capacity-scaling via width non-competitive unless paired with a step-efficiency gain (e.g. larger batch, fewer layers, faster arch). Closing this; edward redirected to a fresh direction.

**Artifacts:** `models/model-wider-192-20260512-175551/metrics.jsonl`
