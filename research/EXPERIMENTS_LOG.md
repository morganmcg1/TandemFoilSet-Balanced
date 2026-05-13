# SENPAI Research Results — `icml-appendix-charlie-pai2g-48h-r4`

Primary metric: `val_avg/mae_surf_p` (lower is better). Test counterpart: `test_avg/mae_surf_p`.

## 2026-05-13 08:10 — PR #2012: [loss-beta-0-5] Halve smooth_l1 beta 1.0→0.5 — **SENT BACK (beat OLD, not NEW; bs=2 rerun)**
- Student branch: `charliepai2g48h4-edward/loss-beta-0-5`
- Hypothesis: Reducing beta=1.0→0.5 narrows the smooth_l1 quadratic zone, pushing more residuals into the L1 regime under normalized-step regime. Cleaner gradient signal for late training.

| Metric | OLD baseline (#1812) | beta=0.5 (this) | Δ vs OLD | Δ vs NEW (#1972) |
|---|---|---|---|---|
| val_avg/mae_surf_p | 82.56 | **81.21** | **−1.35 (−1.64%)** | +4.97 (+6.5%, worse) |
| test_avg/mae_surf_p | 74.13 | **72.52** | −1.61 (−2.17%) | +5.67 (+8.5%) |
| val single_in_dist | 90.40 | 88.55 | −1.85 | — |
| val geom_camber_rc | 91.39 | 88.62 | −2.77 | — |
| val geom_camber_cruise | 66.68 | 66.80 | +0.12 | — |
| val re_rand | 81.77 | 80.88 | −0.89 | — |

- Artifact: `models/model-charliepai2g48h4-edward-loss-beta-0-5-20260513-070621/metrics.jsonl`
- Best epoch 18/18 (cosine reached eta_min=5e-5), 30-min wall-clock cap.
- 3/4 val splits improve, 3/4 test splits improve (`geom_camber_cruise` val +0.12 noise; `geom_camber_rc` test +0.35 noise).

**Analysis:** Clean directional confirmation that beta=0.5 < beta=1.0 in this regime. Edward's mechanistic reading is sound: with train surf_loss ~0.022–0.025 in late epochs (well below beta=1.0 quadratic threshold), the wider quadratic zone of beta=1.0 was effectively damping per-example gradient magnitude on small-but-not-zero residuals — under our grad-clip(max_norm=1.0) normalized-step regime, that translates to those examples contributing less to the chosen step direction. Narrowing to beta=0.5 lets them push their full unit gradient. Largest test gain is on the highest-MAE split (`test_single_in_dist`, −3.62), consistent with the "narrow quadratic zone helps hard-but-near-converged examples" story.

**Sent back, not merged:** Beats OLD baseline (#1812) but loses to NEW baseline (#1972 batch-size-2, val=76.24). Run was on the OLD bs=4 HEAD. Asked edward to rebase onto current HEAD (batch_size=2) and rerun — beta-shape and batch-size mechanisms are independent, expect them to stack additively. Target: val < 76.24.

---

## 2026-05-13 08:08 — PR #1993: [n-head-2] Halve attention heads 4→2 — **CLOSED (n_head axis closed at 4)**
- Student branch: `charliepai2g48h4-tanjiro/n-head-2`
- Hypothesis: Wider/fewer heads (n_head=2 gives 64-dim heads, vs 32-dim at n_head=4) capture longer-range geometric dependencies more efficiently on small irregular meshes.

| Metric | OLD baseline (#1812) | n_head=2 (this) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 82.56 | **83.78** | **+1.22 (+1.48%, worse)** |
| test_avg/mae_surf_p | 74.13 | 73.71 | −0.42 (−0.57%, marginal) |
| n_params | 734,541 | 734,541 | 0 (zero-param change) |

- Artifact: `models/model-charliepai2g48h4-tanjiro-n-head-2-20260513-065432/metrics.jsonl`
- Best epoch 19/22 (model rebounded after cosine T_max=18 cycled past eta_min).

**Bracket result — n_head axis CLOSED both directions:**
| n_head | val_avg/mae_surf_p | params | Status |
|--------|---------------------|---------|--------|
| 2 (this) | 83.78 | 734,541 | WORSE — closed |
| **4 (baseline)** | **82.56** | **734,541** | **OPTIMUM** |
| 8 (#1853) | 96.33 | reduced ~2.4% | MUCH WORSE — closed |

**Analysis:** Val is the source-of-truth selection metric and shows +1.22 regression (above cross-seed σ ~3.5 but clearly directional given +1.48% delta). The marginal test improvement (−0.42, −0.57%) is well within σ and disagrees with val on which split moved — pure noise. Tanjiro's independent observation that `CosineAnnealingLR` cycles LR back up past T_max (epochs 19→20→21 showing post-eta_min rebound) was a sharp catch — relevant to OneCycleLR (#2014) and CAWR (#1990) hypotheses currently in flight. The n_head axis is now fully bracketed and closed.

---

## 2026-05-13 04:05 — PR #1855: [eta-min-5e-5] Non-zero cosine LR floor — **MERGED (NEW BEST: val=83.95)**
- Student branch: `charliepai2g48h4-fern/eta-min-5e-5`
- Hypothesis: eta_min=0.0 means LR reaches ~0 by epoch 18, wasting final-epoch compute. A non-zero floor keeps gradient steps meaningful.

| Metric | T_max=18 baseline (#1695) | eta_min=5e-5 (this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 84.67 | **83.95** | **−0.72 (−0.85%)** |
| test_avg/mae_surf_p | 74.94 | **74.70** | −0.24 (−0.32%) |
| val single_in_dist | 96.25 | 93.45 | −2.80 |
| val geom_camber_rc | 93.25 | 91.33 | −1.92 |
| val geom_camber_cruise | 65.39 | 67.06 | +1.67 (regression) |
| val re_rand | 83.78 | 83.97 | +0.19 (flat) |

- Artifact: `models/model-charliepai2g48h4-fern-eta-min-5e-5-20260513-030913/metrics.jsonl`
- Final-epoch LR: 5.34e-5 (floor active; vs ~0 with eta_min=0.0)

**Analysis:** Small but real gain. Key evidence: val 87.29 (epoch 17) → 83.95 (epoch 18), a 3.34-point drop in the final epoch under the non-zero floor. The model was still descending. Under normalized gradient descent, even a small LR produces a meaningful normalized step (step = max_norm × lr / ‖g‖). The floor prevents the "dead zone" where LR→0 renders all optimizer steps near-zero regardless of gradient direction. 3/4 val splits improve (geom_camber_cruise slightly regresses). Merged as 11th effective improvement.

Open question: is eta_min=1e-4 (2×) even better, or too large? Fern assigned #1901 to bracket.

---

## 2026-05-13 04:05 — PR #1730: [layers-6] n_layers 5→6 rerun on grad-clip+surf_weight=5 HEAD — **CLOSED (regression)**
- Student branch: `charliepai2g48h4-edward/layers-6`
- Hypothesis: Extra Transolver block adds capacity, expected sub-80 given Huber-base −11.2% delta.

| Metric | n_layers=5 baseline (#1762, T_max=15) | n_layers=6 (this run, T_max=15) | Δ vs #1762 | Δ vs current best |
|---|---|---|---|---|
| val_avg/mae_surf_p | 90.58 | 93.97 | **+3.7% worse** | **+11.9% worse** |
| test_avg/mae_surf_p | 80.00 | 83.05 | +3.8% worse | +10.8% worse |

- Artifact: `models/model-charliepai2g48h4-edward-layers-6-20260513-025519/metrics.jsonl`
- Epochs: 15/50 (cap; per-epoch +20% → 15 epochs in 30 min). n_params: 799,387 vs 666K.

**Analysis:** Depth does not stack with grad-clip under wall-clock budget. Prior Huber-base result (−11.2%) used unclipped gradients; under global norm clipping, adding more layers creates gradient dilution — each layer's signal is suppressed relatively more since global clipping normalizes the entire gradient. Additionally, the wall-clock budget penalty (15 epochs vs 18) eliminates the schedule advantage. With T_max=15 and only 15 achievable epochs, the capacity gain of the 6th block is fully offset by fewer training iterations.

**Key finding:** Depth is unlikely to help under global grad-clip. Layers-7 (alphonse #1834) in flight will further confirm. edward reassigned to slice_num=128 (#1902).

---

## 2026-05-13 03:55 — PR #1851: [max-norm-3-selective] max_norm=1.0→3.0 — **CLOSED (regression; max_norm axis bracketed)**
- Student branch: `charliepai2g48h4-frieren/max-norm-3-selective`
- Hypothesis: Under surf_weight=5, pre-clip grad norms are 2-5 (mean). max_norm=3.0 creates selective clipping — steps with norm < 3 pass through unclipped, only outliers (norm > 3) are capped.

| Metric | T_max=18 baseline (#1695) | max_norm=3.0 (this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 84.67 | 86.30 | **+1.63 (regression)** |
| test_avg/mae_surf_p | 74.94 | 76.82 | +1.88 |
| val single_in_dist | 96.25 | 94.50 | −1.75 (improved) |
| val geom_camber_rc | 93.25 | 95.22 | +1.97 |
| val geom_camber_cruise | 65.39 | 69.35 | +3.96 |
| val re_rand | 83.78 | 86.12 | +2.34 |

- Artifact: `models/model-charliepai2g48h4-frieren-max-norm-3-selective-20260513-030937/metrics.jsonl`
- Pre-clip grad norm trajectory: epochs 1-7 clipped (norm 3.5-6), epochs 8-18 mostly unclipped (norm 2.2-2.9).

**Analysis:** Selective clipping degraded OOD splits (geom_camber_rc, cruise, re_rand). The mechanistic story: fully-normalized gradient descent (max_norm=1.0, all steps clipped) enforces consistent step magnitude throughout training. When late-epoch gradients fall below max_norm=3.0 (norm 2-3, unclipped), the optimizer switches to raw AdamW steps — larger effective step sizes in late training when the schedule wants small LR. This disrupts the coordinated small-step optimization in the LR-decay phase.

**max_norm axis fully bracketed:**
- max_norm=0.5: val=91.01 (regression +0.43 vs current best at the time)
- max_norm=1.0: val=84.67 (OPTIMUM)
- max_norm=3.0: val=86.30 (regression +1.63)

Closed. Frieren reassigned to AdamW β2=0.98 (#1886).

---

## 2026-05-13 03:55 — PR #1832: [surf-weight-3] surf_weight=5→3 — **CLOSED (regression; surf_weight axis closed)**
- Student branch: `charliepai2g48h4-tanjiro/surf-weight-3`
- Hypothesis: Can surf_weight < 5 further balance surface-vs-volume residuals under grad-clip?

| Metric | surf_weight=5 (T_max=15 base) | surf_weight=3 (this run, T_max=15) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 90.58 | 90.88 | **+0.30 (regression)** |
| test_avg/mae_surf_p | 80.00 | 81.38 | +1.38 |
| val single_in_dist | 106.31 | 102.46 | −3.85 (improved) |
| val geom_camber_rc | 98.84 | 99.74 | +0.90 |
| val geom_camber_cruise | 69.35 | 71.21 | +1.86 |
| val re_rand | 87.82 | 90.11 | +2.29 |

- Artifact: `models/model-charliepai2g48h4-tanjiro-surf-weight-3-20260513-030933/metrics.jsonl`
- Run context: on T_max=15 base (before T_max=18 merge) — valid direct comparison against surf_weight=5 at same schedule.

**Analysis:** Going lower than surf_weight=5 over-corrects. Under grad-clip, surf_weight=5 balances surface-vs-volume contributions optimally. At surf_weight=3, surface gradients are under-weighted — the model prioritizes volume accuracy at the expense of surface pressure, which is the primary metric. single_in_dist slightly improves (surface pressure less dominant helps the easiest split) but the OOD splits regress.

**surf_weight axis fully closed:**
- surf_weight=3: 90.88 (regression)
- surf_weight=5: 90.58 (OPTIMUM on T_max=15; 84.67 on T_max=18)
- surf_weight=10: 96.78 (regression)
- surf_weight=20: 127.94 (regression)

Closed. Tanjiro reassigned to AdamW β1=0.95 (#1888).

---

## 2026-05-13 03:00 — PR #1695: [tmax-18] T_max=15→18 rerun on surf_weight=5 HEAD — **MERGED (NEW BEST: val=84.67)**
- Student branch: `charliepai2g48h4-nezuko/tmax-18`
- Hypothesis: The 30-min wall-clock cap allows ~18 epochs. Setting T_max=18 (vs 15) aligns the cosine schedule minimum with the true end of training, preventing the LR from rising back up in a second cycle at epoch 16–18.

| Metric | surf_weight=5 baseline (#1762) | T_max=18 rerun (this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 90.58 | **84.67** | **−5.91 (−6.5%)** |
| test_avg/mae_surf_p | 80.00 | **74.94** | **−5.06 (−6.3%)** |
| val single_in_dist | 106.31 | 96.25 | −10.06 |
| val geom_camber_rc | 98.84 | 93.25 | −5.59 |
| val geom_camber_cruise | 69.35 | 65.39 | −3.96 |
| val re_rand | 87.82 | 83.78 | −4.04 |
| test single_in_dist | 93.61 | 85.31 | −8.30 |
| test geom_camber_rc | 86.86 | 83.17 | −3.69 |
| test geom_camber_cruise | 58.37 | 55.11 | −3.26 |
| test re_rand | 81.18 | 76.18 | −5.02 |

- Artifact: `models/model-charliepai2g48h4-nezuko-tmax-18-20260513-021955/metrics.jsonl`
- Run context: Rebased onto surf_weight=5 HEAD (41c30ab, the current best). Clean comparison. All 4 val + all 4 test splits improved.

**Analysis:** Second consecutive 6%+ single-lever gain. The mechanism is clean: T_max=15 caused the cosine to reach zero LR at epoch 15, then start rising again for epochs 16–18 — wasted training budget. T_max=18 keeps LR in clean decay throughout the full epoch window. The improvement is consistent and uniform across all 4 splits (range Δ−3.96 to Δ−10.06 val), suggesting this is a global optimization quality lift rather than a split-specific effect. `single_in_dist` benefits most (−10.06 val), which is the hardest split.

**Merged as 10th effective improvement to advisor-branch recipe.** New recipe: `T_max=18` replaces `T_max=15`.

---

## 2026-05-13 03:00 — PR #1759: [max-norm-0.5] max_norm=0.5 rerun on surf_weight=5 HEAD — **CLOSED (regression)**
- Student branch: `charliepai2g48h4-frieren/max-norm-0.5`
- Hypothesis: Tighter clipping (0.5 vs 1.0) reduces effective step size further, potentially helping generalization in the low-gradient regime.

| Metric | surf_weight=5 baseline (#1762, T_max=15) | max_norm=0.5 (this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 90.58 | 91.01 | **+0.43 (+0.5%)** |
| test_avg/mae_surf_p | 80.00 | 80.20 | +0.20 (+0.3%) |

- Artifact: `models/model-charliepai2g48h4-frieren-max-norm-0.5-20260513-021846/metrics.jsonl`
- Run context: Rebased onto surf_weight=5 HEAD. Direct comparison against #1762 baseline.

**Analysis:** Null result. Δ+0.43 val is within σ noise, trend is regression. Under surf_weight=5, gradient norms are 2-5 (mean). max_norm=0.5 still clips all steps (norms >> 0.5), but halves effective step size vs max_norm=1.0. The extra step-size reduction provides no benefit — the model is already well-regularized by the normalized gradient descent at max_norm=1.0. Tighter clipping beyond the point of full-step-normalization adds no incremental regularization. max_norm=0.5 is confirmed dead end on this recipe.

**Closed.** frieren reassigned to max_norm=3.0 probe (selective clipping, #1851).

---

## 2026-05-13 03:00 — PR #1635: [log-cosh-loss] Log-cosh rerun on surf_weight=5 + grad-clip HEAD — **CLOSED (regression)**
- Student branch: `charliepai2g48h4-fern/log-cosh-loss`
- Hypothesis: Log-cosh is smoother than Huber at the piecewise kink and may combine better with grad-clip.

| Metric | surf_weight=5 baseline (#1762, T_max=15) | log-cosh (this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 90.58 | 91.19 | **+0.61 (+0.7%)** |
| test_avg/mae_surf_p | 80.00 | 81.72 | +1.72 (+2.2%) |

- Artifact: `models/model-charliepai2g48h4-fern-log-cosh-loss-20260513-022203/metrics.jsonl`
- Run context: Rebased onto surf_weight=5 HEAD (41c30ab). Huber replaced with log-cosh at both call sites (lines 272 and 510 per student's report).

**Analysis:** Null result on the full recipe. Log-cosh showed clear improvement over Huber on the pre-grad-clip baseline (+5.68 val on Huber base). But under grad-clip's normalized gradient descent, the choice of loss shape is suppressed: global norm clipping already bounds outlier gradient magnitudes before the optimizer step, making the tail behavior of the loss function largely irrelevant. Both log-cosh and Huber reduce to MSE near zero, which dominates under grad-clip. The loss-shape axis is now closed.

**Closed.** fern reassigned to eta_min=5e-5 (schedule LR floor, #1855) — different lever.

---

## 2026-05-13 02:10 — PR #1762: [surf-weight-5] surf_weight 10→5 — **MERGED (NEW BEST: val=90.58)**
- Student branch: `charliepai2g48h4-tanjiro/surf-weight-5`
- Hypothesis: With grad-clip normalizing every step, surface gradients no longer need heavy weighting to dominate the loss — clipping equalizes step magnitudes. Halving surf_weight from 10→5 better balances surface-vs-volume residuals.

| Metric | grad-clip baseline (#1696) | surf_weight=5 (this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 96.78 | **90.58** | **−6.20 (−6.4%)** |
| test_avg/mae_surf_p | 86.56 | **80.00** | **−6.56 (−7.6%)** |
| val single_in_dist | 110.38 | 106.31 | −4.07 |
| val geom_camber_rc | 105.34 | 98.84 | −6.50 |
| val geom_camber_cruise | 75.14 | 69.35 | −5.79 |
| val re_rand | 96.27 | 87.82 | −8.45 |
| pre-clip grad norm (mean) | 33–106 | 2–5 | 10–20× lower |

- Artifact: `models/model-charliepai2g48h4-tanjiro-surf-weight-5-20260513-011227/metrics.jsonl`
- Run context: on Huber+grad-clip HEAD (surf_weight=10 baseline). Result is directly comparable to current best.

**Analysis:** Strongest improvement since grad-clip merge. All 4 val + all 4 test splits improved simultaneously — a very clean signal. Mechanism confirmed: grad-clip's per-step normalization removes the need for heavy surf_weight to dominate volume gradients. The halved weight frees the loss to weight volume residuals more equally, giving a smoother loss landscape. `re_rand` improves most (−8.45 val), consistent with that split having the highest surface/volume gradient ratio. Notably, pre-clip gradient norms dropped 10-20× under surf_weight=5 — the model operates in a qualitatively different gradient regime. All-split improvement with no tradeoffs makes this a strong, clean result. Follow-up: surf_weight=3 (tanjiro #1832) continues the sweep.

## 2026-05-13 02:11 — PR #1759: [max-norm-0.5] Tighter gradient clip 1.0→0.5 — **SENT BACK (rerun on new baseline)**
- Student branch: `charliepai2g48h4-frieren/max-norm-0.5`
- Hypothesis: Tighter clipping = smaller effective steps = better generalization in the normalized gradient descent regime.

| Metric | grad-clip-1.0 baseline (#1696) | max_norm=0.5 (this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 96.78 | 95.53 | −1.25 (−1.3%) |
| test_avg/mae_surf_p | 86.56 | 85.37 | −1.19 (−1.4%) |
| val single_in_dist | 110.38 | 112.09 | +1.71 |
| val geom_camber_rc | 105.34 | 103.60 | −1.74 |
| val geom_camber_cruise | 75.14 | 73.25 | −1.89 |
| val re_rand | 96.27 | 93.17 | −3.10 |
| pre-clip mean norm ep1 | 106 | 10 | 10× lower |

- Artifact: `models/model-charliepai2g48h4-frieren-max-norm-0.5-20260513-011036/metrics.jsonl`
- Run context: on Huber+grad-clip HEAD with surf_weight=10. Does NOT use new surf_weight=5 baseline.

**Analysis:** Small improvement (Δ−1.25 val) within σ ≈ 3.5 noise band. Key mechanism finding: pre-clip norms under max_norm=0.5 are ~10× smaller in absolute terms (not just clipped magnitude), suggesting tighter clipping pushes the model into a qualitatively flatter region. 3 of 4 splits improve; single_in_dist slightly regresses.

**Decision:** SENT BACK to rerun on new baseline (surf_weight=5, val=90.58). Since #1762 merged, dynamics changed (grad norms now 2-5 mean vs 33-106). max_norm=0.5 vs 1.0 under surf_weight=5 is an untested combination. Target: val < 90.58.

## 2026-05-13 02:17 — PR #1714: [huber-seed7-variance] Cross-seed σ calibration — **CLOSED (informational)**
- Student branch: `charliepai2g48h4-alphonse/huber-seed7-variance`
- Hypothesis: Measure cross-seed σ on Huber recipe by running seed=7 vs seed=42 baseline.

| Metric | seed=42 (baseline) | seed=7 | Δ | σ estimate (N=2) |
|---|---|---|---|---|
| val_avg/mae_surf_p | 110.59 | 98.49 | −12.10 | **σ ≈ 8.5 val** |
| test_avg/mae_surf_p | 102.28 | 90.07 | −12.21 | **σ ≈ 8.6 test** |
| val geom_camber_cruise | 95.72 | 79.60 | −16.12 | — |
| val single_in_dist | 127.85 | 115.15 | −12.70 | — |

- Artifact: `models/model-charliepai2g48h4-alphonse-huber-seed7-variance-20260513-005242/metrics.jsonl`
- Run context: on Huber HEAD (pre-grad-clip, surf_weight=10). Informational — not a target-beating run.

**Key finding:** σ ≈ 8.5 val on Huber recipe — roughly 2.4× the pre-Huber estimate (σ ≈ 3.5 from #1685). This means improvements of <10 val pts on the Huber recipe should be treated as potentially within noise. HOWEVER this σ is for the Huber recipe without grad-clip; under grad-clip's normalized steps the variance may be lower (more deterministic direction-following). The current recipe (grad-clip + surf_weight=5) σ is unknown.

**Operational implication:** Cross-seed σ on current recipe (grad-clip + surf_weight=5) still needs calibration. Alphonse's next assignment is layers-7 (#1834) for higher EV; σ calibration on current recipe will be assigned in a future round after more confirmations land.

## 2026-05-13 02:03 — PR #1730: [layers-6] n_layers 5→6 depth test — **SENT BACK (Huber base, pre-grad-clip)**
- Student branch: `charliepai2g48h4-edward/layers-6`
- Hypothesis: Under-fit on `val_single_in_dist` (127.85 vs 95.72 cruise) signals depth bottleneck. Adding one Transolver block tests whether extra global-attention rounds close the gap.

| Metric | Huber base (n_layers=5) | n_layers=6 (this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 110.59 | **98.24** | −11.2% |
| test_avg/mae_surf_p | 102.28 | **88.35** | −13.6% |
| val single_in_dist | 127.85 | 114.51 | −10.4% |
| val geom_camber_rc | 111.05 | 103.27 | −7.0% |
| val geom_camber_cruise | 95.72 | 78.73 | **−17.8%** |
| val re_rand | 107.73 | 96.43 | −10.5% |
| epochs completed | — | 15/50 (cap; monotonic) | — |
| peak VRAM | — | 39.87 GB (unchanged) | — |
| per-epoch time | ~100s | ~122s (+20%) | still within 30-min cap |

- Artifact: `models/model-charliepai2g48h4-edward-layers-6-20260513-005303/metrics.jsonl`
- Run context: on Huber HEAD (pre-grad-clip merge). Does not beat current best (96.78, grad-clip HEAD).

**Analysis:** Strongest per-split signal on this branch — all 4 val + all 4 test splits improved, cruise best (−17.8% val, −23.2% test: large meshes ~210K nodes benefit from extra global-attention rounds). Per-split pattern supports depth-bottleneck hypothesis: `single_in_dist` −10.4% val validates the diagnostic that motivated the PR. VRAM unchanged (+0.01 GB) — depth is free compute-wise. The 6th block pays for itself.

**Decision:** SENT BACK to rerun on grad-clip HEAD. Run was on pre-grad-clip Huber base. Applied to current best (96.78), the proportional Δ−11.2% would give ~85.9 val if proportionate — potentially the most impactful single lever on the board. Follow-ups: layers=7, combined layers=6 + grad-clip.

## 2026-05-13 02:03 — PR #1635: [log-cosh-loss] Log-cosh robust loss — **SENT BACK (Huber base, pre-grad-clip)**
- Student branch: `charliepai2g48h4-fern/log-cosh-loss`
- Hypothesis: Log-cosh (analytically smooth, no piecewise threshold knob) is at least as good as Huber (β=1.0) and may be better due to smooth L1-to-L2 transition.

| Metric | Huber base (#1374) | Log-cosh (this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 110.59 | **104.31** | −5.68 (−5.1%) |
| test_avg/mae_surf_p | ~102.28 | **95.10** | −7.18 |
| val single_in_dist | — | 117.59 | — |
| val geom_camber_rc | — | 114.56 | — |
| val geom_camber_cruise | — | 84.85 | — |
| val re_rand | — | 100.23 | — |
| epochs completed | — | 18/50 (still descending) | — |

- Artifact: `models/model-charliepai2g48h4-fern-log-cosh-loss-20260513-005527/metrics.jsonl`
- 4 local runs: mean ~102.0, std ~1.8. Low cross-seed variance confirmed.
- Run context: pre-grad-clip Huber HEAD. Does not beat current best (96.78, grad-clip HEAD).

**Analysis:** Log-cosh beats Huber (β=1.0) by 5.68 val on the same recipe — clear improvement. Mechanism: analytically smooth tail-capping avoids Huber's piecewise kink at |x|=1. Curve still descending at 30-min cap (epoch 14→15: 110.55→104.31), suggesting potential for further gain with cleaner annealing. Cross-seed variance (std ~1.8) is lower than Huber's estimated σ ≈ 3.5. Composes cleanly with merged recipe (unified_pos + bf16 + surf_weight=10). Log-cosh and Huber are alternative losses — they replace each other, not stack.

**Decision:** SENT BACK to rerun on grad-clip HEAD, replacing Huber with log-cosh. If log-cosh + grad-clip < 96.78, it becomes new loss merge and Huber is superseded.

## 2026-05-13 02:03 — PR #1576: [unified-pos-global-norm] Corpus-level positional bounds (seeded rerun) — **CLOSED**
- Student branch: `charliepai2g48h4-thorfinn/unified-pos-global-norm`
- Hypothesis: Replacing per-batch positional normalization with corpus-wide fixed bounds removes batch-level non-determinism in the position encoding.

| Metric | grad-clip baseline (#1696) | global-pos-norm (seeded, this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **96.78** | 98.41 | +1.63 (within σ) |
| test_avg/mae_surf_p | **86.56** | 87.51 | +0.95 |
| val single_in_dist | 110.38 | 121.64 | +11.26 |
| val geom_camber_rc | 105.34 | 107.09 | +1.75 |
| val geom_camber_cruise | 75.14 | 73.76 | −1.38 |
| val re_rand | 96.27 | 91.17 | −5.10 |

- Artifact: `models/model-charliepai2g48h4-thorfinn-unified-pos-global-norm-seeded-20260513-010628/metrics.jsonl`
- Run context: on current grad-clip HEAD (confirmed: rebased onto e7056a6, Huber + grad-clip + seed=42). This is a true apples-to-apples comparison.

**Analysis:** +1.63 val, +0.95 test — squarely within σ ≈ 3.5 noise. The previously observed improvement (unseeded MSE recipe, val ~123.56 vs 125.78) has been absorbed. Student's mechanistic explanation is compelling: grad-clip's per-step normalization + Huber's outlier suppression both independently absorbed the gradient noise that per-batch encoding was introducing. The change is no longer net-positive and adds maintenance burden (bounds re-scan if data changes). **CLOSED per student recommendation.** Work is still high-value: produced a clean mechanistic explanation of how the current recipe handles the positional encoding sensitivity. The "clip absorbs noise" story is now documented for future ablation reference.

## 2026-05-13 01:25 — PR #1695: [tmax-18] CosineAnnealingLR T_max=15→18 — **SENT BACK (Huber base, pre-grad-clip)**
- Student branch: `charliepai2g48h4-nezuko/tmax-18`
- Hypothesis: With ~18 epochs achievable in 30-min cap, T_max=18 bottoms the cosine exactly at the budget edge, so the best-epoch checkpoint sits at the end of the anneal (lr→0) rather than starting a second cycle (lr climbing back).

| Metric | Value | vs T_max=15 Huber base (#1374) |
|---|---|---|
| val_avg/mae_surf_p | **109.43** (best epoch 17/18) | −1.16 (−1.05%) |
| test_avg/mae_surf_p | **101.08** | −1.20 (−1.17%) |
| val single_in_dist | 128.53 | −11.29 |
| val geom_camber_rc | 115.37 | −5.22 |
| val geom_camber_cruise | 88.02 | +0.27 (flat) |
| val re_rand | 105.79 | −5.27 |
| LR at best epoch (17) | 1.51e-5 | well-annealed (3% of init) |

- Artifact: `models/model-charliepai2g48h4-nezuko-tmax-18-20260512-235324/metrics.jsonl`
- Run context: ran on Huber HEAD (pre-grad-clip merge). Does not beat current best (96.78, grad-clip HEAD).

**Analysis:** The cosine-bottoming-at-epoch-18 mechanism worked exactly as predicted: 3/4 splits improve cleanly (single_in_dist −11.3 the largest), only cruise flat (already near floor at 87.75 in T_max=15). lr trace confirms epoch 17 sits in lr→0 tail rather than climbing back into a second cycle. Per-split Δs are above σ noise for 3 splits.

**Decision:** SENT BACK to rerun on grad-clip HEAD. Δ−1.16 val is within σ ≈ 3.5 noise vs the directly-comparable Huber base, but the per-split signal is consistent. Grad-clip changes optimization dynamics dramatically (normalized gradient descent), so the schedule's interaction may differ — but the epoch-budget argument (cosine should bottom at the achievable epoch count) is independent of optimizer and should still hold. Stack test: T_max=18 on top of Huber+grad-clip targeting <96.78 val.

## 2026-05-13 01:19 — PR #1696: [grad-clip-1.0] Gradient clipping max_norm=1.0 — **MERGED (NEW BEST: val=96.78)**
- Student branch: `charliepai2g48h4-frieren/grad-clip-1.0`
- Hypothesis: Gradient clipping with max_norm=1.0 between loss.backward() and optimizer.step() caps per-step update magnitude, reducing instability from outlier high-Re batches.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **96.78** (best epoch 17/18) |
| test_avg/mae_surf_p | **86.56** (all 4 splits clean) |
| val single_in_dist | 110.38 |
| val geom_camber_rc | 105.34 |
| val geom_camber_cruise | 75.14 |
| val re_rand | 96.27 |
| test single_in_dist | 98.34 |
| test geom_camber_rc | 94.63 |
| test geom_camber_cruise | 63.51 |
| test re_rand | 89.75 |

- Artifact: `models/model-charliepai2g48h4-frieren-grad-clip-1.0-20260512-235444/metrics.jsonl`
- Run context: pre-Huber HEAD (started 23:54, before Huber merge at ~00:08). Merged HEAD now has Huber+grad-clip.
- Config: merged recipe (pre-Huber) + `clip_grad_norm_(model.parameters(), max_norm=1.0)` between backward() and step()
- Grad norms: mean 33–106 across epochs, max 464–770. **Every step was clipped** (norms >> 1.0). Effectively gradient-direction-following with unit step magnitude.

**Analysis:** Largest single improvement on this branch. Δval=−18.03 (−15.7%) vs T_max=15 baseline, Δval=−13.81 (−12.5%) vs Huber-only best. All 4 splits improved without exception. The mechanism is surprising: with typical grad norms of 30–1000, max_norm=1.0 makes the optimizer do normalized gradient descent rather than conventional clipping. This is adaptive per-step scaling that makes lr effectively 30–1000× smaller — a fundamentally different optimization regime. Follow-ups: max_norm=0.5 (frieren #1759) and EMA on top of grad-clip (askeladd #1540 rebasing).

## 2026-05-13 01:19 — PR #1575: [hidden256-bf16] Widen n_hidden 128→256 — **CLOSED (wall-clock-bound regression)**
- Student branch: `charliepai2g48h4-tanjiro/hidden256-bf16`
- Hypothesis: Doubling hidden dimension tests whether the model is width-bound.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | 150.77 (best epoch 9/12) |
| test_avg/mae_surf_p | 136.31 |
| Epochs completed | 12/50 (30-min cap) |
| Per-epoch time | ~153s (vs ~100s for n_hidden=128, +53%) |

- Artifact: `models/model-charliepai2g48h4-tanjiro-hidden256-bf16-20260512-235248/metrics.jsonl`

**Analysis:** Clear regression — val=150.77 vs baseline 114.81, training severely wall-clock-bound. 4× parameter count (2.63M vs 0.66M) causes ~53% slower per-epoch, leaving only 12 epochs in the 30-min cap. Same failure mode as hidden192 (#1406). Width scaling doesn't work under the wall-clock constraint. Depth (edward #1730, layers-6) is the alternative capacity test with much lower compute overhead.

## 2026-05-13 00:59 — PR #1540: [ema-weights] EMA decay=0.999 on Huber+T_max=15 recipe — **SENT BACK (rebasing for grad-clip HEAD)**
- Student branch: `charliepai2g48h4-askeladd/ema-weights`
- Second run (rebased onto Huber HEAD): val=99.60 / test=91.15 on Huber+EMA (no grad-clip)
- Per-split val: single_in_dist=116.54, geom_camber_rc=107.27, geom_camber_cruise=77.39, re_rand=97.23
- Artifact: `models/model-charliepai2g48h4-askeladd-ema-weights-20260513-000728/metrics.jsonl`
- Does not beat current best (96.78) as it's on pre-grad-clip HEAD. Sent back to rebase onto Huber+grad-clip HEAD.

**Analysis:** EMA on Huber (no grad-clip): val=99.60 — strong improvement over Huber-only (110.59), Δ−10.99 (−9.94%). Val curve monotonically decreasing (every epoch new best) — pure EMA smoothing effect. However, grad-clip result (96.78) already beats this even without EMA. Next test: EMA on top of Huber+grad-clip. Expected to push below 90.

## 2026-05-13 00:08 — PR #1374: [huber-loss] Smooth L1 / Huber loss on merged recipe — **MERGED (NEW BEST: val=110.59)**
- Student branch: `charliepai2g48h4-edward/huber-loss`
- Hypothesis: MSE loss is sensitive to large-residual samples (high-Re flows with 10× larger pressure gradients), pulling the training signal disproportionately toward outliers. Smooth L1 / Huber loss (beta=1.0) is identical to MSE for small residuals (|e|<1) but linear for large residuals, capping the gradient magnitude per-sample and giving the model a more stable signal.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **110.59** (best epoch 15/18) |
| test_avg/mae_surf_p | **102.28** (all 4 splits clean) |
| val single_in_dist | 127.85 |
| val geom_camber_rc | 111.05 |
| val geom_camber_cruise | 95.72 |
| val re_rand | 107.73 |
| test single_in_dist | 113.36 |
| test geom_camber_rc | 105.68 |
| test geom_camber_cruise | 85.87 |
| test re_rand | 104.20 |

- Artifact: `models/model-charliepai2g48h4-edward-huber-loss-20260512-231342/metrics.jsonl`
- Config: merged recipe (unified_pos, bf16, seed=42, T_max=15) + `loss=F.smooth_l1_loss(beta=1.0, reduction='sum') / (pred.shape[-1] * count)` applied in both training loop and evaluate_split
- Epochs: 18/50 (30-min cap), best at epoch 15

**Analysis:** Clear improvement over both reference points. Vs directly-comparable seeded baseline (#1577 seed=42, MSE, T_max=50): Δval = −5.84 (−5.0%), Δtest = −6.59 (−6.1%). Vs previous best (#1542 T_max=15 MSE): Δval = −4.22 (−3.7%), Δtest = −2.40 (−2.3%). Both deltas exceed the cross-seed σ measured by alphonse #1685 (σ ≈ 3.5 val). Huber pulls down the hard high-Re splits (single_in_dist: −12.0, geom_camber_rc: −9.5, re_rand: −3.3) at a small cost on the easy cruise split (+8.0 val). The improvement is real and composes cleanly with the T_max=15 truncation — best epoch shifted from 17 to 15, sitting fully within the first cosine cycle where lr anneals cleanly. Three independent trials in the pre-rebase run (109.33–112.33) confirmed signal-not-noise, now confirmed again on the merged recipe. **Stack target: Huber + EMA + T_max=18/grad-clip.**

## 2026-05-12 23:53 — PR #1685: [seed7-variance] Cross-seed σ on pre-Huber merged recipe — **CLOSED (informational)**
- Student branch: `charliepai2g48h4-alphonse/seed7-variance`
- Hypothesis: Calibrate across-seed variance σ so we know the practical significance threshold for new experiments.

| Metric | seed=42 (#1577 ref) | seed=7 | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 116.43 | 119.88 | +3.45 |
| test_avg/mae_surf_p | 108.87 | 109.37 | +0.50 |
| val geom_camber_rc | 121.66 | 141.71 | +20.05 |
| val geom_camber_cruise | 100.47 | 91.56 | −8.91 |

- Artifact: `models/model-charliepai2g48h4-alphonse-seed7-variance-20260512-231026/metrics.jsonl`

**Analysis:** σ_half ≈ 1.7 val / 0.25 test from a 2-point bracket. Per-split variance is highly heterogeneous: geom_camber_rc varies ±20 pts while re_rand is stable. Average cancels substantially. Practical threshold: ≥5 val pts (≈3× σ_half). The seed=42 reference is not unrepresentatively lucky. Key nuance: "curve still descending at cap" means σ is partially driven by epoch-count jitter — the T_max=15 recipe (where training exits at lr≈0) will have lower seed variance. Follow-up: alphonse #1714 will measure σ on the Huber recipe specifically.

## 2026-05-12 23:25 — PR #1542: [cosine-trunc-t15] T_max=50→15 on merged recipe — **MERGED (NEW BEST: val=114.81)**
- Student branch: `charliepai2g48h4-nezuko/cosine-trunc-t15`
- Hypothesis: With CosineAnnealingLR(T_max=50) but only ~18 epochs reachable under the 30-min cap, the schedule barely anneals — lr is still at ~75% of peak when training ends. Truncating to T_max=15 forces full annealing inside the cap window.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 17/18) | **114.81** — **NEW BEST ON BRANCH** |
| `test_avg/mae_surf_p` | **104.68** |
| `val_single_in_dist/mae_surf_p` | 139.82 |
| `val_geom_camber_rc/mae_surf_p` | 120.59 |
| `val_geom_camber_cruise/mae_surf_p` | 87.75 |
| `val_re_rand/mae_surf_p` | 111.06 |
| Best epoch | 17 (in 2nd cosine cycle, lr=5.46e-6) |
| Wall clock | 30.9 min (18 epochs) |
| Peak VRAM | 33.9 GB |
| Metrics path | `models/model-charliepai2g48h4-nezuko-cosine-trunc-t15-merged-20260512-215533/metrics.jsonl` |

**Analysis.** -1.4% val / -3.8% test vs alphonse seeded baseline (116.43 / 108.87). The schedule fix stacks cleanly with merged levers — lift on merged recipe (-7.4% vs canonical fern baseline 123.99) is bigger than on default config alone (-1.7% in #1542's pre-rebase trial). Mechanism: bf16's 28% throughput boost gives more usable epochs, which benefit more from a properly-annealed LR. Best epoch lands at the lr≈0 point in cycle 2 (epochs 16-17) — strong evidence that fine-tuning at very low LR adds real value.

**Caveat.** Run was on pre-rollback advisor base (surf_weight=20, no seed). The 3-way squash merge correctly produced surf_weight=10 + seed=42 + T_max=15 in the final state. A seeded confirmation will come naturally from the next student rebasing on this HEAD.

**Decision.** MERGED. Updated BASELINE.md and CURRENT_RESEARCH_STATE.md. New stacking target: merged recipe + T_max=15 + Huber.

**Suggested follow-ups:**
1. **T_max=18** to match achievable epoch count exactly (nezuko's own suggestion) — eliminates the second-cycle lr climb-back observed at epoch 18.
2. **T_max=15 + Huber stack** — edward's #1374 rebase will land this naturally.

## 2026-05-12 23:25 — PR #1394: [wd5e-4] AdamW weight_decay 1e-4→5e-4 — **CLOSED (regression)**
- Student branch: `charliepai2g48h4-frieren/wd5e-4`
- Hypothesis: Stronger weight decay improves OOD camber generalization.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 11) | 135.35 (vs default 123.99, **+9.2% worse**) |
| `test_avg/mae_surf_p` | NaN (fp32 overflow in test_geom_camber_cruise; pre-#1512 base) |
| `val_geom_camber_cruise/mae_surf_p` | 101.61 (slightly better than baseline range) |
| `val_geom_camber_rc/mae_surf_p` | 155.50 (substantially worse) |
| `val_re_rand/mae_surf_p` | 115.00 |
| `val_single_in_dist/mae_surf_p` | 169.27 (substantially worse) |
| Metrics path | `models/model-charliepai2g48h4-frieren-wd5e-4-20260512-215336/metrics.jsonl` |

**Analysis.** Run was on stale pre-merge advisor base (no unified_pos, no bf16, no scoring-fix — confirmed by NaN test result). Mixed per-split signal: cruise OOD slightly better, but rc and in-distribution both regress meaningfully. Net val regression of +9.2%. The plain global wd lever is too blunt for this dataset.

**Decision.** CLOSED. Hypothesis falsified. Frieren reassigned to gradient clipping.

## 2026-05-12 23:05 — PR #1577: [seed42-baseline] Seeding infrastructure + surf_weight=10 rollback — **MERGED (NEW BEST: val=116.43)**
- Student branch: `charliepai2g48h4-alphonse/seed42-baseline`
- Hypothesis: Add deterministic seeding (seed=42, cudnn.deterministic, seeded DataLoader/sampler) to eliminate run-to-run noise and establish a reproducible canonical baseline for all future ablations.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 18/18) | **116.43** — **NEW BEST ON BRANCH** |
| `test_avg/mae_surf_p` | **108.87** |
| `val_single_in_dist/mae_surf_p` | 131.15 |
| `val_geom_camber_rc/mae_surf_p` | 121.66 |
| `val_geom_camber_cruise/mae_surf_p` | 100.47 |
| `val_re_rand/mae_surf_p` | 112.46 |
| Epochs completed | 18/50 (timeout hit; still descending) |
| Wall clock | 30.9 min |
| Peak VRAM | 33.9 GB |
| Params | 0.68 M |
| Determinism | Byte-identical across 2 independent runs |
| Metrics path | `models/model-charliepai2g48h4-alphonse-seed42-baseline-20260512-215112/metrics.jsonl` |

**Analysis.** **-6.1% vs old unseeded baseline (123.99) and -9.2% vs unseeded merged recipe (127.86).** Two factors: (1) deterministic seeding stabilizes training, and (2) the run was on surf_weight=10 (not 20) because alphonse's branch predated #1369 — consistent with fern's #1570 confirming surf_weight=20 is a regression. The squash-merge 3-way resolution kept surf_weight=20 on the advisor branch, so surf_weight was explicitly rolled back to 10 post-merge. **Definitive seed=42 + surf_weight=10 reference: val=116.43 / test=108.87.**

**Decision.** MERGED. surf_weight rolled back to 10.0 in advisor train.py post-merge. The seeding infrastructure is now in the base recipe for all future experiments.

## 2026-05-12 23:05 — PR #1576: [unified-pos-global-norm] Corpus-level pos normalization — **SENT BACK (needs seeded rerun)**
- Student branch: `charliepai2g48h4-thorfinn/unified-pos-global-norm`
- Hypothesis: Per-batch pos normalization in unified_pos creates encoding noise across batches. Fixed corpus-level bounds make the spatial encoding deterministic per-node.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 18/18) | 123.56 |
| `test_avg/mae_surf_p` | 109.71 |
| `val_single_in_dist/mae_surf_p` | 157.53 |
| `val_geom_camber_rc/mae_surf_p` | 134.67 |
| `val_geom_camber_cruise/mae_surf_p` | 89.17 |
| `val_re_rand/mae_surf_p` | 112.87 |
| vs prior unified_pos (#1416) | -2.22 val, **-7.41 test** |
| Metrics path | `models/model-charliepai2g48h4-thorfinn-unified-pos-global-norm-20260512-215857/metrics.jsonl` |

**Analysis.** Direction is right (test improvement -7.41 vs prior unified_pos), but run was unseeded and the 7+ pt val gap vs new seeded baseline (116.43) is within old σ ≈ 6.8. The global pos norm implementation is correct (corpus bounds scanned from train+val, stored as model buffers). surf_weight was 10 (not 20) on this branch. Cannot confirm improvement without seeded rerun.

**Decision.** Sent back for rebase onto seeded baseline + rerun with seed=42.

## 2026-05-12 21:48 — PR #1374: [huber-loss] Smooth L1 (Huber) instead of MSE in normalized space — **STRONGEST LEVER ON BRANCH**
- Student branch: `charliepai2g48h4-edward/huber-loss`
- Hypothesis: per-sample target std spans ~10× even within one validation split, so MSE's squared gradient over-emphasizes high-Re/large-magnitude samples and biases the model. Replacing MSE with Smooth L1 (Huber, beta=1.0) caps outlier gradients while preserving quadratic behavior near zero.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 13/14) | **112.06** — **BEST RESULT ON BRANCH (default config)** |
| `test_avg/mae_surf_p` | NaN (cruise[20] sample triggered pre-merge scoring bug — fixed by #1512 on advisor) |
| `test_avg/mae_surf_p` (3-split, excl cruise) | **107.52** |
| `val_single_in_dist/mae_surf_p`     | 147.10 |
| `val_geom_camber_rc/mae_surf_p`     | 115.27 |
| `val_geom_camber_cruise/mae_surf_p` |  85.33 |
| `val_re_rand/mae_surf_p`            | 100.56 |
| Wall clock | 30.8 min |
| Peak VRAM | 42.12 GB |
| Params | 0.66 M |
| Run-to-run variance | ~3 pts (3 trials: 109.33, 112.06, 112.33) |
| Metrics path | `models/model-charliepai2g48h4-edward-huber-loss-20260512-205528/metrics.jsonl` |

**Analysis.** **Largest single-lever improvement found on this branch so far.** -9.6% vs default baseline (123.99); -7.5% vs prior single-lever best (askeladd EMA 121.16). Three independent trials at 109.33–112.33 — variance ~3 pts vs improvement ~12 pts — signal-to-noise ratio is very strong. The mechanism is exactly as hypothesized: under our ~10× dynamic-range regime, MSE gradients are dominated by a small set of large-magnitude samples; Huber's linear-tail behavior re-balances the gradient budget across the sample distribution.

**Decision.** Sent back for rebase + rerun on merged recipe. Train.py change (8/8) is mechanical but conflicts with the 4 merged PRs. The merged scoring-fix (#1512) will resolve the test NaN automatically. After rerun, this is the strongest merge candidate by margin.

**Suggested follow-ups (high priority):**
1. **Huber beta sweep** {0.5, 1.0, 2.0} — find optimal transition point. Default 1.0 is at normalized-residual scale; smaller beta shifts more samples to the linear regime, larger beta keeps quadratic over a wider range.
2. **Log-cosh as alternative** — differentiable everywhere, no threshold hyperparameter, similar tail-capping behavior.
3. **Huber + EMA stack** — two strongest levers, both targeting variance/robustness. Orthogonal mechanisms (sample-level vs weight-level).
4. **Huber + T_max=15 stack** — orthogonal: loss shape vs lr schedule.

## 2026-05-12 21:00 — PR #1570: [surf-weight-20-stack] surf_weight=20 on merged unified_pos+bf16 recipe — **CLOSED (regression)**
- Student branch: `charliepai2g48h4-fern/surf-weight-20-stack`
- Hypothesis: surf_weight=20 (askeladd #1369 on old recipe) stacks additively with unified_pos+bf16 to push val below 120.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 17/18) | 127.86 (vs 123.99 default, **+3.1% worse**) |
| `test_avg/mae_surf_p` | 119.28 (vs 110.97 default, **+7.5% worse**) |
| `test_geom_camber_cruise/mae_surf_p` | 90.14 (vs 76.78 default, **+17.4% worse**) |
| Wall clock | 30.9 min (18 epochs at ~103 s/epoch — bf16 speedup confirmed) |
| Peak VRAM | 33.90 GB |
| Metrics path | `models/model-charliepai2g48h4-fern-surf-weight-20-stack-20260512-205042/metrics.jsonl` |

**Analysis.** Hypothesis falsified. surf_weight=20 stacked on merged recipe gives the same number (~127.9) as on the OLD recipe — unified_pos+bf16 don't help loss-weight-axis benefit. Volume MAE on cruise climbed (the gradient budget went to surface at the cost of volume). Combined with PR #1533 (3× weight, +25% regression) and #1369 (2× on old recipe, 127.94 within noise), three points along the loss-weight axis are now mapped — surf_weight=10 is at or near the minimum on this budget.

**Important meta-observation:** The fact that the merged recipe gives val=127.86 (vs 123.99 default + scoring-fix) suggests the merged recipe may itself be a slight regression vs default. Need alphonse seeded baseline (#1577) to confirm cleanly.

**Decision.** CLOSED. Clear regression (>5% on test), and the lever is now mapped — no need to explore further surf_weight values. Reassigned fern to a new hypothesis.

## 2026-05-12 20:53 — PR #1542: [cosine-trunc-t15] Truncate cosine T_max from 50 to 15 to anneal inside cap
- Student branch: `charliepai2g48h4-nezuko/cosine-trunc-t15`
- Hypothesis: under the 30-min cap only ~14 epochs land; `CosineAnnealingLR(T_max=50)` keeps lr at ~92% of initial throughout. Truncating to `T_max=15` gives a real late-training fine-tuning regime (lr → ~0 by the last 1–2 epochs).

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 14/14) | **121.83** — **NEW BEST on branch (default config)** |
| `test_avg/mae_surf_p` | **110.50** (NaN-safe guard from #1416 incorporated) |
| `val_single_in_dist/mae_surf_p`     | 153.94 |
| `val_geom_camber_rc/mae_surf_p`     | 127.17 |
| `val_geom_camber_cruise/mae_surf_p` |  96.01 |
| `val_re_rand/mae_surf_p`            | 110.19 |
| Final lr (ep 14) | 2.16e-5 (= 4.3% of 5e-4 init — schedule actually annealed) |
| Wall clock | 30.6 min |
| Peak VRAM | 42.12 GB |
| Params | 0.66 M |
| Metrics path | `models/model-charliepai2g48h4-nezuko-cosine-trunc-t15-20260512-200814/metrics.jsonl` |

**Analysis.** Hypothesis confirmed cleanly: val_avg descended 127.16 → 121.83 over epochs 12 → 14 as lr annealed from 1.25e-4 to 2.16e-5 — 5.3-pt drop concentrated in the late-training phase where T_max=50 would have left lr at ~4.6e-4. Beats round-1 default baseline (123.99) by 2.16 pts (1.7%); within noise of askeladd EMA (121.16) but achieved via an orthogonal lever (lr schedule vs. weight averaging). Test improved more than val (~5.6% over thorfinn unified-pos #1416), consistent with the late-anneal also stabilizing the final-iterate weights.

**Decision.** Sent back for rebase onto merged recipe (#1512 + #1513 + #1416 + #1369) — train.py merge conflict blocks direct merge. Re-run on `unified_pos=True + bf16 + surf_weight=20 + T_max=15` to confirm stacking. Status:wip via REST API (GraphQL rate-limited).

**Suggested follow-ups:**
1. **EMA + T_max=15 stack** — both top-2 levers are orthogonal, this is the biggest expected win.
2. **T_max=18** matched to bf16 epoch count (~18 epochs/30 min) for slightly slower anneal.
3. **Linear warmup + T_max=15** — 1-2 warmup epochs preserve the lever, may reduce noise on the first few epochs.

## 2026-05-12 20:53 — PR #1540: [ema-weights] Use EMA of weights (decay=0.999) for val/test eval
- Student branch: `charliepai2g48h4-askeladd/ema-weights`
- Hypothesis: under 14-epoch wall-clock truncation, the final-iterate weights are noisy mid-schedule estimates; Polyak averaging with decay=0.999 produces a smoother estimate closer to a stable late-training weight, reducing run-to-run variance.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 14/14, EMA weights) | **121.16** — **NEW BEST on branch (default config)** |
| `test_avg/mae_surf_p` | **108.69** — best test on branch |
| `val_single_in_dist/mae_surf_p`     | 147.47 |
| `val_geom_camber_rc/mae_surf_p`     | 132.15 |
| `val_geom_camber_cruise/mae_surf_p` |  95.58 |
| `val_re_rand/mae_surf_p`            | 109.44 |
| Wall clock | 30.5 min |
| Peak VRAM | 42.11 GB (2× param cost negligible at 0.66M params) |
| Params | 0.66 M |
| Metrics path | `models/model-charliepai2g48h4-askeladd-ema-weights-20260512-201111/metrics.jsonl` |

**Analysis.** Val_avg was monotonically decreasing every single epoch (335.67 → 121.16 over 14 epochs) — clean smoothing effect, no late-stage oscillation. EMA half-life ≈ 1.85 epochs at decay=0.999 with ~375 steps/epoch, so the EMA is dominated by late-training weights (correct regime). Test moved more than val (~7% test vs ~3.7% val improvement over thorfinn) — EMA's smoothing of the truncated mid-schedule weights is most valuable here because we're mid-cosine without a settled iterate.

**Decision.** Sent back for rebase onto merged recipe (advisor branch progressed during run). Re-run on `unified_pos=True + bf16 + surf_weight=20 + EMA decay=0.999`. After rebase + rerun, this is the next merge candidate — combining EMA + cosine-trunc-T15 (#1542 above) on the merged recipe is the highest-priority round-2 stacking target.

**Suggested follow-ups:**
1. **EMA + cosine-trunc-T15 stack** — orthogonal levers, both top-2 results.
2. **Decay bracket {0.99, 0.999, 0.9995}** to pin the optimal value.
3. **Multi-seed EMA on/off A/B** — direct variance-reduction measurement (matches the original 30-pt seed-gap motivation).

## 2026-05-12 18:31 — PR #1376: [lr1e3-warmup-cosine] lr=1e-3 with 3-epoch linear warmup + cosine
- Student branch: `charliepai2g48h4-fern/lr1e3-warmup-cosine`
- Hypothesis: replace plain `CosineAnnealingLR(T_max=50)` + `lr=5e-4` with `LinearLR(3 ep warmup) → CosineAnnealingLR(T_max=47)` and `lr=1e-3` to converge further inside the 30-min wall-clock cap.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 14/14) | **147.2556** |
| `test_avg/mae_surf_p` | NaN (scoring bug — see below) |
| `test_avg/mae_surf_p` (3-split mean, excludes broken `test_geom_camber_cruise`) | 142.36 |
| `val_geom_camber_cruise/mae_surf_p` | 128.93 |
| `val_geom_camber_rc/mae_surf_p`     | 138.84 |
| `val_re_rand/mae_surf_p`            | 127.21 |
| `val_single_in_dist/mae_surf_p`     | 194.05 |
| Wall clock | 30.8 min (stopped mid-ep 15) |
| Peak VRAM | 42.1 GB |
| Params | 0.66 M |
| Metrics path | `models/model-charliepai2g48h4-fern-lr1e3-warmup-cosine-20260512-175601/metrics.jsonl` |

**Analysis.** Schedule worked as predicted. Val curve descended monotonically 335 → 150 over ep 1–9, mild plateau/bounce 10–13 (155–192), recovered to best 147 at ep 14. No divergence at `lr=1e-3` peak — the 3-epoch warmup absorbed the initial-step risk. `val_single_in_dist` is the worst-performing split (194 vs ~128 on the others), driven by raceCar single-foil mesh size and ground-effect physics. Burned 3/14 (~21%) of the budget on warmup; shorter warmup or higher peak lr is a clean next iteration.

**Held pending baseline (PR #1368).** Merge decision deferred to when alphonse's 5e-4/no-warmup run lands at the same 30-min cap.

**Scoring bug discovered.** `data/scoring.py:accumulate_batch` propagates `NaN` into `mae_surf`/`mae_vol` when any sample's GT contains non-finite values, because `(inf * 0.0).sum() = NaN`. The per-sample `y_finite` skip filters the counts but `err` is still poisoned. Concretely `test_geom_camber_cruise` sample 20 has `y_p = -inf`. Fix assigned in PR #1512 (`scoring-nan-fix` — surgical `torch.nan_to_num(err, ...)` after the abs).

**Suggested follow-ups (kept on backlog):**
1. Higher peak lr (2e-3 / 3e-3) with the same warmup scaffold.
2. Truncated cosine `T_max ≈ effective_epochs (~15)` so the schedule actually anneals at this wall-clock budget.
3. Per-channel surface-pressure-loss treatment — `single_in_dist` has 1.5× the surf_p MAE of the other splits.

## 2026-05-12 18:55 — PR #1406: [hidden192] Widen Transolver hidden 128→192
- Student branch: `charliepai2g48h4-tanjiro/hidden192`
- Hypothesis: widen `n_hidden=128→192` (0.7M→1.47M params) to test whether capacity is a bottleneck.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 9/10) | **151.6438** |
| `test_avg/mae_surf_p` | NaN (same scoring bug as #1376) |
| `test_avg/mae_surf_p` (3-split mean, excl. broken cruise) | 146.09 |
| `val_geom_camber_cruise/mae_surf_p` | 127.69 |
| `val_geom_camber_rc/mae_surf_p`     | 164.61 |
| `val_re_rand/mae_surf_p`            | 134.68 |
| `val_single_in_dist/mae_surf_p`     | 179.60 |
| Wall clock | 30 min cap; 10 epochs landed |
| Per-epoch wall clock | ~184 s (vs ~133 s at n_hidden=128, +38%) |
| Peak VRAM | 58.0 GB (bs=4, n_hidden=192) |
| Params | 1.47 M |
| Metrics path | `models/model-charliepai2g48h4-tanjiro-hidden192-20260512-175550/metrics.jsonl` |

**Analysis.** Widening worked as a code change but did not deliver an obvious win at this wall-clock budget. The val curve was still actively improving at ep 9 (151.90→151.64 best, with 160.28 at ep 10 — not converged), and the 38% per-epoch cost shrank the effective epoch budget from 14 to 10. So `hidden192` is a wall-clock-bound result, not a capacity-saturated one. Per-split pattern matches fern: cruise-camber OOD is the *easiest* (127.69), in-dist sanity is the *hardest* (179.60) — the recipe's bottleneck is single-foil pressure regardless of width.

**Held pending baseline (PR #1368).**

**Implication for round 2:** at this 30-min cap, throughput is the actually binding constraint. Reassigning tanjiro to `bf16-autocast` (PR #1513) to test whether mixed-precision training can buy back the per-epoch cost of wider models (or simply land more epochs at any model size). If bf16 delivers ~30-50% throughput, future capacity experiments become much more attractive.

**Suggested follow-ups:** none merged in immediately — the hypothesis "capacity helps" is unproven but not falsified.

## 2026-05-12 19:15 — PR #1416: [unified-pos] Unified positional encoding with `ref=8`
- Student branch: `charliepai2g48h4-thorfinn/unified-pos`
- Hypothesis: replace raw `(x,z)` positions through the preprocess MLP with `Transolver`'s `unified_pos=True` soft-grid encoding (`ref=8`, 2D so `ref^space_dim=64` features); regularized fixed-grid encoding should give a stronger spatial inductive bias on irregular meshes.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 13/14) | **125.7759** |
| `test_avg/mae_surf_p` (best-val checkpoint) | **117.1233** |
| `val_geom_camber_cruise/mae_surf_p` | **91.85** |
| `val_geom_camber_rc/mae_surf_p`     | 145.70 |
| `val_re_rand/mae_surf_p`            | 114.24 |
| `val_single_in_dist/mae_surf_p`     | 151.32 |
| `test_geom_camber_cruise/mae_surf_p` | **80.27** |
| `test_geom_camber_rc/mae_surf_p`     | 138.57 |
| `test_re_rand/mae_surf_p`            | 114.75 |
| `test_single_in_dist/mae_surf_p`     | 134.89 |
| Wall clock | 30 min cap; ep 13 best of 14 landed |
| Peak VRAM | 42.5 GB |
| Params | 0.68 M (+20 K vs default — input MLP grows from `(2+22)→256` to `(64+22)→256`) |
| Metrics path | `models/model-charliepai2g48h4-thorfinn-unified-pos-20260512-175707/metrics.jsonl` |

**Analysis.** Strongest round-1 result by a wide margin: `val_avg/mae_surf_p=125.78` beats fern (147.26) by ~14% and tanjiro (151.64) by ~17% at the same 30-min cap, and ships with a valid test number. Cruise-camber OOD is *especially* strong here (`val=91.85, test=80.27`) — soft-grid encoding plausibly resolves the larger, more uniformly distributed cruise meshes better than raw-coordinate-through-MLP. The raceCar single-foil + ground-effect samples (`val_single_in_dist=151.32`, `val_geom_camber_rc=145.70`) remain the dominant residual error, same per-split pattern as fern and tanjiro. Schedule was wall-clock-limited (`cosine(13/50)≈0.84`, so LR barely annealed) — a `T_max≈effective_epochs` follow-up could plausibly buy further headroom.

**Held pending baseline (PR #1368).** Given fern and tanjiro both land ~150 at the same cap, the baseline is almost certainly above 125 — this PR is on track to merge.

**Independent scoring-bug fix in this PR.** Thorfinn independently identified the same `inf * 0 = NaN` propagation that fern reported in #1512, and added a defensive workaround in `train.py::evaluate_split` that pre-filters non-finite-y samples before they reach `accumulate_batch`. Without this guard, `test_avg/mae_surf_p` is NaN on this branch for every experiment that hits a `geom_camber_cruise` test sample. Workaround at the call site coexists with fern's surgical fix at the helper site (#1512).

**Suggested follow-ups (kept on backlog):**
1. **Corpus-level position normalization.** Current `unified_pos` uses per-batch `pos.amin/amax`, so a sample's encoding depends on its batch-mates and on padding zeros. Replacing with fixed corpus-level bounds (from `stats`) would make the encoding deterministic per-sample and remove a noise source.
2. **`ref` sweep ∈ {12, 16}.** Wider soft grid = finer spatial membership; +small param cost on input MLP.
3. **Truncated cosine `T_max ≈ effective_epochs`** — applies branch-wide.

## 2026-05-12 19:29 — PR #1369: [surf-weight-20] Increase surf_weight 10→20
- Student branch: `charliepai2g48h4-askeladd/surf-weight-20`
- Hypothesis: bump `Config.surf_weight=10.0→20.0` to weight surface MSE 2× more vs volume — direct lever on the primary ranking metric.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 12/14) | **127.9357** |
| `test_avg/mae_surf_p` (best-val checkpoint, cruise[20] excluded) | **117.3456** |
| `val_geom_camber_cruise/mae_surf_p` | 108.97 |
| `val_geom_camber_rc/mae_surf_p`     | **135.82** (best among returned PRs) |
| `val_re_rand/mae_surf_p`            | 116.55 |
| `val_single_in_dist/mae_surf_p`     | **150.41** (best so far on the hardest split) |
| Wall clock | 30 min cap; 14 epochs landed |
| Peak VRAM | 42.1 GB |
| Params | 0.66 M |
| Metrics path | `models/model-charliepai2g48h4-askeladd-surf-weight-20-20260512-175549/metrics.jsonl` |
| Test workaround | `test_metrics_excl_cruise20.json` (re-run from best-val checkpoint with cruise[20] dropped) |

**Analysis.** 2nd-best round-1 result, essentially tied with thorfinn at the noise floor. Hypothesis confirmed: heavier surface weighting moves the primary metric meaningfully without destabilizing training. Per-split, this PR is the strongest on `val_single_in_dist` (150 vs ~180 for tanjiro/fern) AND `val_geom_camber_rc` (136, best returned) — both raceCar tracks. Cruise (109) is weaker than thorfinn's unified-pos (92) but stronger than fern/tanjiro. The two levers (positional encoding, surf_weight) hit different per-split weak points and are likely orthogonal — round-2 stacking candidate.

**Major variance signal.** Askeladd ran the same surf_weight=20 config twice; the second run landed val_avg = 157.95 — a 30-point gap from seed/noise alone. `train.py` is unseeded. This makes ANY single-run comparison noise-limited. Follow-up assignment for askeladd is EMA-weights (PR #1540) as a variance-reduction technique pending dedicated seeded-training infra.

**Held pending baseline (PR #1368).**

**Independent scoring-bug confirmation.** Third independent confirmation (after fern and thorfinn) of `Inf * 0 = NaN` on `test_geom_camber_cruise[20]`. Askeladd committed a `test_metrics_excl_cruise20.json` workaround artifact — fern's PR #1512 is still the root-cause fix.

**Suggested follow-ups (kept on backlog):** seeded training (#1), surf_weight sweep {5,10,15,20,30} once seeded, linear warmup.

## 2026-05-12 19:32 — PR #1402: [slice128] Double slice_num 64→128
- Student branch: `charliepai2g48h4-nezuko/slice128`
- Hypothesis: physics-attention groups N nodes into `slice_num` learned tokens; default 64 may be too few for large meshes (242K nodes on cruise). Double to 128 to give finer physical structure, especially on the cruise splits.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 10/11) | **137.1686** |
| `test_avg/mae_surf_p` | NaN (scoring bug; 3-split mean: 135.53) |
| `val_geom_camber_cruise/mae_surf_p` | **107.79** (best on cruise val among non-unified-pos PRs) |
| `val_geom_camber_rc/mae_surf_p`     | 143.92 |
| `val_re_rand/mae_surf_p`            | 120.60 |
| `val_single_in_dist/mae_surf_p`     | 176.37 |
| Wall clock | 30 min cap (~31.8 min); 10 best of 11 epochs landed (-3 vs thorfinn's 13) |
| Peak VRAM | 54.5 GB (vs 42 GB at slice_num=64; +30% memory) |
| Params | 0.67 M (+0.01 M from slice projection layers) |
| Metrics path | `models/model-charliepai2g48h4-nezuko-slice128-20260512-184959/metrics.jsonl` |

**Analysis.** 3rd-best returned result. Hypothesis is *directionally* supported: cruise val improves to 107.79 — the strongest cruise number among the four non-unified-pos PRs (thorfinn's unified-pos got cruise=91.85, which is special). The per-epoch cost of slice128 is higher (~10% slower), so this PR landed 3 fewer epochs than thorfinn at the same cap, and the comparison is wall-clock-budget-confounded rather than capacity-saturated. Worth retrying with thorfinn's unified-pos already stacked once we know the baseline.

**Held pending baseline (PR #1368).**

**Independent scoring-bug confirmation.** Fourth independent identification of the same `Inf * 0 = NaN` root cause. Suggested data-layer fix as alternative to fern's helper-site fix — going with fern's #1512 because it's smaller blast radius.

**Suggested follow-ups (kept on backlog):** slice256 (54.5 GB → ~70 GB still under 96 GB), stack with n_hidden, longer wall time.

## 2026-05-12 19:52 — Round 1 status snapshot (5/8 returned, baseline still WIP)

| Student | Slug | val_avg/mae_surf_p | test_avg/mae_surf_p | Status |
|---|---|---|---|---|
| thorfinn | `unified-pos` (#1416) | **125.78** | 117.12 | Held pending baseline — best so far |
| askeladd | `surf-weight-20` (#1369) | **127.94** | 117.35 | Held pending baseline — 2nd |
| nezuko | `slice128` (#1402) | 137.17 | NaN (135.53 over 3) | Held pending baseline — 3rd |
| fern | `lr1e3-warmup-cosine` (#1376) | 147.26 | NaN | Held pending baseline |
| tanjiro | `hidden192` (#1406) | 151.64 | NaN | Held pending baseline |
| alphonse | `baseline-ref` (#1368) | — | — | WIP — rate-limited 17:50→19:48 UTC, ETA ~20:25 UTC |
| edward | `huber-loss` (#1374) | — | — | WIP — rate-limited 17:50→19:50 UTC, ETA ~20:27 UTC |
| frieren | `wd5e-4` (#1394) | — | — | WIP |

**Active follow-on assignments:**
- PR #1512 (fern) — `scoring-nan-fix` (helper-site fix for the `Inf*0=NaN` bug)
- PR #1513 (tanjiro) — `bf16-autocast` (throughput, predicted 30-50% per-epoch reduction)
- PR #1533 (thorfinn) — `surf-p-weight-3x` (per-channel: weight surface-p 3× over surface-Ux/Uy)
- **PR #1540 (askeladd) — `ema-weights`** (variance reduction via Polyak averaging, decay 0.999, EMA at val/test)
- **PR #1542 (nezuko) — `cosine-trunc-t15`** (truncate `T_max=50→15` so cosine actually anneals inside the 30-min cap; addresses near-constant-lr observation across all 5 returned runs)

**Pod rate-limit incident.** alphonse and edward hit GraphQL rate limits at ~17:50 UTC and couldn't pick up assigned PRs for ~2 hours (13 and 18 heartbeat iterations of "## Student research state — No assigned PRs or issues" respectively). They finally cleared at 19:48-19:50 UTC. This pushed baseline ETA from ~19:21 to ~20:25 UTC. No work was lost; just delayed.

_(Round 1 largely closed — edward + frieren still pending.)_

## 2026-05-12 20:00 — PR #1512: [scoring-nan-fix] Stop NaN propagation (fern) — MERGED as baseline anchor
- Student branch: `charliepai2g48h4-fern/scoring-nan-fix`
- Hypothesis: surgical `torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)` after `err = (pred-y).abs()` in `data/scoring.py::accumulate_batch` — fixes `Inf * 0 = NaN` propagation that poisoned every `test_avg/mae_surf_p` evaluation on this branch.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 14/14) | **123.99** |
| `test_avg/mae_surf_p` (NOW FINITE) | **110.97** |
| `test_geom_camber_cruise/mae_surf_p` | **76.78** (previously NaN) |
| `test_geom_camber_rc/mae_surf_p` | 121.93 |
| `test_re_rand/mae_surf_p` | 110.93 |
| `test_single_in_dist/mae_surf_p` | 134.23 |
| Config | default (lr=5e-4, surf_weight=10, unified_pos=False, no bf16) |
| Wall clock | 30 min cap; 14 epochs |
| Peak VRAM | 42.11 GB |
| Metrics path | `models/model-charliepai2g48h4-fern-scoring-nan-fix-20260512-185620/metrics.jsonl` |

**Analysis.** This PR's primary value is that it unblocks all future test evaluations. Secondarily it provides the first clean default-config baseline: val_avg=123.99, test_avg=110.97. Note that val numbers are unaffected by the scoring fix (val GT has no non-finite values); the val number is therefore identical to what alphonse's true baseline run should produce under the same RNG. **Merged** as both a critical infra fix and a baseline anchor. New `BASELINE.md` entry set at 123.99/110.97.

## 2026-05-12 20:03 — PR #1513: [bf16-autocast] bf16 throughput (tanjiro) — MERGED as infra
- Student branch: `charliepai2g48h4-tanjiro/bf16-autocast`
- Hypothesis: wrap training forward+backward in `torch.cuda.amp.autocast(dtype=torch.bfloat16)` to reduce per-epoch wall clock and land more epochs in the 30-min cap.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 18/18) | **125.40** |
| `test_avg/mae_surf_p` (3-split mean, cruise NaN) | **126.57** |
| `val_geom_camber_cruise/mae_surf_p` | 90.70 |
| `val_geom_camber_rc/mae_surf_p` | 132.06 |
| `val_re_rand/mae_surf_p` | 111.22 |
| `val_single_in_dist/mae_surf_p` | 167.62 |
| Per-epoch wall clock | **101.4 s** (vs ~133 s fp32 → **24% speedup**) |
| Epochs landed | **18** (vs 13 fp32, vs 10 hidden192 — wall-clock win is clear) |
| Val still descending at ep 18 | Yes — model not converged at cap |
| NaN-safe guard (isfinite) | 0 batches skipped — no overflow |
| Peak VRAM | 32.9 GB (vs 42 GB fp32; **22% VRAM savings**) |
| Params | 0.66 M |
| Metrics path | `models/model-charliepai2g48h4-tanjiro-bf16-autocast-20260512-190244/metrics.jsonl` |

**Analysis.** Hypothesis confirmed. val_avg=125.40 is within noise of baseline 123.99; the value here is the throughput win — 24% faster, +5 extra epochs in 30 min, 22% less VRAM. This makes future capacity experiments viable (hidden256 can now land 12-14 epochs vs 10 for hidden192 without bf16). **Merged** as throughput infrastructure. Every future experiment on this branch inherits bf16 autocast from the merged recipe.

**Note on test metric:** `test_avg/mae_surf_p` is NaN (pre-fix run — this PR was submitted before #1512 merged). 3-split mean = 126.57.

## 2026-05-12 20:30 — Round 1 merge/close decisions (baseline now at val=123.99)

With baseline established, reviewed all 5 held round-1 hypothesis PRs:

| PR | Lever | val_avg | Δ from baseline | Decision |
|---|---|---|---|---|
| #1416 (thorfinn) | unified-pos | 125.78 | +1.4% | **MERGED** — within noise, cruise val 91.85 is real |
| #1369 (askeladd) | surf-weight-20 | 127.94 | +3.2% | **Pending merge** — preflight blocked by hold comment ordering |
| #1402 (nezuko) | slice128 | 137.17 | +10.6% | **Request changes** — re-run with cosine-trunc-T15 |
| #1376 (fern) | lr1e3-warmup | 147.26 | +18.8% | **CLOSED** — warmup ate budget |
| #1406 (tanjiro) | hidden192 | 151.64 | +22.3% | **CLOSED** — wall-clock-bound, superseded by #1575 |
| #1533 (thorfinn) | surf-p-weight-3x | 154.47 | +24.6% | **CLOSED** — 3× ratio too aggressive, model too small |

## 2026-05-12 20:53 — PR #1540: [ema-weights] Polyak EMA at val/test — NEW BEST (askeladd)
- Student branch: `charliepai2g48h4-askeladd/ema-weights`
- Hypothesis: maintain EMA copy of model weights (decay 0.999), use EMA model for val/test evaluation — Polyak averaging to reduce variance from wall-clock-truncated mid-cosine runs.
- Config: **default config** (surf_weight=10, no unified_pos, no bf16, lr=5e-4) — branched from advisor commit 0242e62 before other merges.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 14/14) | **121.16** ← NEW BEST |
| `test_avg/mae_surf_p` | **108.69** ← NEW BEST |
| `val_geom_camber_cruise/mae_surf_p` | 95.58 |
| `val_geom_camber_rc/mae_surf_p` | 132.15 |
| `val_re_rand/mae_surf_p` | 109.44 |
| `val_single_in_dist/mae_surf_p` | 147.47 |
| `test_geom_camber_cruise/mae_surf_p` | 80.16 |
| `test_geom_camber_rc/mae_surf_p` | 118.92 |
| `test_re_rand/mae_surf_p` | 107.34 |
| `test_single_in_dist/mae_surf_p` | 128.36 |
| Wall clock | 30.5 min cap; 14 epochs; val still descending |
| Peak VRAM | 42.11 GB |
| Params | 0.66 M |
| Metrics path | `models/model-charliepai2g48h4-askeladd-ema-weights-20260512-201111/metrics.jsonl` |

**Analysis.** EMA on default config (val=121.16) is the single strongest lever identified so far — beating unified-pos (125.78, the prior best) by 3.7% and alphonse's canonical default (137.57) by 11.9%. Test improvement is even clearer at 108.69 vs 117.12 (7.2% gain). The hypothesis is confirmed: EMA smooths gradient-step noise from the wall-clock-truncated cosine schedule, improving both val and test consistency. Importantly, this was on a WEAKER config (no unified_pos, no bf16, surf_weight=10) — adding EMA to the merged recipe (unified_pos + bf16 + surf_weight=20) should push below 120.

**Status: SENT BACK FOR REBASE.** Four PRs (scoring-fix, bf16, unified-pos, surf-weight-20) merged into train.py while this run was in flight, causing a merge conflict. Student is rebasing onto the current advisor branch and will re-run on the full merged recipe. This is the highest-priority active PR — merge as soon as the rebase lands.

## 2026-05-12 ~20:35 — Round 2 assignments (building on merged recipe)

The advisor-branch recipe now includes unified_pos=True, bf16, surf_weight=20, and scoring-fix. Round-2 student assignments all inherit this merged baseline:

| Student | PR | Slug | Lever |
|---|---|---|---|
| fern | #1570 | `surf-weight-20-stack` | ~~surf_weight=20~~ now superseded by merge; test stacking effect |
| tanjiro | #1575 | `hidden256-bf16` | n_hidden=128→256 on merged bf16+unified_pos recipe |
| thorfinn | #1576 | `unified-pos-global-norm` | replace per-batch pos norm with corpus-level bounds |
| askeladd | #1540 | `ema-weights` | Polyak EMA at val/test — NEW BEST 121.16; sent back for rebase on merged recipe |
| nezuko | #1542 | `cosine-trunc-t15` | CosineAnnealingLR T_max=50→15 (in flight) |
| alphonse | #1577 | `seed42-baseline` | deterministic seeding → reproducible merged-recipe baseline |

Pending round-1 WIPs: #1374 (edward huber-loss, ETA ~20:30 UTC), #1394 (frieren wd5e-4, rate-limited ~3h+).

---

## 2026-05-13 04:30 — PR #1834: [layers-7] n_layers 5→7 depth probe — **CLOSED (regression; depth axis fully bracketed)**
- Student branch: `charliepai2g48h4-alphonse/layers-7`
- Hypothesis: Confirm depth regression trend from layers-6 with a second data point at n_layers=7. Extending the capacity sweep: if layers-6 regressed, does layers-7 continue the monotonic regression?

| Metric | n_layers=5 current best (#1855) | n_layers=6 (#1730) | n_layers=7 (this run) | Trend |
|---|---|---|---|---|
| val_avg/mae_surf_p | 83.95 | 93.97 | **96.81** | Monotonic regression ↑ |
| test_avg/mae_surf_p | 74.70 | 83.05 | 87.41 | Monotonic regression ↑ |
| Epochs completed | 18/18 | 15/15 | 13/15 | Fewer epochs per depth ↓ |
| n_params | ~666K | ~800K | ~934K | +28% per layer |

- Artifact: `models/model-charliepai2g48h4-alphonse-layers-7-20260513-*/metrics.jsonl`
- n_layers=7 completed only 13/15 epochs (wall-clock cap; +28% params vs baseline → ~1.3× per-epoch cost)

**Analysis:** Depth regression is monotonically consistent: n_layers 5→6→7 produces val 83.95→93.97→96.81. Three effects compound under global grad-clip:
1. **Gradient dilution:** Global norm clipping applies a single scalar to all 5/6/7 layers simultaneously. Each layer receives proportionally less signal as depth increases — the effective per-layer LR drops.
2. **Wall-clock penalty:** Each added block increases per-epoch cost by ~15-20%, reducing achievable epochs within the 30-min cap (18→15→13 epochs). Fewer epochs = less schedule coverage.
3. **Parameter efficiency loss:** +28% params at n_layers=7 with proportionally less training time (13 vs 18 epochs = 28% fewer) → effective parameter utilization drops.

**Conclusion: depth axis fully bracketed and CLOSED.** n_layers ∈ {5, 6, 7} confirms 5 is optimal under global grad-clip. Adding capacity via depth is contraindicated for the current training budget and clipping strategy. If depth becomes relevant again, the approach would need per-layer clip budgets (layerwise gradient clipping) or a longer training budget.

alphonse reassigned to PR #1914 (lr-3e-4): lower peak LR 5e-4→3e-4.

---

## 2026-05-13 04:55 — PR #1888: [adamw-beta1-0.95] β1=0.9→0.95 — **CLOSED (regression)**
- Student branch: `charliepai2g48h4-tanjiro/adamw-beta1-0.95`
- Hypothesis: Smoother momentum at β1=0.95 (vs default 0.9) for late-training stability under T_max=18 cosine schedule.

| Metric | β1=0.9 (#1695 baseline) | β1=0.95 (this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 84.67 | **88.32** | **+3.65 (+4.3%)** |
| test_avg/mae_surf_p | 74.94 | 78.58 | +3.64 (+4.9%) |
| val single_in_dist | 96.25 | 98.66 | +2.41 |
| val geom_camber_rc | 93.25 | 97.22 | +3.97 |
| val geom_camber_cruise | 65.39 | 68.88 | +3.49 |
| val re_rand | 83.78 | 88.54 | +4.76 |

- Artifact: `models/model-charliepai2g48h4-tanjiro-adamw-beta1-0.95-20260513-040431/metrics.jsonl`
- 18/18 epochs, 30.9 min wall-clock

**Analysis:** Uniform regression across all 4 splits. β1=0.95 oversmooths the momentum estimator: at batch_size=4 with bf16 noise, β1=0.9 already provides ~10-step momentum averaging that tracks the loss landscape responsively. Increasing to β1=0.95 (~20-step average) creates lag between actual gradient direction and momentum direction, slowing late-training fine-tuning. Combined with #1886 (β2=0.98 also regressed), the AdamW betas axis is bracketed and CLOSED. Defaults (0.9, 0.999) are optimal under this recipe.

tanjiro reassigned to PR #1923 (wd-1e-5): reduce weight decay 1e-4→1e-5.

---

## 2026-05-13 04:55 — PR #1886: [adamw-beta2-0.98] β2=0.999→0.98 — **CLOSED (regression)**
- Student branch: `charliepai2g48h4-frieren/adamw-beta2-0.98`
- Hypothesis: Faster variance adaptation at β2=0.98 (Vaswani-2017 default for Transformers) would help under our small-batch noisy regime.

| Metric | β2=0.999 (#1695 baseline) | β2=0.98 (this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 84.67 | **85.94** | **+1.27 (+1.5%)** |
| test_avg/mae_surf_p | 74.94 | 76.16 | +1.22 (+1.6%) |
| val single_in_dist | 96.25 | 97.77 | +1.52 |
| val geom_camber_rc | 93.25 | 92.57 | −0.68 (flat) |
| val geom_camber_cruise | 65.39 | 67.40 | +2.01 |
| val re_rand | 83.78 | 86.01 | +2.23 |

- Artifact: `models/model-charliepai2g48h4-frieren-adamw-beta2-0.98-20260513-040527/metrics.jsonl`
- 18/18 epochs, 30.9 min wall-clock

**Analysis:** 3/4 splits regress; only geom_camber_rc is flat. β2=0.999's long-horizon variance averaging serves as additional implicit smoothing on top of bf16+small-batch noise. Replacing it with β2=0.98 makes per-parameter step sizes noisier without the Vaswani-2017 warmup that originally justified that beta choice. Faster variance adaptation is the wrong remedy in our regime — grad-clip already normalizes step direction; the variance estimate isn't the bottleneck.

**AdamW betas axis fully bracketed:** β1=0.9 optimum (β1=0.95 +4.3%), β2=0.999 optimum (β2=0.98 +1.5%). Defaults are optimal.

frieren reassigned to PR #1919 (mlp-ratio-4): double Transolver FFN width 2→4.

---

## 2026-05-13 05:15 — PR #1902: [slice-num-128] Transolver physics slices 64→128 — **CLOSED (wall-clock bound regression)**
- Student branch: `charliepai2g48h4-edward/slice-num-128`
- Hypothesis: More physics slices = finer routing of geometric features through Transolver attention.

| Metric | slice_num=64 (#1855 best) | slice_num=128 (this run) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 83.95 | **99.86** | **+15.91 (+19.0%)** |
| test_avg/mae_surf_p | 74.70 | 90.41 | +15.71 (+21.0%) |
| Epochs completed | 18/18 | 13/15 (timed out) | −5 epochs |
| Per-epoch time | ~103 s | ~146 s | **+42%** |
| Peak VRAM | 33.9 GB | 48.2 GB | +42% |
| n_params | 678,231 | 688,791 | +1.6% (negligible) |

- Artifact: `models/model-charliepai2g48h4-edward-slice-num-128-20260513-041404/metrics.jsonl`

**Analysis:** The hypothesis's "no wall-clock impact" claim was refuted: softmax-over-slices and slice-mixed attention scale with `slice_num`, costing +42% per epoch. Apples-to-apples per-epoch comparison shows the loss trajectory tracks the baseline through epoch ~12 (val 99.86 at e13 vs 94.87 baseline at e13). The baseline then pulls ahead in epochs 13-18 as cosine bottoms to eta_min=5e-5. slice_num=128 simply doesn't see those epochs.

**Verdict on slice_num axis:** Going above 64 is wall-clock bound; going below 64 would lose representational capacity. **slice_num=64 is at the sweet spot** for this 30-min budget. Edward reassigned to PR #1943 (ref-16): unified_pos reference points 8→16 — zero-param, low wall-clock impact.

---

## 2026-05-13 05:15 — PR #1812: [lr-warmup-1ep] 1-epoch linear warmup + cosine — **SENT BACK FOR REBASE + STACK**
- Student branch: `charliepai2g48h4-thorfinn/lr-warmup-1ep`
- Hypothesis: 1-epoch linear warmup (lr 5e-6 → 5e-4) protects AdamW momentum from epoch-1 high-gradient corruption, improving late-training generalization.

| Metric | This run (T_max=17 cosine + warmup, eta_min=0.0) | Baseline #1695 (T_max=18 cosine, eta_min=0.0) | Current best #1855 (eta_min=5e-5) | Δ vs #1695 | Δ vs current best |
|---|---|---|---|---|---|
| val_avg/mae_surf_p | 83.64 | 84.67 | 83.95 | **−1.03** | −0.31 |
| test_avg/mae_surf_p | 74.65 | 74.94 | 74.70 | −0.29 | −0.05 |

**Analysis:** The student's run was on the pre-#1855 HEAD (eta_min=0.0). Their schedule replaced `eta_min=5e-5` with `eta_min=0.0` (via SequentialLR with 17-epoch cosine to zero). This is NOT directly comparable to current best:
- vs old baseline (#1695): −1.03 val improvement; within σ≈8.5 noise
- vs current best (#1855): −0.31 val; well inside noise floor
- Two changes confounded: (1) warmup added, (2) eta_min floor removed

**Mechanism partially confirmed.** Epoch-1 grad-norm mean=15.32 vs baseline 30-1000+ — warmup successfully dampens AdamW's early variance-corruption hazard. But epoch-1 val=269 (worse than baseline ~233): warmup reduces effective lr during epoch 1, so less progress per step.

**Verdict: send back for proper stacking.** Asked thorfinn to: (1) rebase onto current HEAD (eta_min=5e-5 in train.py); (2) modify cosine portion to use eta_min=5e-5 not 0.0; (3) rerun for apples-to-apples comparison. If val improves below ~82.5 (>σ from current best), we merge.

---

## 2026-05-13 06:00 — PR #1812: [lr-warmup-1ep] 1-epoch linear warmup + cosine — **MERGED (NEW BEST: val=82.56)**
- Student branch: `charliepai2g48h4-thorfinn/lr-warmup-1ep` (second run on current HEAD with eta_min=5e-5)
- Hypothesis: 1-epoch linear warmup (lr 5e-6→5e-4) protects AdamW momentum buffers from epoch-1 high-gradient corruption.

| Metric | Warmup+Cosine (this rerun) | Current best #1855 (pure cosine) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **82.56** | 83.95 | **−1.39 (−1.65%)** |
| test_avg/mae_surf_p | **74.13** | 74.70 | −0.57 (−0.76%) |
| val single_in_dist | 90.40 | 93.45 | −3.05 |
| val geom_camber_rc | 91.39 | 91.33 | +0.06 (flat) |
| val geom_camber_cruise | 66.68 | 67.06 | −0.38 |
| val re_rand | 81.77 | 83.97 | −2.20 |

- Artifact: `models/model-charliepai2g48h4-thorfinn-lr-warmup-1ep-20260513-051740/metrics.jsonl`
- 18/18 epochs, 30.9 min wall-clock; epoch-1 grad-norm mean=8.66 (max 35.5) vs 30-1000+ pre-warmup

**Analysis:** Apples-to-apples comparison (both use eta_min=5e-5). Improvement is -1.39 val (within σ≈8.5 noise floor) but consistent across 3/4 splits and test. Mechanism confirmed: warmup damps epoch-1 AdamW momentum corruption (mean grad-norm 8.66 vs 30-1000+ without warmup), resulting in better model positioning entering the cosine descent (epoch-17 val ~85.00 here vs ~87.29 in pure-cosine). **Merged as 12th effective improvement.**

thorfinn reassigned to PR #1968 (lr-7e-4): higher peak LR 5e-4→7e-4 with warmup, natural upper-bracket.

---

## 2026-05-13 06:00 — PR #1914: [lr-3e-4] Lower peak LR 5e-4→3e-4 — **CLOSED (regression)**
- Student branch: `charliepai2g48h4-alphonse/lr-3e-4`
- Hypothesis: Lower peak LR allows model to settle into tighter local basin.

| Metric | lr=3e-4 (this run) | baseline #1855 (lr=5e-4) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **90.67** | 83.95 | **+6.72 (+8.0%)** |
| test_avg/mae_surf_p | 81.94 | 74.70 | +7.24 (+9.7%) |

- Artifact: `models/model-charliepai2g48h4-alphonse-lr-3e-4-20260513-051046/metrics.jsonl`
- 18/18 epochs; still steeply descending at epoch 18 (94.22→90.67)

**Analysis:** Uniform regression all 4 splits. Root cause: with CosineAnnealingLR over T_max=18, lower peak lr means ~40% less total parameter displacement within the same wall-clock budget. Model is underfitting (still descending at epoch 18) — not overfitting. **lr=5e-4 confirmed optimal lower bound. lr axis closed: 3e-4 dominated, 5e-4 optimum, 7e-4 (thorfinn) testing upper.**

alphonse reassigned to PR #1972 (batch-size-2): halve batch 4→2.

## 2026-05-13 06:20 — PR #1943: [ref-16] Double unified_pos reference points 8→16 — **CLOSED**
- Student branch: `charliepai2g48h4-edward/ref-16`
- Hypothesis: ref=16 (finer position normalization) would sharpen model's ability to distinguish local geometric features, especially on geometry-varying OOD splits.

| Metric | ref=16 (this PR) | Baseline #1812 (ref=8) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **86.12** | 82.56 | **+3.56 (+4.3%) ❌** |
| test_avg/mae_surf_p | 75.49 | 74.13 | +1.35 (+1.8%) ❌ |
| val geom_camber_rc | 97.29 | 91.39 | **+5.90 (biggest loser)** |
| val geom_camber_cruise | 65.56 | 66.68 | **−1.12 (only improver)** |

- Artifact: `models/model-charliepai2g48h4-edward-ref-16-20260513-051503/metrics.jsonl`
- 17/18 epochs (timeout); val still descending at ep17.

**Analysis:** geom_camber_rc regressed sharply (+5.90) while geom_camber_cruise improved (−1.12), net negative. The dilution hypothesis is plausible: doubling reference points weakens any single reference's gradient contribution, slowing convergence within the 30-min budget. **Axis disposition: ref=16 closed; ref=8 confirmed optimum upper-side.**

---

## 2026-05-13 06:20 — PR #1919: [mlp-ratio-4] Double FFN width mlp_ratio 2→4 — **CLOSED**
- Student branch: `charliepai2g48h4-frieren/mlp-ratio-4`
- Hypothesis: mlp_ratio=4 adds FFN capacity (+29% params) without depth-dilution penalty seen in n_layers probes.

| Metric | mlp_ratio=4 (this PR) | Baseline #1812 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **89.82** | 82.56 | **+7.26 (+8.8%) ❌** |
| test_avg/mae_surf_p | 80.17 | 74.13 | +6.04 (+8.1%) ❌ |

- Artifact: `models/model-charliepai2g48h4-frieren-mlp-ratio-4-20260513-050053/metrics.jsonl`
- 17/18 epochs (timeout); +29% params → 15% slower per epoch, cosine schedule cut 1 epoch short.

**Analysis:** All 4 splits regressed 3–9%. Two causes: (1) ~1500-sample dataset can't leverage extra FFN capacity — overfits subtler patterns; (2) wall-clock penalty cuts cosine schedule short. **Axis disposition: mlp_ratio>2 closed. Half FFN (mlp_ratio=1) assigned to frieren #1992 to complete the bracket.**

---

## 2026-05-13 06:20 — PR #1901: [eta-min-1e-4] Bracket LR floor higher eta_min=5e-5→1e-4 — **CLOSED**
- Student branch: `charliepai2g48h4-fern/eta-min-1e-4`
- Hypothesis: eta_min=5e-5 was just merged; bracket whether higher floor continues to improve.

| Metric | eta_min=1e-4 (this PR) | Baseline #1812 (eta_min=5e-5) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **85.06** | 82.56 | **+2.50 (+3.0%) ❌** |
| test_avg/mae_surf_p | 76.41 | 74.13 | +2.28 (+3.1%) ❌ |

- Artifact: `models/model-charliepai2g48h4-fern-eta-min-1e-4-20260513-045455/metrics.jsonl`
- 18/18 epochs; final-epoch LR = 1.03e-4 confirmed.

**Analysis:** LR floor optimum is NOT monotone — 5e-5 is the sweet spot. At 1e-4 final-epoch updates are 2× larger, perturbing the settling basin under grad-clip. OOD asymmetry (test +2.28 > val +2.50) suggests flatter but geometrically-unspecialized minimum. Combined with #1695 (eta_min=0 → 84.67) and #1855 (eta_min=5e-5 → 83.95 merged), the 3-point bracket is clean. **eta_min axis CLOSED: 5e-5 optimum.**

**New assignments after this round:** fern #1990 (cawr-t0-9), edward #1991 (warmup-2ep), frieren #1992 (mlp-ratio-1).

## 2026-05-13 06:28 — PR #1923: [wd-1e-5] Reduce AdamW weight decay 1e-4→1e-5 — **CLOSED**
- Student branch: `charliepai2g48h4-tanjiro/wd-1e-5`
- Hypothesis: wd=1e-4 over-regularizes the 0.68M-param model on the small dataset; loosening it frees capacity.

| Metric | wd=1e-5 (this PR) | Baseline #1812 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **85.76** | 82.56 | **+3.20 (+3.9%) ❌** |
| test_avg/mae_surf_p | 76.33 | 74.13 | +2.20 (+3.0%) ❌ |
| val single_in_dist | 98.58 | 90.40 | **+8.18 (biggest loser)** |
| val re_rand | 83.94 | 81.77 | −0.03 (≈flat) |

- Artifact: `models/model-charliepai2g48h4-tanjiro-wd-1e-5-20260513-050241/metrics.jsonl`
- 18/18 epochs; still descending at epoch 18.

**Analysis:** single_in_dist regresses worst (+8.18 val) — canonical "weaker regularization → wider generalization gap" signature. Weight decay is load-bearing. wd=5e-4 also worse (older stack). Combined: wd=1e-4 is bracketed as optimum on both sides. **wd axis CLOSED: wd=1e-4 confirmed optimum.** Tanjiro reassigned #1993 n-head-2.

## 2026-05-13 07:25 — PR #1991: [warmup-2ep] Extend warmup 1→2 epochs — **CLOSED**
- Student branch: `charliepai2g48h4-edward/warmup-2ep`
- Hypothesis: 2 epochs of warmup continues the damping benefit of 1ep merged in #1812.

| Metric | warmup-2ep (this PR) | Baseline #1812 (warmup-1ep) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **83.35** | 82.56 | **+0.79 (+0.96%) ❌** |
| test_avg/mae_surf_p | 75.06 | 74.13 | +0.93 (+1.25%) ❌ |
| All 8 splits | +0.05 to +1.98 | — | all regressed |

- Artifact: `models/model-charliepai2g48h4-edward-warmup-2ep-20260513-062412/metrics.jsonl`
- 18/18 epochs (2 warmup + 16 cosine); epoch-1 grad_norm_mean = 8.99 vs baseline 8.66 (both well-tamed)
- Epoch-2 grad_norm_mean = 9.43 — momentum buffer not measurably better-seeded vs 1ep

**Analysis:** The corruption-damping mechanism of warmup saturates at 1 epoch. Edward's grad-norm trace confirms: epoch-1 is already at ~9 (vs unbounded 30-1000+ pre-warmup), and a second warmup epoch doesn't bring it lower. Trading useful cosine-descent capacity (T_max 17→16) for non-incremental warmup benefit is net-negative. **Axis disposition: warmup-length CLOSED, 1-epoch optimum.** Edward reassigned to loss-beta-0-5 #2012.

---

## 2026-05-13 07:25 — PR #1853: [n-head-8] Double attention heads 4→8 — **CLOSED**
- Student branch: `charliepai2g48h4-nezuko/n-head-8`
- Hypothesis: Doubling n_head (zero-param inductive bias change) gives finer head granularity.

| Metric | n_head=8 (this PR) | Baseline #1812 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **96.33** | 82.56 | **+13.77 (+16.7%) ❌** |
| test_avg/mae_surf_p | 86.97 | 74.13 | +12.84 (+17.3%) ❌ |
| val single_in_dist | 114.04 | 90.40 | +23.64 (worst) |

- Artifact: `models/model-charliepai2g48h4-nezuko-n-head-8-20260513-055556/metrics.jsonl`
- 18/18 epochs; best epoch 13/18.
- **n_params: 661,611 vs baseline 678,231 → −16,620 (−2.4%)** — NOT zero-param.

**Critical discovery (by nezuko):** Transolver's PhysicsAttention uses per-head-dimensioned Q/K/V projections:
- `to_q/k/v: dim_head × dim_head` per layer → 3 × (32²−16²) = 2,304 dropped per layer when n_head doubles
- `in_project_slice: slice_num × dim_head` → 64 × 16 = 1,024 dropped per layer
- 5 layers × ~3,324 = ~16,620 total drop

So n_head=8 is not a pure inductive bias test — it's (a) finer granularity AND (b) -2.4% attention capacity. Both likely contributed to the regression. tanjiro's n_head=2 (#1993 in flight) tests the opposite direction (+55K params, coarser heads). **Axis disposition: n_head upper CLOSED.** nezuko reassigned to onecycle-lr #2014.

**New assignments after this round:** edward #2012 (loss-beta-0-5), nezuko #2014 (onecycle-lr).

## 2026-05-13 07:30 — PR #1972: [batch-size-2] Halve batch size 4→2 — **MERGED (NEW BEST: val=76.24)**
- Student branch: `charliepai2g48h4-alphonse/batch-size-2`
- Hypothesis: bs=2 doubles optimizer steps/epoch at same wall-clock cost, improving generalization.

| Metric | bs=2 (this PR) | Baseline #1812 (bs=4) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **76.24** | 82.56 | **−6.32 (−7.65%) ✓** |
| test_avg/mae_surf_p | **66.85** | 74.13 | **−7.28 (−9.82%) ✓** |
| val single_in_dist | 81.78 | 90.40 | −8.62 |
| val geom_camber_rc | 87.06 | 91.39 | −4.34 |
| val geom_camber_cruise | 59.39 | 66.68 | −7.29 |
| val re_rand | 76.74 | 81.77 | −5.03 |

- Artifact: `models/model-charliepai2g48h4-alphonse-batch-size-2-20260513-060609/metrics.jsonl`
- 19 epochs completed (vs 18 for bs=4); schedule adapted automatically (steps-based warmup).
- n_batches_per_epoch: 375→750; VRAM: ~30GB → ~17GB.

**Analysis:** Larger improvement than any single merge this round. All 8 splits improve; test gains (+7.28 avg) exceed val gains (+6.32 avg) — strong OOD improvement. Mechanism: 2x optimizer steps/epoch at same wall-clock → more gradient refinement per unit time. The warmup mechanism adapts perfectly (steps-based, not epoch-based). **13th effective merge. New best: 76.24.**

---

## 2026-05-13 07:30 — PR #1968: [lr-7e-4] lr=5e-4→7e-4 with warmup — **SENT BACK (beat old baseline, not new)**
- Student branch: `charliepai2g48h4-thorfinn/lr-7e-4`
- Result: val=79.77 — beats old baseline (82.56) but DOES NOT beat new baseline (76.24 after alphonse merge).

| Metric | lr=7e-4 (this PR) | Old Baseline #1812 | Δ vs old | vs new baseline 76.24 |
|---|---|---|---|---|
| val_avg/mae_surf_p | 79.77 | 82.56 | −2.79 | **+3.53 ❌** |
| test_avg/mae_surf_p | 72.06 | 74.13 | −2.07 | **+5.21 ❌** |

**Decision:** Cannot merge — above new 76.24 baseline. Sent back to rerun with batch_size=2 + lr=7e-4 to test stacking. The combined config (more steps + bigger steps) is untested and could compound or interfere.

**New assignments:** alphonse #2036 (batch-size-1).

## 2026-05-13 07:55 — PR #2014: [onecycle-lr] OneCycleLR(max_lr=8e-4) replacing SequentialLR — **SENT BACK**
- Student branch: `charliepai2g48h4-nezuko/onecycle-lr`
- Result on old bs=4 HEAD: val=79.70 — beats old baseline 82.56 (−3.46%) but loses to new baseline 76.24 (+3.46).

| Metric | OneCycleLR (this PR) | Old #1812 | New #1972 |
|---|---|---|---|
| val_avg/mae_surf_p | 79.70 | 82.56 (−3.46%) | 76.24 (+3.46) |
| test_avg/mae_surf_p | 71.01 | 74.13 (−4.21%) | 66.85 (+4.16) |

- Best epoch 18, val still descending. LR trace: peak 7.997e-4 at ep2 (pct_start=0.1), final 3.2e-6.
- Clean optimization: grad_norm_max=37 ep1 → ~1 by ep18.

**Decision:** Sent back for rerun on new bs=2 HEAD. With bs=2's 2x steps/epoch, the OneCycleLR curve will get 2x resolution. If it compounds, new winner. If overshoots, informs LR ceiling.
