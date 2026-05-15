# SENPAI Research Results — `icml-appendix-willow-pai2i-48h-r3`

Logged per advisor review of each PR.

## 2026-05-15 — Launch: round 3 of willow-pai2i-48h begins

All 8 students idle; no PRs in flight. First round of assignments dispatched (each PR runs dual-arm baseline + variant in the same wandb_group, since no canonical baseline run exists yet on this branch state).

| Student | PR | Hypothesis | Family |
|---|---|---|---|
| alphonse | [#3140](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3140) | Wider hidden + more heads (128→192, 4→6) | Capacity |
| askeladd | [#3147](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3147) | LR warmup + peak 5e-4→1e-3 (3-epoch linear warmup) | Optimization |
| edward | [#3152](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3152) | Per-channel loss weighting (p x3 in MSE) | Loss formulation |
| fern | [#3155](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3155) | Huber loss instead of MSE (delta=1.0) | Loss formulation |
| frieren | [#3161](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3161) | Per-sample loss normalization (equal-weight per sample) | Loss formulation |
| nezuko | [#3165](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3165) | Depth scaling (5->8 layers) | Capacity |
| tanjiro | [#3169](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3169) | MLP ratio (2->4) | Capacity |
| thorfinn | [#3172](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3172) | Fourier position features + slice_num 64->96 | Inputs |

## 2026-05-15 14:30 — PR #3140 (alphonse): Widen Transolver n_hidden 128->192, n_head 4->6 — **CLOSED**

- Branch: `willowpai2i48h3-alphonse/capacity-width-heads`
- W&B group: [`capacity-width-heads`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/capacity-width-heads)
- Baseline run: `xehwt9bi` | Variant run: `t2z5ya27`

**Hypothesis:** Widening Transolver capacity (n_hidden 128->192, n_head 4->6) would reduce val_avg/mae_surf_p by 3-8% if the baseline were underfit.

**Result (variant vs baseline, lower is better):**

| Metric | Baseline | Variant | Δ |
|---|---|---|---|
| best_val_avg/mae_surf_p | **135.30** (ep 13) | 160.61 (ep 8) | +18.7% |
| val_geom_camber_cruise/mae_surf_p | 109.77 | 140.94 | +28.4% |
| val_geom_camber_rc/mae_surf_p | 135.88 | 171.21 | +26.0% |
| val_re_rand/mae_surf_p | 127.66 | 139.93 | +9.6% |
| val_single_in_dist/mae_surf_p | 167.88 | 190.37 | +13.4% |
| test_avg/mae_surf_p (excl cruise) | 135.54 | 154.32 | +13.9% |
| Params | 0.66M | 1.45M | 2.19× |
| Sec/epoch | 131.8 | 203.8 | 1.55× |
| Epochs reached / 50 (30-min cap) | 14 (best @13) | 9 (best @8) | -36% |

**Decision:** Closed. Variant >5% worse on every track, well past the merge threshold.

**Analysis:**
- The wider variant is uniformly worse on val and test under the 30-min wall-clock cap.
- Student's epoch-matched comparison (variant ep 8 vs baseline ep 8) suggests per-epoch the wider model may be slightly better, but the 1.55× slower epoch time costs ~36% of epoch count under the wall-clock budget. The variant best-checkpointed at epoch 8/9 — almost certainly still under-trained.
- **Implication for round 2:** width-side scaling is wall-clock-penalized in this regime. Future round assignments should bias toward changes that don't increase per-step cost (optimizers, losses, data aug, input feature engineering).

**Pre-existing bug surfaced:** `test_geom_camber_cruise/mae_surf_p` returns NaN on the baseline arm. Non-finite pressure prediction on at least one test-cruise sample propagates Inf through `data/scoring.py:accumulate_batch`. Affects ALL PRs equally — tracked separately (see CURRENT_RESEARCH_STATE.md).

**Sets the canonical round-3 baseline:** `val_avg/mae_surf_p = 135.30`, `test_avg/mae_surf_p (excl cruise) = 135.54`. See BASELINE.md.

## 2026-05-15 15:00 — PR #3152 (edward): Per-channel loss weighting (p ×3 in MSE) — **REQUEST CHANGES**

- Branch: `willowpai2i48h3-edward/channel-loss-weight-p`
- W&B group: [`channel-loss-weight-p`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/channel-loss-weight-p)
- Baseline run: `sw37lp52` | Variant run: `4rt8g0xb`

**Hypothesis:** Upweight the p channel by 3× in MSE to align gradients with the primary metric `mae_surf_p`. Predicted −5 to −15%.

**Result (variant minus baseline, lower is better):**

| Metric | Baseline (1:1:1) | Variant (1:1:3) | Δ% |
|---|---|---|---|
| best_val_avg/mae_surf_p | **137.63** (ep 11) | 138.51 (ep 14) | +0.6% |
| val_geom_camber_rc/mae_surf_p | 146.58 | 170.86 | **+16.6%** |
| val_geom_camber_cruise/mae_surf_p | 107.67 | 100.75 | −6.4% |
| val_re_rand/mae_surf_p | 127.19 | 117.40 | −7.7% |
| val_single_in_dist/mae_surf_p | 169.09 | 165.01 | −2.4% |
| val_avg/mae_surf_Ux | 2.207 | 2.658 | +20.4% |
| val_avg/mae_surf_Uy | 0.928 | 1.056 | +13.9% |
| val_avg/mae_vol_p | 137.72 | 145.15 | +5.4% |
| 3-split test mean (excl cruise) | 136.36 | 140.57 | +3.1% |
| Epochs reached / 50 (30-min cap) | 18 (best @11) | 18 (best @14) | — |

**Decision:** REQUEST CHANGES. Sent back as draft with surface-only upweight follow-up.

**Reason for not closing despite regression:**
- Headline Δ on val_avg is only +0.6% — within the noise floor revealed by truncation (both arms stopped at 18/50 epochs).
- Per-split deltas show real signal: cruise and re_rand benefit (-6.4%, -7.7%), suggesting p-focus helps on the smoother/easier OOD geometry, but rc regresses sharply (+16.6%) because the shared velocity-pressure representation is needed for rc.
- Cross-check: this PR's baseline arm hits 137.63 vs canonical baseline 135.30 (+1.7%). Single-seed dual-arm comparisons are noisy at the few-percent level under wall-clock truncation.

**Follow-up requested:** Run a third arm `variant-p3x-surf-only` in the same wandb_group. Apply [1,1,3] weighting **only to the surface portion** of the loss (vol_loss stays balanced [1,1,1]). This preserves velocity learning on volume nodes (~99% of mesh) while targeting surface-p specifically. Decision rule: merge if surface-only variant beats baseline by ≥2% on val_avg/mae_surf_p AND velocity MAEs degrade <5%; close otherwise.

**Notable:** Student independently identified the cruise-test NaN bug (matches alphonse's earlier observation in PR #3140). Cross-channel coupling analysis is a useful contribution — confirms that *normalized-space* MSE underweights p relative to its physical contribution (suggests a future hypothesis around physical-units / scale-aware loss).

## 2026-05-15 16:00 — PR #3155 (fern): Huber loss (SmoothL1 beta=1.0) instead of MSE — **MERGED**

- Branch: `willowpai2i48h3-fern/huber-loss`
- W&B group: [`huber-loss`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/huber-loss)
- Baseline run: `dqe6ejfz` | Variant run: `3nivkqy0`

**Hypothesis:** Huber (SmoothL1, delta=1.0) is more robust to outlier pressure samples than MSE; predicted −5 to −15% on val_avg/mae_surf_p.

**Result (variant vs baseline, lower is better):**

| Metric | Baseline (dqe6ejfz) | Variant (3nivkqy0) | Δ vs in-PR baseline | Δ vs canonical |
|---|---|---|---|---|
| best_val_avg/mae_surf_p | 138.43 (ep 14) | **110.83** (ep 13) | **−19.9%** | **−18.1%** |
| val_single_in_dist/mae_surf_p | 159.77 | 132.06 | −17.3% | |
| val_geom_camber_rc/mae_surf_p | 175.41 | 124.13 | −29.2% | |
| val_geom_camber_cruise/mae_surf_p | 100.26 | 82.72 | −17.5% | |
| val_re_rand/mae_surf_p | 118.30 | 104.40 | −11.7% | |
| test 3-split avg (excl cruise) | 139.80 | **109.75** | **−21.5%** | **−19.0%** |
| test_single_in_dist/mae_surf_p | 142.35 | 118.65 | −16.6% | |
| test_geom_camber_rc/mae_surf_p | 161.60 | 111.97 | −30.7% | |
| test_re_rand/mae_surf_p | 115.46 | 98.64 | −14.6% | |
| test_geom_camber_cruise/mae_surf_p | NaN | NaN | — | pre-existing |
| Epochs (30-min cap) | 14 | 13 | — | — |

**Decision:** MERGED. Clean single-variable diff (sq_err → F.smooth_l1_loss). −18.1% val, −19.0% test vs canonical. All 4 val splits improve (11–29%). No wall-clock penalty (same 30-min cap, 13/14 epochs).

**Analysis:**
- Largest single gain in this launch. Huber robustness to outliers is the dominant driver: the rc and single_in_dist splits (which have the most geometric variation and likely the most outlier pressure samples) benefit the most.
- W&B summary value (126.22) is the last-epoch value after oscillation — `best_val_avg/mae_surf_p = 110.83` at best-epoch checkpoint is the correct metric. Student reported correctly.
- **Sets new canonical baseline:** `val_avg/mae_surf_p = 110.83`, `test_avg/mae_surf_p (excl cruise) = 109.75`. See BASELINE.md.
- **Follow-up priority:** Huber delta sensitivity (delta=0.5 and 2.0) — determine if 1.0 is optimal for this scale.

## 2026-05-15 16:00 — PR #3147 (askeladd): LR warmup + peak bump 5e-4→1e-3 (3-epoch linear warmup) — **MERGED**

- Branch: `willowpai2i48h3-askeladd/lr-warmup-peak`
- W&B group: [`lr-warmup-peak`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/lr-warmup-peak)
- Baseline run: `pckvl17x` | Variant run: `gyl9qikv`

**Hypothesis:** 3-epoch linear warmup to peak lr=1e-3, then cosine anneal to 0. Predicted −5 to −15%.

**Result (variant vs baseline, lower is better):**

| Metric | Baseline (pckvl17x) | Variant (gyl9qikv) | Δ vs in-PR baseline | Δ vs canonical |
|---|---|---|---|---|
| best_val_avg/mae_surf_p | 131.88 (ep 14) | **123.20** (ep 14) | **−6.6%** | **−8.9%** |
| val_single_in_dist/mae_surf_p | 168.53 | 160.29 | −4.9% | |
| val_geom_camber_rc/mae_surf_p | 146.49 | 132.83 | −9.3% | |
| val_geom_camber_cruise/mae_surf_p | 97.22 | 89.11 | −8.3% | |
| val_re_rand/mae_surf_p | 115.27 | 110.56 | −4.1% | |
| test 3-split avg (excl cruise) | 133.85 | **121.06** | **−9.6%** | **−10.7%** |
| test_single_in_dist/mae_surf_p | 153.53 | 137.27 | −10.6% | |
| test_geom_camber_rc/mae_surf_p | 131.94 | 118.62 | −10.1% | |
| test_re_rand/mae_surf_p | 116.07 | 107.29 | −7.6% | |
| test_geom_camber_cruise/mae_surf_p | NaN | NaN | — | pre-existing |
| Epochs (30-min cap) | 14 | 14 | — | — |

**Decision:** MERGED (against canonical baseline 135.30/135.54; variant 123.20/121.06 beats canonical by −8.9%/−10.7%). Clean single-variable diff (LR=1e-3 + 3-epoch LinearLR warmup → SequentialLR). All 4 val splits improve. No wall-clock penalty.

**Analysis:**
- Both arms hit 30-min cap at exactly 14 epochs — neither ran the cosine tail. The gain is conservative: warmup typically helps most in later training where the properly-warmed model finds a better basin.
- Consistent 4–10% improvement across all splits including OOD (rc −9.3%, re_rand −4.1%) suggests warmup is a general optimizer improvement rather than a memorization/overfitting artifact.
- **Compounding with Huber:** Huber+LR-warmup was not tested jointly; these are likely orthogonal (loss vs schedule). Next round will test this directly.
- **Follow-up:** Askeladd should tune warmup duration (2 vs 5 epochs) on the new Huber+LR-warmup baseline.

## 2026-05-15 16:00 — PR #3161 (frieren): Per-sample loss normalization — **CLOSED**

- Branch: `willowpai2i48h3-frieren/per-sample-loss-norm`
- W&B group: [`per-sample-loss-norm`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/per-sample-loss-norm)
- Baseline run: `jmuo7vtx` | Variant run: `x158gao1`

**Hypothesis:** Normalizing loss per sample (equal weight per sample not per node) would improve OOD by preventing large-mesh samples from dominating. Predicted −3 to −8%.

**Result (variant vs baseline, lower is better):**

| Metric | Baseline (jmuo7vtx) | Variant (x158gao1) | Δ% |
|---|---|---|---|
| best_val_avg/mae_surf_p | 130.40 (ep 14) | 147.36 (ep 13) | **+13.0% worse** |
| val_single_in_dist/mae_surf_p | 172.21 | 175.50 | +1.9% |
| val_geom_camber_rc/mae_surf_p | 135.10 | 153.78 | +13.8% |
| val_geom_camber_cruise/mae_surf_p | 99.30 | 118.74 | +19.6% |
| val_re_rand/mae_surf_p | 114.98 | 141.44 | +23.0% |
| test 3-split avg (excl cruise) | 127.49 | 139.44 | **+9.4% worse** |

Variant vs canonical baseline (135.30): variant is +8.9% worse.

**Decision:** CLOSED. Variant uniformly worse across all 4 val splits (1–23% regression). The per-sample normalization direction is empirically falsified in this setting.

**Analysis:**
- The baseline arm here (130.40) is notably stronger than the canonical baseline (135.30) — confirming the evaluation is sound and the regression is real.
- Per-sample normalization treats a 242K-node mesh the same as a 74K-node mesh. This changes the effective batch distribution and may destabilize learning since gradients from structurally different samples are now equally weighted.
- Student's suggested follow-ups (mesh-size upweighting, per-sample magnitude weighting) are worth tracking in the research ideas backlog but are not immediately prioritized.

## 2026-05-15 16:00 — PR #3165 (nezuko): Depth scaling n_layers 5→8 — **CLOSED**

- Branch: `willowpai2i48h3-nezuko/depth-8-layers`
- W&B group: [`depth-8-layers`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/depth-8-layers)
- Baseline run: `gpf8gh2a` | Variant run: `zmlvfufz`

**Hypothesis:** Deeper Transolver (5→8 layers, +60% more layers) reduces underfitting. Predicted −5 to −15% under the 30-min cap.

**Result (variant vs baseline, lower is better):**

| Metric | Baseline (gpf8gh2a) | Variant (zmlvfufz) | Δ% |
|---|---|---|---|
| best_val_avg/mae_surf_p | 125.07 (ep 12) | 156.85 (ep 9) | **+25.4% worse** |
| val_single_in_dist/mae_surf_p | 120.51 | 151.22 | +25.5% |
| val_geom_camber_rc/mae_surf_p | 120.97 | 148.84 | +23.0% |
| val_geom_camber_cruise/mae_surf_p | 105.09 | 157.37 | +49.8% |
| val_re_rand/mae_surf_p | 133.73 | 170.47 | +27.5% |
| test 3-split avg (excl cruise) | 121.09 | 156.35 | **+29.1% worse** |
| Sec/epoch | 132.2s | 205.2s | 1.55× slower |
| Epochs (30-min cap) | 14 (best @12) | 9 (best @9) | −36% |

Variant vs canonical baseline (135.30): variant is +15.9% worse.

**Decision:** CLOSED. +25.4% worse than in-PR baseline, +15.9% worse than canonical. All 4 splits regress. Same 1.55× wall-clock penalty as PR #3140 (width scaling) — depth scaling is equally penalized.

**Analysis:**
- Confirms the pattern from PR #3140: any capacity scaling that slows per-epoch time by ~1.55× loses ~36% of the epoch budget, and the under-trained deeper model cannot compensate.
- The cruise split is worst (+49.8%) — OOD geometry is most sensitive to under-training.
- **Implication:** Capacity scaling of any kind (width, depth, MLP ratio) is essentially disallowed under the 30-min wall-clock cap with the current mesh sizes. Future capacity experiments should either (a) target faster operation per step or (b) explicitly test at epoch-matched (not wall-clock-matched) comparisons in a longer run.
