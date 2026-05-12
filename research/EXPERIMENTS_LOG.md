# SENPAI Research Results — `icml-appendix-charlie-pai2g-48h-r2`

---

## 2026-05-12 18:55 — PR #1418: Per-channel loss weight: upweight pressure 3×

- **Branch:** `charliepai2g48h2-askeladd/pressure-channel-weight`
- **Hypothesis:** Add per-channel loss weights [Ux=1, Uy=1, p=3] (normalized by sum=5) to squared error in training and eval. Upweighting pressure targets the primary `val_avg/mae_surf_p` ranking metric.
- **Status:** MERGED ✅ — new round-1 baseline

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 14) | **122.6395** |
| val_single_in_dist/mae_surf_p | 145.914 |
| val_geom_camber_rc/mae_surf_p | 137.895 |
| val_geom_camber_cruise/mae_surf_p | 94.868 |
| val_re_rand/mae_surf_p | 111.882 |
| test_single_in_dist/mae_surf_p | 126.460 |
| test_geom_camber_rc/mae_surf_p | 127.348 |
| test_geom_camber_cruise/mae_surf_p | NaN ⚠️ (scoring bug) |
| test_re_rand/mae_surf_p | 111.169 |
| test_avg/mae_surf_p (3-split partial) | 121.659 |
| Peak GPU | 42.1 GB |
| Epoch time | ~131s/epoch |
| Epochs completed | 14/20 (30-min timeout) |

**Metrics artifacts:**
- `models/model-charliepai2g48h2-askeladd-pressure-channel-weight-20260512-175622/metrics.jsonl`
- `models/model-charliepai2g48h2-askeladd-pressure-channel-weight-20260512-175622/metrics.yaml`

### Commentary

Channel weighting is a clean win for this metric: the pressure channel receives 60% of the gradient (3 out of 5 total weight) instead of 33%, and the model's best val_avg drops from the initial few-epoch range directly to 122.64 by epoch 14 with a monotonically decreasing curve. The mechanism is straightforward — the optimizer is explicitly told the evaluation metric's channel preference.

The val curve was still descending at timeout, suggesting more epochs would have improved further. Follow-ups should consider pressure-only weighting ([0,0,1]) and channel weight sweeps.

NaN on test_geom_camber_cruise is a GT data quality issue in the test set (sample 000020 contains 761 +inf entries in y), not a model or implementation issue. See BASELINE.md for details.

---

## 2026-05-12 19:42 — PR #1421: Surface loss weight 10 → 25

- **Branch:** `charliepai2g48h2-edward/surf-weight-25`
- **Hypothesis:** Bump `surf_weight` from 10 → 25 to bias gradients toward surface predictions.
- **Status:** SENT BACK for refinement → `surf-only-channel-weight` (decouple vol/surf channel weights)

### Results

| Metric | Value | vs Baseline #1418 |
|---|---|---|
| val_avg/mae_surf_p (best ep 14) | 124.9634 | **+1.9% worse** |
| 3-split partial test (excluding cruise NaN) | 122.89 | +1.0% worse |
| 4-split clean re-eval test (student-side) | 117.00 | — (uses 199/200 cruise samples; baseline can't compare) |
| Epochs completed | 14/20 | (timeout) |

### Commentary

Channel weighting [1,1,3] (PR #1418, the merged baseline) is a stronger lever than blanket surf_weight bump (10→25). They are independent axes; this run tested surf_weight=25 with uniform channels.

**Useful finding:** the student demonstrated a clean test re-eval pattern that skips non-finite GT samples (cruise=99.34 from 199/200 samples). This corroborates askeladd's bug diagnosis and gives us a workable test number for paper-facing purposes; we just can't compare it directly to baseline since baseline's cruise NaN poisoned its 4-split avg.

### Follow-up direction

Decouple channel weighting: apply [1,1,3] only to surf loss, vol loss keeps uniform [1,1,1]. Hypothesis: since the metric is *surface* pressure exclusively, the vol channel weighting may dilute useful volume gradient that indirectly informs surface prediction. Edward picks this up as `surf-only-channel-weight`.

---

## 2026-05-12 22:30 — PR #1414: Smooth L1 rebased (Huber β=0.1 + channel_weights=[1,1,3])

- **Branch:** `charliepai2g48h2-alphonse/smooth-l1-loss` (rebased + re-run)
- **Hypothesis:** Stack Smooth L1 (β=0.1) on top of #1418's channel_weights=[1,1,3]. L1 directly minimizes median absolute error = MAE eval criterion. Both axes are orthogonal.
- **Status:** MERGED ✅ — **new baseline** (val_avg/mae_surf_p = 95.336)

### Results

| Metric | Value | vs Baseline #1424 (102.85) |
|---|---|---|
| val_avg/mae_surf_p (best, ep 13) | **95.336** | **−7.3% BETTER** 🏆 |
| val_single_in_dist | 118.539 | −11.3% |
| val_geom_camber_rc | 105.115 | −7.9% |
| val_geom_camber_cruise | 71.196 | −13.3% |
| val_re_rand | 86.495 | −10.2% |
| test_single_in_dist | 103.264 | — |
| test_geom_camber_rc | 96.989 | — |
| test_geom_camber_cruise | **61.217** ✓ | — |
| test_re_rand | 81.121 | — |
| test_avg/mae_surf_p (clean 4-split) | **85.648** | — (baseline was partial NaN) |
| Epochs completed | 13/14 | (timeout) |

**Metric artifacts:**
- `models/model-charliepai2g48h2-alphonse-smooth-l1-rebased-20260512-211440/metrics.jsonl`
- `models/model-charliepai2g48h2-alphonse-smooth-l1-rebased-20260512-211440/metrics.yaml`

### Commentary

Smooth L1 + CW stacked delivers −7.3% over the warmup/clip baseline (102.85) and −22.3% over the original #1418 MSE baseline. Uniform gain across all 4 val splits (−7.9% to −13.3%), strongest on cruise OOD (−13.3%) consistent with hypothesis.

Notably, this combined result (95.336) is slightly worse than Smooth L1 alone on pre-CW code (90.585 from first run). Alphonse identifies two hypotheses: RNG noise (single seed, ~5% spread plausible) OR mild antagonism between Smooth L1 and CW=[1,1,3]. Mechanism for antagonism: Smooth L1 already de-emphasizes large pressure residuals (linear regime above β=0.1); CW then re-upweights pressure 3×, partially canceling the first effect.

**Incidental win:** NaN-skip fix (`y_finite` + `nan_to_num(y)` before accumulation) now merged. `test_avg/mae_surf_p` is clean 4-split (85.648) for the first time. Previously NaN due to IEEE 754 inf×0=NaN in accumulator.

⚠️ **Note on merged config:** The validated metric (95.336) was from lr=5e-4 run (pre-#1424 state). The merged code combines Smooth L1 + CW + warmup + clip (lr=7e-4). Full-stack validation assigned to alphonse as follow-up PR #1663.

### Follow-up direction

PR #1663 — alphonse to run the FULL stack (all merged changes, no train.py edits needed) to confirm actual post-merge metric. β sweep (β ∈ {0.05, 0.02, 0.3}) pending on that baseline.

---

## 2026-05-12 21:00 — PR #1414: Smooth L1 (Huber β=0.1) loss in normalized space

- **Branch:** `charliepai2g48h2-alphonse/smooth-l1-loss`
- **Hypothesis:** Replace MSE with Smooth L1 (Huber β=0.1) in normalized space. MSE minimizes mean-squared residuals, which only matches MAE when residuals are symmetric and tight. Smooth L1 → L1 above β=0.1, and L1 minimizes MEDIAN absolute error — i.e., directly matches the eval metric.
- **Status:** SENT BACK for rebase + re-run on top of #1418 (was branched from pre-#1418 code; needs to stack with channel weighting)

### Results (on pre-#1418 codebase, uniform channels)

| Metric | Value | vs Baseline #1418 |
|---|---|---|
| val_avg/mae_surf_p (best, ep 14) | **90.5853** | **−26.1% BETTER** 🏆 |
| test_avg/mae_surf_p (full 4-split, clean) | 81.5392 | — (baseline NaN) |
| val_single_in_dist | 108.878 | −25.4% |
| val_geom_camber_rc | 98.268 | −28.7% |
| val_geom_camber_cruise | 71.492 | −24.6% |
| val_re_rand | 83.703 | −25.2% |
| Epochs completed | 14/20 | (timeout) |

### Commentary

The biggest single-axis result seen in this round. The win is uniform across ALL splits (−24% to −29%), not just one outlier. Mechanism is principled: Smooth L1 β=0.1 in normalized space puts most residuals in the L1 (linear) regime (student notes: training loss in linear regime all the way). L1 minimizes median absolute error, which IS the MAE eval criterion. MSE trains the model to minimize mean-squared residuals, which over-weights outlier (high-Re, OOD) samples.

**Key insight**: "match training loss to eval metric" is a classic Kaggle result, well-known for regression tasks. The 26% improvement validates this intuition strongly.

The run was on pre-#1418 code (uniform channels, no channel weighting). If the two axes are independent — and they should be, as channel weighting is a per-channel reweight inside the loss and Smooth L1 changes the loss *shape* — the combined result should be similar or better.

### Incidental fixes

Student also added NaN-skip fix in `train.py:evaluate_split` — same fix as tanjiro (#1432) and thorfinn (#1435), but alphonse's formulation uses `torch.nan_to_num` on `y_safe` before accumulation. Enables clean 4-split test_avg (81.54).

### Follow-up direction

Rebase onto current advisor branch (which has #1418 channel_weights=[1,1,3]) and re-run with both Smooth L1 AND channel weighting stacked. Beta sweep (β ∈ {0.05, 0.02, 1.0}) after stacked baseline confirmed.

---

## 2026-05-12 21:00 — PR #1426: Widen Transolver (n_hidden 128→192, n_head 4→6)

- **Branch:** `charliepai2g48h2-frieren/hidden-192-head-6`
- **Hypothesis:** Widen token dimension to 192 and attention heads to 6. Hypothesis: more capacity → better generalization on geometry-OOD splits.
- **Status:** CLOSED — result significantly worse (+12.81%) and model doesn't fit 30-min budget

### Results

| Metric | Value | vs Baseline #1418 |
|---|---|---|
| val_avg/mae_surf_p (best, ep 9) | 138.31 | **+12.81% worse** |
| test_avg/mae_surf_p (3-split partial) | 139.05 | — |
| Epochs completed | 9/15 | (timeout) |
| Epoch time | ~3.4 min/epoch | +55% over baseline |
| Peak memory | 63.0 GB | |

### Commentary

The widened model (1.45M params, 2.2× baseline) is trainable and stable, but costs 3.4 min/epoch — only 9 epochs fit the 30-min cap, and cosine LR was still in high-LR regime. Not a fair comparison. The widening hypothesis is correct in principle but the 30-min budget makes it unworkable. Frieren's follow-up suggestion #4 (depth vs width) is the right next test.

Frieren also provided an excellent independent diagnosis of the `test_geom_camber_cruise` NaN bug (identical to BASELINE.md and other students' reports).

---

## 2026-05-12 21:00 — PR #1429: Double slice tokens (slice_num 64→128) and MLP ratio (2→4)

- **Branch:** `charliepai2g48h2-nezuko/slice-128-mlp-4`
- **Hypothesis:** Simultaneously double slice_num and mlp_ratio. More slice tokens → finer mesh partitioning; wider MLP → more expressive token features.
- **Status:** CLOSED — result worse (+6.97%) and numerical instability at test time

### Results

| Metric | Value | vs Baseline #1418 |
|---|---|---|
| val_avg/mae_surf_p (best, ep 10) | 131.18 | **+6.97% worse** |
| test_avg/mae_surf_p | NaN | (model output overflow + GT NaN) |
| Epochs completed | 10/20 | (timeout, 3.1 min/epoch) |
| Peak memory | 62.5 GB | |

### Commentary

Only 10/20 epochs completed (3.1 min/epoch). val_cruise shows signal (99.2 vs baseline 94.9 — actually 4.5% worse still; val_single is 168 vs 145, much worse). Test-time overflow: slice_num=128 with current softmax temperature dynamics produces near-zero `slice_norm` on OOD inputs → prediction → ~1e20. The two axes were confounded; nezuko's follow-up #3 (decouple mlp_ratio=4 alone) is the right next step.

---

## 2026-05-12 20:54 — PR #1435: Unified positional encoding (ref=8, 8×8 soft Gaussian grid)

- **Branch:** `charliepai2g48h2-thorfinn/unified-pos-ref8`
- **Hypothesis:** Replace raw `(x,z)` positions with a learned 8×8 (=64) soft-Gaussian grid encoding zero-padded to ref³=512. Hypothesis: geometry-OOD splits benefit from spatially structured priors.
- **Status:** SENT BACK for refinement → `unified-pos-ref16-nopad` (drop zero-pad, widen grid to ref=16)

### Results

| Metric | Value | vs Baseline #1418 |
|---|---|---|
| val_avg/mae_surf_p (best, ep 14) | 124.4938 | **+1.51% worse** |
| test_avg/mae_surf_p (full 4-split, clean) | 113.7291 | — (baseline NaN) |
| val_single_in_dist | 141.887 | −2.8% |
| val_geom_camber_rc | 140.815 | +2.1% |
| val_geom_camber_cruise | 93.156 | **−1.8%** ✓ |
| val_re_rand | 122.117 | **+9.1%** ✗ |
| Peak GPU | 45.67 GB | +8% |
| Epochs completed | 14/20 | (timeout) |

### Commentary

Direction shows real signal on the camber-cruise OOD split (−1.8%), which is exactly where the hypothesis predicted positional encoding would help — stable spatial priors for unfamiliar geometry. But the **fixed-bandwidth Gaussian at ref=8 under-resolves the wake region**, badly hurting val_re_rand (+9.1%, high-Re samples push per-batch min/max apart, smearing Gaussian responses across cells).

The architecture also wastes 448 of 512 preprocess input dims on zero-padding (ref²=64 actual content, ref³=512 expected width). This is dead capacity that the first linear layer must learn to ignore.

### Incidental defensive fixes (universally useful)

Thorfinn added two eval-side fixes in `train.py`:
1. **Drop non-finite-GT samples** (similar to tanjiro's #1432 fix) — produces clean 4-split test_avg.
2. **`nan_to_num` prediction sanitization before scoring** — strictly stronger than the GT-only fix; guards against model output overflow (we saw this in fern's #1424 instability run).

Both fixes ride along with this PR. Will propagate to baseline once any iteration of this branch merges.

### Follow-up direction

Drop zero-pad + widen grid to ref=16 (256 cells, much better wake resolution). Cleaner param efficiency and finer spatial discretization. Predicted Δ: −1% to −4% vs current baseline if wake-region gain dominates the per-cell-bandwidth tradeoff.

---

## 2026-05-12 19:56 — PR #1432: Wall-distance feature

- **Branch:** `charliepai2g48h2-tanjiro/wall-distance-feature`
- **Hypothesis:** Add `log(1 + min_euclidean_distance_to_surface_nodes)` as input feature (dim 25). Boundary-layer physics implies pressure is strongly distance-modulated near the wall.
- **Status:** SENT BACK for rebase + re-run on top of #1418 (was branched from pre-#1418 code; needs to stack with channel weighting)

### Results (on pre-#1418 codebase, uniform channels)

| Metric | Value | vs Baseline #1418 |
|---|---|---|
| val_avg/mae_surf_p (best, ep 13) | **121.4633** | **−0.96%** |
| test_avg/mae_surf_p (full 4-split) | 110.1309 | — (baseline NaN) |
| test_avg/mae_surf_p (3-split partial) | 121.155 | −0.42% |
| val_single_in_dist | 151.269 | +3.7% |
| val_geom_camber_rc | 137.454 | −0.32% |
| val_geom_camber_cruise | 89.784 | −5.4% |
| val_re_rand | 107.346 | −4.1% |
| Epochs completed | 14/20 | (timeout) |
| Epoch time | ~137s/epoch | +5% slower |

### Commentary

Wall-distance is a winning direction. The gain (−0.96%) is smaller than the predicted (−3% to −8%), likely because (a) the `dsdf` shape descriptor already encodes related geometric info, (b) per-batch wall-distance standardization is noisy across heterogeneous batches. **Biggest gains landed on the in-distribution + Re-OOD splits**; geometry-OOD camber_rc was nearly flat — wall-distance helps where geometry is in-distribution.

**Incidental bug fix:** Student implemented batch-level non-finite-y filter in `train.py:evaluate_split` (lines 281–289), preserving `data/scoring.py` read-only contract. This is the first PR on this branch to produce a clean 4-split test_avg/mae_surf_p (110.1309). The fix is universally valuable and should propagate via this PR's eventual merge.

### Follow-up direction

Rebase onto current advisor branch (which has #1418 channel_weights=[1,1,3]) and re-run. Stacked wall-distance + channel weighting predicted to give −1.5% to −3% combined gain. Keep the NaN-skip fix.

---

## 2026-05-12 19:56 — PR #1517: EMA weight averaging (decay=0.999)

- **Branch:** `charliepai2g48h2-askeladd/ema-0.999`
- **Hypothesis:** EMA shadow weights with `decay=0.999` smooth optimizer noise; eval on EMA weights instead of live. Universal in diffusion / timm. Predicted Δ: −2% to −5%.
- **Status:** SENT BACK for refinement → `ema-0.99-adaptive` (timm-style adaptive decay, max=0.99)

### Results

| Metric | Value | vs Baseline #1418 |
|---|---|---|
| val_avg/mae_surf_p (best, ep 14) | 135.4957 | **+10.49% WORSE** |
| test_avg/mae_surf_p (3-split partial) | 134.9157 | +10.9% worse |
| val_single_in_dist | 156.940 | +7.6% |
| val_geom_camber_rc | 155.249 | +12.6% |
| val_geom_camber_cruise | 107.027 | +12.8% |
| val_re_rand | 122.767 | +9.7% |
| Epochs completed | 14/20 | (timeout) |

### Commentary

Implementation was correct (verified by student). The hypothesis failure mode is **horizon mismatch**: `decay=0.999` has effective window `1/(1-decay) = 1000 steps`, but our training is ~5,250 total steps in a rapid-descent regime — the EMA trajectory is monotonically decreasing from 327.87 → 135.50 with no plateau, meaning the EMA is lagging behind a still-improving live model. EMA assumes "smooth weight oscillations near convergence" but we never *reach* convergence under the 30-min cap.

### Follow-up direction

Match EMA window to training horizon. Refined hypothesis: **timm-style adaptive decay** `decay_t = min(0.99, (1+step)/(10+step))` — auto-warms over first ~1000 steps then caps at 0.99 (window ~100 steps). Sidesteps cold-start lag entirely. Secondary arm: fixed `decay=0.99`. Adaptive should dominate.

---

## 2026-05-12 22:00 — PR #1424: Warmup cosine peak LR 7e-4 + grad clip 1.0

- **Branch:** `charliepai2g48h2-fern/warmup-7e-4-clip`
- **Hypothesis:** Lower peak LR to 7e-4 (from 1e-3), extend warmup to 2 epochs (from 1), add gradient clipping (max_norm=1.0). Stacked on top of #1418 channel_weights=[1,1,3]. Hypothesis: stable fast convergence under 30-min cap with larger-than-baseline LR but without the epoch-5 spike seen at 1e-3.
- **Status:** MERGED ✅ — **new baseline** (val_avg/mae_surf_p = 102.8503)

### Results

| Metric | Value | vs Baseline #1418 |
|---|---|---|
| val_avg/mae_surf_p (best, ep 14) | **102.8503** | **−16.13% BETTER** 🏆 |
| val_single_in_dist | 119.682 | −18.0% |
| val_geom_camber_rc | 113.333 | −17.8% |
| val_geom_camber_cruise | 82.087 | −13.5% |
| val_re_rand | 96.299 | −13.9% |
| test_single_in_dist | 104.577 | — |
| test_geom_camber_rc | 97.972 | — |
| test_geom_camber_cruise | NaN ⚠️ (GT bug) | — |
| test_re_rand | 93.588 | — |
| test_avg/mae_surf_p (3-split partial) | 98.712 | — |
| Epoch time | ~131s/epoch | same as baseline |
| Peak GPU | 42.1 GB | same as baseline |
| Epochs completed | 14/20 | still descending at timeout |

**Metric artifacts:**
- `models/model-charliepai2g48h2-fern-warmup-7e-4-clip-20260512-211813/metrics.jsonl`
- `models/model-charliepai2g48h2-fern-warmup-7e-4-clip-20260512-211813/metrics.yaml`

### Commentary

The refinement of #1424-r1 (LR=1e-3, 1-epoch warmup, no clip) fully resolved the training instability. No epoch-5 spike. From epoch 3 onward the loss curve is monotonically decreasing all the way to the timeout at epoch 14 — the model is **still improving** at cutoff, implying T_max=20 with 14 reachable epochs leaves 6 epochs over-annealed. The −16.13% improvement is uniform across all 4 val splits (−13.5% to −18.0%), with strongest gains on the in-distribution and rc-OOD splits.

**Mechanism**: gradient clipping (max_norm=1.0) eliminates the catastrophic gradient spikes that destabilized the 1e-3 run. The 7e-4 peak is still +40% above the baseline 5e-4, giving faster early convergence, while the 2-epoch warmup ramps smoothly to avoid cold-start gradient noise. Together these changes compress effective convergence into the 30-min window.

**Compounding**: this PR stacks on top of #1418 channel_weights=[1,1,3]. All subsequent PRs are now measured against both changes combined. The advisor branch now includes: (i) channel_weights=[1,1,3], (ii) lr_peak=7e-4, (iii) 2-epoch warmup, (iv) grad_clip=1.0.

**Open question**: T_max=20 with only 14 reachable epochs means the cosine LR reached ~45% of the schedule. Setting T_max=14 would give a tighter anneal. However, the model was still descending (not oscillating), suggesting the current LR at epoch 14 is still productively high. T_max alignment could be tried as a follow-up.

---

## 2026-05-12 22:00 — PR #1517: EMA adaptive decay (Arm A: timm-style max=0.99, Arm B: fixed 0.99)

- **Branch:** `charliepai2g48h2-askeladd/ema-0.99-adaptive`
- **Hypothesis:** Timm-style adaptive EMA decay `min(0.99, (1+step)/(10+step))` auto-warms over first ~1000 steps and caps at 0.99 (window ~100 steps), avoiding cold-start lag from fixed high-decay EMA. Secondary arm: fixed `decay=0.99`. Both should benefit generalization without the horizon-mismatch failure of 0.999.
- **Status:** CLOSED — neutral result; best arm (+0.40% worse on val, −0.63% better on test 3-split)

### Results

| Metric | Arm A (adaptive) | Arm B (fixed 0.99) | vs Baseline #1418 |
|---|---|---|---|
| val_avg/mae_surf_p (best) | 123.1314 | 124.0113 | +0.40% / +1.12% worse |
| val_single_in_dist | 145.19 | 148.81 | — |
| val_geom_camber_rc | 139.72 | 140.69 | — |
| val_geom_camber_cruise | 94.53 | 94.90 | — |
| val_re_rand | 113.08 | 111.75 | — |
| test_avg (3-split partial) | 120.8885 | 123.99 | −0.63% (Arm A better) |
| Epochs completed | 14/20 | 14/20 | — |

**Metric artifacts:**
- `models/model-charliepai2g48h2-askeladd-ema-0.99-adaptive-*/metrics.jsonl` (two runs)

### Commentary

The hypothesis refinement (max=0.99 vs 0.999) correctly fixed the cold-start lag — Arm A is much closer to baseline than the −10.5% disaster of the original run. However, even adaptive EMA cannot overcome the fundamental issue: with only 14 reachable epochs in a still-descending regime, any weight averaging that incorporates early (worse) weights degrades the final model. The test partial 3-split shows EMA marginally helps OOD generalization (test partial 0.63% better), but the val_avg metric that we rank against is 0.40% worse. Net result: **neutral**. Against the new baseline of 102.85 (PR #1424), a neutral result from the old baseline is even further behind.

**Pattern**: EMA helps OOD splits (camber cruise consistent) but hurts in-dist. This suggests the live model slightly overfits to in-dist while EMA under-fits it. A post-hoc checkpoint ensemble (e.g., average last 3 epoch checkpoints) might capture this benefit without the live-model drag. However, this adds inference complexity for marginal expected gain.

### Why closed

New baseline is 102.8503. EMA alone (no warmup, no LR change) cannot bridge to that bar. Resources better deployed on orthogonal axes.

---

## 2026-05-12 22:00 — PR #1598: MLP ratio 2→4 alone (decoupled from slice_num)

- **Branch:** `charliepai2g48h2-nezuko/mlp-ratio-4-alone`
- **Hypothesis:** Decouple mlp_ratio=4 from the failed #1429 (slice_num=128 + mlp_ratio=4). Wider post-attention MLP at stable slice_num=64. Predicted −1% to −3% vs baseline from wider hidden capacity (128*2 → 128*4 = 512 MLP dim per block).
- **Status:** CLOSED — +7.0% worse vs old baseline; new baseline 102.8503 makes this approach insufficient

### Results

| Metric | Value | vs Baseline #1418 |
|---|---|---|
| val_avg/mae_surf_p (best, ep 9) | 131.2161 | **+7.0% WORSE** |
| val_single_in_dist | 164.234 | +12.6% |
| val_geom_camber_rc | 144.434 | +4.8% |
| val_geom_camber_cruise | 98.040 | +3.3% |
| val_re_rand | 118.155 | +5.6% |
| test_avg (3-split partial) | 128.806 | — |
| Epoch time | ~148s/epoch | +13% over baseline |
| Params | ~991K | +50% over baseline (662K) |
| Epochs completed | 13/20 | (timeout) |

**Metric artifacts:**
- `models/model-charliepai2g48h2-nezuko-mlp-ratio-4-alone-*/metrics.jsonl`

### Commentary

Under-trained: 13/20 epochs at 148s/epoch vs baseline 131s/epoch. The cosine LR was only at ~27% of its T_max=20 schedule at cutoff. Stability confirmed — no softmax/slice_norm overflow at slice_num=64 (max pred abs value logged, all finite). The +7% worse result is almost certainly an under-training artifact; the student's epoch-13 metrics show the model was still declining.

However, even accounting for under-training, a compute-matched rerun (student suggestion: --epochs 12, T_max=12) is unlikely to bridge to the **new baseline of 102.8503** — the original prediction was −1% to −3% from the OLD 122.64 baseline, which would put the expected result around 119-121, still 16-18% above 102.85. The single MLP-ratio axis without warmup/LR/clip stacking is insufficient.

**Key finding**: stability at slice_num=64 confirmed. The slice_num=96 (intermediate) is a lower-risk probe for the attention token count axis.

### Why closed

New baseline (102.85) renders this isolated experiment non-competitive. Resources better deployed on higher-leverage axes.

---

## 2026-05-12 18:53 — PR #1424: Warmup + cosine, peak LR 1e-3

- **Branch:** `charliepai2g48h2-fern/warmup-cosine-1e-3`
- **Hypothesis:** Peak LR 1e-3 with 1-epoch linear warmup and cosine decay. Hypothesis: faster early convergence under the 30-min cap.
- **Status:** SENT BACK for refinement → `warmup-7e-4-clip`

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 14) | 127.170 |
| val_geom_camber_cruise/mae_surf_p | 95.15 |
| val_geom_camber_rc/mae_surf_p | 134.36 |
| val_re_rand/mae_surf_p | 109.46 |
| val_single_in_dist/mae_surf_p | 169.72 |
| test_avg/mae_surf_p (3-split partial) | 126.36 |
| Epochs completed | 14/20 |

### Commentary

3.7% worse than #1418 at the same epoch count. The 1e-3 peak LR caused instability (epoch-5 spike to 267 from 202, slow re-stabilization from epoch 11). The model was still descending at timeout — with a stable schedule it could have caught up, but the instability indicates the LR overshoots the loss landscape at this scale (0.66M params, 96GB-VRAM single GPU). Notably, cruise test overflowed to NaN for a different reason than #1418 (model output overflow, not GT NaN), confirming the instability was real.

Direction is still worth pursuing: faster convergence under the 30-min cap is genuinely valuable. Refinement: peak LR 7e-4 (20% increase over baseline), 2-epoch warmup, gradient clipping at 1.0.

---

## 2026-05-12 23:01 — PR #1597: Depth experiment: n_layers 5→6 (frieren)

- **Branch:** `charliepai2g48h2-frieren/depth-6-layers`
- **Hypothesis:** Adding a 6th Transolver block increases representational capacity for camber-OOD splits with manageable compute overhead.
- **Status:** CLOSED ❌ — dead end (+5.91% worse than baseline #1418; +36% worse than current baseline 95.336)

### Results

| Run | val_avg/mae_surf_p | Note |
|---|---|---|
| ep12 contingency (T_max=12 matched budget) | **129.889** | +5.91% vs #1418 baseline, +36% vs current 95.336 |
| ep20 sanity (T_max=20, only 12 epochs fit) | 157.140 | confirms T_max must match epochs |
| Param count | 783.5K (was 662K) | +121K from 6th block |
| Epoch time | 154-158s | vs baseline 131s |

**Per-split (ep12 best):**
- val_single_in_dist: 154.907 (+6.16%)
- val_geom_camber_rc: 137.211 (-0.50%)  ← only neutral split
- val_geom_camber_cruise: 108.299 (+14.16%)  ← worst regression
- val_re_rand: 119.141 (+6.49%)

### Commentary

Depth 5→6 at width 128 regresses ~5.9% — the model is NOT capacity-bottlenecked on this 1500-sample dataset. Extra block burns ~121K params and ~26s/epoch without recovering fit. Per-split pattern contradicted a-priori hypothesis ("strongest signal on camber-OOD"): cruise regressed +14%, rc was neutral. At fixed compute, trading 1 epoch (~20% of 12-epoch cosine schedule) for more capacity is the wrong direction.

### Key finding for advisor

**Frieren's sharp insight**: T_max must match feasible epoch count under the 30-min cap. The 20-epoch sanity run with truncated cosine confirms this — T_max=20 leaves LR at ~50% peak when training ends, dramatically worse than T_max=12 with the same wall-clock. **This is an immediately actionable lever** — assigning frieren to test T_max alignment directly.

### Why closed

Capacity axis exhausted. Frieren's suggestion #3 (pivot to data/loss/training axes, not capacity probing) is well-supported. Reassigning frieren to T_max alignment experiment.

---

## 2026-05-12 22:57 — PR #1432: Wall-distance feature rebased + stacked (tanjiro)

- **Branch:** `charliepai2g48h2-tanjiro/wall-distance-feature` (rebased onto #1418 + #1424)
- **Hypothesis:** Wall-distance (-0.96% on pre-#1418) should stack additively with channel_weights=[1,1,3] + warmup/clip (#1424). Expected -1.5% to -3% combined gain.
- **Status:** CLOSED ❌ — dead end (negative stacking: +4.59% worse than #1424 baseline; +12.8% worse than current 95.336)

### Results

| Metric | Value | vs Baseline #1424 |
|---|---|---|
| val_avg/mae_surf_p (best ep 14) | **107.5735** | +4.59% worse |
| test_avg/mae_surf_p (clean 4-split via NaN-skip) | 95.0447 | — |
| Epochs completed | 14/14 (30-min cap) | |
| Param count | 662,615 | unchanged |
| Epoch time | ~136s | vs baseline ~131s |

**Per-split val (vs #1424):**
- val_single_in_dist: 128.722 (+7.56%)
- val_geom_camber_rc: 126.387 (+11.52%)
- **val_geom_camber_cruise: 76.680 (-6.59%)** ✓ ← only beneficial split
- val_re_rand: 98.504 (+2.29%)

### Commentary

**Negative stacking confirmed.** Wall-distance + channel_weights+warmup interact destructively on 3 of 4 splits. Tanjiro's analysis is correct:

1. **Capacity competition** at preprocess MLP (25→128 with biased gradient toward pressure)
2. **Per-batch standardization noise** is amplified in steeper warmed-up loss landscape
3. **Cruise-only benefit** (-6.59%) suggests wall-distance helps where boundary layer is well-resolved AND geometry is in-distribution; elsewhere it adds noise

### Why closed

- The signal that initially looked promising on uniform channels disappears under the new training regime.
- The NaN-skip fix tanjiro pioneered was already incorporated into PR #1414 (Smooth L1) and merged — that fix is now canonical on the advisor branch.
- Tanjiro's per-sample standardization variant or capacity-widening variant remain as potential follow-ups, but lower priority than fresh directions (β sweep, pure L1, T_max alignment).

### Lifted artifact

NaN-skip pattern (`y_finite` + `nan_to_num(y)`) pioneered in this PR is now canonical via #1414. Tanjiro is being reassigned to a fresh direction.

---
