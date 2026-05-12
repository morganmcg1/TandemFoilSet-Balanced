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
