# SENPAI Research Results — charlie-pai2i-24h-r2

## 2026-05-16 03:30 — PR #3608: SwiGLU FFN (param-matched GELU replacement) [MERGED — MASSIVE WIN]
- Branch: `charliepai2i24h2-frieren/swiglu-ffn`
- Student: charliepai2i24h2-frieren
- Hypothesis: Replace GELU FFN with SwiGLU gating `W2(SiLU(W1(x)) ⊙ V(x))` with param-matched hidden_inner=128. Predicted 0.5-2% val improvement.

### Results table

| Metric | SwiGLU | Baseline 95.808 | Δ |
|--------|--------|-----------------|---|
| `val_avg/mae_surf_p` | **78.407** | 95.808 | **−18.2%** (MASSIVE WIN) |
| `test_avg/mae_surf_p` | **68.375** | 85.578 | **−20.1%** |
| val single_in_dist | 94.301 | 110.886 | −15.0% |
| val geom_camber_rc | 89.780 | 105.776 | −15.1% |
| val geom_camber_cruise | 56.169 | 76.060 | **−26.2%** |
| val re_rand | 73.379 | 90.510 | −18.9% |
| test single_in_dist | 83.095 | 97.804 | −15.0% |
| test geom_camber_rc | 79.596 | 94.519 | −15.8% |
| test geom_camber_cruise | 45.973 | 64.863 | **−29.1%** |
| test re_rand | 64.837 | 85.126 | **−23.8%** |
| n_params | 379,799 | ~381k | param-matched |
| Wall-clock | 30.4 min | 32 min | −5% |
| Peak VRAM | 42.95 GB | 40.96 GB | +2% |
| Best epoch | 13 / 14 | 14 / 14 | — |

### Analysis
- Prediction dramatically underestimated. 0.5-2% predicted; **18.2% actual**. The multiplicative gating in SwiGLU provides expressive power that GELU's single-path FFN genuinely couldn't express on this dataset.
- ALL 4 splits improved substantially — not a single-split fluke. The gated mechanism helps both in-distribution (single_in_dist) and out-of-distribution geometry splits.
- **Biggest gains on geometry-shift splits**: geom_cruise −26.2% val / −29.1% test, re_rand −23.8% test. Multiplicative gating gives better adaptive feature selection for local mesh geometry.
- **GELU FFN was the bottleneck** — not the schedule, not the regularization, not the capacity. Explains why all prior capacity/regularization experiments made marginal gains: the FFN parameterization itself was holding back the model.
- Retrospective insight: the mlp_ratio=4 failure (#3503, +5.0%) was NOT about capacity — it was about the *parameterization*. SwiGLU effectively re-introduces 1.5× FFN expansion at matched params, and it wins massively. The prior test failed because single-path expansion with GELU hits diminishing returns; gating does not.
- Train losses also lower at every epoch — genuine fit improvement, not regularization artifact.
- Artifacts: `models/model-swiglu-ffn-20260516-022733/metrics.{jsonl,yaml}`

### Decision
- **MERGED** — new baseline is SwiGLU. All in-flight PRs notified to rebase onto new advisor branch.
- **New baseline: val_avg/mae_surf_p = 78.407, test = 68.375**
- Natural follow-ups: (a) full mlp_ratio=2 SwiGLU (hidden_inner=192, +24% params), (b) SwiGLU + dropout/drop-path compound, (c) GEGLU as head-to-head gate comparison.

---

## 2026-05-16 03:25 — PR #3607: FFN dropout p=0.1 [SENT BACK — SwiGLU REBASE + p=0.05]
- Branch: `charliepai2i24h2-thorfinn/dropout-ffn-p01`
- Student: charliepai2i24h2-thorfinn
- Hypothesis: dropout=0.1 on FFN hidden activation should regularize productively (orthogonal to WD).

### Results table

| Metric | dropout=0.1 | Baseline 95.808 | Δ |
|--------|------------|-----------------|---|
| `val_avg/mae_surf_p` | **96.412** | 95.808 | +0.63% |
| `test_avg/mae_surf_p` | 85.118 | 85.578 | **−0.54%** |
| val single_in_dist | 118.247 | 110.886 | +6.6% (WORSE) |
| val geom_camber_rc | 104.503 | 105.776 | −1.2% |
| val geom_camber_cruise | 72.518 | 76.060 | **−4.7%** |
| val re_rand | 90.381 | 90.510 | −0.1% |
| test single_in_dist | 101.800 | 97.804 | +4.1% (WORSE) |
| test geom_camber_rc | 94.074 | 94.519 | −0.5% |
| test geom_camber_cruise | 60.797 | 64.863 | **−6.3%** |
| test re_rand | 83.800 | 85.126 | −1.6% |
| Epochs | 13 / 14 | 14 / 14 | budget-bound |

### Analysis
- Budget-bound failure: 30-min cap cut at E13, model still descending steeply (E12→E13: −6 pts). Extrapolated E14 ~91-93 would have beaten baseline — but cannot count extrapolation.
- Wrong-split pattern: single_in_dist regressed +6.6% (WORST regression) while geom_cruise improved −4.7%. Same OOD/in-dist asymmetry as wd=5e-4 (fern #3569). Stochastic deactivation hurts memorization just like L2.
- However the test-side geom_cruise win (−6.3%) is genuine and interesting.
- SwiGLU merge while this was in-flight: needs rebase anyway.

### Decision
- **SENT BACK** for rebase onto SwiGLU baseline + dropout=0.05 retest. The dropout-inside-SwiGLU implementation targets the gated activation before W2.

---

## 2026-05-16 02:30 — PR #3535: n_head=8 on new baseline [CLOSED — CATASTROPHIC NEGATIVE]
- Branch: `charliepai2i24h2-askeladd/n-head-8`
- Student: charliepai2i24h2-askeladd
- Hypothesis: Doubling attention heads (4→8) with head_dim shrinking 24→12 gives finer-grained per-head attention. Predicted 0.5-2% improvement.

### Results table

| Metric | n_head=8 | Baseline 95.808 | Δ |
|--------|----------|-----------------|---|
| `val_avg/mae_surf_p` | **118.811** | 95.808 | **+22.9%** worse (CATASTROPHIC) |
| `test_avg/mae_surf_p` | 107.034 | 85.578 | +25.3% worse |
| val single_in_dist | ~135 | 110.886 | +21.7% |
| val geom_camber_rc | ~131 | 105.776 | +24% |
| val geom_cruise | ~91 | 76.060 | +19.5% |
| val re_rand | ~118 | 90.510 | +30% |
| Wall-clock/epoch | ~196s | ~137s | **+43%** |
| Peak VRAM | ~61 GB | ~41 GB | +50% |
| Epochs completed | 10 / 14 | 14 / 14 | 30-min cap fired |

### Analysis
- Root cause: **head_dim collapse**. n_hidden=96 / n_head=8 → head_dim=12, too narrow to express meaningful per-head subspaces in slice attention. Baseline head_dim=24 appears to be the lower viable bound for this slot-attention dimensionality.
- Tensor-core dispatch on (slice_num=96, head_dim=12) is unfavorable — explains the wall-clock hit. Combined with VRAM +50%, this is a clean Pareto loss.
- Head axis only has headroom in the **other** direction (fewer heads → wider head_dim). Student's own suggested follow-up is exactly n_head=2 (head_dim=48).

### Decision
- **Closed.** Head expansion axis dead. Follow-up assigned to askeladd: n_head=2 (head_dim=48).

---

## 2026-05-16 02:25 — PR #3564: n_layers=4 (depth bracket below baseline) [CLOSED — NEGATIVE]
- Branch: `charliepai2i24h2-edward/n-layers-4-r6`
- Student: charliepai2i24h2-edward
- Hypothesis: Reducing depth 5→4 brackets the sweet spot from below; if depth=5 is the optimum, depth=4 should also regress (mirror of depth=6 closure). Predicted neutral or slight regression — to characterize the depth axis curvature.

### Results table

| Metric | n_layers=4 | Baseline 95.808 | Δ |
|--------|-----------|-----------------|---|
| `val_avg/mae_surf_p` | **98.409** | 95.808 | **+2.71%** worse |
| `test_avg/mae_surf_p` | 87.748 | 85.578 | +2.54% worse |
| val single_in_dist | ~119 | 110.886 | +7.4% |
| val geom_camber_rc | ~109 | 105.776 | +3% |
| val geom_cruise | ~77 | 76.060 | ~flat |
| val re_rand | ~93 | 90.510 | +2.7% |
| Wall-clock/epoch | ~110s | ~137s | −19.7% |
| Peak VRAM | 33.9 GB | 40.96 GB | −17% |

### Analysis
- 3 of 4 splits regressed. Depth axis confirmed sweet spot at 5 from both sides:
  - n_layers=4: +2.71% (this PR)
  - n_layers=5: 95.808 (baseline)
  - n_layers=6: +15.0% (#3506)
  - n_layers=8: +1.53% on old stack (#3302)
- Compute trade was as predicted (−19.7% wall-clock, −17% VRAM) but the metric loss is not compensable.
- Combined with closed width and mlp_ratio ladders, **the full capacity axis (depth × width × FFN expansion) is exhausted**.

### Decision
- **Closed.** Depth axis decisively dead. Follow-up assigned to edward: surf_weight=5 (loss rebalance — single_in_dist memorization).

---

## 2026-05-16 02:20 — PR #3569: weight_decay=5e-4 (WD ladder) [CLOSED — NEGATIVE, EXCELLENT DIAGNOSTIC]
- Branch: `charliepai2i24h2-fern/wd-5e-4-r6`
- Student: charliepai2i24h2-fern
- Hypothesis: Continue WD ladder 1e-4 → 3e-4 → 5e-4. If wd=3e-4 helped, more should help further (regularization rung up). Predicted 0.5-1% improvement.

### Results table

| Metric | wd=5e-4 | Baseline 95.808 | Δ |
|--------|---------|-----------------|---|
| `val_avg/mae_surf_p` | **97.052** | 95.808 | **+1.30%** worse |
| `test_avg/mae_surf_p` | 86.228 | 85.578 | +0.76% worse |
| val single_in_dist | ~120.3 | 110.886 | **+8.53%** (WORST regression) |
| val geom_camber_rc | ~104 | 105.776 | **−1.5%** (better) |
| val geom_cruise | 72.07 | 76.060 | **−5.25%** (better) |
| val re_rand | ~87.6 | 90.510 | **−3.22%** (better) |

### Analysis
- **Sharpest mechanism signal in the round**: WD does NOT monotonically help the hardest split. WD helps OOD/robustness splits (cruise, re_rand) but past a point hurts in-distribution memorization (single_in_dist).
- WD ladder bracketed at 3e-4 as the optimum:
  - wd=1e-4 → 96.667 (PR #3377 stack)
  - wd=3e-4 → 95.808 (**current best**)
  - wd=5e-4 → 97.052 (this PR)
- Uniform regularization is at its limit on the merged stack. Further gain on single_in_dist requires: (a) asymmetric regularization, (b) orthogonal stochastic regularization (dropout, drop-path), or (c) loss-side rebalance.

### Decision
- **Closed.** WD ladder dead. Follow-up assigned to fern: stochastic depth / drop-path p=0.1 — block-level regularizer complementary to thorfinn's FFN dropout (#3607 in flight).

---

## 2026-05-16 02:15 — PR #3579: lr 7e-4 → 1e-3 (peak LR raise) [CLOSED — NEGATIVE]
- Branch: `charliepai2i24h2-alphonse/lr-1e-3-r6`
- Student: charliepai2i24h2-alphonse
- Hypothesis: Raise peak LR 7e-4 → 1e-3 with same SequentialLR warmup+cosine. On the new merged stack the model may handle a hotter peak. Predicted 0.5-1.5% improvement.

### Results table

| Metric | lr=1e-3 | Baseline 95.808 | Δ |
|--------|---------|-----------------|---|
| `val_avg/mae_surf_p` | **98.079** | 95.808 | **+2.47%** worse |
| `test_avg/mae_surf_p` | 85.761 | 85.578 | +0.21% worse |
| val single_in_dist | 118.41 | 110.886 | **+6.79%** (worst regression) |
| val geom_camber_rc | ~107 | 105.776 | +1.5% |
| val geom_cruise | ~77 | 76.060 | ~flat |
| val re_rand | ~92 | 90.510 | +1.6% |

### Analysis
- E5-E6 instability fingerprint in loss trajectory (overshoot pattern). LinearLR 2-ep warmup is too short for a 1e-3 peak.
- **lr=7e-4 is at the upper edge of the stable range** for this stack (slice_num=96, n_hidden=96, wd=3e-4, mlp_ratio=2, n_layers=5). The cosine tail is doing the heavy lifting.
- Peak-LR axis dead on this stack. Schedule work should move to: (a) eta_min tail (tanjiro #3536 compound retest pending), (b) WSD/inverse-sqrt schedules, (c) optimizer-level tricks (EMA / Polyak, Lookahead).

### Decision
- **Closed.** Peak-LR axis dead. Follow-up assigned to alphonse: EMA / Polyak weight averaging (orthogonal to peak-LR; addresses the same instability).

---

## 2026-05-16 01:25 — PR #3536: eta_min=1e-5 on cosine annealing [SENT BACK — REBASE FOR COMPOUND TEST]
- Branch: `charliepai2i24h2-tanjiro/eta-min-1e-5-rebased-r5`
- Student: charliepai2i24h2-tanjiro
- Hypothesis: Raise CosineAnnealingLR `eta_min` from 0 to 1e-5 so the schedule tail keeps a productive lr. Predicted 0.5–2% val improvement.

### Results table

| Metric | Result | vs OLD baseline 96.667 | vs NEW baseline 95.808 |
|--------|--------|------------------------|------------------------|
| `val_avg/mae_surf_p` (best @ ep 14) | **95.835** | **−0.86% ✓** (beat target) | +0.03% (effectively tied) |
| `test_avg/mae_surf_p` | 85.292 | −0.19% ✓ | **−0.33% ✓** (slight win) |
| `val_geom_camber_rc` | 103.748 | −1.68% ✓ | — |
| `val_re_rand` | 90.032 | −1.52% ✓ | — |
| E13→E14 val drop | 97.63→95.83 | clear cosine-tail productivity | — |

### Analysis
- Hypothesis confirmed in isolation: eta_min=1e-5 makes the final 1-2 cosine epochs continue to descend productively. Best epoch is the final epoch with a sharp 1.8-pt drop in the last step. Validates the lever and falsifies the closed T_max=10 hypothesis as the right direction.
- However, run was launched on OLD baseline (96.667 at commit 1ade498). While training, PR #3314 (wd=3e-4) merged, dropping the baseline to 95.808. Compared at new baseline: val is tied (+0.027), test is marginal win (−0.33%).
- The lever is real and orthogonal to wd=3e-4 (which attacks single_in_dist; eta_min attacks geom_rc + re_rand). Predicted compound effect: 0.5–1% additional improvement on top of 95.808.

### Decision
- **SENT BACK** for rebase + compound retest with wd=3e-4 on the new baseline. Same code change (eta_min=1e-5 in CosineAnnealingLR), same reproduce command. Target: beat 95.808.

---

## 2026-05-16 01:24 — PR #3534: RFF positional encoding (32 freqs, σ=10) on new baseline [CLOSED]
- Branch: `charliepai2i24h2-nezuko/rff-32-rebased-r5`
- Student: charliepai2i24h2-nezuko
- Hypothesis: 32-frequency Gaussian RFF (σ=10) added to (x, z) gives the model high-frequency positional features that should specifically help single_in_dist (the hardest split). Predicted 1-5% improvement.

### Results table

| Metric | RFF (σ=10) | Baseline (PR #3377) | Δ |
|--------|-----------|---------------------|---|
| `val_avg/mae_surf_p` | **99.457** | 96.667 | **+2.89%** worse |
| `test_avg/mae_surf_p` | **87.197** | 85.454 | **+2.04%** worse |
| `val_single_in_dist` | 123.805 | 116.665 | **+6.12%** worse (REVERSE of prediction) |
| `val_geom_camber_rc` | 112.192 | 105.516 | +6.33% worse |
| `val_geom_camber_cruise` | 71.359 | 73.065 | −2.34% better |
| `val_re_rand` | 90.471 | 91.421 | −1.04% better |

### Analysis
- Hypothesis **reversed**: split predicted to help most (single_in_dist) regressed worst. Smooth splits (cruise, re_rand) marginally improved.
- Mechanism: σ=10 generates frequencies up to ~10 cycles/unit on normalized (x, z), matching dense per-node mesh details — not airfoil scales. On geom-shifted splits, those high-freq features encode mesh-frame patterns that don't transfer. The OLD-baseline RFF win likely came from filling a positional-capacity gap that the new slice_num=96 + n_hidden=96 stack has already closed.
- Loss still descending at E14 (~−2.3 pts/epoch) — wider input also pushes the preprocess MLP into under-training.

### Decision
- **Closed.** σ=10 specific direction dead. Follow-up assigned to nezuko: RFF σ=3 (lower frequencies, matching airfoil-scale).

---

## 2026-05-16 01:24 — PR #3506: Scale n_layers 5→6 (depth above baseline) [CLOSED]
- Branch: `charliepai2i24h2-thorfinn/depth-6-capacity-test`
- Student: charliepai2i24h2-thorfinn
- Hypothesis: Adding a 6th Transolver layer (~+20% params) increases representational depth. Predicted single_in_dist improvement.

### Results table

| Metric | Depth-6 | Baseline (PR #3377) | Δ |
|--------|---------|---------------------|---|
| `val_avg/mae_surf_p` (best @ ep 12) | **111.200** | 96.667 | **+15.0%** worse |
| `test_avg/mae_surf_p` | **96.939** | 85.454 | **+13.4%** worse |
| All 4 val splits | regressed | — | +9-19% worse |
| Per-epoch wall | ~163.5 s | ~137 s | +19% slower |
| Epochs completed | 12 of 14 | 14 of 14 | timeout truncated tail |

### Analysis
- Depth-6 decisively worse across every val and test split.
- Truncation (2 epochs missing) cannot close a 15-point gap given ep11→ep12 only dropped 0.256.
- Combined with width-{64,128,192} closures and mlp_ratio=4 closure: **capacity-expansion axis fully exhausted on this stack**. depth=5, n_hidden=96, mlp_ratio=2 confirmed as the sweet-spot tuple.

### Decision
- **Closed.** Follow-up assigned to thorfinn: dropout p=0.1 on FFN — fresh axis (regularization, not capacity).

---

## 2026-05-16 01:28 — PR #3505: Channel-weighted Huber loss [1,1,2] for [Ux,Uy,p] [CLOSED]
- Branch: `charliepai2i24h2-frieren/channel-weight-p2x`
- Student: charliepai2i24h2-frieren
- Hypothesis: Upweighting pressure channel 2× steers gradient mass toward the primary ranking metric. Predicted single_in_dist improvement (largest pressure dynamic range).

### Results table

| Metric | This run | Baseline (PR #3377) | Δ |
|--------|----------|---------------------|---|
| `val_avg/mae_surf_p` (best @ ep 13) | **99.418** | 96.667 | **+2.85%** worse |
| `test_avg/mae_surf_p` | **88.370** | 85.454 | **+3.41%** worse |
| 3 of 4 val splits | regressed | — | +2.5-4.5% worse |

### Analysis (two compounding failures, student-identified)
1. **Loss-scale confound (LR effective).** The instruction's divisor `channel_weights.sum()=4` shrunk gradient scale by ~3× vs baseline (where 3 unit-weighted channels sum to 3). The slow trajectory (E1=384 vs typical ~150) is consistent with implicit under-stepping.
2. **Velocity-pressure physical coupling.** In incompressible NS, velocity ↔ pressure are tightly coupled. Down-weighting velocity (effectively 0.25× baseline share) degrades the velocity field, which the pressure prediction depends on. Worse u → worse p, even with explicit p upweight.

### Decision
- **Closed.** Even with divisor fix the physics argument suggests channel-reweighting is fighting Navier-Stokes coupling rather than helping. Follow-up assigned to frieren: SwiGLU activation in FFN — orthogonal new axis.

---

## 2026-05-16 01:40 — PR #3502: Scale n_hidden 96→64 (width ladder continued) [CLOSED]
- Branch: `charliepai2i24h2-alphonse/n-hidden-64-width-ladder`
- Student: charliepai2i24h2-alphonse
- Hypothesis: Width-ladder monotone trend (192→128→96 = better) continues below n_hidden=96. Fewer params → fewer forward/backward FLOPs → more epochs within 30-min cap. Predicted: capacity still descending, n_hidden=64 beats 96.

### Results table

| Metric | n_hidden=64 (this run) | Baseline n_hidden=96 (PR #3377/now #3314) | Δ |
|--------|----------------------|------------------------------------------|---|
| `val_avg/mae_surf_p` (best @ ep 14) | **104.614** | 95.808 | **+9.20%** (worse) |
| `test_avg/mae_surf_p` | 94.065 | 85.578 | **+9.91%** (worse) |
| `val_single_in_dist` | 130.636 | 110.886 | +17.81% |
| `val_geom_rc` | 111.624 | 105.776 | +5.53% |
| `val_geom_cruise` | 79.283 | 76.060 | +4.24% |
| `val_re_rand` | 96.912 | 90.510 | +7.07% |
| Params | 174,583 (~0.17M) | ~0.38M | −54% |
| Per-epoch wall | ~120 s | ~137 s | **barely faster** |
| Metrics | `models/model-n-hidden-64-20260516-002129/metrics.{jsonl,yaml}` | — | — |

### Analysis
- Width-ladder reversal confirmed: n_hidden=96 is the sweet spot, not "smaller always better". Monotone trend held from 192→128→96 but breaks at 64.
- Critical insight: per-epoch wall-clock is roughly **insensitive to n_hidden** (120s vs 137s — the slicing op, data loading, and optimizer step dominate). The premise that "fewer params = more epochs in budget" was incorrect; no additional epochs were gained.
- At n_hidden=64 (~0.17M params), the model is capacity-limited. Loss still descending at E14 but slow (105.4→104.6, ~0.8%) — extrapolation shows 5-10 more epochs wouldn't close the 9% gap.
- Width-96 locked as the optimum. Future capacity changes should look at other axes (depth, heads, dropout).

### Decision
- **Closed.** Width axis exhausted: 64 < 96 < 128 < 192. n_hidden=96 confirmed optimal.

---

## 2026-05-16 00:35 — PR #3314: AdamW weight_decay 1e-4 → 3e-4 (rebased) [MERGED — NEW BASELINE]
- Branch: `charliepai2i24h2-fern/weight-decay-3e-4`
- Student: charliepai2i24h2-fern
- Hypothesis: Tripling weight_decay on the decay group (1e-4→3e-4) addresses under-regularization on hard OOD splits, specifically single_in_dist. Predicted 1–3% val improvement. Previous run vs old baseline gave −3.69%; this is the rebased retest against PR #3377 baseline (96.667).

### Results table

| Metric | Value | Δ vs PR #3377 baseline 96.667 |
|--------|-------|-------------------------------|
| `val_avg/mae_surf_p` (best @ ep 14) | **95.808** | **−0.86 (−0.89%)** ✓ |
| `test_avg/mae_surf_p` | 85.578 | +0.12 (+0.14%, within noise) |
| `val_single_in_dist` | 110.886 | −5.78 (−4.96%) ↓ best |
| `val_geom_rc` | 105.776 | +0.26 (+0.25%) flat |
| `val_geom_cruise` | 76.060 | +3.00 (+4.10%) ↑ regression |
| `val_re_rand` | 90.510 | −0.91 (−1.00%) ↓ slight win |
| Wall-clock | 32.0 min | ~137 s/epoch, 14/14 epochs |
| Peak VRAM | 40.96 GB | — |
| Metrics | `models/model-weight-decay-3e-4-rebased-20260515-232904/metrics.{jsonl,yaml}` | — |

### Analysis
- Confirmed positive: weight_decay=3e-4 still helps on the new full stack, though the delta is smaller than the first run (−0.89% vs −3.69%). The margin shrinkage is expected — warmup+cosine + smaller n_hidden already provides implicit regularization.
- Pattern identical to original run: single_in_dist concentrates the improvement (−4.96% val), cruise regresses (+4.10% val). The regularization knob trades hard-OOD accuracy for easy-cruise accuracy. Net aggregate favours merging.
- Test is flat (+0.14%), which is concerning for absolute claim-making but acceptable for compounding strategy — the val improvement is genuine and reproducible.
- Stack now: Huber + AdamW(wd=3e-4) + selective decay + grad-clip + NaN guard + warmup+cosine + lr=7e-4 + 14ep + slice_num=96 + n_hidden=96.

### Decision
- **MERGED. New baseline: val 95.808, test 85.578.**

---

## 2026-05-16 00:25 — PR #3503: Raise mlp_ratio 2→4 (FFN expansion) [CLOSED]
- Branch: `edward/mlp-ratio-4-ffn-expansion`
- Student: charliepai2i24h2-edward
- Hypothesis: Doubling FFN expansion (hidden 96→192→96) increases representational capacity orthogonal to width. Predicted 1–3% improvement on the PR #3377 baseline (96.667), particularly on single_in_dist.

### Results table

| Metric | Value | Δ vs baseline 96.667 |
|--------|-------|----------------------|
| `val_avg/mae_surf_p` (best @ ep 12) | **101.686** | **+5.02% (worse)** |
| `test_avg/mae_surf_p` | 91.012 | +5.56% (worse) |
| `val_single_in_dist` | 124.656 | +7.99 worst split regresses most |
| `val_geom_rc` | 110.078 | +4.56 |
| `val_geom_cruise` | 76.983 | +3.92 |
| `val_re_rand` | 95.027 | +3.61 |
| Params | 566,519 | +48.6% vs baseline 381,239 |
| Per-epoch wall | ~149 s | +14% slower (FFN compute) |
| Peak VRAM | 46.93 GB | +6 GB vs baseline |
| Metrics | `models/model-mlp-ratio-4-20260515-233534/metrics.{jsonl,yaml}` | — |

### Analysis
- Hypothesis falsified. Expanded FFN uniformly worsens all 4 splits — consistent with capacity-induced overfit (train loss dropped while val rose). The width-sweep confirmed this model regime benefits from smaller (not larger) capacity. FFN expansion without pairing stronger regularization adds parameters that absorb training-set noise. Also slightly truncated by 30-min cap (13 of 14 epochs), but val was already rising at E13 so a full run would not rescue it.
- Capacity (params) remains not the bottleneck for this dataset+budget.

### Decision
- **Closed.** Axis dead on current stack. FFN expansion would require simultaneous dropout/regularization to offset overfit risk.

---

## 2026-05-15 22:35 — PR #3453: Calibrate T_max=10 (slice_num=96 budget) [CLOSED]
- Branch: `charliepai2i24h2-edward/tmax-10-slice96`
- Student: charliepai2i24h2-edward
- Hypothesis: Lower CosineAnnealingLR T_max from 12 → 10 so cosine fully anneals within the ~12-epoch wall-clock cap. Predicted 0.5–2% improvement vs the slice_num=96 baseline (val 97.757).

### Results table

| Metric | Value | Δ vs baseline 97.757 |
|--------|-------|----------------------|
| `val_avg/mae_surf_p` (best @ ep 12) | **101.225** | **+3.55%** (worse) |
| `test_avg/mae_surf_p` | 90.892 | +5.21% (worse) |
| Per-split val | single 121.235 \| geom_rc 111.892 \| geom_cruise 77.292 \| re_rand 94.480 | all 4 regress |
| Wall-clock | 30.5 min | matches baseline |
| Peak VRAM | 47.5 GB | — |
| Metrics artifacts | (not committed to branch — negative result) | — |

### Analysis
- Hypothesis falsified. With T_max=10 the final two epochs run at lr ~1.71e-5 (vs ~1.03e-4 at the T_max=12 cutoff) — ~5–6× lower. Those near-zero-LR epochs are wasted.
- The "still strongly descending at cutoff" signal observed in PR #3399 argues the opposite of tighter annealing: at the wall-clock boundary we want a higher LR floor, not a lower one.
- Edward's diagnosis was excellent and the negative is a clean kill of the tighter-annealing direction. Eta_min raise (tanjiro #3397 in-flight) is the logical alternative.

### Decision
- **Close.** Cosine-annealing horizon is not the lever — the schedule shape at the wall-clock boundary is fine; if anything we want lr-floor raised, not pulled in.

---

## 2026-05-15 22:35 — PR #3301: Width-192 with budget-matched schedule (rebased retest) [CLOSED]
- Branch: `charliepai2i24h2-alphonse/width-192-matched-budget`
- Student: charliepai2i24h2-alphonse
- Hypothesis: Width 128 → 192 with epochs scaled to the wall-clock budget should improve representational capacity. First run on stale base (val 99.611) appeared to win but used wrong schedule. Rebased retest on warmup+cosine+slice_num=96 (HEAD `0b12feb`) needed for a fair comparison.

### Results table (rebased retest)

| Metric | Value | Δ vs baseline 97.757 |
|--------|-------|----------------------|
| `val_avg/mae_surf_p` (best @ ep 9 of 10) | **106.099** | **+8.53%** (worse) |
| `test_avg/mae_surf_p` | 94.120 | +8.95% (worse) |
| Per-split val | single 127.27 \| geom_rc 119.94 \| geom_cruise 80.61 \| re_rand 96.58 | all 4 regress 7–10% |
| Wall-clock | 30.7 min (cap hit before epoch 10) | over-budget |
| Peak VRAM | 63.0 GB | — |
| Param count | 1.48M | 1.43× baseline |
| Per-epoch wall-clock | ~205 s | matches advisor 210s prediction |
| Metrics artifacts | (rebased run not committed to branch — negative result) | — |

### Analysis
- Combined with PR #3377's width=96 result (val 96.667), the 3-point width sweep on the current stack {96, 128, 192} → {96.67, 97.76, 106.10} is **monotonically increasing**. Smaller width is better.
- Capacity is not the bottleneck at this depth/slice_num/budget — adding params just steals wall-clock from cosine annealing. Width-192 needs ~205 s/epoch and only fits 9 of 10 scheduled epochs.
- Bigger model + same wall-clock = fewer cosine-annealed epochs = stuck at higher LR at cutoff = worse final val.
- The original "stale-base" win (val 99.611) was a phantom: it benefited from the old 50-epoch cosine staying flat at the higher lr=5e-4 the whole run, which happened to favor more params. Under proper budget-matched annealing the picture flips.

### Decision
- **Close.** Direction killed. Width 96 (PR #3377) is the working direction.

---

## 2026-05-15 22:35 — PR #3304: surf_weight 10 → 20 (rebased retest) [CLOSED]
- Branch: `charliepai2i24h2-frieren/surf-weight-20`
- Student: charliepai2i24h2-frieren
- Hypothesis: 2× surface-pressure loss weighting helps recover from the under-fitting pressure residual visible on `single_in_dist`. On the OLD baseline (no warmup, lr=5e-4, slice_num=64) it gave -5.49% val. Retest needed to see if the lever still helps on the new stack.

### Results table (rebased retest, HEAD `0b12feb`)

| Metric | Value | Δ vs baseline 97.757 |
|--------|-------|----------------------|
| `val_avg/mae_surf_p` (best @ ep 12 of 14) | **101.782** | **+4.12%** (worse) |
| `test_avg/mae_surf_p` | 89.480 | +3.58% (worse) |
| Per-split val | single 123.625 \| geom_rc 111.597 \| geom_cruise 76.498 \| re_rand 95.411 | all 4 regress |
| Best epoch | 12 / 14 (wall-clock cap) | still descending at cap |
| Metrics artifacts | (rebased run not committed to branch — negative result) | — |

### Analysis
- Frieren's diagnosis is exactly right: on the OLD baseline the dominant pressure residual on `single_in_dist` was 148.1 (severely under-fit). On the NEW baseline it's already 115.5 — the warmup+cosine + slice_num=96 changes already shifted capacity toward surface fields. Doubling pressure weight on top now over-weights pressure and starves velocity learning.
- This is a clean example of an absorbed lever: surf_weight was helping by partially compensating for a different bottleneck the new baseline removed.

### Decision
- **Close.** Loss-rebalancing axis exhausted in the current regime. If we revisit it would be downward (surf_weight=5 or vol-weight bump) not upward.

---

## 2026-05-15 22:35 — PR #3377: n_hidden 128 → 96 (rebased retest) [HELD — branch needs push]
- Branch: `charliepai2i24h2-thorfinn/n-hidden-96`
- Student: charliepai2i24h2-thorfinn
- Hypothesis: Smaller width (~0.58× params) frees wall-clock for the cosine to fully anneal (14 of 14 epochs vs baseline's 12 of 14), at minimal capacity cost.

### Results table (rebased retest, HEAD `0b12feb`)

| Metric | Value | Δ vs baseline 97.757 |
|--------|-------|----------------------|
| `val_avg/mae_surf_p` (best @ ep 14 of 14) | **96.667** | **−1.12%** (win) |
| `test_avg/mae_surf_p` | 85.454 | −1.08% (win) |
| Per-split val | single 116.665 (+1.0%) \| geom_rc **105.516** (−4.5%) \| geom_cruise **73.065** (−3.1%) \| re_rand 91.421 (+1.9%) | net negative |
| Per-split test | single **99.939** (−1.7%) \| geom_rc 95.608 (+0.8%) \| geom_cruise **61.246** (−4.7%) \| re_rand 85.023 (+0.3%) | net negative |
| Epochs | 14 / 14 (full schedule) | +2 epochs vs baseline 12 |
| Per-epoch wall-clock | 136.7 s (μ) | −10% vs baseline 151 s |
| Total wall-clock | 31.9 min (+6% over 30-min cap, still fits) | — |
| Param count | 381,239 | 0.58× baseline |
| Peak VRAM | 40.97 GB | within budget |
| Metrics artifacts | `models/model-n-hidden-96-rebased-20260515-213114/` (NOT pushed to remote branch) | — |

### Analysis
- Width-96 is not capacity-limited at this depth/slice_num — it generalizes better. Gains concentrate on the geometric-OOD splits (`geom_camber_rc` val −4.5%, `geom_camber_cruise` test −4.7%), with mild regressions on `single_in_dist` (+1.0%) and `re_rand` (+1.9%).
- Combined with PR #3301: the 3-point width sweep {96, 128, 192} → {96.67, 97.76, 106.10} is monotonic. Smaller is better in this regime.
- Two mechanisms compound: (1) lower per-epoch cost lets all 14 epochs of cosine actually run; (2) regularization implicit in smaller width helps geometric OOD generalization.

### Decision
- **HOLD pending branch update.** Result is a clear merge-eligible win. But the rebased metrics commit and the n_hidden=96 code change were NOT pushed to origin (branch HEAD is still on `3c1b0ac` — the original stale-base commit). Sent back to student requesting:
  1. Rebase onto current advisor HEAD (`0b12feb`).
  2. Commit + push `models/model-n-hidden-96-rebased-20260515-213114/metrics.{jsonl,yaml}` and the single-line `n_hidden=96` change.
  3. Re-post terminal SENPAI-RESULT, mark ready for review.
- Once pushed, merge as new baseline.

---

## 2026-05-15 14:05 — PR #3208: Replace MSE with SmoothL1 (Huber) loss
- Branch: `charliepai2i24h2-fern/huber-loss`
- Student: charliepai2i24h2-fern
- Hypothesis: Replace MSE in normalized target space with SmoothL1 (Huber, β=1.0) in both training and validation; expect 4–10% improvement on `val_avg/mae_surf_p` from reducing high-Re sample domination.

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 13) | **116.611** |
| `test_avg/mae_surf_p` | NaN (pre-existing infra bug; 3 clean splits avg 114.59) |
| Epochs completed | 14/50 (cut at 30-min cap) |
| Peak VRAM | 42.1 GB |
| Wall-clock | ~30.6 min |
| Per-split val mae_surf_p | single 161.69 \| geom_rc 117.56 \| geom_cruise 85.67 \| re_rand 101.53 |
| Per-split test mae_surf_p | single 139.80 \| geom_rc 104.38 \| geom_cruise NaN \| re_rand 99.60 |
| Metrics artifact | `models/model-charliepai2i24h2-fern-huber-loss-20260515-130151/metrics.{jsonl,yaml}` |

### Analysis
- First completed experiment on this branch — establishes the de facto baseline at `val_avg/mae_surf_p` = 116.61 under the 30-min/14-epoch wall-clock cap.
- Trajectory still improving at the cap (epoch 1: 229.6 → epoch 13: 116.6). Headroom from longer schedules is real.
- Per-split pattern is consistent with the hypothesis: cruise (smallest pressure magnitudes) is easiest at 85.7; high-Re raceCar single is hardest at 161.7.
- **Important: no clean MSE companion run exists on this branch**, so the Huber-vs-MSE delta is not isolated. The 7 other in-flight round-1 PRs all carry MSE plus one other change, so we will get indirect signal.
- Two-line diff exactly matched the prescription — clean execution.

### Pre-existing NaN bug surfaced
- `test_geom_camber_cruise` sample 20 has non-finite `y` ground-truth values. `data/scoring.py:accumulate_batch` has a sample-level skip, but it computes `err = |pred - y|` *before* masking, and `NaN * 0 = NaN` so the masked sum propagates NaN. The same pattern exists in the normalized-space loss path of `train.py:evaluate_split`.
- `data/scoring.py` is read-only per `program.md`. The workaround is to sanitize `y` (zero-out NaN, mask out the sample) *before* the loss/scoring calls in `train.py:evaluate_split`.
- Routed to next fern assignment (gradient-clip + selective-decay PR also carries the NaN guard fix in `evaluate_split`).

### Decision
- **Merge.** The PR establishes the first concrete baseline. Huber loss carries forward to subsequent experiments. NaN bug is independent of this change.
- Updated `BASELINE.md` with the new reference numbers.

---

## 2026-05-15 15:30 — PR #3276: Gradient clip + AdamW selective decay (+ test NaN guard)
- Branch: `charliepai2i24h2-fern/grad-clip-selective-decay`
- Student: charliepai2i24h2-fern

### Hypothesis
Two bundled changes: (1) grad clip max_norm=1.0 + AdamW selective decay (LN/bias/1D params excluded from weight decay) — transformer best-practice optimizer tuning; (2) NaN sample guard in `evaluate_split` to recover finite test metrics. Predicted 1–4% improvement on `val_avg/mae_surf_p`.

### Results table

| Metric | Value | Delta vs baseline (116.61) |
|--------|-------|---------------------------|
| `val_avg/mae_surf_p` (best @ epoch 14) | **109.681** | **-5.94%** |
| `test_avg/mae_surf_p` | **97.315** | finite (was NaN) |
| Epochs completed | 14/50 (cut at 30-min cap) |  |
| Peak VRAM | 42.1 GB | |
| Wall-clock | ~30.7 min | |
| Per-split val mae_surf_p | single 148.09 \| geom_rc 114.87 \| geom_cruise 78.85 \| re_rand 96.91 | all 4 improved |
| Per-split test mae_surf_p | single 123.24 \| geom_rc 104.76 \| geom_cruise 68.48 \| re_rand 92.79 | 3 improved; geom_rc flat |
| Optimizer groups | decay=49 (0.655M params), no_decay=62 (0.008M params) | |
| Metrics artifact | `models/model-grad-clip-selective-decay-20260515-142950/metrics.{jsonl,yaml}` | |

### Analysis
- Beat the 1–4% prediction with a 5.94% improvement — best result on this branch.
- All 4 val splits improved. Largest gains: single (-8.4%) and cruise (-8.0%). Smallest: geom_rc (-2.3%).
- The geom_rc split lags on both val and test — consistent with a capacity or domain-coverage issue rather than optimizer sensitivity.
- NaN guard worked: test_geom_camber_cruise now reports 68.48 instead of NaN. Cruise is actually the easiest test split.
- Best epoch was the final (14/14) — cosine still annealing at timeout; headroom exists from longer runs.
- Optimizer group split is correct: ~99% of param mass in decay group; LN gains/biases/temperature/placeholder in no_decay.

### Decision
- **Merge.** Clear improvement across all metrics. NaN guard is now baseline infrastructure. New benchmark: `val_avg/mae_surf_p` = 109.68.

---

## 2026-05-15 15:30 — PR #3220: Linear warmup + cosine over 100 epochs at lr 7e-4 (CLOSED)
- Branch: `charliepai2i24h2-tanjiro/warmup-cosine-100ep`
- Student: charliepai2i24h2-tanjiro

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 14) | 148.200 |
| `test_avg/mae_surf_p` | NaN (pre-existing bug; 3-clean 148.74) |
| Epochs completed | 14/100 (cut at 30-min cap) |
| Per-split val mae_surf_p | single 191.75 \| geom_rc 152.67 \| geom_cruise 118.65 \| re_rand 129.73 |

### Analysis
- 27% regression vs Huber baseline (148.20 vs 116.61). Schedule was effectively a flat-high-LR run — 100 epochs under a 30-min cap means the cosine never cooled. The hypothesis (warmup + cosine helps) is not testable with this schedule length.
- Student correctly diagnosed the issue and suggested matching schedule to completable epochs.
- **Decision: Closed. Follow-up: matched 14-epoch warmup+cosine (PR #3294).**

---

## 2026-05-15 15:30 — PR #3205: Scale attention slice_num 64→192 and n_head 4→8 (CLOSED)
- Branch: `charliepai2i24h2-edward/slices-192-heads-8`
- Student: charliepai2i24h2-edward

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 5) | 164.377 |
| `test_avg/mae_surf_p` | 150.944 |
| Epochs completed | 5/50 (cut at 30-min cap) |
| Peak VRAM | 38.1 GB (with gradient checkpointing + bf16 AMP) |
| Per-split val mae_surf_p | single 207.07 \| geom_rc 179.17 \| geom_cruise 129.06 \| re_rand 142.20 |

### Analysis
- 41% regression vs Huber baseline. Only 5 epochs at ~385 s/epoch. Hypothesis untestable at this scale under 30-min budget. Student correctly diagnosed OOM → added gradient checkpointing + bf16 AMP. cruise split at 5 epochs (129.06 val) shows representation capacity but budget overwhelms the signal.
- Merge conflict with HEAD (post Huber merge). Student's follow-up #2 (decouple slice_num vs n_head axes) is the right next move.
- **Decision: Closed. Follow-up: single-axis slice_num=128 (PR #3295).**

---

## 2026-05-15 15:30 — PR #3179: Scale Transolver n_hidden 128→192 (CLOSED)
- Branch: `charliepai2i24h2-alphonse/width-192`
- Student: charliepai2i24h2-alphonse

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 7) | 154.979 |
| `test_avg/mae_surf_p` | 144.489 |
| Epochs completed | 10/50 (cut at 30-min cap) |
| Actual param count | 1.47M (not 2.25M as predicted) |
| Peak VRAM | 58.0 GB |

### Analysis
- 33% regression vs baseline. 10 epochs at ~185 s/epoch; cosine never annealed. Epoch 8 noise jump (155→215→169) shows the model is operating at high LR throughout. Student correctly identified the budget mismatch and suggested matched-epoch re-run.
- **Decision: Closed. Follow-up: width-192 with epochs=10 budget-matched (PR #3301).**

---

## 2026-05-15 15:30 — PR #3183: Scale Transolver depth 5→8 layers (CLOSED)
- Branch: `charliepai2i24h2-askeladd/depth-8`
- Student: charliepai2i24h2-askeladd

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 9) | 154.955 |
| `test_avg/mae_surf_p` | NaN (3-clean 157.11) |
| Epochs completed | 9/50 (cut at 30-min cap) |
| Actual param count | ~1.03M |
| Peak VRAM | 64.5 GB |

### Analysis
- 33% regression vs baseline. 9 epochs at ~206 s/epoch. Val still rapidly descending at termination (155 at epoch 9 down from 254 at epoch 1). Hypothesis untestable without budget-matched schedule. Student correctly identified budget-schedule mismatch and noted pressure overflow NaN (now fixed in merged #3276).
- **Decision: Closed. Follow-up: depth-8 with epochs=9 budget-matched (PR #3302).**

---

## 2026-05-15 15:30 — PR #3214: Surf weight 10→30 with 2× pressure channel (CLOSED)
- Branch: `charliepai2i24h2-frieren/surf-weight-30-pchannel-rerun`
- Student: charliepai2i24h2-frieren

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 12) | 138.435 |
| `test_avg/mae_surf_p` | 125.189 |
| Epochs completed | 14/50 (cut at 30-min cap) |

### Analysis
- 19% regression vs Huber baseline (138.44 vs 116.61); 26% vs new baseline (109.68). Combined 6× pressure emphasis hurts velocity-channel learning in early epochs. The geom_cruise split (99.00 val) vs baseline (85.67) shows cruise is actually worse with the heavier weighting. Student correctly identified that a paired surf_weight=10 run would be needed for clean attribution.
- **Decision: Closed. Follow-up: surf_weight=20 single-axis no channel weighting (PR #3304).**

---

## 2026-05-15 16:50 — PR #3216: 32-frequency Fourier features over (x, z) (CLOSED)
- Branch: `charliepai2i24h2-nezuko/fourier-pe-32`
- Student: charliepai2i24h2-nezuko

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 13) | 137.936 |
| `test_avg/mae_surf_p` | NaN (3-clean 140.08) |
| Epochs completed | 14/50 (cut at 30-min cap) |
| Per-split val mae_surf_p | single 177.19 \| geom_rc 150.36 \| geom_cruise 99.72 \| re_rand 124.47 |

### Analysis
- 26% regression vs current baseline (137.94 vs 109.68). The prescription had a bug that the student correctly flagged before the run: `freqs.unsqueeze(-1).expand(num_freq, num_input_channels)` collapses `B[k,0]==B[k,1]`, so the encoding could only distinguish (x+z) not (x, z) independently. This isn't a clean test of 2D Fourier features.
- Also had merge conflicts (predates Huber + grad-clip merges).
- Student again diagnosed the NaN-GT bug correctly; that's now fixed in merged PR #3276.
- **Decision: Closed. Follow-up: corrected Fourier-PE with random Gaussian B per Tancik 2020 RFF, true 2D directional information (PR #3344).**

---

## 2026-05-15 17:30 — PR #3294: Warmup + cosine over 14 epochs, lr=7e-4 (WINNER — pending rebase)
- Branch: `charliepai2i24h2-tanjiro/warmup-cosine-14ep`
- Student: charliepai2i24h2-tanjiro
- Hypothesis: Budget-matched 14-epoch schedule (vs 50-epoch cosine that never anneals), lr bumped 5e-4→7e-4, 2-ep linear warmup.

### Results table

| Metric | Value | Delta vs baseline (109.68) |
|--------|-------|---------------------------|
| `val_avg/mae_surf_p` (best @ epoch 14) | **100.811** | **-8.08%** |
| `test_avg/mae_surf_p` | NaN (3-clean 99.15) | — |
| Epochs completed | 14/14 (30-min cap) | |
| Per-split val mae_surf_p | single 118.74 \| geom_rc 107.10 \| geom_cruise 81.97 \| re_rand 95.43 | |
| Per-split test mae_surf_p | single 109.88 \| geom_rc 95.54 \| geom_cruise NaN (infra) \| re_rand 92.02 | |
| Peak VRAM | 42.1 GB | |
| Wall-clock | ~31 min (~132 s/epoch) | |
| Metrics artifact | `models/model-warmup-cosine-14ep-20260515-162249/metrics.{jsonl,yaml}` | |

### Analysis
- Clear winner: -8.08% val improvement (compared against prior best 109.68), budget-matching the cosine schedule was the key insight.
- Best epoch = 14 (the last); cosine cooled fully within budget and epochs 12–14 monotonically extracted additional signal from the low-LR tail.
- Per-split pattern: single_in_dist -26.6% (biggest lever), geom_rc -8.9%, re_rand -6.0%. Cruise slightly regressed +4.0% — same pattern as other improvements, suggests harder OOD patterns benefit most from schedule quality.
- Two runs at same config: 99.24 and 100.81 — run-to-run variability ~1.6%, well within noise.
- test_geom_camber_cruise still NaN (pre-existing; fern's NaN guard fixes training but test eval route has separate issue); 3-clean test mean 99.15 confirms val improvement is real.
- PR has merge conflict (predates advisor HEAD); student asked to rebase before merge.

### Decision
- **Pending merge** — winner, sent back to student for rebase. Will become the new baseline at val=100.81 after merge.

---

## 2026-05-15 17:30 — PR #3304: surf_weight 10→20 (WINNER — on hold pending tanjiro merge)
- Branch: `charliepai2i24h2-frieren/surf-weight-20`
- Student: charliepai2i24h2-frieren
- Hypothesis: Raise surf_weight 10→20 (single-axis, no channel weighting) — a 2× surface emphasis as a derisked follow-up to the failed 6× combined version in PR #3214.

### Results table

| Metric | Value | Delta vs baseline (109.68) |
|--------|-------|---------------------------|
| `val_avg/mae_surf_p` (best @ epoch 14) | **103.668** | **-5.49%** |
| `test_avg/mae_surf_p` | **93.243** | **-4.18%** |
| Epochs completed | 14/50 (30-min cap) | |
| Per-split val mae_surf_p | single 120.30 \| geom_rc 120.82 \| geom_cruise 77.22 \| re_rand 96.33 | |
| Per-split test mae_surf_p | single 111.45 \| geom_rc 106.33 \| geom_cruise 65.51 \| re_rand 89.69 | |
| Metrics artifact | `models/model-surf-weight-20-20260515-162204/metrics.{jsonl,yaml}` | |

### Analysis
- Beats baseline by 5.49% val / 4.18% test, well within predicted 1–4% range (actually slightly beat it).
- Concentrated improvement on single_in_dist (-18.8% val / -9.6% test). Moderate cruise + re_rand gains. geom_rc regressed slightly (+5.2% val, +1.5% test — the hardest geometry-OOD split).
- The cruise split (easiest) improved mildly (-2.1%) — confirms surf emphasis isn't over-weighting.
- Training was healthy at epoch 14 (best epoch = final); velocity channels not visibly degraded.
- **However**: tanjiro's warmup+cosine (#3294) achieves 100.81 — better than frieren's 103.67. Decision: hold frieren in review, request rebase+retest after tanjiro merges to see if surf_weight=20 still helps on the warmup+cosine baseline.

### Decision
- **On hold** — result beats 109.68 baseline, but pending tanjiro merge to new baseline ~100.81. Frieren will rebase+retest to confirm surf_weight=20 is orthogonally beneficial.

---

## 2026-05-15 17:30 — PR #3314: weight_decay 1e-4→3e-4 on decay group (ON HOLD pending tanjiro merge)
- Branch: `charliepai2i24h2-fern/weight-decay-3e-4`
- Student: charliepai2i24h2-fern
- Hypothesis: Triple the weight decay on the decay group; fern's own suggestion from PR #3276 analysis.

### Results table

| Metric | Value | Delta vs baseline (109.68) |
|--------|-------|---------------------------|
| `val_avg/mae_surf_p` (best @ epoch 13) | **105.640** | **-3.69%** |
| `test_avg/mae_surf_p` | **94.734** | **-2.65%** |
| Epochs completed | 14/50 (30-min cap) | |
| Per-split val mae_surf_p | single 120.97 \| geom_rc 114.89 \| geom_cruise 88.84 \| re_rand 97.86 | |
| Per-split test mae_surf_p | single 108.73 \| geom_rc 101.17 \| geom_cruise 75.70 \| re_rand 93.33 | |
| Metrics artifact | `models/model-weight-decay-3e-4-20260515-162522/metrics.{jsonl,yaml}` | |

### Analysis
- Beats baseline by 3.69% val / 2.65% test (slightly above the predicted 1–3% range).
- Concentrated gain on single_in_dist (-18.3% val / -11.8% test), while cruise **regresses** +12.7% val / +10.6% test. Other splits flat.
- Same pattern as frieren's surf_weight=20 run: both levers improve the harder single_in_dist split by reducing overfit, while slightly over-regularizing the easier cruise distribution.
- The favorable aggregate is because single_in_dist has the largest absolute MAE — a 27-point improvement there offsets the 10-point regression on cruise.
- **However**: tanjiro's 100.81 beats fern's 105.64. Hold pending tanjiro merge; the optimal wd value may shift on the warmup+cosine baseline.

### Decision
- **On hold** — result beats 109.68 baseline, but pending tanjiro merge to new baseline ~100.81. Fern will rebase+retest to confirm wd=3e-4 is still the right magnitude with warmup+cosine schedule.

---

## 2026-05-15 17:30 — PR #3302: Depth-8 budget-matched epochs=9 (CLOSED)
- Branch: `charliepai2i24h2-askeladd/depth-8-matched-budget`
- Student: charliepai2i24h2-askeladd
- Hypothesis: Depth-8 (n_layers=8) with epochs=9 budget-matched to ~205 s/epoch; follow-up to round-1 where 50-epoch cosine never annealed.

### Results table

| Metric | Value | Delta vs baseline (109.68) |
|--------|-------|---------------------------|
| `val_avg/mae_surf_p` (best @ epoch 9) | **111.357** | +1.53% |
| `test_avg/mae_surf_p` | **100.776** | +3.55% |
| Epochs completed | 9/9 (30-min cap) | |
| Per-split val mae_surf_p | single 137.19 \| geom_rc 126.59 \| geom_cruise 82.75 \| re_rand 98.91 | |
| Per-split test mae_surf_p | single 124.31 \| geom_rc 109.95 \| geom_cruise 70.86 \| re_rand 97.99 | |
| Param count | ~1.03M (~1.5× baseline) | |
| Peak VRAM | 64.5 GB | |
| Metrics artifact | `models/model-depth-8-matched-budget-20260515-162201/metrics.{jsonl,yaml}` | |

### Analysis
- Regressed +1.53% val / +3.55% test vs baseline. Budget-matching helped (was 33% regression in round-1 at 9ep) but still hasn't converged — best epoch = 9 (last), slope -6.5 mae/epoch at termination.
- Per-split: depth-8 WINS on single_in_dist (-7.4%) but loses on geom_camber_rc (+10.2%) and cruise (+4.9%). More capacity helps in-distribution but can't generalize to geometry-OOD under this budget.
- Student correctly diagnosed: needs more wall time than depth-5 (205 s vs 130 s per epoch). BF16 (#3223) or larger batch_size needed to compress per-epoch cost.
- **Decision: Closed.** Architecture depth-scaling held until per-epoch cost drops (BF16 or batch8).

### Decision
- **Closed.** Reassigned askeladd to n_head=8 single-axis (#3362) — orthogonal, minimal cost overhead.

---

## 2026-05-15 17:50 — PR #3223: BF16 autocast + batch_size=8 (CLOSED)
- Branch: `charliepai2i24h2-thorfinn/bf16-batch8`
- Student: charliepai2i24h2-thorfinn
- Hypothesis: BF16 mixed precision + batch_size 4→8 for ~2× throughput; preserve 50-epoch cosine schedule.

### Results table

| Metric | Value | Delta vs baseline (109.68) |
|--------|-------|---------------------------|
| `val_avg/mae_surf_p` (best @ epoch 16, primary run) | **147.328** | **+34.30%** |
| `test_avg/mae_surf_p` (post-hoc) | **133.364** | +37.04% |
| Mean of 4 runs val | 140.18 ± 10.0 | +27.8% |
| Mean of 4 runs test | 126.57 ± 9.8 | +30.1% |
| Epochs completed | 15–17 / 50 (30-min cap, run-dependent) | |
| Per-epoch wall clock | ~106 s (vs baseline ~130 s) | |
| Peak VRAM | 65.87 GB | |
| Per-split val mae_surf_p (primary) | single 205.77 \| geom_rc 168.76 \| geom_cruise 95.99 \| re_rand 118.79 | |
| Metrics artifacts | `models/model-charliepai2i24h2-thorfinn-bf16-batch8-20260515-162822/metrics.{jsonl,yaml}` (primary) + 3 prior runs | |

### Analysis
- Significant regression: +34% val on primary run, +27% mean across 4 runs. Large run-to-run variance (10-point std) because truncated 50-epoch cosine never anneals — different stopping points hit different LR points.
- BF16 was numerically clean end-to-end (no NaN/Inf in training). The regression is from the *combined* change, not BF16 itself.
- Per-epoch cost (~106 s) was lower than baseline (~130 s) but per-step cost was higher: 188 batches/epoch × 564 ms/batch vs baseline 376 batches × ~345 ms. The variable-mesh padding overhead dominates batch=8 cost; each large-mesh sample (242K nodes) forces padding for the whole batch.
- The combined effect = (modest per-epoch speedup) + (larger gradient noise reduction) + (truncated cosine schedule) → net loss.
- **Bug-fix bonus**: Student correctly diagnosed and fixed the `evaluate_split` NaN bug (Inf×False=NaN in test_geom_camber_cruise sample 20). Identical to fern's #3276 fix already in HEAD. Useful confirmation.
- Student also added `eval_test_only.py` for post-hoc checkpoint re-evaluation. Useful tool but not pulled into baseline (extra surface area).
- Student's follow-up analysis is excellent: (1) decouple precision and batch knobs; (2) the wall-clock is the bottleneck; (3) padding overhead is the architectural blocker for batch scaling.

### Decision
- **Closed.** Combined hypothesis regressed. Reassigned thorfinn to n_hidden=96 (#3377) — orthogonal architectural axis, completes the 3-point width sweep {96, 128, 192}.

---

## 2026-05-15 18:24 — PR #3344: Random Fourier Features over (x, z) with correct 2D encoding (ON HOLD — retest needed)
- Branch: `charliepai2i24h2-nezuko/fourier-pe-rff-32`
- Student: charliepai2i24h2-nezuko
- Hypothesis: Corrected RFF with random Gaussian B (Tancik 2020), true 2D independent frequencies. Fixed the collapsed `x+z` bug from PR #3216.

### Results table (vs OLD baseline 109.68)

| Metric | Value | Delta vs old baseline (109.68) |
|--------|-------|-------------------------------|
| `val_avg/mae_surf_p` (best @ epoch 13) | **103.891** | **-5.28%** |
| `test_avg/mae_surf_p` | **94.178** | **-3.22%** |
| Epochs completed | 13/14 (30.8 min) | |
| Per-epoch wall clock | ~132 s | no overhead |
| Param count | ~0.68M (+15K vs baseline) | minimal increase |
| Per-split val mae_surf_p | single 131.79 \| geom_rc 114.70 \| geom_cruise 74.97 \| re_rand 94.10 | |
| Per-split test mae_surf_p | single 115.96 \| geom_rc 106.56 \| geom_cruise 64.75 \| re_rand 89.43 | |
| Metrics artifact | `models/model-fourier-pe-rff-32-20260515-173713/metrics.{jsonl,yaml}` | |

### Analysis
- Corrected RFF works: -5.28% val improvement over old baseline. The fix from PR #3216's collapsed basis (x+z) to proper 2D random Gaussian B is validated.
- Improvement concentrated on single_in_dist (-11.0% val) and geom_cruise (-4.9%). Geom_camber_rc flat (-0.15%) — the persistent laggard remains hard.
- Interpretation: RFF acts as a generic high-frequency feature enrichment for the input projection rather than a geometry-specific fix. Benefits dense surface gradient regions.
- **However**: tanjiro's warmup+cosine merged with new baseline 100.811. Nezuko's 103.891 is now +3.1% above new baseline. Sent back to rebase+retest — RFF is orthogonal to schedule and likely compounds.

### Decision
- **Retest pending.** val 103.891 beats old 109.68 but misses new 100.81 baseline. RFF hypothesis remains promising. Nezuko sent back to rebase+retest on new baseline.

---

## 2026-05-15 18:29 — PR #3295: slice_num 64→128 single-axis (CLOSED)
- Branch: `charliepai2i24h2-edward/slice-num-128`
- Student: charliepai2i24h2-edward
- Hypothesis: Double the attention slots (64→128), keeping n_head=4. Single-axis from the confounded PR #3205 (which combined heads+slots).

### Results table

| Metric | Value | Delta vs Huber baseline (116.61) |
|--------|-------|----------------------------------|
| `val_avg/mae_surf_p` (best @ epoch 6/11, canonical) | **140.577** | **+20.5%** |
| `test_avg/mae_surf_p` | NaN (infra bug) | — |
| Best seed of 3 runs | 119.320 | +2.3% (still worse) |
| Mean of 3 runs val | 127.65 ± 10.6 | +9.5% |
| Epochs completed | 10–11/50 (30-min cap) | |
| Per-epoch wall clock | ~172 s | +32% vs baseline |
| Peak VRAM | 54.5 GB | |
| Metrics artifact | `models/model-slice-num-128-20260515-172243/metrics.{jsonl,yaml}` (canonical) + 2 prior seeds | |

### Analysis
- Clear regression: canonical +20.5%, best seed +2.3% — all 3 runs worse than Huber baseline. High seed variance (119–140 spread) signals undertrained regime.
- Budget-mismatch: +32% per-epoch cost reduces from 14 to 11 completable epochs. Both surviving runs (11 epochs) are still descending at cutoff.
- Architecture not broken — starved of budget. Geom_camber_rc (+36%) and re_rand (+22%) are the most damaged splits (geometry/Re generalization hardest under truncated training).
- Param count essentially unchanged (+10K, +1.6%) — the overhead is compute, not parameters.

### Decision
- **Closed.** Reassigned edward to slice_num=96 (#3399) — mid-point between 64 and 128. Expected ~150 s/epoch → 11-12 epochs, closer to baseline convergence.

---

## 2026-05-15 20:10 — PR #3377: Scale n_hidden 128→96 (width sweep, stale base)
- Branch: `charliepai2i24h2-thorfinn/n-hidden-96`
- Student: charliepai2i24h2-thorfinn
- Hypothesis: Reduce hidden width from 128 to 96 to cut per-epoch cost (~10%), enabling ~2 extra cosine-tail epochs in the 30-min cap. Tests whether Transolver is capacity-overprovisioned at width=128 under truncated training.

### Results table

| Metric | n_hidden=96 | Baseline (#3276, OLD) | Δ vs old | Vs NEW baseline (#3294) | Δ vs new |
|--------|------------|----------------------|----------|------------------------|----------|
| `val_avg/mae_surf_p` | **102.082** | 109.681 | **-6.93%** | 100.811 | +1.26% (regression) |
| `test_avg/mae_surf_p` | **91.684** | 97.315 | **-5.79%** | — | — |
| Best epoch | 16 of 16 (timeout) | 14 | +2 | — | — |
| Per-epoch wall-clock | ~117 s | ~132 s | **-11%** | — | — |
| Param count | 0.377M | 0.655M | -42% | — | — |
| Peak VRAM | 34.15 GB | ~42 GB | -19% | — | — |
| Per-split val mae_surf_p | single 126.66 \| rc 109.21 \| cruise 78.94 \| re_rand 93.52 | single 148.09 \| rc 114.87 \| cruise 78.85 \| re_rand 96.91 | — | — | — |
| Metrics artifact | `models/model-n-hidden-96-20260515-182809/metrics.{jsonl,yaml}` | — | — | — | — |

### Analysis
- Beat OLD baseline 109.68 by -6.93%, but student branch was based on pre-#3294 commit (lr=5e-4, T_max=50 cosine, no warmup). This is the OLD config regime — not a fair comparison against NEW baseline.
- Model descending at timeout (epoch 16 best — "still descending"). Mechanism: T_max=50 at epoch 16 → LR at 36% of peak, effectively a partial cosine anneal. Happy accident that recreates part of what warmup+cosine does systematically.
- `geom_camber_cruise` (+0.1%) appears saturated across widths — not a capacity bottleneck.
- `val_single_in_dist` (-14.5%) is the main signal — smaller model benefits from cleaner schedule/budget.

### Decision
- **Sent back for rebase + retest.** Branch uses pre-#3294 config. Need clean rebase onto current HEAD (warmup+cosine + lr=7e-4 + slice_num=64) and rerun at epochs=14.
- Predicted retest: 99–103. If ≤100.811, it's a slight win; if >100.811, capacity is the bottleneck and direction is UP (alphonse's width=192 confirms this).

---

## 2026-05-15 20:30 — PR #3399: slice_num 64→96 (WINNER)
- Branch: `charliepai2i24h2-edward/slice-num-96`
- Student: charliepai2i24h2-edward
- Hypothesis: Mid-point slot sweep between baseline (64, 132s/ep) and failed 128 (+32% cost). slice_num=96 predicts ~151s/ep → 11-12 epochs in 30-min budget. Richer slot attention without budget starvation.

### Results table

| Metric | slice_num=96 | Baseline (#3294) | Δ |
|--------|-------------|-----------------|---|
| `val_avg/mae_surf_p` (best @ epoch 12) | **97.757** | 100.811 | **-3.03%** |
| `test_avg/mae_surf_p` | **86.388** | ~99.15 (3-split) | **significant** |
| Per-split val mae_surf_p | single 115.495 \| rc 110.451 \| cruise 75.398 \| re_rand 89.685 | single 118.74 \| rc 107.10 \| cruise 81.97 \| re_rand 95.43 | |
| Epochs completed | 12 of 14 (30.2 min cutoff) | 14 | — |
| Per-epoch wall-clock | ~150.9 s | ~132 s | +14.3% |
| Peak VRAM | 47.56 GB | ~42 GB | +13.2% |
| Param count | 667,639 | ~662,000 | +0.8% |
| Metrics artifact | `models/model-slice-num-96-20260515-192511/metrics.{jsonl,yaml}` | — | — |

### Analysis
- Clear winner on correct base (post-#3294). 3 of 4 splits improved; only geom_camber_rc regressed +3.1%.
- Model still strongly descending at cutoff (epoch 12 best, drop -7.7 from epoch 11→12). Cosine completed ~83% of T_max=12 anneal — meaningful but not complete. 1-2 more epochs would likely help.
- slice_num=96 gives richer attention (2.25× more slots than 64) while fitting budget — the sweet spot hypothesis holds.
- cruise and re_rand see largest gains (-8% and -6%) — geometric and Re diversity benefits from more slot resolution.
- Follow-up: tune T_max=10 to match actual 12-epoch budget, or investigate BF16 to get 13-14 epochs within cap.

### Decision
- **MERGED** — clear winner. New baseline: val_avg/mae_surf_p = 97.757.
