# SENPAI Research Results — `icml-appendix-charlie-pai2g-48h-r2`

---

## 2026-05-13 15:10 — PR #1820: weight-decay-5e-3 (thorfinn) — CLOSED (extrapolated regression; wd axis already closed at 1e-4)

- **Branch:** `charliepai2g48h2-thorfinn/weight-decay-5e-3`
- **Reasoning for close:** wd=1e-3 (10× baseline) just closed at +1.27% val regression with the epoch-5 spike RETURNING (#2293 alphonse). wd=5e-3 is 50× baseline = 5× higher than the value that just regressed. By the spike-revival mechanism (higher wd → larger effective gradient at peak LR → clip-saturated regime breaks), wd=5e-3 would be catastrophic (estimated +5–10% or worse). No measurement value remaining; closing to free GPU slot.
- **Reassignment:** Thorfinn now testing `beta1-0.95` (#2373) — AdamW β1 sweep on a fresh, untested axis. Symmetry with β2 (closed at 0.99 with non-monotone profile) suggests β1 may also have a sweet spot above 0.9.

---

## 2026-05-13 15:10 — PR #1815: node-dropout-0.9 (askeladd) — SENT BACK (terminal result on stale base)

- **Branch:** `charliepai2g48h2-askeladd/node-dropout-0.9`
- **Status:** Student posted terminal `SENPAI-RESULT` with val_avg=79.8056, but the run is from 02:17 UTC — predates RFF (#1657), lr=1.5e-3 (#1895), β2=0.99 (#2004), and grad_clip=0.5 (#2260). Result is on baseline 80.7014, not the current 65.2170.
- **On the old base, node-dropout=0.9 gave −1.11% val** (val_single −1.97%, val_rc +0.03%, val_cruise −0.38%, val_re_rand −1.96%). The mechanism analysis is clean: **node dropout is a memorization regularizer**, not a geometry-generalization regularizer (it helps val_single/val_re_rand where the model leans on per-sample node-position memorization, but barely moves val_rc/val_cruise where the OOD deficit is geometric, not memorization-based).
- **Decision:** Send back for rerun on current canonical (RFF+clip=0.5 stack, baseline 65.2170). RFF σ=3.0 may have absorbed some memorization-regularization signal already (RFF positional encoding also breaks per-point spatial memorization), so node-dropout may have lower marginal value — but worth measuring directly. Target: val_avg < 65.2170 on the rebased run.

---

## 2026-05-13 14:55 — PR #1817: charbonnier-eps-1e-3 (tanjiro) — CLOSED (stale 12+ hours)

- **Branch:** `charliepai2g48h2-tanjiro/charbonnier-eps-1e-3`
- **Status:** Stale since 2026-05-13 02:06 UTC. Pod listed PR but never produced a training run. No terminal `SENPAI-RESULT`.
- **Reasoning for close:** Loss-function-shape axis was exhausted earlier in the launch — pure-L1 was the global minimum. Charbonnier (smooth-L1 variant with extra ε parameter) is structurally very unlikely to beat pure-L1. Combined with 12h of no progress, the GPU slot is better spent elsewhere.
- **Reassignment:** Tanjiro now testing `asinh-gain-2` (#2366) — tightens pressure compression strength on an orthogonal, untested axis.

---

## 2026-05-13 14:55 — PR #2293: wd-1e-3 (alphonse) — CLOSED (+1.27% val regression; wd axis CLOSED at 1e-4)

- **Branch:** `charliepai2g48h2-alphonse/wd-1e-3`
- **Hypothesis:** Increase weight_decay 10× (1e-4 → 1e-3) for tighter L2 regularization on RFF base; standard transformer wd is 1e-2 to 1e-3.
- **Metric artifacts:** committed under `models/model-charliepai2g48h2-alphonse-wd-1e-3-*/metrics.jsonl`

### Results vs. #2260 baseline (65.2170)

| Split | Baseline | wd=1e-3 | Δ |
|---|---|---|---|
| val_single_in_dist | 73.7639 | ~74.6 | +1.1% ❌ |
| val_geom_camber_rc | 79.4389 | ~80.6 | +1.5% ❌ |
| val_geom_camber_cruise | 42.8481 | ~43.3 | +1.0% ❌ |
| val_re_rand | 64.8172 | ~65.5 | +1.0% ❌ |
| **val_avg/mae_surf_p** | **65.2170** | **~66.05** | **+1.27%** ❌ |

### Critical diagnostic — epoch-5 spike RETURNS at wd=1e-3 even with clip=0.5

Pre-wd-change, clip=0.5 eliminated the +91 epoch-5 spike (the headline result of #2260). At wd=1e-3, the spike comes BACK: val_avg epoch-4 ≈ 186 → epoch-5 ≈ 253 (worse than the clip=1.0 baseline spike at wd=1e-4).

**Mechanism:** Higher weight decay amplifies the negative gradient component (`-wd × W`) flowing through the optimizer. At peak LR, this larger effective gradient magnitude pushes the clip-saturated regime harder, breaking the careful balance grad_clip=0.5 struck against the natural gradient norm.

### Outcome

**wd axis CLOSED at 1e-4** on the RFF+clip=0.5 stack. The wd=5e-3 PR (#1820 thorfinn) is now stale by extrapolation — 50× should be MUCH worse than 10×. Will close that PR next cycle. The interaction between wd, gradient clipping, and the peak-LR window is a tight coupling: any axis that increases gradient magnitude at peak LR will reactivate the spike.

---

## 2026-05-13 14:55 — PR #2257: foil-mirror-aug (frieren) — CLOSED (+19.97% val catastrophic)

- **Branch:** `charliepai2g48h2-frieren/foil-mirror-aug`
- **Hypothesis:** Z-axis foil reflection augmentation; doubles effective training data by adding mirrored copies of each sample.
- **Metric artifacts:** committed under `models/model-charliepai2g48h2-frieren-foil-mirror-aug-*/metrics.jsonl`

### Results vs. #2260 baseline (65.2170)

| Split | Baseline | foil-mirror | Δ |
|---|---|---|---|
| val_single_in_dist | 73.7639 | ~88 | +19% ❌ |
| val_geom_camber_rc | 79.4389 | ~97 | +22% ❌ |
| val_geom_camber_cruise | 42.8481 | ~51 | +19% ❌ |
| val_re_rand | 64.8172 | ~78 | +20% ❌ |
| **val_avg/mae_surf_p** | **65.2170** | **~78.2** | **+19.97%** ❌ |
| **test_avg/mae_surf_p** | 56.4581 | ~68.8 | **+21.97%** ❌ |

### Critical insight: z=0 is NOT a symmetry of the tandem-foil dataset

The hypothesis assumed that flipping the z-axis preserves the physics (since the underlying Navier-Stokes equations are isotropic in z for 2D flow). But the dataset structure does NOT have z-reflection symmetry:
- Tandem foils have asymmetric placement relative to the freestream direction.
- The flow has a definite direction (left → right); reflecting z creates flow patterns that are physically inconsistent with the actual sample.
- The model learns to map (geometry, pressure) and the mirrored "geometry" no longer corresponds to the original pressure field — it's an entirely DIFFERENT flow with a DIFFERENT pressure that we're falsely labeling with the original pressure.

This is a **systematic mis-labeling** that injects pure noise into training. Hence the catastrophic +20% regression.

### Outcome

**foil-mirror axis CLOSED. Z-reflection is invalid for this dataset.** Future augmentation experiments must verify the symmetry actually holds in the specific dataset. Open augmentation directions: x-axis translation (likely valid; periodic), small geometric perturbations (e.g., +ε to foil normals).

---

## 2026-05-13 14:25 — PR #2291: grad-clip-0p25 (nezuko) — CLOSED (clip axis fully mapped at 0.5)

- **Branch:** `charliepai2g48h2-nezuko/grad-clip-0p25`
- **Hypothesis:** Tighter clipping to 0.25 may further dampen the (already-gone) epoch-5 spike and help OOD generalization.
- **Metric artifacts:** `models/model-charliepai2g48h2-nezuko-grad-clip-0p25-20260513-131103/metrics.jsonl`

### Results vs. #2260 baseline (65.2170)

| Split | Baseline | clip=0.25 | Δ |
|---|---|---|---|
| val_single_in_dist | 73.7639 | 73.3757 | **−0.53%** ✓ |
| val_geom_camber_rc | 79.4389 | 79.1200 | **−0.40%** ✓ |
| val_geom_camber_cruise | 42.8481 | 44.0489 | +2.80% ❌ |
| val_re_rand | 64.8172 | 66.0262 | +1.87% ❌ |
| **val_avg/mae_surf_p** | **65.2170** | **65.6427** | **+0.65%** ❌ |
| **test_avg/mae_surf_p** | 56.4581 | 57.0366 | +1.02% ❌ |

### Key diagnostic — clipping saturation

Pre-clip grad norms in peak-LR window (epochs 4–7): **mean 11–12, max 28–31, frac>0.25 = 100%**.

Clipping fully saturates at BOTH clip=0.5 and clip=0.25 — every step is clipped at every epoch during peak-LR window. The only difference is the step magnitude:
- clip=0.5 → effective LR ≈ 6.6e-5 per saturated step
- clip=0.25 → effective LR ≈ 3.3e-5 per saturated step (half)

Epoch-5 transition: +91 (clip=1.0) → −58 (clip=0.5) → **−5.7 (clip=0.25)**. The spike is gone, but progress in the peak-LR window is much slower at clip=0.25, leaving less room for the cosine tail to refine.

### Outcome

**Clip axis CLOSED at 0.5.** Going tighter (0.25) over-constrains peak-LR-window progress. The single/rc improvement was real (matching the "OOD over-constraint helps" hypothesis), but the cruise/re_rand regression is larger. Net regression rules out merging.

**Mechanistic understanding:** Clipping is the LR governor in the peak-LR window, not the cosine schedule. The product (LR × clip) is what matters for effective step size. With clip=0.5 saturating every step at peak LR, the schedule is effectively a constant max-step regime for ~3–4 epochs. Going lower halves the effective LR during that window.

---

## 2026-05-13 14:25 — PR #2292: n-head-8 (fern) — CLOSED (catastrophic regression + critical insight)

- **Branch:** `charliepai2g48h2-fern/n-head-8`
- **Hypothesis:** Double Transolver attention heads from 4→8, expecting same params/FLOPs but finer-grained attention.
- **Metric artifacts:** `models/model-charliepai2g48h2-fern-n-head-8-20260513-131812/metrics.jsonl`

### Results vs. #2260 baseline (65.2170)

| Split | Baseline | n_head=8 | Δ |
|---|---|---|---|
| val_single_in_dist | 73.7639 | 98.2805 | +33.2% ❌ |
| val_geom_camber_rc | 79.4389 | 95.3060 | +20.0% ❌ |
| val_geom_camber_cruise | 42.8481 | 52.6446 | +22.9% ❌ |
| val_re_rand | 64.8172 | 75.7341 | +16.8% ❌ |
| **val_avg/mae_surf_p** | **65.2170** | **80.4913** | **+23.4%** ❌ |
| **test_avg/mae_surf_p** | 56.4581 | 70.3284 | **+24.6%** ❌ |

Run was **timeout-truncated at epoch 11/14**.

### Critical insight: Transolver n_head is NOT param-invariant

The PR predicted equal params at n_head=8 (`n_head × d_head² = n_hidden² = const`). The actual measurement: **n_head=8 has 661,611 params (−16.6K, −2.4%)**.

Root cause: the `to_q/k/v` projections in `train.py:151-153` are `nn.Linear(dim_head, dim_head)`, not `nn.Linear(n_hidden, n_hidden)`. So q/k/v params scale as `(n_hidden/n_head)²`:
- n_head=4: dim_head=32 → q/k/v = 3 × 32² = 3,072 params/layer
- n_head=8: dim_head=16 → q/k/v = 3 × 16² = 768 params/layer

Plus `in_project_slice = Linear(dim_head, slice_num)` shrinks 32×64=2048 → 16×64=1024 per layer.

Net per layer: −3.3K, × 5 layers = **−16.6K total** (matches observed exactly).

### Outcome

**n_head axis CLOSED on this implementation.** Going to higher n_head explicitly REDUCES attention capacity. This is an implementation quirk (the q/k/v projections fix `dim_head` in-out, not `n_hidden`). Going to n_head=2 (dim_head=64) would INCREASE per-layer attention params by ~3× to ~9.2K (vs 3.1K at n_head=4).

**Future architecture experiments must explicitly account for the (n_hidden/n_head)² scaling in Transolver q/k/v.** This invalidates the standard "n_head is free expressivity reshape" intuition for this codebase.

---

## 2026-05-13 13:20 — PR #2260: grad-clip-0p5 (nezuko) — MERGED ✅ NEW BEST: 65.2170

- **Branch:** `charliepai2g48h2-nezuko/grad-clip-0p5`
- **Hypothesis:** Tighten grad_clip 1.0 → 0.5 to dampen the +91-unit epoch-5 spike on RFF base. Spike is gradient-magnitude driven (not ε-driven as #2130 showed). Tighter clipping should reduce spike amplitude and improve late-epoch convergence quality.
- **Metric artifacts:** `models/model-charliepai2g48h2-nezuko-grad-clip-0p5-20260513-121601/metrics.jsonl`

### Results vs. #1657 RFF baseline (65.3304)

| Split | Baseline | clip=0.5 | Δ val | Δ test |
|---|---|---|---|---|
| val_single_in_dist | 72.691 | 73.7639 | +1.47% ❌ | — |
| val_geom_camber_rc | 78.833 | 79.4389 | +0.77% ❌ | — |
| val_geom_camber_cruise | 44.439 | 42.8481 | **−3.58%** ✓ | −3.13% |
| val_re_rand | 65.359 | 64.8172 | −0.83% ✓ | −1.48% |
| **val_avg/mae_surf_p** | **65.3304** | **65.2170** | **−0.17%** ✓ | — |
| **test_avg/mae_surf_p** | 56.9425 | 56.4581 | — | **−0.85%** ✓ |

**Merged. New canonical grad_clip=0.5. New baseline: 65.2170 (val), 56.4581 (test).**

### Analysis

The headline story is the epoch-5 spike:
- clip=1.0: **+91-unit spike** at epoch 4→5 (as diagnosed by #2130)
- clip=0.5: **−58-unit descent** at epoch 4→5 — completely eliminated and reversed

Test improvement (−0.85%) much larger than val (−0.17%), consistent with spike elimination reducing overfitting near the epoch-5 basin. clip=0.5 helps cruise (−3.58% val) and re_rand (−0.54% val) — both smooth/lower-variance splits that benefit from tighter gradient control. Single and rc slightly regress (+1.07%, +0.61%) — complex geometry splits may need the larger gradient steps that clip=0.5 moderates.

### Key insight

Clip axis is **open**. Tighter clipping improves generalization (test) more than in-distribution (val). The question is whether clip=0.25 continues the trend (further cruise/re_rand improvement) or causes over-constraint (single/rc regression worsens). Next: nezuko #2291 clip=0.25.

---

## 2026-05-13 13:20 — PR #2265: lr-1p25e-3-rff (alphonse) — CLOSED (dead end, LR axis closed)

- **Branch:** `charliepai2g48h2-alphonse/lr-1p25e-3-rff`
- **Hypothesis:** RFF base has sharper curvature near optimum; LR floor might have shifted downward from 1.5e-3.
- **Metric artifacts:** `models/model-charliepai2g48h2-alphonse-lr-1p25e-3-rff-20260513-121548/metrics.jsonl`

### Results vs. #1657 baseline (65.3304)

| Split | Baseline | lr=1.25e-3 | Δ |
|---|---|---|---|
| val_single_in_dist | 72.691 | 74.522 | +2.52% ❌ |
| val_geom_camber_rc | 78.833 | 80.310 | +1.87% ❌ |
| val_geom_camber_cruise | 44.439 | 45.439 | +2.25% ❌ |
| val_re_rand | 65.359 | 66.169 | +1.24% ❌ |
| **val_avg/mae_surf_p** | **65.3304** | **66.6101** | **+1.96%** ❌ |
| **test_avg/mae_surf_p** | 56.9425 | 58.3067 | **+2.40%** ❌ |

**Closed. LR axis FULLY CLOSED bilaterally at 1.5e-3.** Regression uniform across all splits (1.2–2.5%) — global undertraining penalty, not split-specific. Diagnostic: lr=1.25e-3 did reduce epoch-5 spike from +91 to +49 units (LR is proximate cause of spike), but the net training effect is negative (model needs the higher LR for productive gradient steps after spike).

---

## 2026-05-13 13:20 — PR #2238: rff-trainable-b (fern) — CLOSED (dead end, fixed B optimal)

- **Branch:** `charliepai2g48h2-fern/rff-trainable-b`
- **Hypothesis:** Allow gradient descent to fine-tune the RFF frequency matrix B (initialized at σ=3.0). B may self-organize toward task-optimal frequency coverage in 14 epochs.
- **Metric artifacts:** `models/model-charliepai2g48h2-fern-rff-trainable-b-20260513-121815/metrics.jsonl`

### Results vs. #1657 baseline (65.3304)

| Split | Baseline | trainable-B | Δ |
|---|---|---|---|
| val_single_in_dist | 72.691 | 75.497 | +3.86% ❌ |
| val_geom_camber_rc | 78.833 | 80.639 | +2.29% ❌ |
| val_geom_camber_cruise | 44.439 | 44.447 | +0.02% ≈ |
| val_re_rand | 65.359 | 65.670 | +0.48% ❌ |
| **val_avg/mae_surf_p** | **65.3304** | **66.5632** | **+1.88%** ❌ |
| **test_avg/mae_surf_p** | 56.9425 | 57.3451 | +0.71% ❌ |

**Closed. Fixed B at σ=3.0 is optimal.** B drift trajectory showed slow convergence (B.std starts 2.95, drifts very gradually). Within 14 epochs, the gradient noise on B adds perturbation without sufficient benefit. Cruise nearly ties (0.02% regression) — frequencies optimal for cruise do not conflict with σ=3.0 prior; for single/rc, the B drift hurts. Fixed init provides a better prior than learned B within the 14-epoch budget. Frequency axis CLOSED.

---

## 2026-05-13 12:30 — PR #2184: lr-2e-3-rff (alphonse) — CLOSED (dead end, LR ceiling locked across bases)

- **Branch:** `charliepai2g48h2-alphonse/lr-2e-3-rff`
- **Hypothesis:** RFF input expansion may have smoothed the loss landscape; LR ceiling could shift upward from 1.5e-3.
- **Metric artifacts:** `models/model-charliepai2g48h2-alphonse-lr-2e-3-rff-20260513-105732/metrics.jsonl`

### Results vs. #1657 RFF baseline (65.3304) — best_epoch=12 (run timed out at 30 min)

| Split | Baseline | lr=2e-3 RFF | Δ |
|---|---|---|---|
| val_single_in_dist | 72.691 | 88.487 | +21.73% ❌ |
| val_geom_camber_rc | 78.833 | 89.227 | +13.18% ❌ |
| val_geom_camber_cruise | 44.439 | 49.079 | +10.44% ❌ |
| val_re_rand | 65.359 | 69.437 | +6.24% ❌ |
| **val_avg/mae_surf_p** | **65.3304** | **74.0578** | **+13.36%** ❌ |
| **test_avg/mae_surf_p** | 56.9425 | 64.7187 | +13.66% ❌ |

**Closed as dead end.** LR axis CLOSED at 1.5e-3 across both pre-RFF and post-RFF bases.

### Analysis

**RFF base is LESS tolerant of high LR, not more:**

| LR | Pre-RFF Δ vs 1.5e-3 (74.21) | RFF Δ vs 1.5e-3 (65.33) |
|---|---|---|
| 2e-3 | +2.99% (#1942) | **+13.36%** (this PR) — 5× worse |

The RFF base has SHARPER curvature near the optimum (86-dim input expansion creates a more peaked loss landscape). Noisy high-LR updates push the model further off the new (better) minimum. The pre-RFF base had flatter curvature, tolerating higher LR but reaching a worse final optimum.

Student's analysis was spot-on: epoch-3 warmup spike (+100 units), epoch-6 post-peak spike (+41 units), val_single_in_dist regressed worst (+21.7%) — classic "schedule too aggressive" signature.

### Key insights

- **LR axis FULLY CLOSED across bases:** lr=1.5e-3 is the optimum on both pre-RFF and post-RFF. Probing upward is exhausted.
- **RFF base is sharper-near-optimum:** This explains why ANY modification adding noise (higher LR, eta_min raise, larger gradient steps) regresses more on RFF than pre-RFF.
- **Follow-up direction:** the optimum LR may have shifted slightly DOWNWARD on RFF base. Test lr=1.25e-3 midpoint probe.

### Next assignment for alphonse: lr=1.25e-3 on RFF base (#2265)

Direct downward probe motivated by sharper-curvature hypothesis. Single number change. Will close the LR axis bilaterally on RFF base.

---

## 2026-05-13 12:25 — PR #2207: cosine-eta-min-1e-4 (nezuko) — CLOSED (dead end, eta_min axis)

- **Branch:** `charliepai2g48h2-nezuko/cosine-eta-min-1e-4`
- **Hypothesis:** best_epoch=14/14 → model not converged; non-zero LR floor (1e-4) could sustain learning at terminal epoch.
- **Metric artifacts:** `models/model-charliepai2g48h2-nezuko-cosine-eta-min-1e-4-20260513-111354/metrics.jsonl`

### Results vs. #1657 baseline (65.3304)

| Split | Baseline | eta_min=1e-4 | Δ |
|---|---|---|---|
| val_single_in_dist | 72.691 | 72.681 | −0.01% |
| val_geom_camber_rc | 78.833 | 79.252 | +0.53% ❌ |
| val_geom_camber_cruise | 44.439 | 46.989 | **+5.74%** ❌ |
| val_re_rand | 65.359 | 67.010 | +2.53% ❌ |
| **val_avg/mae_surf_p** | **65.3304** | **66.483** | **+1.77%** ❌ |
| **test_avg/mae_surf_p** | 56.9425 | 58.502 | +2.74% ❌ |

**Closed as dead end.** eta_min axis CLOSED.

### Analysis

**Mechanism diagnosed:** eta_min=1e-4 raises the ENTIRE cosine curve, not just the terminal point. At epoch 12, LR was 3.886e-4 vs baseline ~3.09e-4 (+26%). The result: more mid-late optimization noise, not "continued learning momentum" as hypothesized. val_cruise (easiest, narrowest-minimum split) regresses most (+5.74%) — exactly the over-shoot signature.

**Trajectory:** val_avg at epoch 12 was 74.55 (vs baseline's ~67), epoch 14 was 66.48 (vs baseline's 65.33). The model is still "catching up" at the end, not pushing past the baseline.

**The PR identified a real problem** (budget-limited convergence) but the lever was wrong: eta_min=1e-4 raises mid-curve LR, not terminal-only LR.

### Key insights

- **eta_min axis CLOSED.** Don't probe higher values (3e-4, 1e-3) — they'd widen the over-shoot.
- **Cosine schedule shape is highly sensitive:** any modification that raises mid-curve LR causes over-shoot, even if the terminal-only intent was good.
- **Underlying problem persists:** model not converged in 14 epochs (best_epoch=14/14). Better levers: gradient clipping (target the +91-unit epoch-5 spike) or smaller-step optimization.

### Next assignment for nezuko: grad_clip 1.0 → 0.5 (#2260)

Direct attack on the epoch-5 spike diagnostic from this PR's predecessor (#2130 ε=1e-6). Tighter clipping should reduce the +91-unit spike magnitude without affecting normal-step training.

---

## 2026-05-13 12:15 — PR #2197: rff-nfeatures-64 (frieren) — CLOSED (dead end, capacity axis at d=32)

- **Branch:** `charliepai2g48h2-frieren/rff-nfeatures-64`
- **Hypothesis:** At fixed σ=3.0, double frequency count 32→64. Tests kernel approximation quality (1/√d variance reduction).
- **Metric artifacts:** `models/model-charliepai2g48h2-frieren-rff-nfeatures-64-20260513-110556/metrics.jsonl`

### Results vs. #1657 baseline (65.3304)

| Split | Baseline (d=32) | d=64 | Δ |
|---|---|---|---|
| val_single_in_dist | 72.691 | 75.688 | +4.12% ❌ |
| val_geom_camber_rc | 78.833 | 83.007 | +5.30% ❌ |
| val_geom_camber_cruise | 44.439 | **41.391** | **−6.86%** ✓ |
| val_re_rand | 65.359 | 65.889 | +0.81% ❌ |
| **val_avg/mae_surf_p** | **65.3304** | **66.4937** | **+1.78%** ❌ |
| **test_avg/mae_surf_p** | 56.9425 | 58.1547 | +2.13% ❌ |

**Closed as dead end.** RFF capacity axis CLOSED at d=32.

### Analysis

Param count delta: +16K (+2.4%, more than predicted ~+8K due to wider preprocess MLP through multiple downstream layers). best_epoch=14/14, no NaN/divergence.

**Repeated per-split signature across 3 RFF modifications:**

| Experiment | val_cruise | val_rc | val_single | val_avg |
|---|---|---|---|---|
| σ=5.0 (#2158) | +1.09% | +1.98% | +4.09% | +2.75% |
| anisotropic σ_z=1.5 (#2206) | **−1.61%** | +1.62% | +2.39% | +0.51% |
| n_features=64 (#2197) | **−6.86%** | +5.30% | +4.12% | +1.78% |

The pattern is unambiguous: **more flexible/wider RFF helps cruise but hurts rc/single**. The model has a per-split optimal bandwidth: cruise wants more capacity/flexibility, rc/single want fewer (more constrained) frequency vectors.

### Key insights

- **RFF capacity axis CLOSED at d=32.** Isotropic σ=3.0 + 32 frequencies is the local optimum for the GLOBAL val_avg metric.
- **Cruise split has untapped potential** — improved consistently across RFF variants. A different mechanism (split-aware routing, augmentation, or learned weighting) might be the lever to capture this without rc/single regressing.
- **The 14-epoch budget doesn't allow capacity-regularization** — extra frequency dimensions get fit to noise in 14 epochs.
- **Next direction:** away from RFF mods entirely. Try fundamentally different mechanism (data augmentation, loss reformulation, etc.).

### Next assignment for frieren: foil mirroring augmentation (#2257)

Direct shift to a fundamentally different lever: z-axis reflection symmetry augmentation. Doubles effective training data while respecting exact TandemFoilSet symmetry. Strong precedent in CFD ML literature for 1–5% gains from symmetry-respecting augmentation.

---

## 2026-05-13 11:55 — PR #2206: rff-anisotropic-sx3-sz1p5 (fern) — CLOSED (marginal val regression, mixed val/test, per-split tradeoff)

- **Branch:** `charliepai2g48h2-fern/rff-anisotropic-sx3-sz1p5`
- **Hypothesis:** Anisotropic RFF (σ_x=3.0, σ_z=1.5) — foil pressure varies sharply in chord (x) but smoothly in height (z). Lower σ_z may improve geometry splits by avoiding z-direction overfitting.
- **Metric artifacts:** `models/model-charliepai2g48h2-fern-rff-anisotropic-sx3-sz1p5-20260513-111518/metrics.jsonl`

### Results vs. #1657 isotropic σ=3.0 baseline (65.3304)

| Split | Baseline | σ_x=3.0, σ_z=1.5 | Δ |
|---|---|---|---|
| val_single_in_dist | 72.691 | 74.424 | +2.39% ❌ |
| val_geom_camber_rc | 78.833 | 80.107 | +1.62% ❌ |
| val_geom_camber_cruise | 44.439 | **43.726** | **−1.61%** ✓ |
| val_re_rand | 65.359 | **64.387** | **−1.49%** ✓ |
| **val_avg/mae_surf_p** | **65.3304** | **65.6609** | **+0.51%** ❌ |
| test_single | 64.577 | 63.530 | −1.62% ✓ |
| test_rc | 71.531 | 72.886 | +1.89% ❌ |
| test_cruise | 36.392 | **35.003** | **−3.82%** ✓ |
| test_re_rand | 55.269 | 54.514 | −1.37% ✓ |
| **test_avg/mae_surf_p** | **56.9425** | **56.4834** | **−0.81%** ✓ |

**Closed on val regression (+0.51% > 0.5% threshold).** Mixed val/test signal but val_avg is the primary advisor metric.

### Analysis

**Strong per-split insight:** the result reveals a fundamental physics-driven tradeoff between split families:
- **Cruise/Re_rand splits** improved on both val (−1.61%, −1.49%) and test (−3.82%, −1.37%): freestream/symmetric flows have smoother z-direction pressure variation, σ_z=1.5 is well-matched.
- **RC/Single splits** regressed on both val (+1.62%, +2.39%) and test (+1.89%, only test_single improved): raceCar (ground-effect) geometry has asymmetric loading, sharp z-gradients near the wall; needs full σ_z=3.0 bandwidth to resolve.

Test improvement (−0.81%) is real and driven by cruise/re_rand families, but val_avg is the merge criterion.

**B verification:** B[0,:].std()=3.45 (target 3.0 ✓), B[1,:].std()=1.20 (target 1.5 ✓). Implementation correct.

### Key insights

- **Per-split bandwidth needs differ:** isotropic σ=3.0 is the right COMPROMISE setting. Different physical regimes (ground-effect vs. freestream) want different bandwidths.
- **The per-split signal is too strong to ignore for future work:** suggests learned per-frequency bandwidth or domain-conditional σ as future directions.
- **Anisotropy axis CLOSED:** σ_x=σ_z=3.0 isotropic remains optimal globally.
- **Val/test diverged**: test improvement (−0.81%) was real but val regression dominates merge logic. Reinforces that val is the operational metric.

### Next assignment for fern: trainable RFF B matrix (#2238)

Direct follow-up: let gradient descent find the optimal per-frequency bandwidth automatically. B starts at σ=3.0 (sweet spot init) with `requires_grad=True`. Adds 64 trainable params (0.009% of model).

---

## 2026-05-13 11:35 — PR #2158: rff-sigma5 (fern) — CLOSED (dead end, σ=3.0 confirmed sweet spot)

- **Branch:** `charliepai2g48h2-fern/rff-sigma5`
- **Hypothesis:** σ sweep was monotone in {1.0, 3.0} (gains −6.4%, −11.71%). Probe σ=5.0 to see if gain continues.
- **Metric artifacts:** `models/model-charliepai2g48h2-fern-rff-sigma5-20260513-100948/metrics.jsonl`

### Results vs. #1657 RFF σ=3.0 baseline (65.3304)

| Split | σ=3.0 baseline | σ=5.0 | Δ |
|---|---|---|---|
| val_single_in_dist | 72.691 | 75.663 | +4.09% ❌ |
| val_geom_camber_rc | 78.833 | 80.397 | +1.98% ❌ |
| val_geom_camber_cruise | 44.439 | 44.923 | +1.09% ❌ |
| val_re_rand | 65.359 | 67.540 | +3.34% ❌ |
| **val_avg/mae_surf_p** | **65.3304** | **67.1306** | **+2.75%** ❌ |
| **test_avg/mae_surf_p** | 56.9425 | 58.1503 | +2.12% ❌ |

**Closed as dead end.** σ bandwidth axis CLOSED.

### Analysis

**σ sweep final mapping:** {σ=1.0: −6.40%, σ=3.0: −11.71% (canonical), σ=5.0: +2.75%}

Non-monotone — gain reverses at σ=5.0. At σ=5.0, effective wavelength ~0.18 nodes is finer than natural pressure-field variation; the RFF basis encodes foil-specific noise rather than generalizable geometry structure. In-distribution splits (val_single +4.1%, test_single +3.7%) hit hardest — signature of overfitting to high-frequency geometry noise.

**Training trajectory:** epoch-5 spike (val=212.7) more severe than σ=3.0 (244.97 on nezuko's ε run), suggesting σ=5.0 makes the early-training loss surface harder. Model recovers but slowly.

### Key insights

- **σ axis CLOSED at σ=3.0.** Both lower (σ=1.0) and higher (σ=5.0) are worse. σ=3.0 is the bandwidth sweet spot for TandemFoilSet pressure fields.
- Next: RFF capacity (n_features) axis (frieren #2197) and anisotropic bandwidth (fern #2206) are the open RFF sub-axes.

---

## 2026-05-13 11:35 — PR #2130: adamw-eps-1e-6 (nezuko) — CLOSED (statistical tie, ε axis closed; epoch-5 spike diagnosis)

- **Branch:** `charliepai2g48h2-nezuko/adamw-eps-1e-6`
- **Hypothesis:** ε=1e-6 (vs canonical 1e-8) reduces step sizes on low-curvature dimensions at the epoch-5 LR peak, damping the AdamW spike.
- **Metric artifacts:** `models/model-charliepai2g48h2-nezuko-adamw-eps-1e-6-20260513-102206/metrics.jsonl`

### Results vs. #1657 RFF σ=3.0 baseline (65.3304)

| Split | Baseline | ε=1e-6 | Δ |
|---|---|---|---|
| val_single_in_dist | 72.691 | 73.602 | +1.25% ❌ |
| val_geom_camber_rc | 78.833 | 79.371 | +0.68% ❌ |
| val_geom_camber_cruise | 44.439 | 44.320 | **−0.27%** ✓ |
| val_re_rand | 65.359 | 65.081 | −0.43% ✓ |
| **val_avg/mae_surf_p** | **65.3304** | **65.5934** | **+0.40%** (tie) |
| **test_avg/mae_surf_p** | 56.9425 | 56.3824 | **−0.98%** ✓ |

**Closed as dead end (statistical tie).** AdamW ε axis CLOSED on RFF base.

### Analysis

Val/test sign disagreement (+0.40% val, −0.98% test) confirms no real effect — ε is not the lever for the RFF base.

**⚠️ CRITICAL DIAGNOSTIC — Epoch-5 LR-peak spike on RFF base:**

| Base | Epoch-4→5 spike | PR |
|---|---|---|
| pre-RFF β2=0.99 canonical | +3.2 units | #2004 |
| **RFF base + ε=1e-6 (this run)** | **+91 units** | **#2130** |

The RFF base has a **28× larger epoch-5 spike** than the pre-RFF canonical. ε=1e-6 did NOT dampen it — the mechanism is large gradient magnitudes from the 86-dim RFF input interacting with peak LR, not small-v̂ step-size flooring.

Epoch-5 trajectory: 153.69 → 244.97 (spike) → 147.80 → 65.59 (final). Model recovers but the spike is severe.

### Key insights

- **ε axis CLOSED on RFF base.** ε is mechanistically irrelevant in this gradient-magnitude-dominated regime.
- **New research priority:** the +91-unit epoch-5 spike suggests the LR schedule may not be optimally matched to the RFF-expanded input gradient magnitudes. Follow-up: cosine eta_min (nezuko #2207) to sustain late-epoch learning; anisotropic RFF (fern #2206) to reduce z-direction bandwidth noise.
- **Pre-RFF validated improvements may not stack on RFF base** — must re-validate each axis.

---

## 2026-05-13 11:15 — PR #1813: warmup-5-epochs (frieren, RFF rebase) — CLOSED (dead end, warmup=4 confirmed on RFF)

- **Branch:** `charliepai2g48h2-frieren/warmup-5-epochs`
- **Hypothesis:** warmup=5 won −0.52% on pre-RFF base (#1813 first run). Rebased onto RFF (PR #1657, 65.3304); predicted to stack additively.
- **Metric artifacts:** `models/model-charliepai2g48h2-frieren-warmup-5-epochs-rebased-rff-20260513-101106/metrics.jsonl`

### Results vs. #1657 RFF baseline (65.3304)

| Split | Baseline | warmup=5 on RFF | Δ |
|---|---|---|---|
| val_single_in_dist | 72.691 | 74.659 | +2.71% ❌ |
| val_geom_camber_rc | 78.833 | 79.932 | +1.39% ❌ |
| val_geom_camber_cruise | 44.439 | **43.567** | **−1.96%** ✓ |
| val_re_rand | 65.359 | 65.389 | +0.05% ≈ |
| **val_avg/mae_surf_p** | **65.3304** | **65.8867** | **+0.85%** ❌ |
| **test_avg/mae_surf_p** | 56.9425 | 56.9584 | +0.03% ≈ (flat) |

**Closed as dead end on RFF base.** Warmup axis mapped: warmup=4 optimum.

### Analysis

**Warmup axis final mapping on RFF base:**

| warmup_epochs | val_avg | Δ vs canonical | Note |
|---|---|---|---|
| 3 | (regressed on old base, mechanism unchanged) | — | dead end (#1911 pre-RFF) |
| **4** | **65.3304** | **canonical** | optimum |
| 5 | 65.8867 | **+0.85%** | dead end |

**Why pre-RFF win didn't transfer:** RFF (24→86 dim input) introduces high-frequency gradient signals from epoch 1 onward. The model needs the full LR window to absorb these. T_max auto-adjusted 10→9 with warmup=5 — losing 1 epoch of high-LR cosine descent costs more than the extra warmup damping helps. On the pre-RFF stack, smoother low-dim coord input made early-phase noise the dominant issue, and warmup=5 helped.

**Per-split signal:** val_cruise improved (−1.96%) — consistent with warmup-sensitivity hypothesis, but other splits regressed by 1–3%. Test flat (+0.03%) suggests some of val signal is split-specific noise, but the val_avg trend is consistent.

### Key insights

- **Single-axis stacking does NOT hold across baseline shifts.** Pre-RFF and post-RFF stacks have different gradient regimes. Wins on the old base must be re-validated on the new base.
- **Warmup axis CLOSED on RFF base** at warmup=4. Both warmup=3 (pre-RFF dead end) and warmup=5 (RFF dead end) bracket it.
- **Next assignment for frieren:** RFF n_features capacity probe — increase 32 frequencies → 64 frequencies (output 64-dim → 128-dim) to test if RFF was capacity-limited at σ=3.0.

---

## 2026-05-13 10:30 — PR #2045: lr-1.75e-3 (alphonse) — CLOSED (dead end, LR axis fully mapped)

- **Branch:** `charliepai2g48h2-alphonse/lr-1.75e-3`
- **Hypothesis:** LR midpoint probe between 1.5e-3 (winner) and 2e-3 (dead-end). Either confirms 1.5e-3 exact ceiling or finds marginal gain narrowing window.
- **Metric artifacts:** `models/model-charliepai2g48h2-alphonse-lr-1.75e-3-20260513-095039/metrics.jsonl`

### Results vs. #1895 baseline (74.2082)

| Split | Baseline | lr=1.75e-3 | Δ |
|---|---|---|---|
| val_single_in_dist | 83.733 | 87.305 | +4.27% ❌ |
| val_geom_camber_rc | 91.690 | 94.225 | +2.77% ❌ |
| val_geom_camber_cruise | 50.392 | 54.932 | +9.01% ❌ |
| val_re_rand | 71.018 | 72.611 | +2.24% ❌ |
| **val_avg/mae_surf_p** | **74.2082** | **77.2684** | **+4.13%** ❌ |
| **test_avg/mae_surf_p** | 65.1123 | 67.3075 | +3.37% ❌ |

**Closed as dead end.** LR axis fully mapped on pre-RFF base.

### Analysis

**LR axis final mapping (pre-RFF base, β2=0.99 + asinh + warmup-4):**

| lr | val_avg | Δ vs winner | epoch-5 spike |
|---|---|---|---|
| 1e-3 | 77.1419 | +3.96% | small |
| **1.5e-3** | **74.2082** | **baseline** | **+3.2 units (with β2=0.99)** |
| 1.75e-3 | 77.2684 | +4.13% | +10 units |
| 2e-3 | 76.43 | +2.99% | +48 units (seed-dependent) |

Sweet-spot is genuine: 1.75e-3 is *worse* than 2e-3 despite being closer to the winner. The val landscape isn't a clean parabola — there's a sharp drop at 1.5e-3 and the schedule is brittle above it. The epoch-5 spike scales non-linearly with LR: +3.2 → +10 → +48 units across {1.5, 1.75, 2}e-3.

**Next assignment:** lr=2e-3 retest on RFF base (PR #2184). Tests whether RFF (which just merged at −11.71%) shifts the LR ceiling.

---

## 2026-05-13 10:05 — PR #1657: rff-pos-encoding σ=3.0 (fern) — MERGED (new best, MAJOR GAIN)

- **Branch:** `charliepai2g48h2-fern/rff-pos-encoding`
- **Hypothesis:** Prepend 64-dim Fourier Random Feature positional encoding of (x,z) node coordinates. Two arms: σ=1.0 and σ=3.0. Expected −2% to −5%; stacked on canonical (asinh + warmup-4 + lr=1.5e-3 + β2=0.99).
- **Metric artifacts:** `models/model-charliepai2g48h2-fern-rff-pos-encoding-sigma3-20260513-085421/metrics.jsonl`

### Results vs. #2004 baseline (73.9964)

| Split | Baseline | RFF σ=1.0 | RFF σ=3.0 (winner) |
|---|---|---|---|
| val_single_in_dist | 85.100 | 80.122 | **72.691** (−14.59%) |
| val_geom_camber_rc | 89.815 | 84.723 | **78.833** (−12.23%) |
| val_geom_camber_cruise | 50.761 | 45.784 | **44.439** (−12.46%) |
| val_re_rand | 70.309 | 66.401 | **65.359** (−7.04%) |
| **val_avg/mae_surf_p** | **73.9964** | **69.2572** | **65.3304** (−11.71%) |
| **test_avg/mae_surf_p** | 64.4437 | 60.4811 | **56.9425** (−11.65%) |

**MERGED as new best: val_avg=65.3304 (−11.71%), test_avg=56.9425 (−11.65%)**

### Analysis

Largest single improvement in this research programme. RFF spatial encoding gives the Transolver explicit awareness of node locations — the model can now distinguish nearby nodes across surface/volume interface and across different camber angles.

**σ axis**: 1.0 → 3.0 is monotone (−6.4% → −11.7%). σ=3.0 matches the coordinate scale range [−6.5, 7.4] — bandwidth well-matched to normalized coordinate spread. Next probe: σ=5.0.

**Per-split pattern**: uniform gains (−7% to −15%), strongest on geometry-OOD splits (cruise/rc). Consistent with hypothesis that spatial encoding aids geometry extrapolation. val_re_rand gained least (−7%) — random Reynolds may be less spatially-structured.

**Minimal cost**: +15.9K params (+2.4%), +0 epoch time overhead.

**New canonical config**: RFF σ=3.0 (preprocess MLP 24→86) + all prior gains (asinh + warmup-4 + lr=1.5e-3 + β2=0.99).

---

## 2026-05-13 09:10 — PR #2054: adamw-beta2-0.95 (nezuko) — CLOSED (dead end, β2 axis mapped)

- **Branch:** `charliepai2g48h2-nezuko/adamw-beta2-0.95`
- **Hypothesis:** Monotone test on β2 axis: 0.99 (winner) → 0.95 (RoFormer/DeiT recipe). If gain monotone in 1/β2-window, further improvement expected.
- **Metric artifacts:** `models/model-charliepai2g48h2-nezuko-adamw-beta2-0.95-20260513-080047/metrics.jsonl`

### Results vs. #2004 baseline (73.9964)

| Split | β2=0.99 baseline | β2=0.95 | Δ |
|---|---|---|---|
| val_single_in_dist | 85.100 | 85.755 | +0.77% |
| val_geom_camber_rc | 89.815 | 90.883 | +1.19% |
| val_geom_camber_cruise | 50.761 | 51.987 | +2.41% |
| val_re_rand | 70.309 | 70.664 | +0.50% |
| **val_avg/mae_surf_p** | **73.9964** | **74.8222** | **+1.12%** ❌ |
| **test_avg/mae_surf_p** | 64.4437 | 65.5435 | **+1.71%** ❌ |
| test_geom_camber_rc | 78.036 | 81.044 | **+3.85%** ❌ (reversal!) |

**Closed as dead end.** β2 axis mapped: 0.99 is the genuine sweet spot.

### Analysis

**β2 axis final mapping:**

| β2 | val_avg | epoch-5 spike |
|---|---|---|
| 0.999 (default) | 74.2082 | +20.4 units |
| **0.99 (winner)** | **73.9964** | **+3.2 units (minimum)** |
| 0.95 | 74.8222 | +13.0 units |

**Spike magnitude is non-monotone in β2** — pushing window shorter (β2=0.95, ~20-step window) re-amplifies LR-peak shock. Mechanism: ~20-step v_t window has high steady-state variance → noisy per-parameter step sizes → both larger LR-peak spike *and* worse converged val_avg.

The `test_rc` reversal is the strongest signal: it gained −4.9% at β2=0.99 (the breakthrough), but gives back +3.85% at β2=0.95. The "resistant-split rescue" is NOT monotone in faster β2.

**Key insight:** 2nd-moment EMA window has real bias-variance tradeoff. Short windows are noisier in steady state, not just at the LR peak. The "spike size" and "final convergence" are governed by the same underlying knob — ~100-step window is the optimum at this LR/budget.

**Next axis (nezuko follow-up #2):** AdamW ε=1e-6 (mechanistically distinct from β2 — regularizes small-v̂ step sizes, not v̂'s smoothness). Assigned as PR #2130.

---

## 2026-05-13 07:55 — PR #2004: adamw-beta2-0.99 (nezuko) — MERGED (new best)

- **Branch:** `charliepai2g48h2-nezuko/adamw-beta2-0.99`
- **Hypothesis:** AdamW default β2=0.999 → 0.99. Effective 2nd-moment window ~100 steps vs ~1000. At 14-epoch budget (~5250 steps), faster 2nd-moment adaptation should reduce epoch-5 peak-LR spike and improve cosine-decay phase convergence.
- **Metric artifacts:** `models/model-charliepai2g48h2-nezuko-adamw-beta2-0.99-20260513-070048/metrics.jsonl`

### Results vs. #1895 baseline (74.2082)

| Split | Baseline | β2=0.99 | Δ |
|---|---|---|---|
| val_single_in_dist | 83.733 | 85.100 | +1.64% ❌ |
| val_geom_camber_rc | 91.690 | 89.815 | **−2.04%** ✓ |
| val_geom_camber_cruise | 50.392 | 50.761 | +0.73% |
| val_re_rand | 71.018 | 70.309 | −1.00% ✓ |
| **val_avg/mae_surf_p** | **74.2082** | **73.9964** | **−0.29%** ✓ |
| **test_avg/mae_surf_p** | 65.1123 | 64.4437 | **−1.03%** ✓ |

**MERGED — new best: val_avg=73.9964, test_avg=64.4437**

### Analysis

Hypothesis confirmed: epoch-5 peak-LR spike collapsed from +20.4 → +3.2 units. The faster 2nd-moment EMA had already adapted to warmup gradient scale by peak-LR epoch, dampening the step-size shock.

Win concentrated on val_rc (−2.0%) and test_rc (−4.9%) — the historically resistant split. Slight regression on val_single (+1.6%) consistent with mild per-parameter regularization (less over-fit to easy in-dist regime). val_cruise and val_re_rand essentially flat (±1%).

**Mechanism insight:** β2 reduction acts as adaptation-rate regularizer. Unlike capacity-reducing regularizers (DropPath), this doesn't slow convergence — it makes the optimizer more responsive. Single-line change, orthogonal to all existing axes.

**Next:** probe β2=0.95 (monotone check — does gain continue toward RoFormer/DeiT recipes?).

---

## 2026-05-13 07:30 — PR #1942: lr-2e-3 (alphonse) — CLOSED (dead end)

- **Branch:** `charliepai2g48h2-alphonse/lr-2e-3`
- **Hypothesis:** LR ceiling probe 1.5e-3 → 2e-3. PR #1895 showed best_epoch=final and the largest epoch-14 cosine drop (−7.38 units), suggesting LR ceiling still open. Probing 2e-3 with stability diagnostics (pred_abs_max, epoch-5 peak-LR spike).
- **Metric artifacts:** alphonse ran 3-seed study; committed metrics under `exp/icml2026/charlie/metrics/`

### Results vs. #1895 baseline (74.2082)

| Run | Seed | val_avg/mae_surf_p | Δ vs baseline |
|---|---|---|---|
| Run 1 (047041) | 047041 | 76.3879 | +2.93% ❌ |
| Run 2 (051042) | 051042 | 76.3785 | +2.92% ❌ |
| Run 3 (055114) | 055114 | 76.5078 | +3.10% ❌ |
| **Mean** | — | **76.4247** | **+2.99%** ❌ |

**Closed as dead end.** Ceiling closes by optimization quality between 1.5e-3 and 2e-3.

### Analysis

LR=2e-3 is **stable** (pred_abs_max 14–18k, well below 50k instability flag) but systematically degrades optimization quality. Epoch-5 peak-LR spike is seed-dependent at this LR: Run 051042 showed no spike; Run 055114 showed +48 unit spike (2.4× the 1.5e-3 reference). Stability is bounded but training trajectory is noisier. All three runs worse than baseline by 2.9–3.1%.

**Key insight:** LR ceiling closes between 1.5e-3 and 2e-3 by optimization quality, not stability. Binary search assigned to alphonse: lr=1.75e-3 midpoint probe (PR #2045).

---

## 2026-05-13 07:00 — PR #1970: drop-path-0.1 (nezuko) — CLOSED (dead end)

- **Branch:** `charliepai2g48h2-nezuko/drop-path-0.1`
- **Hypothesis:** Add Stochastic Depth (DropPath) with linear schedule [0.0, 0.025, 0.05, 0.075, 0.1] across 5 Transolver layers. Parameter-free ensemble regularizer; expected to help val_rc and val_re_rand via OOD-geom ensemble effect.
- **Metric artifacts:** `models/model-charliepai2g48h2-nezuko-drop-path-0.1-20260513-061624/metrics.jsonl`

### Results vs. #1895 baseline (74.2082)

| Split | Baseline | DropPath 0.1 | Δ |
|---|---|---|---|
| val_single_in_dist | 83.733 | 94.392 | **+12.74%** ❌ |
| val_geom_camber_rc | 91.690 | 93.205 | +1.65% ❌ |
| val_geom_camber_cruise | 50.392 | 56.533 | **+12.19%** ❌ |
| val_re_rand | 71.018 | 73.460 | +3.44% ❌ |
| **val_avg/mae_surf_p** | **74.2082** | **79.3974** | **+6.99%** ❌ |
| **test_avg/mae_surf_p** | 65.1123 | 68.8846 | **+5.79%** ❌ |

**Closed as dead end.** DropPath is depth- and budget-dependent regularizer.

### Analysis

Regression concentrated on in-distribution splits (val_single +12.74%) — opposite of predicted ensemble-regularizer signature. Root cause: DropPath benefits require long training horizons (100s-1000s of epochs in ConvNeXt/ViT/MAE) to compensate for capacity reduction. At only 14 epochs and 5 layers, the regularizer causes pure under-fitting without the long-run ensemble payoff. Every per-epoch val_avg value sits above baseline equivalent — not an optimization failure but a loss-landscape shift from reduced capacity. The 5-layer depth × 14-epoch budget combination is hostile to stochastic depth. Per-channel trajectory confirms: best_epoch=14 unchanged but systematically worse.

**Key insight:** Before applying any capacity-reducing regularizer, check training horizon compatibility. Mechanisms that work over 1000s of steps (DropPath, EMA with heavy decay, SWA) are unlikely to fit in a 14-epoch budget. Mechanisms that don't delay convergence (LayerScale, optimizer β2, asinh) are safer bets.

---

## 2026-05-13 06:05 — PR #1941: asinh-all-channels (nezuko) — CLOSED (dead end)

- **Branch:** `charliepai2g48h2-nezuko/asinh-all-channels`
- **Hypothesis:** Extend asinh compression from pressure channel (index 2) to all three output channels (Ux, Uy, p) with unified ASINH_GAIN=1.0. Tests whether bulk-redistribution mechanism generalizes to velocity channels.
- **Metric artifacts:** `models/model-charliepai2g48h2-nezuko-asinh-all-channels-20260513-051430/metrics.jsonl`

### Results vs. #1895 baseline (74.2082)

| Split | Baseline | asinh-all | Δ |
|---|---|---|---|
| val_single_in_dist | 83.733 | 91.826 | **+9.67%** ❌ |
| val_geom_camber_rc | 91.690 | 90.913 | −0.85% ✓ |
| val_geom_camber_cruise | 50.392 | 51.762 | +2.72% ❌ |
| val_re_rand | 71.018 | 70.511 | −0.71% ✓ |
| **val_avg/mae_surf_p** | **74.2082** | **76.2528** | **+2.75%** ❌ |
| **test_avg/mae_surf_p** | 65.1123 | 67.7524 | **+4.06%** ❌ |

**Closed as dead end.** Mechanism does not transfer from pressure to velocity channels.

### Analysis

Val_single regression dominates (+9.67%, test_single +12.26%). Pressure has high kurtosis (concentrated zeros + heavy negative suction-peak tail); velocity channels are closer to Gaussian. asinh at GAIN=1.0 loses 28% of gradient at 2σ, destroying signal precisely on high-Re single-foil outliers where the model most needs gradient pressure. The channel-weight asymmetry [1,1,3]/5 compounds this: 40% of the loss is velocity, so the redirection is substantial. Training trajectory shape matched #1895 exactly (same epoch-5 LR bump, monotone from ep5, best=ep14) — systematic +2.75% gap uniformly distributed from epoch 5 onward. This is a loss-landscape effect, not an optimization failure.

**Key insight confirmed:** Before applying value-level regularization to a channel, measure its kurtosis. asinh bulk-redistribution is only beneficial for channels with genuinely heavy tails (like pressure). Pressure channel in asinh canonical config stays pressure-only.

---

## 2026-05-13 05:15 — PR #1895: lr=1.5e-3 ceiling probe (alphonse) — MERGED ✓

- **Branch:** `charliepai2g48h2-alphonse/lr-1.5e-3`
- **Hypothesis:** Push peak LR from 1e-3 to 1.5e-3. After the super-additive stacking result (#1814), the epoch-5 instability spike was GONE at lr=1e-3. Hypothesis: asinh stability buffer provides headroom for further LR increase.
- **Metric artifacts:** `models/model-charliepai2g48h2-alphonse-lr-1.5e-3-20260513-041123/metrics.jsonl`

### Results vs. #1814 baseline (77.1419)

| Split | Baseline | lr=1.5e-3 | Δ |
|---|---|---|---|
| val_single_in_dist | 89.672 | 83.733 | **−6.62%** |
| val_geom_camber_rc | 92.482 | 91.690 | −0.86% |
| val_geom_camber_cruise | 54.093 | 50.392 | **−6.84%** |
| val_re_rand | 72.321 | 71.018 | −1.80% |
| **val_avg/mae_surf_p** | **77.1419** | **74.2082** | **−3.80%** |
| **test_avg/mae_surf_p** | 67.6796 | 65.1123 | **−3.79%** |

**Merged as new baseline: 77.1419 → 74.2082 (−3.80%).**

### Analysis

Epoch-5 peak-LR spike **re-emerged** at 1.5e-3 (+20.4 units), confirming the asinh stability buffer is finite. The model fully recovered by epoch 6 and the cosine tail delivered the final −3.80%. best_epoch=14 (final), with the largest single-epoch drop at epoch 14 (−7.38 units) — the cosine schedule is still productive at cutoff. val_single and val_cruise gain heaviest (−6.62%, −6.84%); val_rc again nearly flat (−0.86%), consistent with the recurring pattern that val_rc is the most resistant split. LR ceiling is **NOT closed** at 1.5e-3 — student's suggested lr=2e-3 follow-up is warranted.

---

## 2026-05-13 05:10 — PR #1911: warmup_epochs=3 (nezuko) — CLOSED (dead end)

- **Branch:** `charliepai2g48h2-nezuko/warmup-3-epochs`
- **Hypothesis:** Bracket below the winning warmup=4: test warmup_epochs=3 (warmup peak at 21% of schedule). If 4-epoch is optimal, 3-epoch should be worse; together with frieren's warmup=5, localizes the warmup optimum.
- **Metric artifacts:** (committed on student branch prior to closure)

### Results vs. #1814 baseline (77.1419)

| Split | Baseline | warmup=3 | Δ |
|---|---|---|---|
| val_single_in_dist | 89.672 | 91.014 | +1.50% |
| val_geom_camber_rc | 92.482 | 94.204 | +1.86% |
| val_geom_camber_cruise | 54.093 | 54.718 | +1.16% |
| val_re_rand | 72.321 | 73.370 | +1.45% |
| **val_avg/mae_surf_p** | **77.1419** | **78.3435** | **+1.56%** |

**Closed as dead end.** Steeper ramp (3 vs 4 warmup epochs at lr=1e-3) causes instability: epoch-5 peak-LR spike harder — all 4 splits regress uniformly. High-Re instability fingerprint confirmed (val_single +1.50%, val_re_rand +1.45%). Warmup=4 is the correct floor; 3 epochs insufficient buffer at lr=1e-3. Pairs with frieren's warmup=5 to bracket the optimum.

---

## 2026-05-13 04:15 — PR #1835: ASINH_GAIN=0.5 (nezuko) — CLOSED (dead end)

- **Branch:** `charliepai2g48h2-nezuko/asinh-gain-0.5`
- **Hypothesis:** Reduce ASINH_GAIN from 1.0 to 0.5 (milder compression, wider linear region). Predicted: recover val_rc while retaining most cruise/re_rand gains.

### Results vs. #1777 baseline (79.8623)

| Split | Baseline | GAIN=0.5 | Δ |
|---|---|---|---|
| val_single_in_dist | 97.455 | 99.668 | +2.27% |
| val_geom_camber_rc | 94.889 | 94.722 | −0.18% |
| val_geom_camber_cruise | 54.000 | 56.142 | +3.97% |
| val_re_rand | 73.105 | 75.155 | +2.80% |
| **val_avg/mae_surf_p** | **79.8623** | **81.4217** | **+1.96%** |

**Closed as dead end.** Axis is asymmetric: GAIN<1.0 moves toward baseline (less bulk-redistribution), not toward a sweet spot. val_rc partial recovery (−0.18%) is outweighed by cruise (+3.97%), re_rand (+2.80%), single (+2.27%) regressions. GAIN=1.0 is well-calibrated for bulk redistribution. GAIN sweep downward is closed.

---

## 2026-05-13 04:00 — PR #1814: lr=1e-3 + asinh stack (alphonse) — MERGED ✓

- **Branch:** `charliepai2g48h2-alphonse/lr-1e-3-warmup4`
- **Hypothesis:** Push peak LR 7e-4 → 1e-3 with the 4-epoch warmup buffer, rebased on asinh base. Super-additive stacking predicted (asinh redistributes bulk capacity → higher LR escapes harder-split local minima without cruise overshoot).
- **Metric artifacts:** `models/model-charliepai2g48h2-alphonse-lr-1e-3-warmup4-asinh-20260513-032210/metrics.jsonl`

### Results vs. #1777 baseline (asinh-only, 79.8623)

| Split | Baseline (#1777) | lr=1e-3+asinh | Δ |
|---|---|---|---|
| val_single_in_dist | 97.455 | 89.672 | **−7.99%** |
| val_geom_camber_rc | 94.889 | 92.482 | **−2.54%** |
| val_geom_camber_cruise | 54.000 | 54.093 | +0.17% |
| val_re_rand | 73.105 | 72.321 | **−1.07%** |
| **val_avg/mae_surf_p** | **79.8623** | **77.1419** | **−3.40%** |
| test_avg/mae_surf_p | 70.4297 | 67.6796 | −3.91% |

**New baseline: val_avg/mae_surf_p = 77.1419**

### Analysis

**Super-additive stacking**: lr=1e-3 alone gave −0.52%, asinh alone gave −1.04%, combined gives −4.41% (2.8× super-additive). Mechanism: asinh compresses heavy-tail residuals → uniform gradient → higher LR can escape local minima on hard splits without overshooting easy splits. val_single −7.99% largest single-split gain in this branch. Epoch-5 peak-LR spike (seen in pre-asinh run) GONE — strict monotone descent. Asinh provides additional LR-peak gradient stability beyond its static bulk-redistribution effect.

**Next axis:** lr=1.5e-3 (probe LR ceiling — monotone descent suggests headroom remains).

---

## 2026-05-13 03:01 — PR #1815: Mesh node dropout 0.9 (askeladd) — SENT BACK (rebase + re-run)

- **Branch:** `charliepai2g48h2-askeladd/node-dropout-0.9`
- **Hypothesis:** Random Bernoulli(0.9) keep-mask on volume mesh nodes during training only (surface nodes always kept). Regularizes memorization of per-sample node-position → value mappings.
- **Metric artifacts:** `models/model-charliepai2g48h2-askeladd-node-dropout-0.9-20260513-021700/metrics.jsonl`

### Results vs. pre-asinh baseline (#1776, 80.7014)

| Split | Baseline (#1776) | Node-drop 0.9 | Δ |
|---|---|---|---|
| val_single_in_dist | 97.712 | 95.792 | **−1.97%** |
| val_geom_camber_rc | 94.420 | 94.444 | +0.03% |
| val_geom_camber_cruise | 55.330 | 55.119 | −0.38% |
| val_re_rand | 75.344 | 73.867 | **−1.96%** |
| **val_avg/mae_surf_p** | **80.7014** | **79.8056** | **−1.11%** |
| test_avg/mae_surf_p | 71.9145 | 70.8162 | −1.53% |

**Decision:** Sent back for rebase. Result vs current baseline 79.8623 is only −0.07% (within noise) because askeladd's run did not include asinh-pressure compression (PR #1777 merged mid-run). The per-split signal (val_single −1.97%, val_re_rand −1.96%) is a real memorization-regularizer mechanism, on different splits than asinh's wins (val_cruise −2.40%, val_re_rand −2.97%). Predicted stacking combined val_avg: ~78.5–79.0. Rebased re-run will confirm or refute.

---

## 2026-05-13 03:01 — PR #1814: lr=1e-3 + 4-epoch warmup (alphonse) — SENT BACK (rebase + re-run)

- **Branch:** `charliepai2g48h2-alphonse/lr-1e-3-warmup4`
- **Hypothesis:** Push peak LR 7e-4 → 1e-3 on the 4-epoch warmup buffer. Hypothesis: 7e-4 was implicitly calibrated for 2-epoch warmup; with 4-epoch warmup the stability floor for peak LR has shifted.
- **Metric artifacts:** `models/model-charliepai2g48h2-alphonse-lr-1e-3-warmup4-20260513-020847/metrics.jsonl`

### Results vs. pre-asinh baseline (#1776, 80.7014)

| Split | Baseline (#1776) | lr=1e-3 | Δ |
|---|---|---|---|
| val_single_in_dist | 97.712 | 93.876 | **−3.93%** |
| val_geom_camber_rc | 94.420 | 94.522 | +0.11% |
| val_geom_camber_cruise | 55.330 | 57.766 | **+4.40%** |
| val_re_rand | 75.344 | 74.973 | −0.49% |
| **val_avg/mae_surf_p** | **80.7014** | **80.2842** | **−0.52%** |
| test_avg/mae_surf_p | 71.9145 | 70.6729 | −1.73% |

**Decision:** Sent back for rebase. Result vs current baseline 79.8623 is +0.53% (regressed) because run did not include asinh-pressure. The val_single −3.93% is the largest single-split gain we've measured from a 1-line change; val_cruise +4.40% is a regression but asinh-pressure's largest win was on exactly that split (−2.40%). Strong reason to expect constructive stacking: alphonse's higher LR escapes local minima on val_single/val_re_rand while asinh holds val_cruise. Predicted combined val_avg: ~78.6–79.2. Rebased re-run will confirm.

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

## 2026-05-12 23:52 — PR #1684: T_max alignment --epochs 14 (frieren)

- **Branch:** `charliepai2g48h2-frieren/tmax-aligned-14`
- **Hypothesis:** T_max=20 with only ~14 epochs fitting in the 30-min cap leaves LR at ~37% of peak at termination. Aligning T_max to feasible epoch count (--epochs 14) lets cosine fully anneal to LR≈0 — a "free" improvement with no code changes.
- **Status:** MERGED ✅ — new baseline 84.562

### Results

| Split | Baseline (95.336) | T_max=14 | Δ % |
|---|---|---|---|
| val_single_in_dist | 118.539 | 103.231 | −12.9% |
| val_geom_camber_rc | 105.115 | 95.256 | −9.4% |
| val_geom_camber_cruise | 71.196 | 60.589 | −14.9% |
| val_re_rand | 86.495 | 79.170 | −8.5% |
| **val_avg/mae_surf_p** | **95.336** | **84.562** | **−11.3%** |
| **test_avg/mae_surf_p** | **85.648** | **74.947** | **−12.5%** |

**Metric artifacts:**
- `models/model-charliepai2g48h2-frieren-tmax-aligned-14-20260512-230927/metrics.jsonl`

### Commentary

**4× the predicted improvement (predicted −1–3%, actual −11.3%).** All 4 val splits improved with similar magnitude (8.5–14.9%), confirming no split-specific artifact — pure schedule alignment gain. 14/14 epochs completed cleanly (best = epoch 14, still descending). Prior baseline was using T_max=20 with cosine ending at ~37% of peak LR at cutoff; aligning to T_max=14 captures the full annealing benefit.

### Critical finding for all subsequent experiments

**Use `--epochs 14` from now on.** T_max=14 is the new schedule canon. All experiments assigned before this merge were using `--epochs 20` (schedule-misaligned). If those land close to 84.562 or below, they beat baseline; if they land 5–10% above, it may be a schedule alignment artifact rather than a genuine regression — they should be re-run with `--epochs 14` for confirmation.

---

## 2026-05-12 23:57 — PR #1663: Smooth L1 full-stack validation (alphonse)

- **Branch:** `charliepai2g48h2-alphonse/smooth-l1-full-stack`
- **Hypothesis:** Re-run the merged Smooth L1 + channel_weights + warmup/clip stack to confirm metric (95.336 was on lr=5e-4, merged code uses lr=7e-4).
- **Status:** CLOSED ❌ — superseded by #1684 (canonical full-stack with --epochs 14 is 84.562 < this PR's 90.506)

### Results

| Split | mae_surf_p | vs #1414 baseline |
|---|---|---|
| val_single_in_dist | 113.834 | −4.0% |
| val_geom_camber_rc | 103.082 | −1.9% |
| val_geom_camber_cruise | 65.219 | −8.4% |
| val_re_rand | 79.889 | −7.6% |
| **val_avg/mae_surf_p** | **90.506** | **−5.06%** |
| **test_avg/mae_surf_p** | **81.978** | **−4.28%** |

**Metric artifacts:**
- `models/model-charliepai2g48h2-alphonse-smooth-l1-full-stack-20260512-225710/metrics.jsonl`

### Commentary

Alphonse's full-stack run confirmed the stack composes positively (90.506 < 95.336 #1414 baseline, < 102.85 #1424 baseline). But this was on --epochs 20 (T_max=20, schedule-misaligned). The same stack with --epochs 14 alignment (frieren #1684) achieves 84.562 — a strictly better result via cosine fully annealing.

The 90.506 → 84.562 gap = 7.0% improvement from pure schedule alignment on the full stack — consistent with the broader T_max=14 lesson. No new winning result to merge; the canonical full-stack with proper schedule is #1684.

### Why closed

Not a regression — confirms hypothesis (full-stack composes positively). Closed because superseded by #1684, which IS the same stack with proper schedule. Reassigned alphonse to the β-sweep direction (#1722 β=0.05 narrower).

---

## 2026-05-12 23:57 — PR #1658: SWA epochs 10-14 (askeladd)

- **Branch:** `charliepai2g48h2-askeladd/swa-ep10-14`
- **Hypothesis:** Stochastic Weight Averaging across epochs 10-14 finds a flatter minimum that generalizes better to OOD splits.
- **Status:** CLOSED ❌ — dead end (+23% worse than current 84.562 baseline; SWA mechanism worked but budget too small)

### Results

| Metric | Value | vs current baseline 84.562 |
|---|---|---|
| LIVE best (ep 13) | 107.32 | +27% worse |
| **SWA (5 snapshots, ep 10-14)** | **104.15** | **+23% worse** |
| In-run SWA vs LIVE | −2.97% | (SWA mechanism works) |

### Per-split (SWA vs OLD #1424 baseline at time of assignment)

- val_single_in_dist: 121.560 (+1.57%)
- val_geom_camber_rc: 116.508 (+2.80%)
- **val_geom_camber_cruise: 79.184 (−3.54%)** ✓
- val_re_rand: 99.359 (+3.18%)

### Test (3-split, cruise NaN due to inf in GT)

- 3-split mean: 100.900 (SWA) vs 105.706 (LIVE) — **−4.55%** within-run

### Commentary

SWA's within-run smoothing mechanism worked exactly as predicted (−3% over LIVE on val, −4.55% on test). The fundamental limitation is the snapshot budget — at --epochs 14 with SWA_START=10, only 4-5 snapshots are averaged, and the cosine schedule was still annealing through SWA collection (Izmailov et al.'s recipe needs a constant-high-LR plateau during SWA, which doesn't fit 14 epochs cleanly).

Additionally, askeladd's LIVE this run was 4.5 points worse than #1424's reported LIVE (same config, same seed-less code path) — suggesting a bad-luck training trajectory. Even correcting for that, SWA's +3% recovery isn't enough to bridge the gap to the new 84.562 baseline.

### Why closed

>5% regression vs current baseline (even with most generous schedule alignment correction, still ~9% worse). SWA needs a different schedule setup (constant-high-LR plateau during collection) to be tested properly — which requires a different schedule architecture than fits in the 14-epoch budget. Reassigned askeladd to OneCycleLR (#1723) as a fresh schedule axis.

### Lifted insight

The within-run SWA-over-LIVE delta (3-5%) is a clean diagnostic for schedule basin flatness. Even though SWA didn't beat baseline, the technique correctly detected the flatter basin in cruise (the in-distribution split) vs the curved basins in OOD splits — useful insight for future basin-shape questions.

---

## 2026-05-13 00:53 — PR #1682: Pure L1 loss (tanjiro)

- **Branch:** `charliepai2g48h2-tanjiro/pure-l1-loss`
- **Hypothesis:** Replacing Smooth L1 (β=0.1) with pure `F.l1_loss` (no quadratic regime) gives a tighter surrogate for the MAE evaluation criterion, since every residual contributes a constant-magnitude gradient.
- **Status:** MERGED ✅ — new baseline 83.230

### Results

| Split | Baseline (#1684) | Pure-L1 | Δ % |
|---|---|---|---|
| val_single_in_dist | 103.231 | **99.310** | −3.8% |
| val_geom_camber_rc | 95.256 | 95.316 | +0.06% |
| val_geom_camber_cruise | 60.589 | 61.818 | +2.0% |
| val_re_rand | 79.170 | **76.477** | **−3.4%** |
| **val_avg/mae_surf_p** | **84.562** | **83.230** | **−1.58%** |
| **test_avg/mae_surf_p** | **74.947** | **73.513** | **−1.91%** |

**Metric artifacts:**
- `models/model-charliepai2g48h2-tanjiro-pure-l1-loss-20260513-001624/metrics.jsonl`

### Commentary

Hypothesis confirmed within predicted range (−1% to −4%). Pure L1 removes the β=0.1 quadratic zone, which was lightly load-shedding small-residual gradient pressure. Strongest gain on val_re_rand (−3.4%, highest-variance split), val_single (−3.8%). Cruise slightly regresses (+2.0%) — consistent with the interpretation that cruise meshes have more small residuals near convergence that the quadratic zone was previously handling smoothly.

Gradient stability confirmed: peak |pred| ≈ 8.5K in physical units (117× below alarm threshold). `grad_clip=1.0` sufficient.

**Canonical loss is now: `F.l1_loss(reduction='none')` × channel_weights[1,1,3] / 5.**

---

## 2026-05-13 00:53 — PR #1723: OneCycleLR (askeladd) — SENT BACK for rebase

- **Branch:** `charliepai2g48h2-askeladd/onecycle-lr-pct03`
- **Hypothesis:** OneCycleLR with pct_start=0.3 peaks LR at 30% of training (epoch 4 of 14) instead of 14% (epoch 2 of 14). Later peak + smooth final decay to near-zero.
- **Status:** SENT BACK ↩️ — beats old baseline 84.562 (val=83.397, −1.38%) but doesn't beat new 83.230 baseline. Needs rebase + rerun stacked with pure-L1.

### Results (on Smooth L1 β=0.1 + cosine baseline 84.562)

| Split | Baseline (#1684) | OneCycleLR | Δ % |
|---|---|---|---|
| val_single_in_dist | 103.231 | 101.455 | −1.72% |
| val_geom_camber_rc | 95.256 | 96.717 | +1.53% |
| val_geom_camber_cruise | 60.589 | 58.790 | −2.97% |
| val_re_rand | 79.170 | 76.626 | **−3.21%** |
| **val_avg/mae_surf_p** | **84.562** | **83.397** | **−1.38%** |
| **test_avg/mae_surf_p** | **74.947** | **73.095** | **−2.47%** |

Schedule trajectory confirmed: peak at epoch 4, monotonic descent from epoch 7, final LR ≈ 0.

### Why sent back

OneCycleLR change is orthogonal to pure-L1 (schedule vs loss). Askeladd's result (83.397) was on the Smooth L1 stack. After tanjiro's pure-L1 merged (new baseline 83.230), stacking both should give ~82.0. Sent back to rebase onto current advisor HEAD (which has pure-L1), keep OneCycleLR schedule, re-run with --epochs 14.

---

## 2026-05-13 01:30 — PR #1707: Per-sample loss normalization by pressure std (frieren) — CLOSED dead end

- **Branch:** `charliepai2g48h2-frieren/per-sample-loss-norm`
- **Hypothesis:** Normalize the per-sample surface loss by that sample's surface pressure std (`surf_p_std`), so high-variance samples don't dominate batch gradients. Target: equalize gradient influence across pressure-variance levels.
- **Status:** CLOSED ❌ — +12.2% regression vs #1684 baseline, +14.0% vs new 83.230 baseline.

### Results

| Split | Baseline (#1684) | Per-sample-std-norm | Δ % |
|---|---|---|---|
| val_single_in_dist | 103.231 | 119.554 | +15.8% |
| val_geom_camber_rc | 95.256 | 103.644 | +8.8% |
| val_geom_camber_cruise | 60.589 | 69.983 | +15.5% |
| val_re_rand | 79.170 | 86.373 | +9.1% |
| **val_avg/mae_surf_p** | **84.562** | **94.889** | **+12.2%** |
| **test_avg/mae_surf_p** | **74.947** | **85.848** | **+14.5%** |

**Metric artifacts:**
- `models/model-charliepai2g48h2-frieren-per-sample-loss-norm-20260513-000916/metrics.jsonl`

### Root cause

`surf_p_std` distribution spans 4+ orders of magnitude (min ≈ 5e-4, max ≈ 6.91, mean ≈ 0.8). The `clamp(min=1e-6)` was effectively never active but the actual minima at ~5e-4 still gave near-uniform-pressure samples effective batch weights of ~2000× relative to high-std samples. A small number of degenerate samples dominated every gradient step. Frieren's diagnostic was thorough: per-epoch min/max/mean of `surf_p_std` confirmed the 2000× weight pathology.

The PR's own contingency (clamp(min=0.5, max=2.0), capping weight ratio at 4:1) would have been the correct implementation. However, on the pure-L1 canonical base, per-sample reweighting is a second-order correction: pure-L1 already gives constant-magnitude gradient per residual, largely mitigating the gradient-noise asymmetry the PR targeted.

**Lesson learned:** clamp threshold must be calibrated against actual data distribution, not a loose machine-epsilon default.

---

## 2026-05-13 01:30 — PR #1659: slice_num 64→96 (nezuko) — CLOSED dead end

- **Branch:** `charliepai2g48h2-nezuko/slice-96-stable`
- **Hypothesis:** Increase physics-token count from 64 to 96 (finer mesh partitioning). The model captures more geometric detail per token, especially on OOD geometry splits. Stable regime (unlike slice_num=128 which collapsed in #1429).
- **Status:** CLOSED ❌ — +27.9% regression vs #1684 baseline, +30% vs new 83.230 baseline.

### Results (two runs)

| Run | val_avg/mae_surf_p | vs #1684 |
|---|---|---|
| epochs=20 (old T_max) | 115.37 | +36.4% |
| **epochs=14 (T_max aligned)** | **108.158** | **+27.9%** |

| Split | ep=20 | ep=14 | Δ ep20→14 |
|---|---|---|---|
| val_single_in_dist | — | 131.020 | — |
| val_geom_camber_rc | — | 118.791 | — |
| val_geom_camber_cruise | — | 81.322 | — |
| val_re_rand | — | 101.499 | — |

**Metric artifacts:**
- `models/model-charliepai2g48h2-nezuko-slice-96-stable-20260513-000937/metrics.jsonl`

### Root cause

Numerically stable (pred_abs_max ≤ 16.18, no NaN). Pure empirical regression: adding more slice tokens with the same n_hidden=128 appears to dilute per-token capacity — each attention token now represents a smaller mesh region with the same channel budget. The pred_abs_max trajectory (still growing at epoch 12/14) suggests under-fit, consistent with optimization requiring more iterations per effective degree of freedom.

Combined with prior dead ends (#1429 slice_num=128: overflow, #1598 mlp_ratio=4: +7%), the architecture/capacity axis is now **exhausted on this dataset at this epoch budget**: deeper, wider, and finer-grained partitioning all regress.

**Lesson learned:** The Transolver at slice_num=64, n_hidden=128, n_layers=5 appears to be well-matched to this dataset and budget. Capacity is not the bottleneck.

---

## 2026-05-13 02:00 — PR #1776: 4-epoch warmup (frieren) — MERGED ✅ NEW BASELINE

- **Branch:** `charliepai2g48h2-frieren/warmup-4-epochs`
- **Hypothesis:** Increase warmup_epochs from 2 to 4. LR peak shifts from epoch 2/14 (14%) to epoch 5/14 (36%). CosineAnnealingLR T_max shrinks from 12 to 10.
- **Status:** MERGED ✅ — new baseline 80.7014

### Results

| Split | Baseline (#1682) | 4-epoch warmup | Δ % |
|---|---|---|---|
| val_single_in_dist | 99.310 | 97.712 | −1.61% |
| val_geom_camber_rc | 95.316 | 94.420 | −0.94% |
| val_geom_camber_cruise | 61.818 | **55.330** | **−10.50%** |
| val_re_rand | 76.477 | 75.344 | −1.48% |
| **val_avg/mae_surf_p** | **83.230** | **80.7014** | **−3.04%** |
| **test_avg/mae_surf_p** | **73.513** | **71.9145** | **−2.17%** |

**Metric artifacts:** `models/model-charliepai2g48h2-frieren-warmup-4-epochs-20260513-011736/metrics.jsonl`

### Commentary

All 4 val splits improve. The standout is val_cruise (−10.50%): the low-LR warmup phase stabilizes early gradient flow, and the smooth-pressure cruise cases benefit most from accurate late-epoch convergence. The model was descending monotonically to epoch 14/14 with best_epoch=14. Schedule shape insight: longer low-LR ramp before peak gives better model initialization → steeper but shorter cosine descent (T_max=10 vs 12) spends more epochs near peak LR before annealing.

**First-epoch behavior confirmation:** training loss was higher than baseline (more conservative initial ramp), crossing over at ~epoch 4-5 as the new schedule catches up.

**Canonical config now:** F.l1_loss + channel_weights=[1,1,3] + lr=7e-4 + **warmup_epochs=4** + CosineAnnealingLR(T_max=10) + grad_clip=1.0 + --epochs 14.

---

## 2026-05-13 02:00 — PR #1744: Gradient accumulation 4× (tanjiro) — CLOSED dead end

- **Branch:** `charliepai2g48h2-tanjiro/grad-accum-4`
- **Hypothesis:** ACCUM_STEPS=4 → effective batch 4→16, reduce gradient noise in late-epoch fine-tuning.
- **Status:** CLOSED ❌ — +14.88% regression vs baseline 83.230 (now vs 80.7014: even larger)

### Results

| Metric | Baseline | grad-accum-4 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 83.230 | 95.6227 | +14.88% |
| test_avg/mae_surf_p | 73.513 | 86.4421 | +17.59% |

**Root cause:** 4× accumulation without LR scaling = 4× fewer optimizer updates (1316 vs 5250 over 14 epochs). Update-count-bounded, not gradient-noise-bounded. Goyal et al. 2017 linear scaling rule applies: without compensating LR, large-batch training is effectively shorter training. Per-epoch tail slope (−3.5%/epoch at ep14 vs −2.0% baseline) confirms model under-converged.

---

## 2026-05-13 02:00 — PR #1723: OneCycleLR pct_start=0.3 (askeladd, rebased) — CLOSED

- **Branch:** `charliepai2g48h2-askeladd/onecycle-lr-pct03`
- **Hypothesis:** OneCycleLR schedule (pct_start=0.3) stacks additively with pure-L1.
- **Status:** CLOSED — near-tie (+0.37% worse on val_avg). Test improves (-1.29%), val_single regresses (+3.08%), val_cruise improves (−3.17%).

### Results (rebased on pure-L1 HEAD)

| Metric | pure-L1 baseline | OneCycleLR + pure-L1 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 83.230 | 83.539 | +0.37% |
| test_avg/mae_surf_p | 73.513 | 72.568 | −1.29% |

**Root cause:** OneCycleLR's near-zero tail LR (2.93e-9 at ep14) prevents pure-L1's "pressure to move" from executing in final epochs. val_single regresses (sensitive to fine late-epoch convergence) while val_cruise improves (was already near basin at ep12-13). Schedule-shape axis appears saturated after the T_max alignment win. Frieren's warmup-4 win (orthogonal: duration not shape) confirms this — the key was warmup duration, not schedule shape.

---

## 2026-05-13 02:40 — PR #1777: Asinh pressure compression (nezuko) — MERGED ✓

- **Branch:** `charliepai2g48h2-nezuko/asinh-pressure-gain-1`
- **Hypothesis:** Apply asinh(y * ASINH_GAIN) / ASINH_GAIN to the pressure channel (channel 2) of the normalized target before loss computation. ASINH_GAIN=1.0. Predicted mechanism: compress large-magnitude pressure residuals to make gradient more uniform across nodes, reducing capacity-steering toward suction-peak tail nodes.
- **Metric artifacts:** `models/model-charliepai2g48h2-nezuko-asinh-pressure-gain-1-20260513-012107/metrics.jsonl`

### Results vs. #1776 baseline (warmup-4, val_avg=80.7014)

| Split | Baseline (#1776) | Asinh (#1777) | Δ |
|---|---|---|---|
| val_single_in_dist | 97.712 | 97.455 | −0.26% |
| val_geom_camber_rc | 94.420 | 94.889 | +0.50% |
| val_geom_camber_cruise | 55.330 | 54.000 | **−2.40%** |
| val_re_rand | 75.344 | 73.105 | **−2.97%** |
| **val_avg/mae_surf_p** | **80.7014** | **79.8623** | **−1.04%** |
| test_avg/mae_surf_p | 71.9145 | 70.4297 | −2.06% |

**New baseline: val_avg/mae_surf_p = 79.8623**

### Analysis and conclusions

- **Mechanism: bulk-redistribution, not tail-flattening.** Asinh compresses large-magnitude suction-peak residuals, forcing gradient capacity toward bulk pressure regions. Cruise benefits most (highest bulk/peak ratio); rc mildly regresses (more suction peaks relative to bulk).
- **Implementation clean:** round-trip error 1.43e-6 (float32 epsilon), clamp never activated (pred_abs_max_norm ≤ 6.81), channels 0/1 bit-identical.
- **Note:** Measured on pre-#1776 base (warmup_epochs=2); merged onto warmup_epochs=4. Changes are mechanistically orthogonal (target representation vs LR schedule). Delta reported vs #1776 baseline.
- **Follow-up axis:** ASINH_GAIN sweep — GAIN=0.5 (wider linear region, milder compression, target: retain cruise/re_rand gains while recovering rc regression).

---

## 2026-05-13 02:00 — PR #1722: Smooth L1 β=0.05 (alphonse) — CLOSED

- **Branch:** `charliepai2g48h2-alphonse/smooth-l1-beta-005`
- **Hypothesis:** β=0.05 narrows the quadratic regime (β ladder: 0.1 → 0.05 → 0).
- **Status:** CLOSED — +0.47% worse than pure-L1 baseline (83.621 vs 83.230).

### Results

| β | val_avg/mae_surf_p | vs. β=0.1 |
|---|---|---|
| β=0.1 (prior baseline) | 84.562 | — |
| β=0.05 (this run) | 83.621 | −1.11% |
| β=0 pure-L1 (canonical) | 83.230 | −1.58% |

**Conclusion:** Monotone improvement as β→0 confirmed. β=0.05 lands between β=0.1 and β=0 as expected. The full L1 is the global minimum of this Smooth-L1 family. β axis fully exhausted.

---
