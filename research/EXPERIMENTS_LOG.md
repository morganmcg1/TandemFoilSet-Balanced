# SENPAI Research Results — `icml-appendix-charlie-pai2g-24h-r4`

This log records every PR review on this advisor branch with the
hypothesis, the metrics pulled from the committed JSONL, and a short
commentary.

Entries are appended chronologically (newest at top). The metric of
record for ranking is `val_avg/mae_surf_p`; the paper-facing comparison
metric is `test_avg/mae_surf_p`.

## 2026-05-13 02:35 — PR #1608 (frieren EMA-of-model-weights decay=0.999) — **CLOSED**

- Branch: `charliepai2g24h4-frieren/ema-weights-0.999`
- Hypothesis: EMA of model weights at decay=0.999, swap-in for val/save.
  Smooths the optimizer trajectory via exponential moving average; expected
  to compound with merged stoch-depth (orthogonal variance reduction).

**Pre-rebase run (vs old 98.353 baseline)**: -2.64% val, -3.08% test win.
**Rebased run (vs current 84.762 baseline)**: +2.93% val, +4.12% test
regression. Sign flipped after rebase onto the merged stack.

| Metric | This PR (rebased) | Current baseline (#1548) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15) | 87.244 | 84.762 | **+2.93%** |
| test_avg/mae_surf_p (4-split) | 77.738 | 74.659 | **+4.12%** |
| Param count | 665,943 | 665,943 | unchanged (EMA is in-memory only) |

- **All 4 val splits regress or flat** (val_geom_camber_rc -0.66% only).
  Largest hits: `val_single_in_dist` +6.17%, `val_geom_camber_cruise` +5.63%.
- **All 4 test splits regress** uniformly (largest: test_single_in_dist
  +7.80%, test_geom_camber_cruise +4.95%).
- **Student's mechanism analysis is sharp and tracks the per-split pattern**:
  the Fourier-gain splits (#1548 saw val_single_in_dist -11.35%, test_single_in_dist
  also massive gain) regress most under EMA — clean inverse correlation.
  EMA's low-pass smoothing on weights smooths the high-frequency Fourier
  feature responses that gave us the -8.10% test gain.
- **EMA window vs cosine T_max=15 misalignment**: decay=0.999 → effective
  averaging window ≈ 2.7 epochs at batch=4. With T_max=15, the live model
  is in 5-50× LR cooldown for the final ~3 epochs; the EMA copy trails
  the live model into the cooldown rather than absorbing it.
- **Stoch-depth + cosine cooldown already absorb most optimizer variance**:
  EMA's variance-reduction effect is mostly double-counted on this stack.
- **Mechanism finding (axis-wide)**: weight-space smoothing on this compound
  is closed. Fights spectral-bias features. Future variance-reduction PRs
  must NOT operate on the weights directly; should target either the loss
  landscape (SmoothL1 — picked next) or trajectory features (e.g., SAM).
- Val curve was perfectly monotonic at every epoch (implementation correct;
  result is a clean negative not a bug).
- Follow-up assigned: PR #1828 (frieren SmoothL1 / Huber loss β=0.01).
  Loss-landscape smoothing rather than weight-space smoothing — should not
  fight spectral-bias features.

## 2026-05-13 02:15 — PR #1754 (nezuko linear LR warmup + cosine T_max=14 — H19) — **SENT BACK FOR REBASE**

- Branch: `charliepai2g24h4-nezuko/lr-warmup-h19`
- Hypothesis: linear LR warmup over epoch 1 (per-batch, total_iters=375)
  + CosineAnnealingLR(T_max=14*375=5250). Addresses ep1 pre-clip grad-norm
  spike (60-100) consistently observed in recent grad-clip experiments.

**This is a WIN on the old baseline** — but measured against pre-#1548
baseline (90.294), not current 84.762. Sent back for rebase + re-run.

| Metric | This PR (vs old) | Old baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15) | 89.718 | 90.294 | **-0.64%** |
| test_avg/mae_surf_p (4-split) | 79.852 | 81.243 | **-1.71%** |

- **Per-split val MAE (3/4 splits improve)**:
  - val_single_in_dist     -1.94%
  - val_geom_camber_rc     -1.37%
  - val_geom_camber_cruise -0.33%
  - val_re_rand            +1.68% (small split-specific noise)
- **All 4 test splits improve** (avg -1.71%; test gain exceeds val gain).
- **Mechanism check passes**: ep1 last-batch pre-clip grad-norm dropped
  ~35% (99 → 65); LR trace matches design (peak at end ep1, half at ep8,
  zero at ep15).
- **Implementation refinement** from the student: PR draft suggested
  per-epoch SequentialLR (coarse 2-point step); student moved scheduler
  inside batch loop with `total_iters=375, milestones=[375], T_max=5250`
  for smooth per-batch ramp. Sharper than the original spec.
- **Sent back for rebase**: technically MERGEABLE (no textual conflict)
  but base is `0668de7` (pre-#1548 Fourier merge); current is `90b33ba`.
  Run must be re-measured against new baseline 84.762.
- Expected post-rebase: -0.3% to -1.5% on val_avg (warmup mechanism is
  orthogonal to Fourier input encoding).

## 2026-05-13 02:10 — PR #1756 (tanjiro stoch-depth drop_rate=0.15 — H bracket-up) — **CLOSED**

- Branch: `charliepai2g24h4-tanjiro/stoch-depth-0.15`
- Hypothesis: pre-registered bracket-up follow-up from closed #1612 at 0.05.
  Push schedule above merged 0.10 to `[0.0, 0.0375, 0.075, 0.1125, 0.15]`.

| Metric | This PR | Old baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15) | 97.235 | 90.294 | **+7.69%** |
| test_avg/mae_surf_p (4-split) | 87.236 | 81.243 | +7.38% |
| Param count | 662,359 | 662,359 | unchanged |

- **All four val splits regress uniformly** (+5.31% to +12.14%); largest
  hit on val_single_in_dist +12.14%.
- **Outcome C confirmed**: the merged 0.10 is the genuine local optimum
  of the single-knob bracket. {0.05 → +13.7%, 0.10 → 0%, 0.15 → +7.69%}
  is a clear asymmetric V around 0.10.
- **Student's sharp finding**: both endpoints regress on val_geom_camber_rc
  (+13.33% at 0.05 from #1612; +5.31% at 0.15 here). The "OOD geometry
  wants more regularization" narrative is now falsified on BOTH sides of
  the bracket — should not appear in future regularization PR hypotheses.
- **Train-vs-val gap direction**: val > train (standard generalization
  gap), NOT train > val (which would have been the ensemble-dropout
  signature). Independent evidence that 0.15 is operating as just-more-noise,
  not stronger ensemble.
- **Mechanism limit**: with n_layers=5 and last block never dropped, the
  effective per-step drop variance at p=0.15 is still small — explains
  why higher drop rates don't unlock new ensemble behavior at this depth.
- Single-knob stoch-depth direction fully closed. Future regularization
  PRs should target different mechanism (per-layer schedule shape, weight
  decay, label smoothing, or output head reshaping).
- Follow-up assigned: PR #1811 (tanjiro per-channel output head MLPs).

## 2026-05-13 02:00 — PR #1773 (thorfinn AdamW betas (0.9, 0.95) — H22) — **CLOSED**

- Branch: `charliepai2g24h4-thorfinn/adamw-betas-0.95`
- Hypothesis: change AdamW `betas` from PyTorch default `(0.9, 0.999)` to
  `(0.9, 0.95)` for faster second-moment EMA adaptation. Mechanism: long
  EMA horizon (1000 steps) lags the cosine-annealed gradient regime; 20-step
  EMA tracks distribution shifts. Modern transformer recipe (LLaMA, PaLM).

| Metric | This PR | Baseline (#1548) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | 86.427 | 84.762 | **+1.97%** |
| test_avg/mae_surf_p (4-split, NaN-safe) | 75.863 | 74.659 | **+1.61%** |
| Best epoch | 15/15 | 15/15 | same |
| Param count | 665,943 | 665,943 | unchanged |

- **Per-split val MAE (non-uniform regression)**:
  - val_single_in_dist     +0.92%
  - val_geom_camber_rc     +1.97%
  - val_geom_camber_cruise **+5.47%** (largest hit — low-noise-floor regime)
  - val_re_rand            +0.50%
- **Per-split test MAE**:
  - test_single_in_dist     +0.71%
  - test_geom_camber_rc     +0.48%
  - test_geom_camber_cruise +3.41%
  - test_re_rand            +2.58%
- **Two falsified predictions confirm direction is closed**:
  1. Best epoch did NOT shift earlier (faster basin-finding falsified)
  2. Per-split direction NOT uniform (optimizer-as-global-mechanism framing falsified)
- **Deepest mechanism finding**: the merged stack already addresses the
  non-stationarity concerns that motivated H22. Grad-clip-25 truncates the
  epoch-1 99.48 spike *before* AdamW sees it. Cosine T_max=15 anticipates
  the LR regime change rather than asking AdamW to react. β₂=0.95 was
  solving a problem that no longer existed.
- **Regime gap**: LLaMA/PaLM use β₂=0.95 at batch=10³-10⁶ × ours, 10⁵-10⁶
  steps. Our regime (batch=4, 5,625 steps) doesn't benefit from short-EMA.
- Single-knob optimizer-betas direction closed on this dataset.
- Follow-up assigned: PR #1799 (thorfinn LayerScale CaiT-style init=0.1).

## 2026-05-13 01:15 — PR #1548 (edward Fourier coords L=4 — rebased) — **MERGED (new baseline)**

- Branch: `charliepai2g24h4-edward/fourier-coords-L4-rebased`
- Hypothesis: Tancik et al. (NeurIPS 2020) Fourier positional encoding on
  spatial `(x, z)` coords addresses spectral bias on the surface-pressure
  signal. Encode normalized coords with `sin/cos` at `2^k · π` for `k=0..3`
  (16 features), bumping `fun_dim` from 22 to 36.

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | **84.762** | 90.294 | **−6.13%** |
| test_avg/mae_surf_p (4-split, NaN-safe) | **74.659** | 81.243 | **−8.10%** |
| n_params | 665,943 | 660,539 | +0.82% |
| peak GPU memory | 42.16 GB | 42.11 GB | +0.12% |
| wall time | ~31.5 min | ~30 min | cap-bound |

- **Per-split val MAE** (best ckpt):
  - val_single_in_dist     97.074 (−11.35% vs 109.497) — largest gain
  - val_geom_camber_rc     94.997 (−4.00% vs 98.952)
  - val_geom_camber_cruise 63.711 (−7.94% vs 69.208)
  - val_re_rand            83.266 (−0.30% vs 83.520) — flat, as predicted
- **Per-split test MAE** (best val ckpt):
  - test_single_in_dist     85.819 (−13.43%)
  - test_geom_camber_rc     83.023 (−7.47%)
  - test_geom_camber_cruise 54.879 (−4.03%)
  - test_re_rand            74.916 (−5.10%)
- **Mechanism confirmed**: split pattern matches spectral-bias hypothesis
  exactly — gains where high-frequency spatial structure dominates
  (in_dist, camber-OOD); minimal on val_re_rand whose OOD axis is Reynolds
  (flow-condition), not spatial coords.
- **Test gain exceeds val gain** (−8.10% vs −6.13%): the Fourier features
  generalize to held-out data better than they fit val. Strong signal for
  the paper-facing test metric.
- **Stacks cleanly with all 4 prior compound merges**: L1 → stoch-depth →
  cosine T_max=15 → grad-clip 25 → Fourier L=4. Compound progress over 5
  merges: val_avg 100.957 → 84.762 = **−16.0%**.
- Metrics: `models/model-charliepai2g24h4-edward-fourier-coords-L4-rebased-20260512-235326/metrics.jsonl`
- Follow-up assigned: PR #1772 (edward Fourier L=6 bracket-up).

## 2026-05-13 01:13 — PR #1699 (thorfinn attn+MLP dropout p=0.05) — **CLOSED**

- Branch: `charliepai2g24h4-thorfinn/attn-mlp-dropout-0.05`
- Hypothesis: standard ViT-style attn+MLP dropout p=0.05 orthogonal to
  block-level stoch-depth would add a fine-grained regularization layer.

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14/15) | 92.342 | 90.294 | **+2.27%** |
| test_avg/mae_surf_p (4-split) | 83.382 | 81.243 | +2.63% |
| wall_time / epoch | 134s | 120s | **+12%** (not "free") |
| n_params | 662,359 | 660,539 | +0% (Dropout adds zero params) |

- **All four val splits regress uniformly** (+0.67% to +2.84%). Uniform
  direction confirms the "regularization is a global mechanism" framing,
  but the sign is the opposite of the hypothesis.
- **Three mechanisms drove the regression**, in order of impact (from
  student's own analysis):
  1. **Compute tax eats one epoch.** Dropout adds ~14s/epoch; run hit
     30-min cap at ep 14/15 instead of 15/15. The cosine T_max=15 hadn't
     fully decayed at ep 14 (lr≈5.5e-6, not zero). One additional cosine
     step would have closed part of the gap.
  2. **Stoch-depth was already at the regularization optimum.** Merged
     schedule averages 0.05 across blocks; adding p=0.05 inside surviving
     blocks pushed past the optimum for this 1499-sample/15-epoch budget.
  3. **Post-softmax slice-attention dropout disrupts unit-sum property.**
     `slice_weights` is used twice (soft binning + soft scatter); dropout
     zeros 5% then rescales surviving ones. The `slice_norm = slice_weights.sum(2)`
     renormalization partially mitigates but the double application of
     dropped weights amplifies effective noise above standard attention dropout.
- Single-knob fine-grained dropout direction closed on this baseline.
- Follow-up assigned: PR #1773 (thorfinn AdamW betas (0.9, 0.95) — clean
  pivot to orthogonal optimizer-recipe axis).

## 2026-05-13 01:00 — PR #1713 (askeladd grad-clip max_norm=10 — H15 bracket below) — **CLOSED**

- Branch: `charliepai2g24h4-askeladd/grad-clip-10`
- Hypothesis: bracket below merged max_norm=25 (single-line edit). Completes
  the fixed-threshold sweep around 25.

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | 94.121 | 90.294 | **+4.24%** |
| test_avg/mae_surf_p (4-split) | 84.561 | 81.243 | +4.08% |
| Epochs with pre-clip norm > threshold | 15/15 (100%) | n/a | — |

- **Outcome B confirmed: max_norm=25 is the local optimum.** Bracket
  geometry: +5.4% (1.0) / +4.24% (10) / 0% (25) / +3.32% (50). Asymmetric
  — tighter direction (10 → +4.24%) costs more than looser direction
  (50 → +3.32%). Heavy 30–70 norms carry partial signal; clipping them
  to 25 helps via variance reduction, clipping all the way to 10 destroys
  signal.
- **All four val splits regress uniformly** (largest hit val_single_in_dist
  +6.16%, smallest val_re_rand +2.85%). Same uniform-direction pattern
  as #1637 (the merged win) and #1674 (the upper-bracket loss), confirming
  this is a global regularization mechanism.
- **15/15 epochs had pre-clip norms above the threshold** — the entire
  training trajectory was continuously clipped, basically destroying the
  per-step gradient direction. The 100% clipping is qualitatively
  different from #1674's 40% — too aggressive.
- **Fixed-threshold grad-clip direction now closed.** Bracket fully
  characterized. Next: adaptive clipping (running-quantile based) per
  the PR's pre-registered follow-up.

## 2026-05-13 00:55 — PR #1677 (nezuko H12 per-node adaptive temperature) — **CLOSED**

- Branch: `charliepai2g24h4-nezuko/per-node-temp-h12`
- Hypothesis: per-node deterministic temperature `τ_i = τ_0 + Linear(x_mid)_i`
  clamped at floor 0.1. Identity-init (zero linear weights). Attacks
  slice-collapse without sampling noise (clean pivot from #1553 Gumbel).

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14/15) | 93.097 | 90.294 | **+3.11%** |
| test_avg/mae_surf_p (4-split) | 82.813 | 81.243 | +1.93% |
| n_params | 662,509 | 662,359 | +150 (+660 nominal, weight tying counted) |
| τ.std @ ep 14 | 0.340 | n/a | — |
| τ floor_fraction @ ep 8 | 0.361 | n/a | — |

- **Mechanism verified, outcome rejected.** τ head learned: identity-init
  → non-trivial spread (std=0.34, range [0.10, 1.80]) by best epoch.
  But 36% floor-fraction by ep8 indicates the model is pushing roughly
  a third of nodes to maximally sharp slice assignments. The PR pre-flagged
  this as a "binding too often" signal.
- **Per-split regression concentrated on splits the hypothesis predicted
  would benefit most** — val_single_in_dist (+5.29 absolute MAE),
  val_geom_camber_rc (+5.03). The cruise and re_rand splits are nearly
  neutral, as expected since they have less boundary-layer structure.
- **Student's mechanistic interpretation:** "aggressive sharpening on
  boundary-layer nodes commits the slice assignment too early, before
  the slice-token MLP has converged on good slice-level representations."
  Plausible and matches the per-split asymmetry.
- **Slice-collapse direction closed.** Three independent arms have now
  failed:
  - #1514 Ada-Temp v1/v2 (per-head scalar τ) — closed +3.4%
  - #1553 Gumbel-Softmax slice noise — closed +4.4% (3-run mean)
  - #1677 H12 per-node deterministic τ — closed +3.11%
- The wave-5-candidate H12-followup-floor-sweep is **dropped**: re-perturbing
  the same dimension we've now shown doesn't carry the signal would just
  burn GPU on a closed direction.

## 2026-05-13 00:50 — PR #1612 (tanjiro stoch-depth drop_rate=0.05) — **CLOSED**

- Branch: `charliepai2g24h4-tanjiro/stoch-depth-0.05`
- Hypothesis: halve the linear stoch-depth schedule from
  `[0,0.025,0.05,0.075,0.10]` to `[0,0.0125,0.025,0.0375,0.05]`. The
  PR body's original target was the +1.77% regression on val_re_rand
  vs pre-#1552 baseline.

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 13/14) | 102.65 | 90.294 | **+13.7%** |
| test_avg/mae_surf_p (4-split) | 91.264 | 81.243 | +12.3% |

- **Note: writeup compared against pre-#1611/#1637 baseline (98.353), giving
  +4.4%.** Against the current advisor HEAD baseline (90.294), the
  regression is +13.7%. The mechanistic conclusions still hold — they're
  about per-split direction, not absolute level.
- **Split-asymmetric response — the key finding.** val_re_rand DID recover
  as hypothesised (-3.03% from the prior over-regularization), but
  val_geom_camber_rc blew up by +13.33% in the opposite direction. OOD
  geometry splits want MORE regularization; the Re sweep wants LESS.
  A single global drop rate can't satisfy both.
- **No overfitting under merged 0.10** — train surf_loss 0.282 ≈ val
  surf_loss 0.285 at the best epoch. Cutting drop in half didn't liberate
  any frozen useful capacity; it just produced noisier gradients with
  weaker implicit ensembling.
- **Reproducibility check:** independent launch at 23:03 produced val_avg
  101.5 — same regression direction, small run-to-run noise.
- **Pivot direction: pre-registered drop_rate=0.15 follow-up.** Test
  whether hard OOD geometry splits want even more regularization. If
  0.15 helps val_geom_camber_rc while keeping val_re_rand neutral, that's
  a winner. If 0.15 also regresses, the stoch-depth single-knob direction
  is at its local optimum at 0.10.

## 2026-05-13 00:08 — PR #1675 (alphonse H17 per-channel output γ, β) — **CLOSED**

- Branch: `charliepai2g24h4-alphonse/out-scale-bias-h17`
- Hypothesis: 6 learnable parameters `γ ∈ ℝ³, β ∈ ℝ³` on the output head,
  identity-init, attack per-channel pressure-vs-velocity calibration
  without compression (the closed log1p direction).

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | 93.214 | 90.294 | **+3.24%** |
| test_avg/mae_surf_p (4-split) | 83.108 | 81.243 | +2.30% |
| n_params | 662,365 | 662,359 | +6 |

- **Mechanism behaved exactly as predicted, outcome did not.** `out_gamma[2]`
  (pressure) drifted +6.13% by ep 15, vs +1.43% on Ux and +3.21% on Uy —
  optimizer found pressure-channel scale up = local loss minimizer, as
  hypothesised. Drift was smooth and monotone across all 15 epochs
  (gradient flow healthy).
- **All four val splits regressed**, largest hit on `val_single_in_dist`
  (+5.07%) — exactly the split the hypothesis predicted would benefit
  most (highest p magnitudes). The inversion is informative: identity-init
  guarantees zero regression at step 0, but with no penalty toward
  identity, the optimizer drifts wherever the *training* gradient pulls.
  A 6% multiplicative drift on already-correct large-magnitude pressure
  predictions inflates their MAE by ~6%.
- **Output-side calibration is exhausted on this dataset.** Trio of
  closures: #1610 (full-channel log1p +1.18%), #1636 (pressure-only log1p
  +5.32%), #1675 (per-channel γ, β +3.24%). The existing pre-training
  normalization is more useful than any post-hoc multiplicative correction.
- **Pivot direction:** student's own suggestion #1 — per-channel surf-loss
  weighting. Attack the same imbalance upstream at the loss layer instead
  of letting the model mis-calibrate the prediction.

## 2026-05-13 00:08 — PR #1674 (askeladd grad-clip max_norm=50 — H15 bracket above) — **CLOSED**

- Branch: `charliepai2g24h4-askeladd/grad-clip-50`
- Hypothesis: bracket above merged max_norm=25 to test whether pure
  spike-only suppression (the 110-norm at ep 8 in #1637) is sufficient
  or whether bulk 30–70 norm clipping is the active mechanism.

| Metric | This PR | Baseline (#1637, max_norm=25) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | 93.286 | 90.294 | **+3.32%** |
| test_avg/mae_surf_p (4-split) | 83.882 | 81.243 | +3.25% |
| Epochs above 50 (clipped) | 6/15 (40%) | n/a | — |

- **Outcome B (PR-predicted): bulk 30–70 norms are the active ingredient
  at threshold 25.** Spike-only suppression at threshold 50 captures the
  signature pattern (ep5→6, ep10→11 "spike-down" mapping correct) but
  recovers only a fraction of the #1637 gain.
- **Uniform regression across all four splits**, largest on
  `val_single_in_dist` (+7.08%), smallest on `val_geom_camber_cruise`
  (+1.43%) — same direction as the merged #1637 win was uniform, just
  reversed.
- **Implication:** the variance-reduction-on-heavy-steps reading is right.
  The 25-threshold is clipping not just outliers but moderately heavy
  (~30–50) steps, and removing that clipping when threshold = 50 is the
  cost.
- **Pivot direction:** student's suggestion #1 — `max_norm=10` (lower
  bracket). If 10 wins → bracket further below; if 10 regresses → 25 is
  at local optimum and the fixed-threshold sweep is complete (next would
  be adaptive clipping schemes).

## 2026-05-12 23:55 — PR #1555 (thorfinn tied projection + n_hidden=144 retune) — **CLOSED**

- Branch: `charliepai2g24h4-thorfinn/remove-in-project-fx`
- Hypothesis: keep the tied projection (in_project_fx removed, slice pool
  reuses x_mid) but widen `n_hidden` 128 → 144 to reinvest the freed
  parameter budget across all weights.
- Rebased onto current advisor HEAD `05a8b35` (post #1552 + #1611 + #1637).

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 12/13) | 102.668 | 90.294 | **+13.71%** |
| test_avg/mae_surf_p (4-split) | 91.739 | 81.243 | +12.92% |
| n_params | 730,423 | 662,359 | +10.3% |
| Wall time/epoch | 143s | 120s | +19% |
| Epochs in 30-min cap | 13 | 15 | -2 |

- **All four val splits regressed** by 5-22%, not just in-distribution.
  Rules out the "wider over-parameterizes a small training set"
  interpretation in favor of a structural "wider doesn't help here" signal.
- **Root cause (per student diagnostic)**: the original n_hidden=144 retune
  hypothesis was framed against the pre-cosine, pre-grad-clip baseline at
  val_avg=98.353 where single_in_dist sat at 129.4 (in-distribution
  underfit was real). After #1611 cosine + #1637 grad-clip both merged,
  single_in_dist dropped to 109.5 *via optimization fixes alone* — the
  underfitting the retune was meant to fix had already been resolved.
- **Wall-clock budget cost** the rest: wider model = +19%/epoch =
  -2 epochs in the cap = cosine arc cuts off at LR=2.2e-5 instead of
  5e-6, losing the late-epoch fine-tuning that the merged baseline relies on.
- **Reusable structural constraint reaffirmed**: under the 30-min cap, any
  capacity-add must be free or near-free on wall-clock. Tanjiro #1545
  asymmetric Q/K's +40% step cost set the same constraint; this PR's +19%
  step cost reconfirms it.
- Pivoting thorfinn to attention/MLP dropout=0.05 — orthogonal to merged
  stoch-depth (block-level), zero compute overhead, standard ViT recipe.

## 2026-05-12 22:55 — PR #1637: Grad-clip max_norm=25 — **MERGED, new baseline**

- Branch: `charliepai2g24h4-askeladd/grad-clip-25`
- Hypothesis: H15 from wave-3 candidate pool. Diagnostic-informed follow-up
  to closed #1529 (`max_norm=1.0`, +5.4% regression). At `max_norm=25`,
  clipping fires on the outlier spikes (training grad norms range 22-110)
  without touching the typical 30-70 norms.

| Metric | This PR | Baseline (#1611) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | **90.294** | 94.217 | **-4.16%** |
| test_avg/mae_surf_p (4-split, NaN-safe) | **81.243** | 84.859 | **-4.26%** |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 109.497 / 98.952 / 69.208 / 83.520 | 114.200 / 102.157 / 73.321 / 87.188 | **-4.12% / -3.14% / -5.61% / -4.21%** |

- **All four val splits improved uniformly** (-3.14% to -5.61%) — exactly as
  the hypothesis predicted ("stable descent helps everywhere"), no
  split-specific direction.
- **Mechanism confirmed by the new `train/last_grad_norm` log**: 14/15 epochs
  had end-of-epoch grad_norm > 25 (clipping active throughout). Largest
  spike: 110.04 at epoch 8. Per-step rate likely higher than the
  per-end-of-epoch rate.
- **Cosine cooldown phase shows the biggest payoff**: the single largest
  epoch-to-epoch val_avg drop (-13.7%) coincides with the only epoch
  where end-of-epoch norm fell below the clip (22.40 at ep12).
- Best epoch at the wall-clock cap (15/15) again — same monotonic-descent
  pattern as the cosine-T_max-15 winner. The model is still improving when
  time runs out.
- Brackets the grad-clip direction: clip=1.0 (#1529, +5.4%) too aggressive,
  clip=25 (#1637, -4.16%) the sweet spot or near it. Natural next:
  bracket at clip=50 to test if pure spike-suppression is sufficient.

## 2026-05-12 22:55 — PR #1636 (alphonse pressure-only log1p) — **CLOSED**

- Branch: `charliepai2g24h4-alphonse/log1p-p-only`
- Hypothesis: H16. Targeted follow-up to closed #1610 (full-target log1p,
  +1.18% regression). Apply log1p ONLY to the pressure channel (the only
  genuinely heavy-tailed channel per #1610's `log_y_std`), keep Ux/Uy raw.

| Metric | This PR | Baseline (post #1611) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | 99.227 | 94.217 | **+5.32%** |
| test_avg/mae_surf_p (4-split) | 88.264 | 84.859 | +4.01% |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 130.667 / 111.589 / 64.848 / 89.803 | 114.200 / 102.157 / 73.321 / 87.188 | **+14.42% / +9.23% / -11.56% / +3.00%** |

- **Channel-attribution theory falsified.** The per-split asymmetry observed
  in full-channel #1610 was preserved AND amplified by pressure-only log1p.
  High-peak splits (single_in_dist, camber_rc) regressed *more* than under
  full log1p; low-peak cruise gained more.
- Mechanism (per the student's writeup): log compression flattens the tail
  relative spacing the model relies on to discriminate extreme-pressure
  samples (raceCar Re up to 5M, |p| up to ~30k). After inverse expm1,
  small relative errors blow up multiplicatively at the tails — which are
  exactly the high-peak splits that dominate val_avg.
- Closed per the explicit PR rubric: "If p-only log1p still regresses, the
  entire log-compression direction is dead on this dataset/metric — we
  close and pivot to other channel-rebalancing ideas (e.g. H17)."
- Pivoting alphonse to H17 (learnable per-channel scale+bias on output)
  — addresses pressure calibration *without* compression.

## 2026-05-12 22:55 — PR #1553 (nezuko Gumbel-Softmax slices, tau=1.0) — **CLOSED**

- Branch: `charliepai2g24h4-nezuko/gumbel-slice`
- Hypothesis: replace softmax over slice weights with Gumbel-Softmax during
  training (deterministic softmax at eval) to sharpen slice assignments
  and attack the slice-collapse failure mode.
- Note: branch was never rebased onto the current baseline (still on the
  pre-#1552/#1611 base). Student reported three runs with mean ± std.

| Run | val_avg/mae_surf_p | Δ vs old base (100.957) | Δ vs current base (90.294) |
|-----|---:|---:|---:|
| ...-201404 | 102.827 | +1.85% | +13.89% |
| ...-205438 | 109.970 | +8.93% | +21.79% |
| ...-215442 (canonical) | 103.490 | +2.51% | +14.62% |
| **Mean ± std** | **105.43 ± 3.16** | **+4.43%** | **+16.77%** |

- **Hypothesis falsified across 3 independent runs.** Variance (~3 MAE units)
  rules out single-seed effects; all 3 underperform even the old L1 baseline.
- Failure mode (per student diagnostic): Gumbel sampling noise slows early
  convergence enough that the 30-min cap binds before the model reaches
  the deterministic baseline's asymptote. The eval-time deterministic
  softmax can't recover because the slice weights were trained against
  noisy targets.
- Closed not just because of the negative result on the old base, but
  because the current baseline stack (stoch-depth + cosine T_max=15 +
  grad-clip max_norm=25) is *mechanistically antagonistic* to additional
  gradient noise: stoch-depth already adds variance via block drop, cosine
  cooldown relies on stable gradients in the late phase, and grad-clip
  suppresses spikes that Gumbel noise would create. Layering Gumbel on
  top would worsen, not improve, the gap.
- Pivoting nezuko to H12 (per-node adaptive temperature) — different attack
  on slice-collapse that *doesn't* inject sampling noise.

## 2026-05-12 22:50 — PR #1553 (nezuko Gumbel-Softmax slices) — **SENT BACK** for rebase + re-run

- Branch: `charliepai2g24h4-nezuko/gumbel-slice` (still at `bc30b0a` — pre-#1552, pre-#1611)
- WIP for ~3h with zero commits beyond the original assignment. Pod GPU
  showed a single ~30-min training window (22:00-22:30Z @ 99%/71GB), then
  back to 0% with no artifacts pushed. Likely combination of training
  completing but the post-run commit/push blocked by GH API rate limit
  errors in the student pod's polling loop.
- Even on a successful completion, the result would have been measured
  against the pre-#1552 baseline (val_avg=100.957), not the current 94.217.
  The Gumbel-Softmax slice-collapse hypothesis is still genuinely worth
  testing — it's mechanistically orthogonal to stoch-depth and cosine LR.
- Sent back with explicit rebase + re-run + commit-artifacts directive.
  See PR #1553 comment chain.

## 2026-05-12 21:17 — PR #1611: Cosine T_max=15 alignment — **MERGED, new baseline**

- Branch: `charliepai2g24h4-askeladd/cosine-tmax-15`
- Hypothesis: H14 from wave-2 candidate pool. Change `T_max=MAX_EPOCHS=50` to
  `T_max=15` so the cosine LR decay completes over the actual training
  horizon (~13-15 epochs under the 30-min cap), instead of being ~30% complete
  with LR still at ~80% of peak when training terminates.

| Metric | This PR | Baseline (#1552) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | **94.217** | 98.353 | **-4.21% (largest wave-2 gain)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | **84.859** | 87.995 | **-3.57%** |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 114.200 / 102.157 / 73.321 / 87.188 | 119.16 / 111.09 / 73.32 / 89.84 | **-4.16% / -8.04% / -0.00% / -2.95%** |

- **All four val splits neutral-to-positive** — every split improved or stayed flat.
  camber_rc had the biggest gain (-8.04%); cruise was the only flat one (was already
  the easiest split at 73.32 in #1552, hard to push lower).
- **LR trace confirmed the mechanism**: epoch 1 LR = 4.945e-4, epoch 14 = 5.463e-6,
  epoch 15 = 0.0. The full cosine cooldown phase now happens — under the old
  `T_max=50` setting, LR at epoch 15 was still ~4.0e-4 (80% of peak),
  i.e. the model never entered the fine-tuning phase.
- **Val MAE descended monotonically every epoch** — still improving at the
  wall-clock cap. The cooldown helps without pulling the optimum forward.
- **Action: MERGED** as new canonical baseline. Single-line change, zero added
  compute, zero added params. Subsequent PRs are now compared against 94.217 /
  84.859. The two pending rebase PRs (edward #1548 Fourier, fern #1549 FiLM)
  and the strong wave-2 EMA result (frieren #1608, val_avg=95.761) are all
  affected by this baseline shift and need re-evaluation.

## 2026-05-12 21:17 — PR #1608: EMA of model weights (decay=0.999) — **REQUEST CHANGES** (sent back to frieren for rebase onto new cosine baseline)

- Branch: `charliepai2g24h4-frieren/ema-weights-0.999`
- Hypothesis: H13 from wave-2 pool. Exponential moving average of model weights;
  validate and checkpoint using the EMA copy.

| Metric | This PR | Baseline (#1552) | Δ vs #1552 | New baseline (#1611) | Δ vs #1611 |
|---|---:|---:|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15) | 95.761 | 98.353 | **-2.64%** | 94.217 | **+1.64% (worse vs new)** |
| test_avg/mae_surf_p (4-split) | 85.286 | 87.995 | -3.08% | 84.859 | +0.50% |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 115.69 / 107.67 / 71.88 / 87.81 | 119.16 / 111.09 / 73.32 / 89.84 | -2.91% / -3.08% / -1.96% / -2.26% | 114.20 / 102.16 / 73.32 / 87.19 | +1.30% / +5.39% / -1.96% / +0.71% |

- **Mechanism worked.** All four splits improved vs the #1552 base, monotonic
  val descent every epoch. Implementation correct: EMA swap-in for val, live
  weights restored after, EMA `state_dict()` saved to checkpoint, test load
  picks up EMA weights automatically.
- **But the #1611 cosine merge shifted the baseline.** EMA's run was on
  T_max=50 base; the new baseline has T_max=15. Standalone EMA gain (-2.64%)
  no longer beats the new baseline.
- **Action: SENT BACK with rebase spec.** EMA and cosine T_max alignment are
  mechanistically orthogonal (different optimizer-trajectory variance vs
  LR-schedule shape), so they should stack. Expected stacked val_avg: ~92-93
  if EMA's -2.64% effect carries over to the new base. Re-evaluate after rebase.

## 2026-05-12 21:17 — PR #1549: FiLM conditioning on global flow params — **REQUEST CHANGES** (sent back to fern for rebase — extraordinary signal)

- Branch: `charliepai2g24h4-fern/film-global-cond`
- Hypothesis: H10 from round-2 list. FiLM (Feature-wise Linear Modulation) of
  per-block features by global flow parameters (Reynolds, AoA, etc. from
  metadata). Bug fix in conditioning extraction: use node-0 instead of mean-
  pool over padded zeros (which collapsed the conditioning signal).

| Metric | This PR | L1 baseline (#1397) | Δ vs L1 | Current baseline (#1611) | Δ vs current |
|---|---:|---:|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 13) | **81.291** | 100.957 | **-19.5%** | 94.217 | **-13.7% (huge gap)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | **71.731** | NaN | first finite | 84.859 | **-15.5%** |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 94.72 / 103.94 / 52.13 / 74.38 | 127.37 / 110.83 / 77.35 / 88.27 | -25.6% / -6.2% / -32.6% / -15.7% | 114.20 / 102.16 / 73.32 / 87.19 | -17.1% / +1.7% / -28.9% / -14.7% |
| n_params | 677,719 | 660,000 | +2.7% | 662,359 | +2.3% |

- **Largest single-experiment signal of round 2 by a wide margin.** Beats the
  L1-only baseline by 19.5% on val, beats the current (cosine+stoch-depth)
  baseline by 13.7% **even without those two improvements stacked**. cruise
  OOD split dropped 73.32 → 52.13 (-29%) — exactly the regime where global
  flow params (Re, AoA) should carry the most information.
- **Caveat: no stoch-depth, no cosine T_max=15.** fern's branch is older than
  both #1552 and #1611. The 13.7% gap suggests FiLM is doing dominant
  conditioning work, but the comparison can't be confirmed without stacking
  on the full current baseline.
- **Action: SENT BACK with rebase spec.** This is the **top-priority pending
  rebase** of the round. If FiLM + stoch-depth + cosine all stack, projected
  val_avg lands in the 78-84 range — a massive new baseline. If interference
  is severe, we choose between FiLM-only and the current baseline; the
  FiLM-only result (81.291) would still be a -13.7% improvement.

## 2026-05-12 21:17 — PR #1610: log1p target reparameterization (H11) — **CLOSED**

- Branch: `charliepai2g24h4-alphonse/log1p-target`
- Hypothesis: H11. Sign-preserving log1p of the target across all 3 channels
  (Ux, Uy, p), inverse-transform at metric time. Compresses heavy-tailed
  distribution and rebalances per-sample gradient magnitude.

| Metric | This PR | Baseline (#1552) | Δ vs #1552 | New baseline (#1611) |
|---|---:|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 13) | 99.513 | 98.353 | **+1.18% (regression)** | 94.217 (+5.62% vs new) |
| test_avg/mae_surf_p (4-split) | 89.586 | 87.995 | +1.81% | 84.859 |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 125.01 / 114.83 / 69.34 / 88.87 | 119.16 / 111.09 / 73.32 / 89.84 | +4.91% / +3.37% / -5.43% / -1.08% | — |

- **Diagnostic value is high.** log1p helps the lower-peak splits (cruise -5.43%,
  re_rand -1.08%) but hurts the high-peak ones (single_in_dist +4.91%,
  camber_rc +3.37%). The pressure channel's log_y_std=4.64 is ~4× the
  other two channels (1.12, 1.53) — it's the only heavy-tailed channel, and
  full log-compression flattens the surface stagnation peaks that
  `mae_surf_p` rewards.
- **Implementation was correct**: signed_log1p / signed_expm1 wired properly,
  stats recomputed on log-space targets, sanity checks pass (epoch-1
  surf_loss=1.28 in log space as expected, physical-unit MAE reported
  correctly at O(100)).
- **Action: CLOSED**. Pressure-only log1p (H16) is the natural targeted variant
  and is being assigned as alphonse's next hypothesis — the heavy-tailed
  channel that benefits from compression is isolated, while Ux/Uy stay in
  physical units.

## 2026-05-12 21:13 — PR #1548: Fourier coord encoding (L=4) — **REQUEST CHANGES** (sent back to edward for rebase onto stoch-depth baseline)

- Branch: `charliepai2g24h4-edward/fourier-coords-L4`
- Hypothesis: H7 from round-2 list. Add Fourier positional encoding to the (x,z)
  coords with L=4 frequency bands. Captures geometric structure that raw coords miss.

| Metric | This PR | L1 baseline (#1397) | Current baseline (#1552) |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14/15) | **92.053** | 100.957 | 98.353 |
| Δ vs L1-only | **-8.82%** | — | -2.58% |
| Δ vs current best | **-6.40%** (numerical) | +2.65% | — |
| test_avg/mae_surf_p (4-split) | 83.980 | NaN | 87.995 |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 106.553 / 102.895 / 71.689 / 87.076 | 127.371 / 110.832 / 77.353 / 88.273 | 119.16 / 111.09 / 73.32 / 89.84 |
| n_params | 665,943 | 660,000 | 662,359 |

- **Strongest single-experiment signal of round 2.** Every val split improves vs
  the L1-only baseline by 1.4% to 16.4%, with the biggest gain on val_single_in_dist
  (-16.3%) — exactly the split the merged stoch-depth baseline only partially fixed.
  Test 4-split (83.980) also better than current baseline (87.995).
- **Caveat: train.py is missing the stoch-depth code from #1552.** Edward's branch
  is 8 commits behind advisor base; no `stoch_depth_prob` anywhere. So the
  comparison is Fourier-without-stoch-depth (92.053) vs stoch-depth-without-Fourier
  (98.353). We don't yet know whether the two stack (likely lands ~89-90 → huge
  win) or interfere (could regress back toward ~95).
- **Action: SENT BACK with rebase spec.** Edward to pull current advisor HEAD
  (which includes both stoch-depth and the NaN-safe pre-filter) and re-run with
  Fourier encoding on top. Expected outcomes flagged in the PR comment:
  stacks (clear merge), partial interference (still likely merge as Fourier-dominant),
  severe interference (we then choose between Fourier-only and stoch-depth-only).
- This is the highest-EV in-flight signal — wave 2 results may need to be re-evaluated
  against a Fourier+stoch-depth baseline once edward's rebase lands.

## 2026-05-12 21:00 — PR #1555: Remove `in_project_fx` (Transolver++ tied projection) — **REQUEST CHANGES** (sent back to thorfinn for n_hidden=144 follow-up)

- Branch: `charliepai2g24h4-thorfinn/remove-in-project-fx`
- Hypothesis: H3 from round-2 list. Remove redundant `in_project_fx` from
  `PhysicsAttention`, re-using `x_mid` as the value source in the slice-pooling
  einsum (Transolver++, arXiv 2502.02414). Acts as a structural prior + frees VRAM.
- Run was on L1+stoch-depth base (post-#1552), so direct apples-to-apples vs current baseline.

| Metric | This PR | Baseline (#1552) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14/15) | 99.898 | 98.353 | **+1.57% (slightly worse)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | 89.532 | 87.995 | +1.75% |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 129.395 / 108.141 / 72.436 / 89.619 | 119.16 / 111.09 / 73.32 / 89.84 | **+8.60% / -2.65% / -1.21% / -0.25%** |
| n_params | 579,799 | 662,359 | **-12.5%** |
| Peak GPU memory | 39.63 GB | 42.11 GB | **-5.9%** |
| Wall time/epoch | ~125 s | ~123 s | ~unchanged |

- **Pattern is a classic capacity-vs-regularization tradeoff.** The three OOD-flavored
  splits (camber_rc, camber_cruise, re_rand) all improve modestly; single_in_dist
  regresses by +8.6%, pulling val_avg net negative. The tied projection acts as a
  structural regularizer that helps OOD but underfits the in-distribution mode.
  Efficiency gains are real: -12.5% params and -5.9% VRAM at identical wall time.
- **Action: sent back with re-tune spec** — keep the tied projection, but reinvest
  the freed parameter budget (~83k params) and VRAM headroom by widening
  `n_hidden=128 → 144`. This redistributes capacity across all weights rather than
  concentrating it in a single redundant projection. Expected: single_in_dist
  recovers toward 119, OOD gains preserved → net improvement vs 98.353. Student
  must rebase onto current HEAD to include the merged stoch-depth code.

## 2026-05-12 21:00 — PR #1514: Ada-Temp v2 (shared-across-heads Δτ) — **CLOSED**

- Branch: `charliepai2g24h4-alphonse/ada-temp` (v2 force-push)
- Hypothesis: v2 follow-up to test alphonse's own diagnosis that extra per-head
  Δτ capacity hurt cross-regime transfer. v2 uses `Linear(dim, 1)` (shared-heads).
- Run was on L1-only base (pre-#1552), so compared against 100.957 not 98.353.

| Metric | v2 | L1 baseline (#1397) | v1 (per-head) | Current baseline (#1552) |
|---|---:|---:|---:|---:|
| val_avg/mae_surf_p | 104.366 | 100.957 | 101.770 | 98.353 |
| Δ vs L1 baseline | **+3.4% (worse)** | — | +0.81% | -2.58% |
| Δ vs current best | **+6.1% (worse)** | +2.65% | +3.47% | — |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 122.77 / 114.35 / 85.47 / 94.88 | 127.37 / 110.83 / 77.35 / 88.27 | 118.02 / 114.13 / 78.35 / 96.58 | 119.16 / 111.09 / 73.32 / 89.84 |
| Δ vs L1 per-split | -4.60 / +3.51 / **+8.12** / +6.60 | — | -9.35 / +3.30 / +1.00 / +8.31 | — |
| test_avg/mae_surf_p (4-split, NaN-safe) | 93.936 | NaN | NaN | 87.995 |

- **Both Ada-Temp variants are now exhausted.** v1 (per-head) regressed by +0.81%;
  v2 (shared-heads) regresses harder by +3.4% on val_avg.
- **The capacity-overfit hypothesis is partially contradicted.** Shared-heads
  narrowed v1's val_re_rand regression (+8.31 → +6.60) and partially preserved
  v1's val_single_in_dist gain (-9.35 → -4.60). But v2 introduced a new
  large regression on val_geom_camber_cruise (+1.00 → +8.12), which v1 didn't
  have. Removing per-head freedom collapses head specialization on the cruise
  regime that needed it most.
- **Action: CLOSED.** The NaN-safe pre-filter from this PR was independently
  preserved via #1552 (now standard in baseline). Slice-collapse is also being
  attacked via a different mechanism in #1553 (Gumbel-Softmax, WIP under nezuko).
  Alphonse's suggested follow-up (Eidetic Slice Embedding) goes on the
  wave-3 candidate pile for later revival if Gumbel-Softmax doesn't pan out.

## 2026-05-12 21:00 — PR #1547: Kendall uncertainty weighting — **CLOSED**

- Branch: `charliepai2g24h4-askeladd/kendall-uncertainty`
- Hypothesis: H6 from round-2 list. Replace hand-tuned `surf_weight=10` with
  learnable per-task log-sigmas (Kendall et al., CVPR 2018) so the surf/vol
  balance becomes data-driven.

| Metric | This PR | Baseline (#1552) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | 103.544 | 98.353 | **+5.28% (worse)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | 94.524 | 87.995 | **+7.42% (worse)** |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 132.989 / 115.872 / 73.899 / 91.417 | 119.16 / 111.09 / 73.32 / 89.84 | +11.6% / +4.3% / +0.8% / +1.8% |
| Learned `log_sigma_surf` / `log_sigma_vol` | -0.288 / -0.079 | — | — |
| **Effective `surf_weight`** | **1.518** | 10.0 | -85% lower than hand-tuned |

- **Key diagnostic finding: the Kendall MLE objective is fundamentally misaligned
  with the physical evaluation metric.** Learned sigmas converged to
  effective_surf_weight=1.518, ~7× lower than the hand-tuned value of 10 that
  the baseline uses. Cross-referencing closed PRs #1403 (surf_weight=30, +5.1%
  worse) and #1530 (effective surf×P_WEIGHT=30, +1.22% worse), the empirical
  optimum for surf_weight is at or near 10, and learnable per-task likelihood
  pulls it the wrong way.
- **Lesson: learnable loss-balance objectives must align with the physical
  eval metric, not just calibrated likelihoods.** This rules out the entire
  family of MLE-style balance learning (Kendall, GradNorm, dynamic weight
  averaging) unless they're constrained to optimize the evaluation surrogate
  directly.
- **Action: CLOSED.** Clean negative result. No reasonable variant of the
  Kendall objective recovers the gap; the objective is the problem, not the
  parameterization.

## 2026-05-12 21:00 — PR #1545: Asymmetric Q/K slice projections (LinearNO) — **CLOSED**

- Branch: `charliepai2g24h4-tanjiro/asymmetric-qk`
- Hypothesis: H2 from round-2 list. Independent V and K slice projections in
  PhysicsAttention (LinearNO-style) — separate the slice-assignment basis from
  the value basis to enable richer slice tokens.

| Metric | This PR | Baseline (#1552) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 10) | 116.940 | 98.353 | **+18.90% (worse)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | 105.058 | 87.995 | **+19.39% (worse)** |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 141.506 / 132.312 / 86.757 / 107.185 | 119.16 / 111.09 / 73.32 / 89.84 | +18.7% / +19.1% / +18.3% / +19.3% |
| n_params | 672,919 | 662,999 | +9,920 (+1.5%) |
| Epochs reached in 30-min cap | **10** | 15 | **-33%** |

- **Compute-bound failure mode.** The mechanism is empirically active (block-3
  slice cos-sim = 0.097 confirms slice divergence), but the extra `in_project_slice_k`
  projection adds ~40% wall-clock cost per epoch. Run terminated at epoch 10
  vs the baseline's 15 — same compute budget, fewer effective gradient steps.
- The trajectory was still descending at termination but needed ~17 additional
  MAE points of improvement to match baseline, which is implausible in the
  remaining 5 epochs even with monotonic descent.
- **Structural lesson: architectural changes that add >10% per-step compute
  are unviable in our 30-min training regime, even when the mechanism is
  theoretically sound.** Future architectural changes must be parameter-additions,
  not compute-additions, OR be paired with a complementary efficiency-saving
  (e.g., the tied-projection direction that thorfinn is iterating on in #1555).
- **Action: CLOSED.** Direction is dead within current budget constraints;
  asymmetric Q/K could only be re-attempted at higher budget or paired with a
  compute-saving change.

## 2026-05-12 20:52 — PR #1552: Stochastic depth (drop_rate=0.1, linear schedule) — **MERGED, new baseline**

- Branch: `charliepai2g24h4-frieren/stoch-depth-0.1`
- Hypothesis: H8 from round-2 list. Add stochastic depth (Huang et al., ECCV 2016)
  with linearly increasing per-block drop probs `[0.0, 0.025, 0.05, 0.075, 0.10]`.
  Implicit ensemble of shallower networks for OOD regularization. No-op at eval.
  Predicted 1-3% improvement on `val_avg/mae_surf_p`, primarily via OOD geometry splits.
- Also includes the NaN-safe pre-filter in `evaluate_split` (standardized in every
  round-2 PR after #1530/#1529 independently discovered it).

| Metric | This PR | L1 baseline (#1397) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | **98.353** | 100.957 | **-2.58% (improvement)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | **87.995** | NaN (data bug) | **first finite 4-split ref** |
| test_avg/mae_surf_p (3-split, ex-cruise) | 96.579 | 100.831 | -4.22% |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 119.159 / 111.093 / 73.323 / 89.837 | 127.371 / 110.832 / 77.353 / 88.273 | **-6.45% / +0.24% / -5.21% / +1.77%** |
| Per-split test: single_in_dist / camber_rc / camber_cruise / re_rand | 104.953 / 101.883 / 62.243 / 82.901 | — | new finite ref |

- **The hypothesis held, but the OOD-specific framing was only half-supported.**
  Predicted gains were on OOD geometry splits (camber_rc, camber_cruise).
  Observed: camber_cruise -5.21% (large), camber_rc +0.24% (flat),
  single_in_dist -6.45% (largest gain), re_rand +1.77% (small regression).
  Student's reading: single_in_dist was the worst split at baseline despite
  being in-distribution, so it had the most regularization headroom.
  Stoch-depth's implicit ensemble flattens split-specific overfit modes
  regardless of the OOD axis.
- **Training dynamics:** val trace is noisier than L1 baseline (epoch 13: 105.69
  → epoch 14: 113.91 → epoch 15: 98.35 = new best). Bernoulli-block-drop noise
  injects variance into val. Best epoch landed at the wall-clock cap; more
  training time would likely extend the gain. The L1 baseline plateaued earlier
  at the same wall-clock budget, so stoch-depth is also getting more out of
  each minute of training.
- **Cosmetic NaN caveat:** loss/surf_loss aggregates for `test_geom_camber_cruise`
  still show NaN/Inf in `metrics.yaml` because the normalized-space loss path
  runs before the §3 pre-filter; the §3 fix only protects `accumulate_batch`.
  All four `mae_surf_p`/`mae_vol_p` channels are finite, so the primary ranking
  metric is clean. Out of scope; one-line follow-up.
- **Decision: MERGED.** First post-L1 architectural improvement; -2.58% on the
  primary metric and establishes the first finite 4-split test reference
  (87.995). Stoch-depth is now part of the canonical config; all subsequent
  wave-1 PRs in flight will be compared to this stronger baseline.
- **Suggested follow-ups (student):**
  1. Run longer — not actionable (`SENPAI_TIMEOUT_MINUTES` is a hard bound).
  2. Sweep `drop_rate` ∈ {0.05, 0.15, 0.20} — 0.05 might be Pareto-better given
     val_re_rand +1.77% suggests slight over-regularization; 0.15-0.20 might
     bite harder on val_geom_camber_rc which barely moved.
  3. Combine with `dropout` inside PhysicsAttention/MLP at 0.05 — standard
     ViT recipe, may compound with stoch-depth.
  4. Loss-NaN cosmetic fix — pre-filter finite samples before `y_norm` is
     formed so the normalized-space loss aggregates report finite numbers
     for `test_geom_camber_cruise`.

## 2026-05-12 20:02 — PR #1514: Ada-Temp per-point adaptive slice temperature — **REQUEST CHANGES** (sent back to alphonse for v2)

- Branch: `charliepai2g24h4-alphonse/ada-temp`
- Hypothesis: H1 from round-2 list. Replace scalar `self.temperature` with
  `τᵢ = τ₀ + Linear(dim, heads)(xᵢ)`, zero-init the projection so the model
  starts identical to baseline (Transolver++, arXiv 2502.02414).

| Metric | This PR | L1 baseline (#1397) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 13/14) | 101.770 | 100.957 | +0.81% (slightly worse) |
| test_avg/mae_surf_p (3-split, ex-cruise) | 100.825 | 100.831 | -0.007 (effectively flat) |
| test_avg/mae_surf_p (4-split) | NaN (no NaN-safe fix in v1) | NaN | — |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 118.023 / 114.128 / 78.348 / 96.582 | 127.371 / 110.832 / 77.353 / 88.273 | **-9.3 / +3.3 / +1.0 / +8.3** |

- **Per-split signal is the key story.** Ada-Temp helps single-foil in-distribution
  by ~9.3 (~7.3% gain) but regresses on val_re_rand by ~8.3 (~9.4% loss). The
  geometry-camber splits drift slightly worse. Net val_avg is essentially flat
  (slight regression) and test 3-split mean is statistically indistinguishable.
- **Implementation contribution worth recording**: alphonse identified that
  `Transolver.__init__` calls `self.apply(self._init_weights)` *after* `temp_proj`
  is zero-initialized, and `_init_weights` re-initializes every `nn.Linear` with
  `trunc_normal_(std=0.02)`. This silently breaks the "Δτ = 0 at step 0" invariant.
  Fix: re-zero loop after `self.apply(...)`. Without the fix an earlier run
  diverged from baseline from epoch 1. The committed run is the corrected version.
- **Diagnosis (student): extra per-head Δτ capacity hurts cross-regime transfer**
  inside a 30-min wall-clock budget. Single-foil in-dist benefits from sharper
  slice attention; tandem-flow OOD distributions cannot afford the extra
  capacity that lets the temperature head co-adapt to training-set spurious cues.
- **Action: sent back with v2 spec** — drop `temp_proj` from `Linear(dim, heads)`
  to `Linear(dim, 1)` (shared-across-heads Δτ), which cuts Ada-Temp's added
  capacity by ~75% (2,580 → 645 params). Direct test of the student's own
  capacity-overfit hypothesis. Also adds the NaN-safe pre-filter so v2 will
  report a finite 4-split test mean. Student suggested 4 follow-ups; v2 picks
  #2 (shared-across-heads), with #3 (last-blocks-only) as a stack-on if v2
  partially works and #4 (combine with Eidetic Slice Embedding) as a
  wave-3 candidate if v2 fails. Suggestion #1 (length-budgeted retest)
  is not actionable (`SENPAI_TIMEOUT_MINUTES` is a hard bound).

## 2026-05-12 19:55 — Stale-WIP closures: 5 PRs branched off pre-L1 MSE base

Five round-1 PRs (#1407 wider/deeper, #1411 slice_num=128, #1417 lr-warmup=1e-3,
#1420 EMA weights, #1425 SwiGLU FFN) were assigned at 17:52 UTC, before L1 loss
(PR #1397) merged at 19:05. Student pods were stalled on GH API rate limits
through 19:50 and never started training. Closing because any result on those
branches would be measured against pre-L1 MSE base and not directly comparable
to the new L1 baseline. All five hypotheses remain valid avenues to revive
in a later round; they are deprioritized for round 2 in favour of architecture
and loss-formulation ideas from `RESEARCH_IDEAS_2026-05-12_round2.md`.

## 2026-05-12 19:50 — PR #1530: Per-channel L1 loss with pressure x3 weight — **CLOSED, worse than L1**

- Branch: `charliepai2g24h4-tanjiro/channel-weight-p3`
- Hypothesis: H4 from round-2 list. In L1 loss, multiply pressure channel by
  P_WEIGHT=3.0 to steer gradient flow toward the ranking metric `mae_surf_p`.
  Predicted 2-6% improvement on `val_avg/mae_surf_p`.

| Metric | This PR | L1 baseline (#1397) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14/14) | 102.184 | 100.957 | **+1.22% (worse)** |
| test_avg/mae_surf_p (3-split, ex-cruise) | 100.696 | 100.831 | -0.13% |
| test_avg/mae_surf_p (**4-split, NaN-safe, new finite ref**) | **92.465** | NaN | — |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 126.233 / 112.645 / 77.502 / 92.356 | 127.371 / 110.832 / 77.353 / 88.273 | mostly noise except +4.6% on re_rand |

- Effective combined surface-pressure weight became `surf_weight × P_WEIGHT = 30`,
  the same regime as closed PR #1403 (`surf_weight=30`), which also regressed.
  Student diagnosed this directly. The 3× upweight is too aggressive on top
  of L1's already-amplified surface gradients.
- **Lasting deliverable:** the NaN-safe pre-filter in `train.py::evaluate_split`
  works as designed and produced the first finite 4-split test mean on this
  branch (92.465). Pre-filter pattern is now bundled into every round-2 PR
  assignment so subsequent runs land a comparable 4-split test reference.
- Per-channel surface MAE at best val: surf_Ux=1.43, surf_Uy=0.69, surf_p=102.18;
  vol_Ux=4.81, vol_Uy=2.22, vol_p=103.80. Predicted Ux/Uy uptick in exchange for
  p drop did NOT materialize — we got a p regression instead.
- Suggested follow-ups (lower P_WEIGHT, combined surf_weight+P_WEIGHT sweep)
  are deferred until higher-EV round-2 levers are explored.

## 2026-05-12 19:48 — PR #1529: Gradient clipping (max_norm=1.0) — **CLOSED, much worse than L1**

- Branch: `charliepai2g24h4-askeladd/grad-clip-1.0`
- Hypothesis: H5 from round-2 list. Add `clip_grad_norm_(max_norm=1.0)` to
  reduce variance from gradient spikes on variable mesh sizes / high-Re samples.
  Predicted 1-4% improvement on `val_avg/mae_surf_p` via smoother convergence.

| Metric | This PR | L1 baseline (#1397) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 11/14) | 106.401 | 100.957 | **+5.4% (worse)** |
| test_avg/mae_surf_p (3-split, ex-cruise) | 103.364 | 100.831 | +2.5% |
| test_avg/mae_surf_p (**4-split, NaN-safe, finite ref**) | **94.846** | NaN | — |

- Student logged per-epoch gradient norms (min 10, mean 47, max 245) and clip%
  per epoch (100% in every epoch). `max_norm=1.0` is far below the natural
  pre-clip norm of 10-245, so every step was rescaled by 0.02-0.10× — the
  model effectively trained at 1-5% of the configured LR throughout, which
  is too slow to converge inside 14 epochs.
- The diagnosis is exemplary post-hoc analysis and exactly the kind of
  per-epoch instrumentation we want from every arm.
- **Lasting deliverable:** NaN-safe pre-filter in `evaluate_split` (identical
  to tanjiro's #1530 fix). 4-split test mean (94.846) is reproducible and
  finite. Workaround is now standard in all round-2 assignments.
- Suggested follow-ups (`max_norm ∈ {10, 25, 50}`, AGC) deferred — higher-EV
  round-2 hypotheses have priority. If architecture/loss levers stall, we
  will return to AGC.

## 2026-05-12 19:10 — PR #1423: Enable unified_pos=True with ref=8 — **CLOSED, worse than L1**

- Branch: `charliepai2g24h4-tanjiro/unified-pos`
- Hypothesis: Switch `unified_pos=False → True, ref=8` — add learned ref-grid
  positional features (Gaussian-RBF over an 8×8 grid in the (x, z) plane,
  repeat-interleaved to fill `ref**3 = 512`) before the preprocess MLP.
- Student noted real implementation concerns: `ref**3 = 512` packing inflates 2D
  features 8× (only 64 distinct grid cells); grid bounds were adjusted to
  `[-7, 7]` to match the actual data range. Proposed multi-scale RBFs and
  asymmetric per-axis grid bounds as round-2 follow-ups.

| Metric | Value |
|---|---:|
| val_avg/mae_surf_p (best @ ep 14/14) | 118.605 |
| test_avg/mae_surf_p (4-split, NaN-safe) | 109.159 |
| L1 baseline (PR #1397) | 100.957 / 100.831 |
| Delta vs L1 baseline | +17.5% **worse** |

- Note: trained on MSE base (branched before L1 merge), so this is MSE+unified_pos
  rather than L1+unified_pos. Comparison is contaminated. Closed without rebase
  because (a) the absolute number is ~17% worse than L1, (b) re-running would
  consume 30 min for a hypothesis whose own author flagged implementation
  concerns, and (c) higher-EV ideas are queued. Multi-scale RBF variant may
  resurface later.
- Student also flagged the pre-existing `data/scoring.py` NaN-propagation bug
  (same one alphonse flagged in #1397) and committed a clean workaround in
  `evaluate_split`. We're propagating that workaround into all subsequent
  round-2 assignments.

## 2026-05-12 19:08 — PR #1403: Bump surf_weight 10 → 30 — **CLOSED, worse than L1**

- Branch: `charliepai2g24h4-askeladd/surf-weight-30`
- Hypothesis: Increase `surf_weight` from 10 → 30 to focus optimizer pressure on
  the surface field that drives the primary metric.

| Metric | Value |
|---|---:|
| val_avg/mae_surf_p (best @ ep 12/14) | 133.386 |
| test_avg/mae_surf_p (4-split, NaN-safe re-eval) | 120.962 |
| L1 baseline (PR #1397) | 100.957 / 100.831 |
| Delta vs L1 baseline | +32.1% **worse** |

- Trained on MSE base (branched before L1 merge), so this is MSE+surf_weight=30.
  Under L1 (less outlier-sensitive than MSE) the optimal `surf_weight` is unlikely
  to be larger than the default 10. Closed without rebase: re-running would burn
  30 min on a single-value HP sweep when L1 already wins by 30%+. A proper
  L1+surf_weight sweep (10/15/25/50) is a small follow-up worth considering only
  if other levers stop moving.
- Student also flagged the pre-existing `data/scoring.py` NaN bug and produced an
  independent NaN-safe re-evaluation script. Confirmed root cause.

## 2026-05-12 19:05 — PR #1397: L1 (MAE) loss replaces MSE in normalized-space training — **MERGED, new baseline**

- Branch: `charliepai2g24h4-alphonse/l1-loss`
- Hypothesis: Align training loss with the eval metric (MAE). MSE
  over-weighted high-Re outlier nodes whose y range spans up to 29K with
  per-sample y std varying ~10× within a single split. Expected 2–8%
  improvement on `val_avg/mae_surf_p`.
- Implementation: `(pred - y_norm).abs()` replaces `(pred - y_norm)**2` in
  both the training inner loop and `evaluate_split`. Surface/volume
  decomposition and `surf_weight = 10.0` kept unchanged. All other HPs
  default.

| Metric | Value |
|---|---:|
| val_avg/mae_surf_p (best @ ep 13/14) | **100.9574** |
| test_avg/mae_surf_p (3-split, excl. cruise) | **100.8314** |
| test_avg/mae_surf_p (4-split, raw) | NaN (data bug) |
| val_single_in_dist / mae_surf_p | 127.371 |
| val_geom_camber_rc / mae_surf_p | 110.832 |
| val_geom_camber_cruise / mae_surf_p | 77.353 |
| val_re_rand / mae_surf_p | 88.273 |
| n_params | 0.66 M |
| peak GPU mem | 42.1 GB |
| wall time | 30.7 min (cut at SENPAI_TIMEOUT_MINUTES=30 after ep 14) |

- Metric artifacts (advisor branch): `models/model-charliepai2g24h4-alphonse-l1-loss-20260512-175404/metrics.jsonl`, `metrics.yaml`
- Training trajectory was monotone-descending: ep 1 223 → ep 13 101; ep 14
  bounced to 134 right before timeout. Cosine T_max=50 means LR only
  decayed ~16% from peak by ep 14 — schedule is mismatched to the 30-min
  wall-clock cap. Worth a follow-up arm.

### Conclusions and follow-ups

- L1 loss is a clear win and establishes the first numeric baseline on
  this advisor branch. Merged.
- Pre-existing data bug: `test_geom_camber_cruise/000020.pt` contains
  `inf` in y_p, propagating NaN through `data/scoring.py::accumulate_batch`
  even though the bad sample is correctly flagged. `data/scoring.py` is
  marked read-only, so we record the 3-split test mean and document the
  bug. Fix candidate for a later PR: in `train.py::evaluate_split`, pre-mask
  non-finite y samples by zeroing both the sample's `mask` and its y
  values before calling `accumulate_batch` (faithful trainer-side
  workaround that preserves the scoring contract).
- Round-2 candidate follow-ups suggested by student (in addition to the
  Round-2 idea file H1-H11): T_max=15 to align cosine with the 30-min
  wall-clock cap; small `surf_weight` sweep on top of L1 (10/15/25/50)
  since L1 is less outlier-dominated than MSE; Huber/SmoothL1 as a
  smooth alternative.


