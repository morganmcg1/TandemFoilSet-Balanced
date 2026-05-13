# SENPAI Research State

- **Date**: 2026-05-13 05:45 UTC
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r3` (base `icml-appendix-charlie`)
- **Research tag**: `charlie-pai2g-24h-r3`
- **Students (8)**: charliepai2g24h3-{alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn}
- **Per-run budget**: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50` (caps)
- **Logging**: local JSONL only (no W&B in this arm)

## Latest human direction

None received.

## Current best baseline

**val_avg/mae_surf_p = 88.175** — PR #1662 (nezuko, Fourier PE surface-only L=4 on merged stack).
Test 4-split safe re-eval = **83.362** (−2.63% vs PR #1745 baseline 85.611).
Stack: `grad_clip=1.0 + wd=1e-3 + augment + cosine T_max=14 + EMA=0.999 + huber_delta=0.5 + surf_weight_warmup_epochs=5 + surf_weight_init=1.0 + surf_weight=20.0 + pos_freq_bands=4 + pos_freq_surface_only=True`.
**Per-split val:** single=104.911, rc=99.544, **cruise=64.603** (−9.21% — largest gain, super-additive!), re_rand=83.642.
**Per-split test:** single=93.872, rc=86.766, cruise=74.616, re_rand=78.192.
**Key finding:** Fourier PE + Huber + curriculum compose super-additively on camber_cruise. Surface-only L=4 Fourier features expose boundary-layer high-frequency structure; Huber stabilises per-node gradient distribution; curriculum's surf_weight ramp focuses optimisation on surface nodes. All three target the same surface-pressure-on-OOD-geometry objective and reinforce on cruise (positive but sub-additive on other splits). Strongest bottleneck remaining: `val_single_in_dist = 104.911`.

**Previous baselines:** #1745 (Huber×curriculum) val 91.507 / test 85.611. #1686 (curriculum) val 97.620 / test 91.947. #1484 (Huber alone) val 99.879 / test 93.596.

**Disproved (closed, mechanistic):** PR #1543 v2 (fern) log-cosh + augment @ 106.93 (+3.71%) / test 100.61 (+6.18%). The v2 − v1 delta ≈ 0 (augmentation added nothing on top of log-cosh, vs +9.4 on MSE) proves log-cosh and augment are SUBSTITUTES, not complements: both target high-Re gradient dominance via different mechanisms. Log-cosh's gradient cap defeats augmentation's purpose on rc split (+13.5% worse, the killer).

**Per-channel pressure weighting disproved (closed):** PR #1488 Arm C (askeladd, surf_weight_p=20 alone, no decoupling) @ val 105.72 (+2.62%) / test 99.29 (+4.53%). All three arms (A, B, C) fail pass criterion. Arm B's 102.12 was single-seed noise from cosine schedule, not signal. Adds to the "weight the loss + diversify the data" substitution principle established by PR #1543.

## Current student assignments

| Student | PR | Slug | Status |
|---|---|---|---|
| alphonse | #1931 | `huber-delta-0p6-0p75-up-sweep` | WIP — Huber δ=0.6 (Arm A) / δ=0.75 (Arm B) on #1745 stack. Tests up-direction after #1869 closed (δ=0.25/0.1 both regressed). Resolves inflection-vs-optimum question; expected to close axis for good. |
| askeladd | #1912 | `per-domain-loss-weighting` | WIP — per-domain LOSS reweighting λ=0.3 (Arm A) / λ=1.0 (Arm B) on racecar_single. Tests whether off-domain regression in #1822 was sampler-specific (sample-exclusion) or gradient-share fundamental (P9). λ=1.0 ≈ matches 2× sampler gradient share in-batch. (#1822 closed — sampling-level both arms regressed val_avg despite single_in_dist −10.5 at 3×, established P9.) |
| edward | #1490 | `scale-model-256-v2` | WIP — rebase: n_hidden=192, n_head=6 on new stack |
| fern | #1935 | `slice-num-down-sweep-32-48` | WIP — slice_num=48 (Arm A) / slice_num=32 (Arm B) on merged stack. Tests opposite direction after #1850 was budget-blocked at slice_num=96/128. Budget-respecting (both arms faster than baseline → full 14/14 epochs). Calibrated curve test. |
| frieren | #1492 | `mlp-ratio-4-wider-ffn` | WIP — rebase: mlp_ratio=4 |
| nezuko | #1955 | `fourier-pe-v4-coord-rescaling` | WIP — per-sample [0,1] bbox coord rescaling before Fourier encoding. Arm A: independent x/y; Arm B: isotropic. Tests whether dataset-normalised coords are sub-optimal vs NeRF-convention bbox normalization. (#1662 merged as new baseline.) |
| tanjiro | #1693 | `swiglu-ffn` | WIP v2 — v1 hit val 87.28 / test 82.24 (−10% vs #1686!) but merge conflicts + pre-#1484/#1686 base. Sent back for rebase + rerun on full merged stack (#1745 now baseline). |
| thorfinn | #1885 | `surf-weight-warmup-3-8-epochs` | WIP — warmup_epochs=3 (Arm A) / warmup_epochs=8 (Arm B) at fixed plateau sw=20. Tests ramp shape decoupled from plateau height. (#1827 closed — sw=30/50 both regressed, established P8.) |

## Research themes and findings

### Confirmed winners (merged)
1. **Optimization hygiene** (PR #1491): grad_clip=1.0 + wd=1e-3 → 115.40.
2. **Scheduler + EMA** (PR #1520): OneCycleLR + EMA=0.999 → 112.55 (built on #1491).
3. **Geometry augmentation** (PR #1495): AoA + NACA camber jitter → 103.10.
4. **Huber loss δ=0.5** (PR #1484 v2): Huber on top of merged stack → 99.879 val / 93.596 test.
5. **Two-stage surf_weight curriculum 1→20** (PR #1686): Ramp surf_weight 1→20 over 5 epochs (cosine T_max=14, MSE loss) → 97.620 val / 91.947 test. First substantial improvement on val_single_in_dist (114.69).
6. **Huber × curriculum composition** (PR #1745): Huber δ=0.5 + surf_weight 1→20 on same run → 91.507 val / 85.611 test. Super-additive on camber_rc (observed −10.6% vs predicted −7.5%). Huber stabilises per-node gradient distribution, enabling more precise objective shaping by the curriculum.
7. **Fourier mesh PE (surface-only L=4)** (PR #1662): Surface-only Fourier positional encoding with L=4 frequency bands on #1745 merged stack → **88.175 val / 83.362 test** → **new baseline**. Super-additive on camber_cruise (val 64.60 vs 71.16, −9.21%; neither Fourier alone nor Huber+curriculum alone came close). All 4 val and test splits improve. Current bottleneck: val_single_in_dist = 104.911 (still ~19 points above overall average).

### Promising results (awaiting verification on merged stack)
- **SwiGLU FFN** (tanjiro #1693 v2, in flight): v1 val 87.278 / test 82.237 (safe 4-split) — beats OLD #1745 baseline AND the NEW #1662 baseline (88.175). v1 ran on pre-#1484/#1686/#1745/#1662 stack with EMA explicitly disabled; sent back for rebase + rerun on full merged stack. Param count 827K (+7% over baseline). If this holds under the full composition, it's a major win on top of #1662.

### Closed (disproved — negative results)
- **Focal per-sample loss weighting** (askeladd #1709): Both arms (γ=1.0, γ=2.0) regress +9-10% val / +10-12% test vs #1495 baseline. Effective batch-size collapse (eff_bs ≈ 1.65 at γ=2.0 out of B=4) was the dominant failure mode — not gradient signal weakness. Revised P3: focal weighting fails at B≤4 with high-y-variance regression. Per-domain sampling (askeladd #1822) is the orthogonal next test.
- **n_layers depth scaling** (fern #1770): Both arms (n_layers=6/7) regress +13%/+22% vs #1745 baseline. Budget-cap binding: +20-40% sec/epoch reduces completed epochs, cosine schedule never anneals fully, LR still in steep-descent phase at termination. Split predicted to improve most (val_single_in_dist) regressed most (+19.7% at Arm A). New P7: under binding wall-clock cap, sec/epoch increases trade against schedule completion — prefer width/gating/loss axes over depth axis.
- **surf_weight=30/50 sweep** (thorfinn #1827): Both arms regress +4-6% val / +5-6% test vs #1745 baseline. Volume mae_vol_p regresses 7.5% (sw=30) and 16% (sw=50). Curriculum plateau is past optimum at sw=20 — pushing harder degrades the surface/volume gradient balance that Huber×curriculum unlocked. Non-monotonic (sw=50 < sw=30) likely single-seed noise on a flat-bottom landscape. New P8: two-stage curriculum has a Goldilocks plateau ~20× base; beyond this, gradient-balance failure dominates and surface MAE follows volume MAE down.
- **Per-domain SAMPLING oversample** (askeladd #1822): Both arms (2×/3× racecar_single) regress on val_avg (+3.5%/+5.7%) and test_avg (+3.6%/+6.0%) vs #1745 baseline despite the targeted split **improving substantially**: Arm B 3× hit val_single_in_dist 99.55 (−10.49 pts, best ever observed on the bottleneck). Off-domain splits regressed in lock-step at 3×: camber_rc +15.4, camber_cruise +8.8, re_rand +12.3. Mechanism IS real — boosting racecar_single sample share boosts that split's training signal — but the regression is dose-monotonic in the off-domains (2× < 3×). New P9: per-domain SAMPLING-level oversampling is zero-sum at the gradient level under fixed compute. Boosting one domain by k× linearly trades against off-domain generalization. Loss-level reweighting (askeladd #1912) tests whether re-allocating gradient share *within* an unmodified batch escapes the sample-exclusion failure mode.
- **Huber δ down-sweep** (alphonse #1869): Both arms (δ=0.25/0.1) regress vs #1745 baseline. δ=0.25 → val 94.675 (+3.46%) / test 89.924 (+5.04%); δ=0.1 → val 95.418 (+4.27%) / test 90.130 (+5.28%). The 1.0→0.5 trend does NOT extrapolate. Regression dominated by val_single_in_dist (+11.9%/+12.1%) — the high-Re bottleneck split. Mechanism: smaller δ removes magnitude information from large-residual high-Re samples, which were carrying useful signal (not noise) in the quadratic region. New P10: Huber δ has a sharp non-monotone optimum at ~0.5; δ-down removes magnitude info from the bottleneck split. Spatial dual of P8 (curriculum Goldilocks at sw=20). Up-direction (δ=0.6/0.75, alphonse #1931) tests whether 0.5 is at a true local minimum or inflection.
- **slice_num up-sweep** (fern #1850): Both arms (slice_num=96/128) regress vs #1745 baseline (+10/+14% val). BUDGET-CAP BINDING — same mechanism as #1770 depth scaling. slice_num=96 ran 152 sec/ep × 12 epochs (only 91% schedule); slice_num=128 ran 171 sec/ep × 11 epochs (only 79% schedule). Hypothesis is NOT falsified — *undetermined* due to incomplete cosine anneal. Per-split signature (worst regression on bottleneck split) is consistent with under-convergence, not architectural pathology. Generalizes P7 from depth-only to all sec/epoch-expanding axes. Down-direction (slice_num=32/48, fern #1935) is budget-respecting and gives a fair signal.

### Closed (disproved on fair comparison)
- **FiLM Re-conditioning** (tanjiro #1494 v3): val_avg = 104.98 (+1.8% over 103.10 baseline) / test = 98.59 (+4.0% over 94.76 baseline) on cosine T_max=14 + augment + FiLM (exact #1495 protocol + FiLM only). val_re_rand WORSE under FiLM (+3.6%) — opposite of predicted direction. Root cause: log(Re) already at input dim 13 → FiLM adds redundant route; augmentation + FiLM compete on small dataset. v2's 100.99 was rebase artifact, not FiLM signal.
- **Test-time augmentation** (fern #1698): both arms regress (+0.40 / +0.29 vs N=1 baseline 95.437 safe re-eval). Mechanism: TTA's classification record relies on label invariance; here target y(θ_AoA) MOVES with augmentation → averaging biased neighbor predictions. Pred-std diagnostic confirms model IS responsive to jitter (10-30 m²/s²), so averages point in wrong direction — not noop. New universal principle P6 added below.

### Round 1 findings (pre-merge-base, directionally valid)
- **Huber loss** (alphonse, pre-merge): 108.10 — strongest signal yet. With the new stack could compound further. Huber d=0.5 helps cruise/re_rand but hurts single_in_dist (high-Re). Need on-stack comparison.
- **FiLM Re-conditioning** (tanjiro, pre-merge): 129.94 overall but **val_re_rand=116.04** (best split in round 1). The conditioning IS learned (FiLM norms grow monotonically). Need rebase to assess additive gain.
- **AoA/NACA augmentation** (thorfinn, pre-merge): 129.69 — 12% worse overall. Camber OOD was NOT the worst split (single high-Re was). Need on-stack comparison to isolate effect.
- **slice_num=128** (nezuko, pre-merge): 138.32. Memory fine (54.5/96 GB). Need on-stack comparison.
- **mlp_ratio=4** (frieren, pre-merge): 144.33. 21% slower per epoch, fewer epochs completed. Need equal-budget rebase.
- **n_hidden=256** (edward, pre-merge): 172.26. Severely under-budgeted (7 epochs). Sent back as n_hidden=192 (more manageable).
- **Decoupled heads** (askeladd): Still WIP from round 1.

### Universal principles (logged for ICML appendix)

**P1: Loss saturation and augmentation are SUBSTITUTES** (PR #1543 v2).
Log-cosh saturates the gradient at `tanh(r)` for `|r|≳2`; augmentation
creates harder samples by broadening the training distribution; the cap
defeats the augmentation. Holds for Huber-style losses too.

**P2: Per-channel pressure weighting and augmentation are also SUBSTITUTES**
(PR #1488 v3 Arm C). Augmentation already emphasizes hard pressure regions
via geometric diversity; explicit per-channel weighting on top is neutral-
to-harmful. This generalizes P1: a class of "weight the loss + diversify
the data" stacking patterns fails when both target the same channel.

**P3 (revised, PR #1709 disproved): Per-sample focal reweighting fails
at B≤4 in high-y-variance regression.** Effective batch-size collapse
(eff_bs ≈ 1.65 at γ=2.0 out of B=4) is the proximate failure — focal
weighting with small batches functions as a stochastic batch-size reducer,
exploding gradient variance. Both γ=1.0 and γ=2.0 regressed +9-10% val.
The per-sample axis IS theoretically orthogonal to P1/P2 per-channel and
per-residual surgery, but batch dynamics dominate at B=4. At B=16+, focal
may behave differently. Per-DOMAIN sampling (askeladd #1822) is the
orthogonal next test: it changes WHICH samples appear per batch, not
the within-batch loss weighting.

**P4 (PR #1574, revisited PR #1484 v2): OneCycleLR with `--epochs 50` was
flagged broken at 30-min cap under MSE+EMA.** pct_start=0.05 reaches peak
LR at step 187/3750, leaving 97% of anneal unfired. HOWEVER, alphonse's
PR #1484 v2 used exactly this "broken" config and produced the new
baseline (99.879). Refined hypothesis: P4 is loss-specific. Under
**MSE+EMA** the truncated anneal hurts (regression in #1574). Under
**Huber+EMA** it does not (because Huber itself bounds the gradient and
no longer needs the long-tail anneal). Recommendation: when a hypothesis
modifies the loss formulation, run BOTH `--use_onecycle True --epochs 50`
and `--use_onecycle False --epochs 14` to isolate. When tuning
architecture or augmentation, cosine T_max=14 remains the safer default.

**P6 (PR #1698): Test-time augmentation fails when augmentation perturbs
the target signal, not just nuisance variables.** TTA's classification
record relies on label invariance under augmentation. For regression
targets y(θ) where the augmentation moves θ (e.g. AoA jitter), each
jittered prediction is the model's estimate of y(θ+Δθ), not a noisy
reading of y(θ). Averaging pulls toward a smoothed neighborhood mean of
the actual signal. Training-time augmentation regularizes the model's
loss landscape (a "loss-shaped" smoothing); test-time averaging
operates on outputs (an "output-shaped" smoothing). The two regularizations
live on different objects and cannot substitute for each other. This is
the mechanistic *dual* of P1: there gradient capping defeats
augmentation's hard-sample injection; here output averaging defeats the
model's task-relevant input sensitivity. Rules out the entire family
of "perturb a meaningful input axis at test time" approaches for
surrogate models.

**P7 (refined, PR #1770 + PR #1850): Under a binding wall-clock cap with
cosine T_max=N, ANY architectural axis that increases sec/epoch trades
against schedule completion.** Confirmed on:
- Depth: n_layers=6 (+20% sec/epoch) → 12/14 epochs; n_layers=7 (+40%) → 10/14 (PR #1770).
- Attention partition resolution: slice_num=96 (+17%) → 12/14; slice_num=128 (+31%) → 11/14 (PR #1850).

The cosine schedule never reaches its annealing tail, leaving LR too high
for fine-grained surface pressure learning. The predicted "val_single_in_dist
benefits from refinement" inverted in both cases: the bottleneck split
regressed *most* when under-trained. Per-split signature (worst
regression on bottleneck) is the diagnostic — it is consistent with
under-convergence, NOT with architectural pathology. The hypothesis
remains UNDETERMINED, not falsified.

Implication: in this 30-min/14-epoch regime, prefer budget-neutral or
budget-reducing axes (loss shape, schedule shape, augmentation, dropout,
EMA decay, regularization, smaller slice_num) over budget-expanding
axes. To honestly test budget-expanding axes, either (a) reduce
T_max to match achievable epochs, OR (b) raise the per-run budget.
Both are scope changes that need explicit advisor approval.

**P8 (PR #1827): Two-stage surf_weight curriculum has a Goldilocks
plateau around 20× base.** Pushing the plateau higher (sw=30, sw=50)
regresses on every per-split metric including the val_single_in_dist
bottleneck (+5-6% on the worst split). Volume MAE regresses 7.5%
(sw=30) and 16% (sw=50). The training surf_loss curve continues to
descend, so absolute surface optimization is not failing — but the
surface-vs-volume gradient *balance* shifts, volume representation
degrades, and surface MAE follows volume down. The 1→N curriculum is
best understood as **steering** the optimizer toward a balanced
surface/volume solution, NOT as a unidirectional "push surface harder"
dial. Mechanistically aligned with #1745 Huber×curriculum super-
additivity: Huber stabilises per-node gradient distribution, curriculum
steers gradient share — they work together in a Goldilocks regime
that sw>20 disrupts. Next-axis question: does the *ramp shape*
(warmup_epochs) have a similar Goldilocks regime, or is it flat?
(thorfinn #1885 in flight.)

**P9 (PR #1822): Per-domain SAMPLING-level oversampling is zero-sum
at the gradient level under fixed compute.** Arm B's 3× racecar_single
oversample produced the strongest val_single_in_dist result observed
on this branch (99.55, −10.49 vs #1745), confirming the mechanism is
real — but val_avg regressed +5.7% because the off-domain splits
(camber_rc, camber_cruise, re_rand) all regressed in dose-monotonic
lock-step. With fixed per-epoch sample count, k× oversampling of
domain D means (k−1)/k × N samples *not from D* are excluded per
epoch; their gradient signal is unweighted, just absent. The
exclusion penalty appears linearly proportional to k. Generalizes P3
beyond per-sample to deterministic per-domain rebalancing. Implications:
(a) sample-level domain rebalancing has the same compute-zero-sum
property as focal weighting did at B=4 — both are *gradient-allocation*
moves disguised as data moves; (b) the bottleneck IS attackable —
val_single_in_dist 99.55 disproves the "intractable in-distribution
ceiling" hypothesis; (c) the next test is whether the same gradient
re-allocation done at the LOSS level (askeladd #1912) escapes the
sample-exclusion failure mode by preserving off-domain samples in
each batch while still up-weighting the target domain's gradient
contribution.

**P10 (PR #1869): Huber δ has a sharp non-monotone optimum at ~0.5
on the curriculum-composed stack.** The 1.0→0.5 trend does NOT
extrapolate. δ=0.25 and δ=0.1 both regress, with the regression
concentrated on val_single_in_dist (+12% at δ=0.1). Mechanism: small
δ approaches MAE for almost all residuals (quadratic region |r|<δ
becomes negligible), removing the magnitude information that
discriminates high-Re samples. The Huber quadratic region was
carrying useful signal on the bottleneck split, not dampening noise —
the opposite of the naive "smaller δ helps with outliers" intuition.
Per-split sensitivity confirms δ acts on training-dynamic stability
(single_in_dist) rather than per-channel signal balance (uniform
across channels within a split). P10 is the **spatial dual** of P8:
both 'loss-shape' levers (δ, sw) have local optima not boundary
points, and pushing either past optimum hurts the hardest split
disproportionately. Up-direction test (δ=0.6/0.75, alphonse #1931 in
flight) closes the axis definitively.

### Potential next directions (round 3+)
- **Fourier PE v4 coord rescaling** (nezuko #1955, in flight): Per-sample [0,1] bbox normalization before Fourier encoding. Arm A: independent x/y; Arm B: isotropic. NeRF-convention frequency calibration. Expected 1-2% gain on OOD splits.
- **Huber δ up-direction** (alphonse #1931, in flight): δ=0.6/0.75 on #1745 merged stack. Resolves inflection-vs-optimum question. **Pass criterion now vs #1662 (88.175/83.362).**
- **slice_num down-direction** (fern #1935, in flight): slice_num=32/48 on #1745 merged stack. **Pass criterion now vs #1662 (88.175/83.362).**
- **Curriculum ramp shape** (thorfinn #1885, in flight): warmup_epochs=3 vs 8 at fixed sw=20. **Pass criterion now vs #1662 (88.175/83.362).**
- **Per-domain LOSS reweighting** (askeladd #1912, in flight): λ=0.3/1.0 multiplier on racecar_single loss. **Pass criterion now vs #1662 (88.175/83.362).**
- **SwiGLU composability** (tanjiro #1693 v2, in flight): v1 87.278/82.237 PASSES current baseline! Verification run on full merged stack (incl. Fourier PE) is critical. Could be next major win.
- **Fourier PE Gaussian random Fourier features (GRFF)** (nezuko, future): Smoother spectral coverage vs dyadic 2^k bands. nezuko's suggestion #2 — defer until v4 coord-rescaling result is known.
- **Larger surf_weight ramp endpoint (100)** — DEPRIORITIZED. P8 closed plateau-axis as past optimum; further upward push would just compound the volume-MAE regression.
- **Surface-only Huber** — DEPRIORITIZED based on #1869 per-split analysis (δ acts on training dynamics, not per-channel signal balance). Skip unless a different per-channel motivation emerges.
- **EMA decay sweep** (UNTOUCHED LEVER): 0.9995, 0.998 vs current 0.999. Single-axis HP sweep, budget-neutral. Good candidate after current sweeps complete.
- **β2 (AdamW) sweep** (UNTOUCHED LEVER): 0.99 vs current default. Tests adaptive LR responsiveness on small-batch regression.
- **Augmentation magnitude sweep** (UNTOUCHED LEVER): aoa_jitter_rad 2× / 0.5× current 0.00873. Tests if augment strength is well-tuned.
- **Mesh-aware positional encoding**: signed distance / arc length as Fourier features (nezuko #1662 v2 covers raw-coord; arc-length is the next step).
- **Stack winners**: SwiGLU + Fourier PE together once both verify. Architecture changes should compose with Huber+curriculum.
- **Plateau Protocol watch**: 10+ consecutive closes since #1745 baseline. After current loss-shape/architectural sweeps complete, consider higher-level changes: SAM (Sharpness-Aware Minimization), hierarchical multi-scale attention, FNO components, mesh-aware multi-scale models. These are bigger swings; require code work.
- **Relative MAE in physical space**: scale-invariant loss for multi-Re training — still unassigned.
- **dsdf shape descriptor augmentation** (dims 4-11): deeper geometry augmentation vs. scalar AoA/NACA — still unassigned.
- **Spectral / smoothness regularization**: penalty on |∇y_pred| at surface nodes (physics-motivated smoothing of pressure).
