# SENPAI Research State

- **Date**: 2026-05-13 03:30 UTC
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r3` (base `icml-appendix-charlie`)
- **Research tag**: `charlie-pai2g-24h-r3`
- **Students (8)**: charliepai2g24h3-{alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn}
- **Per-run budget**: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50` (caps)
- **Logging**: local JSONL only (no W&B in this arm)

## Latest human direction

None received.

## Current best baseline

**val_avg/mae_surf_p = 91.507** — PR #1745 (thorfinn, Huber δ=0.5 × surf_weight curriculum 1→20).
Test 4-split safe re-eval = **85.611** (−6.91% vs PR #1686 baseline 91.947).
Stack: `grad_clip=1.0 + wd=1e-3 + augment + cosine T_max=14 + EMA=0.999 + huber_delta=0.5 + surf_weight_warmup_epochs=5 + surf_weight_init=1.0 + surf_weight=20.0`.
**Per-split val:** single=110.04, rc=100.44 (−10.6% vs #1686 — super-additive!), cruise=71.16, re_rand=84.38.
**Per-split test:** single=96.26, rc=88.65, cruise=77.18, re_rand=80.36.
**Key finding:** Huber + curriculum compose super-additively on camber_rc. Huber stabilises per-node gradient distribution; curriculum steers surface/volume gradient share. Together they unlock a regime neither alone was sufficient for (geometry-OOD camber_rc, 111→100). Strongest bottleneck remaining: `val_single_in_dist = 110.04`.

**Previous baseline (#1686, curriculum):** val 97.620 / test 91.947. (#1484 Huber alone: val 99.879 / test 93.596.)

**Disproved (closed, mechanistic):** PR #1543 v2 (fern) log-cosh + augment @ 106.93 (+3.71%) / test 100.61 (+6.18%). The v2 − v1 delta ≈ 0 (augmentation added nothing on top of log-cosh, vs +9.4 on MSE) proves log-cosh and augment are SUBSTITUTES, not complements: both target high-Re gradient dominance via different mechanisms. Log-cosh's gradient cap defeats augmentation's purpose on rc split (+13.5% worse, the killer).

**Per-channel pressure weighting disproved (closed):** PR #1488 Arm C (askeladd, surf_weight_p=20 alone, no decoupling) @ val 105.72 (+2.62%) / test 99.29 (+4.53%). All three arms (A, B, C) fail pass criterion. Arm B's 102.12 was single-seed noise from cosine schedule, not signal. Adds to the "weight the loss + diversify the data" substitution principle established by PR #1543.

## Current student assignments

| Student | PR | Slug | Status |
|---|---|---|---|
| alphonse | #1736 | `huber-delta-smaller` | WIP — Huber δ sweep: 0.25 + 0.1 on merged stack. Should rebase on #1745 to test smaller δ on top of Huber+curriculum stack. |
| askeladd | #1822 | `domain-oversample-racecar-single` | WIP (just assigned) — over-sample racecar_single domain 2x (Arm A) / 3x (Arm B) to target single_in_dist bottleneck (110.04). Different from focal weighting (per-domain sampling, not per-sample loss reweighting). |
| edward | #1490 | `scale-model-256-v2` | WIP — rebase: n_hidden=192, n_head=6 on new stack |
| fern | #1850 | `slice-num-sweep-96-128` | WIP — slice_num=96 (Arm A) / slice_num=128 (Arm B) on merged stack. Tests attention partitioning granularity. |
| frieren | #1492 | `mlp-ratio-4-wider-ffn` | WIP — rebase: mlp_ratio=4 |
| nezuko | #1662 | `fourier-mesh-positional-encoding` | WIP v3 — v2 PASSED (Arm B surface-only L=4: val 95.598 / test 89.895, beats #1686 by −2.07%/−2.23%). Sent back for v3 verification on full merged stack (Huber+curriculum+EMA). |
| tanjiro | #1693 | `swiglu-ffn` | WIP v2 — v1 hit val 87.28 / test 82.24 (−10% vs #1686!) but merge conflicts + pre-#1484/#1686 base. Sent back for rebase + rerun on full merged stack (#1745 now baseline). |
| thorfinn | #1827 | `surf-weight-sweep-30-50` | WIP (just assigned) — surf_weight=30 (Arm A) / surf_weight=50 (Arm B) on #1745 merged stack. Tests whether optimal final surf_weight > 20; direct continuation of thorfinn's winning axis. |

## Research themes and findings

### Confirmed winners (merged)
1. **Optimization hygiene** (PR #1491): grad_clip=1.0 + wd=1e-3 → 115.40.
2. **Scheduler + EMA** (PR #1520): OneCycleLR + EMA=0.999 → 112.55 (built on #1491).
3. **Geometry augmentation** (PR #1495): AoA + NACA camber jitter → 103.10.
4. **Huber loss δ=0.5** (PR #1484 v2): Huber on top of merged stack → 99.879 val / 93.596 test.
5. **Two-stage surf_weight curriculum 1→20** (PR #1686): Ramp surf_weight 1→20 over 5 epochs (cosine T_max=14, MSE loss) → 97.620 val / 91.947 test. First substantial improvement on val_single_in_dist (114.69).
6. **Huber × curriculum composition** (PR #1745): Huber δ=0.5 + surf_weight 1→20 on same run → **91.507 val / 85.611 test** → **new baseline**. Super-additive on camber_rc (observed −10.6% vs predicted −7.5% from individual gains). Huber stabilises per-node gradient distribution, enabling more precise objective shaping by the curriculum. Current bottleneck: val_single_in_dist = 110.04 (still 26 points above the overall average).

### Promising results (sent back for verification on merged stack)
- **Fourier mesh PE** (nezuko #1662 v2): Both arms pass against #1495 baseline.
  - Arm A (uniform L=2): val 95.727 / test 89.989
  - Arm B (surface-only L=4): **val 95.598 / test 89.895** (recommended)
  - vs NEW #1686 baseline: Arm B = −2.07% val / −2.23% test → still a winner.
  - Per-split Arm B: single 111.06 / rc 106.97 / cruise 72.07 / re_rand 92.30 — three of four splits beat #1686 per-split, only re_rand marginally regressed (+1.72%).
  - v2 ran with `--ema_decay 0.0` (no EMA), no Huber, no curriculum — full composability with merged stack untested. v1's failure was the OneCycle@11ep scheduler confound, NOT capacity or scope — both A and B (with different fixes) produce essentially identical wins under cosine T_max=14.
  - Sent back for v3 = surface-only L=4 on merged stack with EMA + Huber + curriculum all active.
- **SwiGLU FFN** (tanjiro #1693 v1): val 87.278 / test 82.237 (safe 4-split) — beats #1686 by **−10.5% val / −10.6% test**, uniform 12-16% gain across ALL 4 splits. If this holds on rebase it's the strongest single-change result on this branch. v1 ran on pre-#1484/#1686 stack with EMA explicitly disabled — sent back for rebase + rerun on full merged stack to verify and to test SwiGLU × curriculum × Huber × EMA composition in one run. Param count 827K (+7% over baseline).

### Closed (disproved — negative results)
- **Focal per-sample loss weighting** (askeladd #1709): Both arms (γ=1.0, γ=2.0) regress +9-10% val / +10-12% test vs #1495 baseline. Effective batch-size collapse (eff_bs ≈ 1.65 at γ=2.0 out of B=4) was the dominant failure mode — not gradient signal weakness. Revised P3: focal weighting fails at B≤4 with high-y-variance regression. Per-domain sampling (askeladd #1822) is the orthogonal next test.
- **n_layers depth scaling** (fern #1770): Both arms (n_layers=6/7) regress +13%/+22% vs #1745 baseline. Budget-cap binding: +20-40% sec/epoch reduces completed epochs, cosine schedule never anneals fully, LR still in steep-descent phase at termination. Split predicted to improve most (val_single_in_dist) regressed most (+19.7% at Arm A). New P7: under binding wall-clock cap, sec/epoch increases trade against schedule completion — prefer width/gating/loss axes over depth axis.

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

**P7 (PR #1770): Under a binding wall-clock cap with cosine T_max=N,
architectural changes that increase sec/epoch trade against schedule
completion.** n_layers=6 (+20% sec/epoch) completed only 12/14 epochs;
n_layers=7 (+40%) only 10/14 epochs. The cosine schedule never reached
its annealing tail, leaving LR too high for fine-grained surface pressure
learning. The predicted "val_single_in_dist benefits from depth" inverted:
the most compute-hungry split (largest in-distribution gradients) regressed
most when under-trained. Implication: in this budget regime, prefer axes
that keep sec/epoch ≈ baseline (width, FFN gating, loss, slice_num) over
axes that scale it (depth, attention resolution, mesh resolution). Depth
scaling requires either a longer per-run budget or adaptive schedule
matching epochs to available compute.

### Potential next directions (round 3+)
- **Even smaller Huber δ** (alphonse #1736, in flight): δ=0.25 and δ=0.1 on #1745 merged stack (Huber+curriculum). Optimum may be below 0.5 now that curriculum handles the training-dynamic stability.
- **surf_weight push** (thorfinn #1827, in flight): surf_weight=30/50 on #1745 stack. Directly tests whether optimal final weight > 20; expected gain from targeting val_single_in_dist bottleneck (110.04).
- **Per-domain data curriculum** (askeladd #1822, in flight): over-sample racecar_single training domain 2x/3x. Direct orthogonal approach to single_in_dist bottleneck after focal weighting failure.
- **SwiGLU composability** (tanjiro #1693 v2, in flight): ~−10% gain expected if v1 held on merged stack.
- **Fourier PE composability** (nezuko #1662 v3, in flight): ~−2% additional gain expected if v2 held on merged stack.
- **slice_num sweep** (fern #1850, just assigned): slice_num=96/128 on #1745 merged stack. Tests attention partitioning granularity. Mechanically efficient (tiny extra compute). Targeting boundary-layer clustering under surf_weight=20 curriculum.
- **Larger surf_weight ramp endpoint (100)** — only if thorfinn's 50 arm passes, test 100.
- **Surface-only Huber**: apply Huber to surface nodes, MSE to volume. Different from whole-loss Huber; could help single_in_dist where surface-pressure-range is large.
- **Mesh-aware positional encoding**: signed distance / arc length as Fourier features (nezuko #1662 v2 covers raw-coord; arc-length is the next step).
- **Stack winners**: SwiGLU + Fourier PE together once both verify. Architecture changes should compose with Huber+curriculum.
- **Relative MAE in physical space**: scale-invariant loss for multi-Re training — still unassigned.
- **dsdf shape descriptor augmentation** (dims 4-11): deeper geometry augmentation vs. scalar AoA/NACA — still unassigned.
