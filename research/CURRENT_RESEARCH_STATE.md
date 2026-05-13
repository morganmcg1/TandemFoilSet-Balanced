# SENPAI Research State

- **Date**: 2026-05-13 00:30 UTC
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r3` (base `icml-appendix-charlie`)
- **Research tag**: `charlie-pai2g-24h-r3`
- **Students (8)**: charliepai2g24h3-{alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn}
- **Per-run budget**: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50` (caps)
- **Logging**: local JSONL only (no W&B in this arm)

## Latest human direction

None received.

## Current best baseline

**val_avg/mae_surf_p = 99.879** — PR #1484 v2 (alphonse, Huber loss δ=0.5 on merged stack).
Test 4-split safe re-eval = **93.596** (−1.23% vs prior #1495 baseline); 3-split proxy = 99.616.
Stack: `grad_clip=1.0 + wd=1e-3 + augment(±0.5° AoA, ±0.002 NACA) + use_onecycle=True + ema_decay=0.999 + huber_delta=0.5`, 14 epochs at 30-min cap.
**Per-split val:** single=123.26, rc=118.37, cruise=69.27 (−11.2% absolute on cruise!), re_rand=88.62.
**Schedule note (P4 revisit):** Arm A used `--use_onecycle True --epochs 50` (the "broken" P4 config) AND won the baseline. Hypothesis: Huber's gradient-clipping property makes the truncated OneCycle anneal benign. Future loss-formulation experiments may not need cosine T_max=14.
**`safe_test_eval.py` is now the standard 4-split test harness** — committed at repo root via PR #1484. All future PRs must run it on best-val checkpoint for paper-facing test metrics.

**Disproved (closed, mechanistic):** PR #1543 v2 (fern) log-cosh + augment @ 106.93 (+3.71%) / test 100.61 (+6.18%). The v2 − v1 delta ≈ 0 (augmentation added nothing on top of log-cosh, vs +9.4 on MSE) proves log-cosh and augment are SUBSTITUTES, not complements: both target high-Re gradient dominance via different mechanisms. Log-cosh's gradient cap defeats augmentation's purpose on rc split (+13.5% worse, the killer).

**Per-channel pressure weighting disproved (closed):** PR #1488 Arm C (askeladd, surf_weight_p=20 alone, no decoupling) @ val 105.72 (+2.62%) / test 99.29 (+4.53%). All three arms (A, B, C) fail pass criterion. Arm B's 102.12 was single-seed noise from cosine schedule, not signal. Adds to the "weight the loss + diversify the data" substitution principle established by PR #1543.

## Current student assignments

| Student | PR | Slug | Status |
|---|---|---|---|
| alphonse | TBD | `huber-delta-smaller` | IDLE — to be assigned: δ=0.25 + δ=0.1 sweep, plus optional cosine T_max=14 isolation arm |
| askeladd | #1709 | `focal-per-sample-loss-weighting` | WIP — focal weighting γ=1.0/2.0 (2 arms). Amplifies hard-sample gradient (mechanistic opposite of log-cosh). |
| edward | #1490 | `scale-model-256-v2` | WIP — rebase: n_hidden=192, n_head=6 on new stack |
| fern | #1698 | `test-time-augmentation` | WIP — TTA with 2 arms (N=5/9, jitter=0.5°/0.75°) at eval time. Pure inference-time, no training changes. |
| frieren | #1492 | `mlp-ratio-4-wider-ffn` | WIP — rebase: mlp_ratio=4 |
| nezuko | #1662 | `fourier-mesh-positional-encoding` | WIP — v1 had +3.97% val (val_single_in_dist −6.26%! best ever on worst split), sent back v2 with 2 arms: L=2 + cosine, L=4 surface-only + cosine. |
| tanjiro | #1693 | `swiglu-ffn` | WIP — SwiGLU gated linear unit FFN replacing GELU MLP (single arm, cosine T_max=14) |
| thorfinn | #1686 | `two-stage-surf-weight-curriculum` | WIP — curriculum ramp surf_weight 1.0→10.0 (Arm A) / 1.0→20.0 (Arm B) over 5 epochs, then hold. cosine T_max=14. |

## Research themes and findings

### Confirmed winners (merged)
1. **Optimization hygiene** (PR #1491): grad_clip=1.0 + wd=1e-3 → 115.40.
2. **Scheduler + EMA** (PR #1520): OneCycleLR + EMA=0.999 → 112.55 (built on #1491).
3. **Geometry augmentation** (PR #1495): AoA + NACA camber jitter → 103.10.
4. **Huber loss δ=0.5** (PR #1484 v2): Huber on top of merged stack → **99.879** val / **93.596** test (4-split safe) → new baseline. Note: ran with `--use_onecycle True --epochs 50` (the P4 "broken" config) AND won, contradicting P4 — to be revisited.

### Promising single-split signal (sent back for v2)
- **Fourier mesh PE** (nezuko #1662 v1): val_avg 107.19 (+3.97%, fails) BUT `val_single_in_dist = 118.03 vs 125.91 = −6.26%` — first substantial improvement on the historically WORST split. OOD splits regressed (rc +13.7%). Schedule confound (OneCycleLR ep=11 instead of cosine T_max=14). v2 with 2 arms: L=2 capacity-fix and L=4 surface-only scope-fix, both on cosine T_max=14.

### Closed (disproved on fair comparison)
- **FiLM Re-conditioning** (tanjiro #1494 v3): val_avg = 104.98 (+1.8% over 103.10 baseline) / test = 98.59 (+4.0% over 94.76 baseline) on cosine T_max=14 + augment + FiLM (exact #1495 protocol + FiLM only). val_re_rand WORSE under FiLM (+3.6%) — opposite of predicted direction. Root cause: log(Re) already at input dim 13 → FiLM adds redundant route; augmentation + FiLM compete on small dataset. v2's 100.99 was rebase artifact, not FiLM signal.

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

**P3 (testable, askeladd #1709): Per-sample reweighting and augmentation
should COMPOUND.** Focal weighting amplifies gradient on hard samples
(mechanism-opposite of log-cosh); augmentation creates hard samples;
the two should compose. This is what P1/P2 *don't* rule out: the per-
sample axis is orthogonal to per-residual and per-channel surgery.

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

### Potential next directions (round 3+)
- **Even smaller Huber δ** (alphonse follow-up, queued): δ=0.25 and δ=0.1. The round-1 fear that "δ=0.5 hurts single_in_dist" is eliminated on the merged stack; the optimum may sit further below 0.5. Both arms still descending at 14-epoch cap.
- **Surface-only Huber**: apply Huber to surface nodes, MSE to volume (the inverse split would also be informative — volume MAE is currently ~3% worse than surface).
- **Per-sample importance weighting**: weight each sample's loss by 1/y_std_sample.
- **Relative MAE in physical space**: scale-invariant loss for multi-Re training.
- **dsdf shape descriptor augmentation** (dims 4-11): deeper geometry augmentation vs. scalar AoA/NACA.
- **n_layers=7** (depth scaling): orthogonal to width scaling experiments.
- **Mesh-aware positional encoding**: signed distance / arc length as Fourier features (nezuko #1662 v2 covers raw-coord Fourier; arc-length variant is the next step).
- **Surface-aware attention**: separate slice tokens for surface vs. volume nodes.
- **Stack winners**: once each in-flight axis settles (SwiGLU, TTA, surf_weight curriculum, focal weighting), test their composability with Huber δ=0.5.
