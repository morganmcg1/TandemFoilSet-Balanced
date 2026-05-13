# SENPAI Research State

- **Date**: 2026-05-13 00:04 UTC
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r3` (base `icml-appendix-charlie`)
- **Research tag**: `charlie-pai2g-24h-r3`
- **Students (8)**: charliepai2g24h3-{alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn}
- **Per-run budget**: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50` (caps)
- **Logging**: local JSONL only (no W&B in this arm)

## Latest human direction

None received.

## Current best baseline

**val_avg/mae_surf_p = 103.100** — PR #1495 (thorfinn, AoA + NACA camber jitter augmentation).
Test 4-split safe re-eval = 94.757; 3-split proxy = 98.520.
Stack: `grad_clip=1.0 + wd=1e-3 + augment(±0.5° AoA, ±0.002 NACA) + cosine T_max=14`.
**Composability note:** ran with cosine, not OneCycleLR+EMA. Merged train.py defaults to use_onecycle=True (from #1520), so reproducing requires `--use_onecycle False --epochs 14`. OneCycleLR+EMA+augment composability is untested.

**Best raw number observed (but not merged):** 100.987 — PR #1494 v2 (tanjiro, FiLM on log(Re), rebased onto #1491 only, without augmentation). Sent back to rebase onto post-#1495 base.

**Disproved (closed, mechanistic):** PR #1543 v2 (fern) log-cosh + augment @ 106.93 (+3.71%) / test 100.61 (+6.18%). The v2 − v1 delta ≈ 0 (augmentation added nothing on top of log-cosh, vs +9.4 on MSE) proves log-cosh and augment are SUBSTITUTES, not complements: both target high-Re gradient dominance via different mechanisms. Log-cosh's gradient cap defeats augmentation's purpose on rc split (+13.5% worse, the killer).

**Per-channel pressure weighting disproved (closed):** PR #1488 Arm C (askeladd, surf_weight_p=20 alone, no decoupling) @ val 105.72 (+2.62%) / test 99.29 (+4.53%). All three arms (A, B, C) fail pass criterion. Arm B's 102.12 was single-seed noise from cosine schedule, not signal. Adds to the "weight the loss + diversify the data" substitution principle established by PR #1543.

## Current student assignments

| Student | PR | Slug | Status |
|---|---|---|---|
| alphonse | #1484 | `huber-pressure-loss` | WIP — rebase: Huber d=0.5+d=1.0 on full merged stack (2 arms) |
| askeladd | #1709 | `focal-per-sample-loss-weighting` | WIP — focal weighting γ=1.0/2.0 (2 arms). Amplifies hard-sample gradient (mechanistic opposite of log-cosh). |
| edward | #1490 | `scale-model-256-v2` | WIP — rebase: n_hidden=192, n_head=6 on new stack |
| fern | #1698 | `test-time-augmentation` | WIP — TTA with 2 arms (N=5/9, jitter=0.5°/0.75°) at eval time. Pure inference-time, no training changes. |
| frieren | #1492 | `mlp-ratio-4-wider-ffn` | WIP — rebase: mlp_ratio=4 |
| nezuko | #1662 | `fourier-mesh-positional-encoding` | WIP — Fourier features on (x,y) coordinates (NeRF-style γ(x), L=6 bands) |
| tanjiro | #1693 | `swiglu-ffn` | WIP — SwiGLU gated linear unit FFN replacing GELU MLP (single arm, cosine T_max=14) |
| thorfinn | #1686 | `two-stage-surf-weight-curriculum` | WIP — curriculum ramp surf_weight 1.0→10.0 (Arm A) / 1.0→20.0 (Arm B) over 5 epochs, then hold. cosine T_max=14. |

## Research themes and findings

### Confirmed winners (merged)
1. **Optimization hygiene** (PR #1491): grad_clip=1.0 + wd=1e-3 → 115.40.
2. **Scheduler + EMA** (PR #1520): OneCycleLR + EMA=0.999 → 112.55 (built on #1491).
3. **Geometry augmentation** (PR #1495): AoA + NACA camber jitter → **103.10** → new baseline. NOTE: thorfinn's best run used cosine T_max=14, not OneCycleLR.

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

**P4 (PR #1574): OneCycleLR with `--epochs 50` is broken at 30-min cap.**
pct_start=0.05 reaches peak LR at step 187/3750, leaving 97% of anneal
unfired. All 30-min runs use cosine T_max=14 unless explicitly testing
schedule mechanics.

### Potential next directions (round 3+)
- **Compose winners**: combine Huber + FiLM + log-cosh + curriculum once individual rebases are scored.
- **Surface-only Huber**: apply Huber to volume nodes, MSE to surface (alphonse follow-up).
- **Per-sample importance weighting**: weight each sample's loss by 1/y_std_sample.
- **Relative MAE in physical space**: scale-invariant loss for multi-Re training.
- **dsdf shape descriptor augmentation** (dims 4-11): deeper geometry augmentation vs. scalar AoA/NACA.
- **n_layers=7** (depth scaling): orthogonal to width scaling experiments.
- **Corrected OneCycleLR**: `pct_start * actual_epochs` matched (e.g. `--pct_start 0.2 --epochs 14`) — current default is broken at 30-min cap.
- **Mesh-aware positional encoding**: signed distance / arc length as Fourier features (nezuko #1662 covers raw-coord Fourier; arc-length variant is the next step).
- **Surface-aware attention**: separate slice tokens for surface vs. volume nodes.
