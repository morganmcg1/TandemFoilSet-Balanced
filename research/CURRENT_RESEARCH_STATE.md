# SENPAI Research State

- **Date**: 2026-05-12 23:26 UTC
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

**Strong loss-curvature signal (sent back):** PR #1543 v1 (fern) log-cosh @ 106.68 on PR #1520 base — clean −5.21% vs #1520, cruise split gets −12.0%. Doesn't beat 103.10 baseline (no augmentation); sent back for log-cosh + augmentation rerun.

**Per-channel pressure weighting signal (sent back, entangled):** PR #1488 v2 Arm B (askeladd) decoupled heads + surf_weight_p=20 @ val 102.12 / test 96.82 on full merged stack with cosine T_max=14. Beats val 103.10 by 0.95% but loses test 4-split by +2.18%. Two changes entangled (head decoupling + weighting). Sent back for surf_weight_p=20-alone ablation. Likely the per-channel weighting is the active ingredient.

## Current student assignments

| Student | PR | Slug | Status |
|---|---|---|---|
| alphonse | #1484 | `huber-pressure-loss` | WIP — rebase: Huber d=0.5+d=1.0 on full merged stack (2 arms) |
| askeladd | #1488 | `decoupled-channel-heads` | WIP — v2 sent back (Arm B 102.12 val/96.82 test, entangled), run Arm C: surf_weight_p=20 alone (no decoupling) |
| edward | #1490 | `scale-model-256-v2` | WIP — rebase: n_hidden=192, n_head=6 on new stack |
| fern | #1543 | `logcosh-loss` | WIP — sent back v1 (106.68 vs baseline 103.10), rebase + re-run with augmentation default ON |
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

### Universal finding (PR #1574)
**OneCycleLR scheduling bug:** `--use_onecycle True --epochs 50` with `pct_start=0.05`
hits peak LR at step 187/3750 → 97% of cosine anneal tail never executes at the
30-min wall-clock cap (~14 epochs). The current merged default `use_onecycle=True`
is actively hurting any run that uses it. PR #1495's 103.10 win used cosine
T_max=14, not OneCycleLR. **Going forward all 30-min experiments should use
`--use_onecycle False --epochs 14` (cosine T_max=14) unless explicitly testing
schedule mechanics.**

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
