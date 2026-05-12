# SENPAI Research State

- **Date**: 2026-05-12 20:15 UTC
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

## Current student assignments

| Student | PR | Slug | Status |
|---|---|---|---|
| alphonse | #1484 | `huber-pressure-loss` | WIP — rebase: Huber d=0.5+d=1.0 on full merged stack (2 arms) |
| askeladd | #1488 | `decoupled-channel-heads` | WIP — rebase: decoupled heads on full merged stack (2 arms) |
| edward | #1490 | `scale-model-256-v2` | WIP — rebase: n_hidden=192, n_head=6 on new stack |
| fern | #1543 | `logcosh-loss` | WIP — log-cosh on full merged stack |
| frieren | #1492 | `mlp-ratio-4-wider-ffn` | WIP — rebase: mlp_ratio=4 |
| nezuko | #1493 | `more-slices-128-v2` | WIP — rebase: slice_num=128 |
| tanjiro | #1494 | `re-film-conditioning-v3` | WIP — rebase: FiLM on top of augmentation (2 arms) |
| thorfinn | #1574 | `augment-onecycle-ema-stack` | WIP — composability: augment + OneCycleLR + EMA (2 arms, incl. EMA warmup ramp) |

## Research themes and findings

### Confirmed winners (merged)
1. **Optimization hygiene** (PR #1491): grad_clip=1.0 + wd=1e-3 → 115.40.
2. **Scheduler + EMA** (PR #1520): OneCycleLR + EMA=0.999 → 112.55 (built on #1491).
3. **Geometry augmentation** (PR #1495): AoA + NACA camber jitter → **103.10** → new baseline. NOTE: thorfinn's best run used cosine T_max=14, not OneCycleLR.

### Strong unmerged signal (sent back for rebase)
- **FiLM Re-conditioning** (tanjiro #1494 v2): **100.99 — best raw number this track has produced**. Beats current #1495 baseline by 2.1%. val_re_rand=92.90 is the best split (matches FiLM hypothesis). Merge blocked by conflicts; v3 will test FiLM on top of augmentation + full stack.

### Round 1 findings (pre-merge-base, directionally valid)
- **Huber loss** (alphonse, pre-merge): 108.10 — strongest signal yet. With the new stack could compound further. Huber d=0.5 helps cruise/re_rand but hurts single_in_dist (high-Re). Need on-stack comparison.
- **FiLM Re-conditioning** (tanjiro, pre-merge): 129.94 overall but **val_re_rand=116.04** (best split in round 1). The conditioning IS learned (FiLM norms grow monotonically). Need rebase to assess additive gain.
- **AoA/NACA augmentation** (thorfinn, pre-merge): 129.69 — 12% worse overall. Camber OOD was NOT the worst split (single high-Re was). Need on-stack comparison to isolate effect.
- **slice_num=128** (nezuko, pre-merge): 138.32. Memory fine (54.5/96 GB). Need on-stack comparison.
- **mlp_ratio=4** (frieren, pre-merge): 144.33. 21% slower per epoch, fewer epochs completed. Need equal-budget rebase.
- **n_hidden=256** (edward, pre-merge): 172.26. Severely under-budgeted (7 epochs). Sent back as n_hidden=192 (more manageable).
- **Decoupled heads** (askeladd): Still WIP from round 1.

### Potential next directions (round 3+)
- **Compose winners**: combine Huber + FiLM + OneCycleLR+EMA once individual rebases are scored.
- **Surface-only Huber**: apply Huber to volume nodes, MSE to surface (alphonse follow-up).
- **Per-sample importance weighting**: weight each sample's loss by 1/y_std_sample.
- **Relative MAE in physical space**: scale-invariant loss for multi-Re training.
- **dsdf shape descriptor augmentation** (dims 4-11): deeper geometry augmentation vs. scalar AoA/NACA.
- **n_layers=7** (depth scaling): orthogonal to width scaling experiments.
- **EMA warmup ramp**: linear ramp of EMA decay from 0 → 0.999 over first 5 epochs (eliminates early-epoch lag).
- **Two-stage curriculum**: fit volume first, then finetune with high surf_weight_p.
- **Mesh-aware positional encoding**: signed distance / arc length as Fourier features.
