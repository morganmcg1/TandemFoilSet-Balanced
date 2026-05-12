# SENPAI Research State

- **Date**: 2026-05-12 20:00 UTC
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r3` (base `icml-appendix-charlie`)
- **Research tag**: `charlie-pai2g-24h-r3`
- **Students (8)**: charliepai2g24h3-{alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn}
- **Per-run budget**: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50` (caps)
- **Logging**: local JSONL only (no W&B in this arm)

## Latest human direction

None received.

## Current best baseline

**val_avg/mae_surf_p = 112.546** — PR #1520 (fern, OneCycleLR + EMA=0.999).
Test 3-split proxy = 110.862. 
Stack: `grad_clip=1.0 + wd=1e-3 + OneCycleLR(peak=5e-3, pct_start=0.05) + EMA(0.999)`.

## Current student assignments

| Student | PR | Slug | Status |
|---|---|---|---|
| alphonse | #1484 | `huber-pressure-loss` | WIP — rebase: Huber d=0.5+d=1.0 on new stack |
| askeladd | #1488 | `decoupled-channel-heads` | WIP |
| edward | #1490 | `scale-model-256-v2` | WIP — rebase: n_hidden=192, n_head=6 on new stack |
| fern | #1543 | `logcosh-loss` | WIP — log-cosh on new stack |
| frieren | #1492 | `mlp-ratio-4-wider-ffn` | WIP — rebase: mlp_ratio=4, --epochs 12 |
| nezuko | #1493 | `more-slices-128-v2` | WIP — rebase: slice_num=128, --epochs 11 |
| tanjiro | #1494 | `re-film-conditioning-v2` | WIP — rebase: FiLM on new stack, --epochs 14 |
| thorfinn | #1495 | `geometry-aoa-augmentation-v2` | WIP — rebase: AoA+NACA jitter on new stack |

## Research themes and findings

### Confirmed winners (merged)
1. **Optimization hygiene** (PR #1491): grad_clip=1.0 + wd=1e-3 → 115.40 → baseline.
2. **Scheduler + EMA** (PR #1520): OneCycleLR + EMA=0.999 → **112.55** → new baseline.

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
