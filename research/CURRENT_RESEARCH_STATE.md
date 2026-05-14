# SENPAI Research State

- **Date:** 2026-05-14 17:15 UTC
- **Branch:** `icml-appendix-charlie-pai2g-48h-r1`
- **Research tag:** `charlie-pai2g-48h-r1` (Charlie no-W&B logging-ablation arm)
- **Most recent human directive:** None — no GitHub issues from human team

## Current Best Baseline — PR #2954 (askeladd, torch.compile, merged 2026-05-14 17:12 UTC)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **65.953** |
| test_avg/mae_surf_p | 56.825 |
| val_geom_camber_cruise | 49.899 |
| val_re_rand | 64.475 |
| val_single_in_dist | 70.437 |
| val_geom_camber_rc | 79.001 |
| **epochs realized** | **25/25** |
| **wall-clock** | **21.7 min** |

**Recipe:** `--epochs 25 --lr 2e-3 --loss l1 --eval_every 2 --compile_model` + bf16 autocast + OneCycleLR
**Critical:** ALL future experiments MUST include `--compile_model`. Without it only 19 epochs fit.
**Key insight:** `torch.compile` gives 1.86× throughput (50 s vs 94 s/epoch), all 25 epochs fit in budget, VRAM drops 32.95→23.8 GB.

## Progress Path

| PR | Merged | val_avg | Improvement |
|----|--------|---------|-------------|
| MSE baseline | — | ~218 | — |
| #1355 pure L1 | ✅ | 94.291 | -57% |
| #1581 OneCycleLR @2e-3 | ✅ | 85.615 | -9.2% |
| #1405 bf16 + epochs=25 | ✅ | 73.295 | -14.4% |
| #2936 eval_every=2 | ✅ | 72.694 | -0.82% |
| #2954 torch.compile | ✅ | **65.953** | **-9.3%** |

## Active Research Focus

**Every improvement to date is a training-efficiency improvement**: more epochs, faster training, better LR schedule utilization within the 30-min wall-clock cap. The pattern is clear: the OneCycleLR tail (low-LR fine-tuning zone) is extremely productive, and maximizing epochs in that regime drives all gains.

**Current frontier questions:**
1. **Horizon extension**: With compile, can we run 35 epochs and get more from the extended LR schedule? (askeladd #2967)
2. **Loss quality**: Does variance-penalized surf loss (mean + λ·std) improve the rc split outlier nodes? (fern #2963)
3. **Other efficiency levers**: EMA, batch_size=8, domain re-weighting — all still in flight on stalled pods

## Students — Current State

| Student | PR | Hypothesis | State |
|---------|-----|-----------|-------|
| askeladd | #2967 | OneCycleLR horizon extension 30/35 ep (new) | WIP |
| fern | #2963 | Variance-penalized surface loss λ=0.5/1.0 | WIP |
| frieren | #2913 | OneCycle epoch-horizon sweep (--epochs 30/40) | WIP ⚠ rate-limited |
| thorfinn | #2915 | EMA model weights (decay 0.999/0.9999) | WIP ⚠ rate-limited |
| tanjiro | #2916 | bf16 batch_size=8 + extended schedule | WIP ⚠ rate-limited |
| edward | #1605 | asinh-p680 + OneCycle re-run on bf16 baseline | WIP ⚠ rate-limited + rebase needed |
| nezuko | #1625 | surf_channel_weight cw=2 re-run on bf16 baseline | WIP ⚠ rate-limited + rebase needed |
| alphonse | #1582 | surf_weight=5 re-run on bf16 baseline | WIP ⚠ rate-limited |

6 stale pods are rate-limited on shared student token (user 20516801). May resume when token window rotates.

**IMPORTANT for stale pods:** When frieren/thorfinn/tanjiro/alphonse/edward/nezuko resume, ALL their experiments must be updated to include `--compile_model`. Without it, they can only reach 19 epochs vs 25 with compile. These re-runs may need to be sent back with updated instructions.

## Key Findings (cumulative)

- **torch.compile is transformative.** 1.86× speedup, 50 s/epoch, full 25-epoch schedule in 21.7 min, VRAM drops 32→24 GB. Zero architecture/hyperparameter change. ALL future experiments must use `--compile_model`.
- **val_geom_camber_rc persistent weakness.** Still the worst split at 79.001 (vs 49.9 for cruise). Improving it is the highest-value research direction.
- **val_single_in_dist improved dramatically.** 79.89 → 70.44 with compile (more tail epochs help this split most).
- **Wall-clock budget now has headroom.** With 21.7 min for 25 epochs, there are 8+ min unused. Extending to 35 epochs will use most of it.
- **Augmentation directions closed.** z-flip (both full and cruise-only) failed. The training dynamics interaction is too fragile.
- **Gradient clipping hurts.** On 25-ep OneCycleLR, the schedule provides enough regularization.
- **Depth/width scaling fails.** Compute overhead reduces realized epochs below the productive tail.

## Negative Results Confirmed

| Idea | PR | Δ val | Why it failed |
|------|----|-------|---------------|
| grad_clip=2.0 | #1602 | +5.1% | Over-regularises on 25-ep schedule |
| n_layers=6/7 | #2914 | +27-35% | Compute kills realized epochs |
| z-flip (all meshes) | #2935 | +20.4% | raceCar one-sided topology |
| z-flip (cruise-only) | #2945 | +4.5%/+18.3% | Mesh node density not z-symmetric; training dynamics disturbed |

## Potential Next Directions (not yet assigned)

1. **OneCycleLR horizon 30/35 epochs** (assigned: askeladd #2967) — use the 8.3 min budget surplus
2. **Variance-penalized loss** (assigned: fern #2963) — target rc split outlier nodes
3. **pct_start warmup tuning** (H2: 0.05 or 0.3) — shorter/longer warmup fraction
4. **Per-domain normalization** (H5) — addresses 4× pressure scale difference cruise vs raceCar
5. **Domain re-weighting** (H9) — increase cruise/tandem weight in sampler
6. **Compound stacking**: SW=5 + CW=2 + asinh (if they validate on new compile+compile baseline)
7. **Higher peak LR with extended schedule**: lr=3e-3 with 35 epochs — more aggressive warmup
8. **pct_start=0.05 to reach peak LR faster** with extended epoch budget

## Open Questions

- Does the OneCycleLR horizon extension improve results when full schedule can actually run? (askeladd #2967 will answer)
- What is the true floor? val_avg was at 218 (MSE baseline), now at 66 — is there a physical lower bound for TandemFoilSet at this architecture scale?
- Do the stalled experiments (EMA, batch_size=8, surf_weight=5, etc.) still have value now that the baseline is 65.95? They need to be re-run with `--compile_model` to be comparable.
