# SENPAI Research State

- **Date:** 2026-05-14 18:38 UTC
- **Branch:** `icml-appendix-charlie-pai2g-48h-r1`
- **Research tag:** `charlie-pai2g-48h-r1` (Charlie no-W&B logging-ablation arm)
- **Most recent human directive:** None — no GitHub issues from human team

## Current Best Baseline — PR #2967 (askeladd, --epochs 35, merged 2026-05-14 18:35 UTC)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **54.475** |
| test_avg/mae_surf_p | 47.043 |
| val_geom_camber_cruise | 37.613 |
| val_re_rand | 53.733 |
| val_single_in_dist | 57.573 |
| val_geom_camber_rc | 68.980 |
| **epochs realized** | **35/35** |
| **wall-clock** | **29.8 min** |

**Recipe:** `--epochs 35 --lr 2e-3 --loss l1 --eval_every 2 --compile_model` + bf16 autocast + OneCycleLR
**Critical:** ALL future experiments MUST include `--compile_model --epochs 35`.
- Without `--compile_model`: only 19 epochs fit, LR schedule starved.
- Without `--epochs 35`: LR floor at ep 25, 10 productive tail epochs wasted.

## Progress Path

| PR | Merged | val_avg | Improvement |
|----|--------|---------|-------------|
| MSE baseline | — | ~218 | — |
| #1355 pure L1 | ✅ | 94.291 | -57% |
| #1581 OneCycleLR @2e-3 | ✅ | 85.615 | -9.2% |
| #1405 bf16 + epochs=25 | ✅ | 73.295 | -14.4% |
| #2936 eval_every=2 | ✅ | 72.694 | -0.82% |
| #2954 torch.compile | ✅ | 65.953 | -9.3% |
| #2967 --epochs 35 | ✅ | **54.475** | **-17.4%** |

## Active Research Focus

**Every improvement to date is a training-efficiency improvement**: more epochs, faster training, better LR schedule utilization within the 30-min wall-clock cap. The pattern is relentless: the OneCycleLR mid-tail (LR ∈ [5e-4, 5e-6]) is extremely productive, and maximizing epochs in that regime drives all gains.

**Current frontier questions:**
1. **final_div_factor tuning**: Can keeping the final epochs at a productive LR (not 8e-9 dead floor) extract more from the 35-epoch schedule? (askeladd #2987 — new)
2. **Cruise domain upweighting**: Does upweighting cruise in the sampler help the rc extrapolation gap? (fern #2982 — running)
3. **pct_start tuning**: Shorter warmup to reach peak LR faster? (frieren #2970 — rate-limited)
4. **Stalled experiments**: EMA, batch_size=8, surf_weight=5, asinh, channel weights — all need `--compile_model --epochs 35` updates when rate-limit clears.

## Students — Current State

| Student | PR | Hypothesis | State |
|---------|-----|-----------|-------|
| askeladd | #2987 | OneCycleLR final_div_factor tuning (100/10) | WIP — newly assigned |
| fern | #2982 | Cruise domain upweighting 2x/3x | WIP — training |
| frieren | #2970 | OneCycleLR pct_start warmup tuning (0.05/0.2) | WIP ⚠ rate-limited |
| thorfinn | #2915 | EMA model weights (decay 0.999/0.9999) | WIP ⚠ rate-limited |
| tanjiro | #2916 | bf16 batch_size=8 + extended schedule | WIP ⚠ rate-limited |
| edward | #1605 | asinh transform + scale sweep | WIP ⚠ rate-limited + rebase needed |
| nezuko | #1625 | surf_channel_weight sweep | WIP — rebased, training |
| alphonse | #1582 | surf_weight=5 re-run | WIP ⚠ rate-limited |

5 pods still rate-limited on shared token. nezuko (#1625) and fern (#2982) are active.

**IMPORTANT for stale pods:** When frieren/thorfinn/tanjiro/alphonse/edward resume, ALL their experiments MUST be updated to `--compile_model --epochs 35` (not 25). The new baseline is 35 epochs. Without both flags, results are not comparable and will likely regress vs the new baseline.

## Key Findings (cumulative)

- **The binding constraint is LR schedule, not VRAM or architecture.** Every win traces back to maximizing epochs in the productive mid-tail of OneCycleLR. torch.compile (50 s/epoch), eval_every=2, and --epochs 35 are all different levers on the same mechanism.
- **Final LR collapse wastes one epoch.** `final_div_factor=1e4` drives LR to 8e-9 at ep 35; the last epoch contributes only 0.10 val points. Reducing to 100 or 10 could keep final epochs productive.
- **val_geom_camber_rc extrapolation gap persists.** Still at 68.980 after all efficiency gains (vs 37.613 for cruise). Mechanism: M=6-8 raceCar cambers unseen in training. Sampler re-weighting and camber augmentation are the right levers.
- **Augmentation/loss-shape changes exhausted.** z-flip (mesh topology), variance penalty (wrong mechanism for rc extrapolation), gradient clipping, depth/width scaling — all confirmed negative.
- **35 epochs is the hard wall-clock ceiling** at 51 s/epoch + compile overhead. ~40 epochs would exceed 30 min. Squeezing more out of the existing schedule (final_div_factor, pct_start) is the remaining efficiency lever.

## Negative Results Confirmed

| Idea | PR | Δ val | Why it failed |
|------|----|-------|---------------|
| grad_clip=2.0 | #1602 | +5.1% | Over-regularises on 25-ep schedule |
| n_layers=6/7 | #2914 | +27-35% | Compute kills realized epochs |
| z-flip (all meshes) | #2935 | +20.4% | raceCar one-sided topology |
| z-flip (cruise-only) | #2945 | +4.5%/+18.3% | Mesh node density not z-symmetric |
| variance-penalized loss λ=0.5/1.0 | #2963 | +5.7%/+17.8% | rc is extrapolation gap, not outlier-fitting |

## Potential Next Directions (not yet assigned)

1. **final_div_factor=100/10** (assigned: askeladd #2987) — keep final epochs productive
2. **Cruise domain upweighting 2x/3x** (assigned: fern #2982) — rc via geometric coverage
3. **pct_start warmup tuning** (assigned: frieren #2970) — shorter/longer warmup fraction
4. **Per-domain normalization** — 4× pressure scale difference cruise vs raceCar
5. **Camber-interpolation augmentation** — synthetic M=5-9 via feature blending
6. **Stale experiments** (EMA, batch_size=8, surf_weight, asinh, channel weights) — need --epochs 35 update

## Open Questions

- Does reducing final_div_factor meaningfully improve the schedule tail? (askeladd #2987 answers this)
- Does cruise upweighting help rc extrapolation, or does it hurt raceCar-specific splits? (fern #2982)
- At val=54.5, how close are we to the physical floor? What do the worst predictions look like spatially?
- Do the stalled experiments still have value vs the new 54.475 baseline? All need --epochs 35 update.
