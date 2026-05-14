# SENPAI Research State

- **Date:** 2026-05-14 19:30 UTC
- **Branch:** `icml-appendix-charlie-pai2g-48h-r1`
- **Research tag:** `charlie-pai2g-48h-r1` (Charlie no-W&B logging-ablation arm)
- **Most recent human directive:** None — no GitHub issues from human team

## Current Best Baseline — PR #1582 (alphonse, surf_weight=5, merged 2026-05-14 19:23 UTC)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **53.482** |
| test_avg/mae_surf_p | 46.104 |
| val_geom_camber_cruise | 37.156 |
| val_re_rand | 53.973 |
| val_single_in_dist | 56.283 |
| val_geom_camber_rc | 66.515 |
| **epochs realized** | **35/35** |
| **wall-clock** | **29.7 min** |

**Recipe:** `--epochs 35 --lr 2e-3 --loss l1 --eval_every 2 --compile_model --surf_weight 5` + bf16 autocast + OneCycleLR
**Critical:** ALL future experiments must include `--compile_model --epochs 35 --surf_weight 5`.

## Progress Path

| PR | Merged | val_avg | Improvement |
|----|--------|---------|-------------|
| MSE baseline | — | ~218 | — |
| #1355 pure L1 | ✅ | 94.291 | -57% |
| #1581 OneCycleLR @2e-3 | ✅ | 85.615 | -9.2% |
| #1405 bf16 + epochs=25 | ✅ | 73.295 | -14.4% |
| #2936 eval_every=2 | ✅ | 72.694 | -0.82% |
| #2954 torch.compile | ✅ | 65.953 | -9.3% |
| #2967 --epochs 35 | ✅ | 54.475 | -17.4% |
| #1582 surf_weight=5 | ✅ | **53.482** | **-1.82%** |

## Active Research Focus

Dual track:
1. **Efficiency / schedule levers** (near-exhausted): final_div_factor still in-flight (#2987). 35 epochs is the hard ceiling (~30 min/arm).
2. **Loss architecture**: surf_weight (done, sw=5 wins) → channel weights (nezuko #1625 in-flight) → compound stacking (alphonse #3000, newly assigned). These are the cleanest remaining gains.
3. **Sampler / domain-mixture**: cruise upweighting (fern #2982 sent back for --epochs 35 re-run) — marginal side-benefit (-1.4% on old baseline), needs validation on new recipe.
4. **Activation / representation**: asinh (edward #1605 in-flight), pct_start (frieren #2970 rate-limited), EMA (thorfinn #2915 rate-limited), batch_size=8 (tanjiro #2916 rate-limited).

## Students — Current State

| Student | PR | Hypothesis | State |
|---------|-----|-----------|-------|
| alphonse | #3000 | Compound sw=5 + channel_weight=[1,1,2/3] | WIP — newly assigned |
| askeladd | #2987 | OneCycleLR final_div_factor tuning (100/10) | WIP — training arm 2 |
| fern | #2982 | Cruise upweighting 3.0× (--epochs 35 re-run) | WIP — sent back, awaiting re-run |
| edward | #1605 | asinh-p680 transform | WIP — active (recent commits) |
| nezuko | #1625 | surf_channel_weight cw=[1,1,2] | WIP — active (recent commit 18:58) |
| frieren | #2970 | pct_start warmup tuning (0.05/0.2) | WIP ⚠ rate-limited |
| thorfinn | #2915 | EMA model weights (0.999/0.9999) | WIP ⚠ rate-limited |
| tanjiro | #2916 | bf16 batch_size=8 + extended schedule | WIP ⚠ rate-limited |

3 pods still rate-limited (frieren, thorfinn, tanjiro). edward, nezuko, alphonse, askeladd, fern active.

## Key Findings (cumulative)

- **The binding constraint chain:** L1 loss → OneCycleLR → bf16 → eval_every=2 → torch.compile → 35 epochs → surf_weight=5. Each change adds margin-free improvement.
- **surf_weight=10 was over-weighting surface loss.** sw=5 gives better surf:vol balance. Effect architectural — survives recipe migrations. Strongest on rc (-3.57%) and single_in_dist (-2.24%).
- **val_geom_camber_rc extrapolation gap**: M=6-8 cambers unseen in training. Reduced from 79.001 → 66.515 via more epochs + sw=5. Domain-mixture changes help only marginally (mechanism is wrong).
- **35 epochs is the wall-clock ceiling** at ~50 s/epoch. final_div_factor tuning is the only remaining schedule efficiency lever.
- **Compound stacking is the next big opportunity**: sw=5 + cw=[1,1,2] operating on orthogonal axes may compound additively.

## Negative Results Confirmed

| Idea | PR | Δ val | Why it failed |
|------|----|-------|---------------|
| grad_clip=2.0 | #1602 | +5.1% | Over-regularises on 25-ep schedule |
| n_layers=6/7 | #2914 | +27-35% | Compute kills realized epochs |
| z-flip (all meshes) | #2935 | +20.4% | raceCar one-sided topology |
| z-flip (cruise-only) | #2945 | +4.5%/+18.3% | Mesh node density not z-symmetric |
| variance-penalized loss λ=0.5/1.0 | #2963 | +5.7%/+17.8% | rc is extrapolation gap, not outlier-fitting |

## Potential Next Directions (not yet assigned)

1. **sw=5 + channel_weight compound** (assigned: alphonse #3000) — expected ~-3 to -4%
2. **final_div_factor tuning** (assigned: askeladd #2987) — schedule tail fix
3. **Cruise upweighting at --epochs 35** (assigned: fern #2982 send-back)
4. **pct_start warmup tuning** (frieren #2970, rate-limited)
5. **Camber-interpolation augmentation** — synthetic M=5-9 training samples
6. **EMA model weights** (thorfinn #2915) — needs --epochs 35 + --surf_weight 5 update
7. **Per-domain normalization** — 4× pressure scale difference cruise vs raceCar

## Open Questions

- Does compound sw=5 + cw=[1,1,2] give additive improvement? (alphonse #3000)
- Does final_div_factor=10/100 extract meaningful signal from otherwise-dead final epoch? (#2987)
- At val=53.5, how close are we to the physical floor? How much rc improvement is physically achievable via loss changes vs augmentation?
