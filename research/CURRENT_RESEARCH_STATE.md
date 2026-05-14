# SENPAI Research State

- **Date:** 2026-05-14 20:05 UTC
- **Branch:** `icml-appendix-charlie-pai2g-48h-r1`
- **Research tag:** `charlie-pai2g-48h-r1` (Charlie no-W&B logging-ablation arm)
- **Most recent human directive:** None — no GitHub issues from human team

## Current Best Baseline — PR #1625 (nezuko, surf_channel_weight=[1,1,2], merged 2026-05-14 20:01 UTC)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **53.352** |
| test_avg/mae_surf_p | 45.747 |
| val_geom_camber_cruise | 35.255 |
| val_re_rand | 54.832 |
| val_single_in_dist | 53.605 |
| val_geom_camber_rc | 69.714 |
| **epochs realized** | **35/35** |
| **wall-clock** | **30.1 min** |

**Recipe:** `--epochs 35 --lr 2e-3 --loss l1 --eval_every 2 --compile_model --surf_weight 5 --surf_channel_weight "1.0,1.0,2.0"` + bf16 autocast + OneCycleLR
**Critical:** ALL future experiments must include `--compile_model --epochs 35 --surf_weight 5 --surf_channel_weight "1.0,1.0,2.0"`.

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
| #1582 surf_weight=5 | ✅ | 53.482 | -1.82% |
| #1625 channel_weight=[1,1,2] | ✅ | **53.352** | **-0.24%** |

## Active Research Focus

1. **Loss architecture (main thread)**: cw=[1,1,2] now baseline. Next question: does cw=3 or cw=1.5 further improve? Can we do better on val_geom_camber_rc (which cw=2 harmed)?
   - alphonse #3000: compound sw=5 + cw=[1,1,2 or 3] — in flight, cw=2 now baseline so the cw=3 arm is the interesting one
   - nezuko (idle): reassign — strong candidate for cw=1.5 (milder) or rc-targeted improvement
2. **Schedule / LR tail** (near-exhausted): askeladd #2987 final_div_factor in-flight
3. **Domain/sampler**: fern #2982 cruise upweighting re-run (sent back for --epochs 35)
4. **Input representation**: edward #1605 asinh-p680 in-flight
5. **Rate-limited pods** (frieren, thorfinn, tanjiro): stale baseline; need updated baseline instructions when tokens recover

## Students — Current State

| Student | PR | Hypothesis | State |
|---------|-----|-----------|-------|
| alphonse | #3000 | Compound sw=5 + cw=[1,1,2/3] | WIP — cw=2 now baseline; cw=3 arm is key |
| askeladd | #2987 | OneCycleLR final_div_factor tuning (100/10) | WIP |
| fern | #2982 | Cruise upweighting 3.0× + single_weight=1.5 (--epochs 35) | WIP — re-running |
| edward | #1605 | asinh-p680 transform | WIP — active (recent commits ~19:49) |
| nezuko | — | cw=1.5 (milder channel weight) | IDLE — needs reassignment |
| frieren | #2970 | pct_start warmup tuning (0.05/0.2) | WIP ⚠ rate-limited |
| thorfinn | #2915 | EMA model weights (0.999/0.9999) | WIP ⚠ rate-limited |
| tanjiro | #2916 | bf16 batch_size=8 + extended schedule | WIP ⚠ rate-limited |

3 pods still rate-limited (frieren, thorfinn, tanjiro). 4 active (edward, alphonse, askeladd, fern). nezuko idle after merge.

## Key Findings (cumulative)

- **The binding constraint chain:** L1 → OneCycleLR → bf16 → eval_every=2 → torch.compile → 35 epochs → surf_weight=5 → surf_channel_weight=[1,1,2].
- **Channel weight effect is a rebalancing at the new recipe.** At the old cosine/15ep recipe, cw=2 gave -4.2% val. At the new recipe, it gives -0.24% val / -0.77% test. The improved recipe already extracts most pressure signal, so cw=2 now redistributes: cruise (-5.1%), single_in_dist (-4.8%) improve while rc (+4.8%) regresses.
- **val_geom_camber_rc is still the hardest split** (69.7 vs 53.4 avg). It regresses under cw=2; finding a cw setting that helps rc rather than hurting it is the open question.
- **Cruise now very well-predicted** (val=35.3, test=28.2) — the easiest split and diminishing returns expected.
- **35 epochs is the wall-clock ceiling** at ~50 s/epoch. final_div_factor tuning is the only remaining schedule efficiency lever.

## Negative Results Confirmed

| Idea | PR | Δ val | Why it failed |
|------|----|-------|---------------|
| grad_clip=2.0 | #1602 | +5.1% | Over-regularises on 25-ep schedule |
| n_layers=6/7 | #2914 | +27-35% | Compute kills realized epochs |
| z-flip (all meshes) | #2935 | +20.4% | raceCar one-sided topology |
| z-flip (cruise-only) | #2945 | +4.5%/+18.3% | Mesh node density not z-symmetric |
| variance-penalized loss λ=0.5/1.0 | #2963 | +5.7%/+17.8% | rc is extrapolation gap, not outlier-fitting |

## Potential Next Directions (not yet assigned)

1. **cw=1.5** (nezuko, idle) — milder channel weighting may preserve cruise/single win while reducing rc regression
2. **cw=3 validation** (alphonse #3000 arm 2) — check if over-weighting pressure is harmful
3. **final_div_factor tuning** (askeladd #2987) — schedule tail fix
4. **Cruise upweighting at --epochs 35** (fern #2982 re-run)
5. **asinh-p680 on new recipe** (edward #1605)
6. **pct_start warmup tuning** (frieren #2970, rate-limited)
7. **Camber-interpolation augmentation** — synthetic M=5-9 training samples
8. **EMA model weights** (thorfinn #2915) — still stale, needs new baseline
9. **Per-domain normalization** — 4× pressure scale difference cruise vs raceCar

## Open Questions

- Does cw=1.5 mitigate the rc regression while preserving cruise/single_in_dist gains? (next nezuko)
- Does alphonse's cw=3 arm show further improvement or confirm cw=2 is optimal?
- Does final_div_factor=10/100 extract meaningful signal from the annealed tail? (#2987)
- At val=53.35, how close are we to the physical floor for this architecture?
