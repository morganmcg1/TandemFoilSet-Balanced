# SENPAI Research State

- **Date:** 2026-05-14 22:50 UTC
- **Branch:** `icml-appendix-charlie-pai2g-48h-r1`
- **Research tag:** `charlie-pai2g-48h-r1` (Charlie no-W&B logging-ablation arm)
- **Most recent human directive:** None — no GitHub issues from human team

## Current Best Baseline — PR #2970 (frieren, pct_start=0.2, merged 2026-05-14 22:38 UTC)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **51.817** |
| test_avg/mae_surf_p | 44.616 |
| val_geom_camber_cruise | 31.929 |
| val_re_rand | 51.773 |
| val_single_in_dist | 55.477 |
| val_geom_camber_rc | 68.090 |
| **epochs realized** | **35/35** |

**Recipe:** `--epochs 35 --lr 2e-3 --loss l1 --eval_every 2 --compile_model --surf_weight 5 --surf_channel_weight "1.0,1.0,2.0" --pct_start 0.2` + bf16 autocast + OneCycleLR
**Critical:** ALL future experiments must include `--compile_model --epochs 35 --surf_weight 5 --surf_channel_weight "1.0,1.0,2.0" --pct_start 0.2`.

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
| #1625 channel_weight=[1,1,2] | ✅ | 53.352 | -0.24% |
| #2970 pct_start=0.2 | ✅ | **51.817** | **-2.88%** |

## Active Research Focus

### 1. Schedule axis — highly productive, still open
- pct_start=0.2 merged, confirming warmup is a productive lever
- **frieren #3060**: pct_start extension 0.25/0.30 — probes whether even more warmup continues to improve
- **askeladd #2987**: final_div_factor 100/10 — orthogonal cooldown axis (now needs `--pct_start 0.2` in recipe)

### 2. OOD regularization — suite under test
- `val_geom_camber_rc` (68.1, down from 69.7 — pct_start already helped) remains the key bottleneck
- **thorfinn #3052**: weight_decay 5e-4/1e-3 — needs rebase for pct_start (sent back)
- **tanjiro #3055**: DropPath 0.05/0.10 — block-level stochastic depth

### 3. Channel weight axis — left half under test
- **nezuko #3051**: cw=1.5/1.25 sweep redux — may preserve cruise/single wins while reducing rc penalty

### 4. Surface weight re-tuning
- **alphonse #3029**: sw=4/6 re-sweep at cw=[1,1,2] + pct_start=0.2

### 5. Domain/sampler
- **fern #2982**: cruise upweighting 3.0× + single_weight=1.5 (stale, needs re-run with full recipe)

### 6. Input representation
- **edward #1605**: asinh-p680 transform — branch is clean (MERGEABLE); needs to run with new recipe including `--pct_start 0.2`

## Students — Current State

| Student | PR | Hypothesis | State |
|---------|-----|-----------|-------|
| alphonse | #3029 | surf_weight re-sweep (sw=4/6) at cw=[1,1,2] | WIP |
| askeladd | #2987 | OneCycleLR final_div_factor tuning (100/10) | WIP — needs to use --pct_start 0.2 |
| fern | #2982 | Cruise upweighting 3.0× + single_weight=1.5 | WIP — stale, needs full recipe |
| edward | #1605 | asinh-p680 on full current recipe | WIP — branch clean, needs to run |
| nezuko | #3051 | surf_channel_weight cw=1.5/1.25 sweep redux | WIP |
| frieren | #3060 | pct_start extension 0.25/0.30 | WIP — newly assigned |
| thorfinn | #3052 | AdamW weight_decay sweep 5e-4/1e-3 | WIP — needs rebase for pct_start |
| tanjiro | #3055 | DropPath 0.05/0.10 (stochastic depth) | WIP |

0 idle students. 8 active experiments.

## Key Findings (cumulative)

- **The binding constraint chain:** L1 → OneCycleLR → bf16 → eval_every=2 → torch.compile → 35 epochs → surf_weight=5 → surf_channel_weight=[1,1,2] → **pct_start=0.2**.
- **Schedule axis is productive:** pct_start=0.2 wins −2.88% via (1) avoids bf16 instability at fast warmup (ep-10 spike in 0.05 arm), (2) keeps ~30% higher LR during productive tail. Crossover at ep 28 → further extension may still gain.
- **pct_start already improved rc:** val_geom_camber_rc went from 69.714 → 68.090 at pct_start=0.2. The OOD split improved without any explicit regularization.
- **Channel weight axis is inverted-U:** cw=2 is local optimum; left half (cw<2) still under test.
- **batch_size=8 fails, EMA fails:** Both closed as dead ends.

## Negative Results Confirmed

| Idea | PR | Δ val | Why it failed |
|------|----|-------|---------------|
| grad_clip=2.0 | #1602 | +5.1% | Over-regularises on 25-ep schedule |
| n_layers=6/7 | #2914 | +27-35% | Compute kills realized epochs |
| z-flip (all meshes) | #2935 | +20.4% | raceCar one-sided topology |
| z-flip (cruise-only) | #2945 | +4.5%/+18.3% | Mesh node density not z-symmetric |
| variance-penalized loss λ=0.5/1.0 | #2963 | +5.7%/+17.8% | rc is extrapolation gap, not outlier-fitting |
| EMA weights (decay 0.999/0.9999) | #2915 | +2.1%/+251% | OneCycleLR cooldown = meaningful descent |
| surf_channel_weight cw=3 | #3000 | +3.81% | cw axis inverted-U; cw=2 is optimum |
| MLP dropout 0.05/0.10 | #3027 | +6.8%/+1.6% | Slows convergence; OOD gain dominated by ID regression |
| batch_size=8 | #2916 | +49.9% | compile OOMs; nocompile halves realized epochs |
| pct_start=0.05 | #2970 arm A | +2.5% | Fast warmup → bf16 instability spike; lower tail LR |

## Open Questions

- Does pct_start continue to improve beyond 0.2 (0.25/0.30)? Crossover at ep 28 suggests yes. (#3060)
- Does final_div_factor compound with pct_start=0.2? (#2987)
- Does DropPath give rc OOD benefit without convergence cost? (#3055)
- Does weight_decay give rc OOD benefit? (#3052)
- Does asinh-p680 still improve on the pct_start=0.2 baseline? (#1605)

## Potential Next Directions (not yet assigned)

1. **Camber-interpolation augmentation** — synthetic M=5-9 training samples directly attack rc extrapolation gap
2. **Per-domain normalization** — 4× pressure scale difference cruise vs raceCar
3. **Combine pct_start + lower div_factor** — initial LR = max_lr/div_factor; lower div_factor starts warmup from a higher floor
4. **Deeper-block-only regularization** — apply DropPath only to blocks 3-5; preserve early feature extraction
5. **pct_start=0.4** — if 0.30 still wins over 0.25, push further
