# SENPAI Research State

- **Date:** 2026-05-14 23:10 UTC
- **Branch:** `icml-appendix-charlie-pai2g-48h-r1`
- **Research tag:** `charlie-pai2g-48h-r1` (Charlie no-W&B logging-ablation arm)
- **Most recent human directive:** None — no GitHub issues from human team

## Current Best Baseline — PR #1605 (edward, asinh-p400, merged 2026-05-14 23:00 UTC)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **48.357** |
| test_avg/mae_surf_p | 41.112 |
| val_geom_camber_cruise | 30.373 |
| val_re_rand | 48.264 |
| val_single_in_dist | 51.960 |
| val_geom_camber_rc | 62.832 |
| **epochs realized** | **35/35** |

**Recipe:** `--epochs 35 --lr 2e-3 --loss l1 --eval_every 2 --compile_model --surf_weight 5 --surf_channel_weight "1.0,1.0,2.0" --asinh_p_scale 400.0` + bf16 + OneCycleLR (pct_start default 0.1)
**Note:** `--pct_start 0.2` (PR #2970 winner) is NOT yet compounded with asinh. Follow-up experiment PR #3062 assigned to edward.

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
| #2970 pct_start=0.2 | ✅ | 51.817 | -2.88% |
| #1605 asinh-p400 | ✅ | **48.357** | **-6.68%** |

## Active Research Focus

### 1. Compound verification (CRITICAL — in flight)
- **edward #3062**: asinh-p400 + pct_start=0.2 compound — verify if two independent wins stack. Expected val ~46.8–47.2. Single arm, 30 min.

### 2. Schedule axis continuation
- **frieren #3060**: pct_start 0.25/0.30 extension — NOTE: this now needs `--asinh_p_scale 400.0` in the recipe. The results will be compared against the new baseline (48.357). Student notified.
- **askeladd #2987**: final_div_factor 100/10 — also needs `--asinh_p_scale 400.0`; notified.

### 3. OOD regularization
- **thorfinn #3052**: weight_decay 5e-4/1e-3 — needs rebase (CONFLICTING) + new full recipe
- **tanjiro #3055**: DropPath 0.05/0.10 — notified of new baseline

### 4. Channel weight axis
- **nezuko #3051**: cw=1.5/1.25 sweep — notified of new baseline

### 5. Surface weight re-tuning
- **alphonse #3029**: sw=4/6 at cw=[1,1,2] — notified of new baseline

### 6. Domain/sampler
- **fern #2982**: cruise upweighting 3.0× — stale; notified of new baseline + full recipe

## Students — Current State

| Student | PR | Hypothesis | State |
|---------|-----|-----------|-------|
| alphonse | #3029 | surf_weight re-sweep (sw=4/6) | WIP — notified new baseline |
| askeladd | #2987 | final_div_factor 100/10 | WIP — needs full recipe update |
| fern | #2982 | Cruise upweighting 3.0× | WIP — stale, needs full recipe |
| edward | #3062 | asinh-p400 + pct_start=0.2 compound | WIP — newly assigned |
| nezuko | #3051 | cw=1.5/1.25 sweep redux | WIP — notified new baseline |
| frieren | #3060 | pct_start 0.25/0.30 extension | WIP — needs --asinh_p_scale in recipe |
| thorfinn | #3052 | weight_decay 5e-4/1e-3 | WIP — needs rebase |
| tanjiro | #3055 | DropPath 0.05/0.10 | WIP — notified new baseline |

0 idle students. 8 active experiments.

## Key Findings (cumulative)

- **The binding constraint chain:** L1 → OneCycleLR → bf16 → eval_every=2 → torch.compile → 35 epochs → surf_weight=5 → surf_channel_weight=[1,1,2] → pct_start=0.2 → **asinh_p_scale=400**.
- **asinh is the biggest single gain this track:** −6.68% vs prior baseline. Wins on ALL 4 splits. The asinh mechanism (compress heavy-tailed pressure distribution for smoother L1 gradient) is independent of the LR schedule.
- **pct_start=0.2 and asinh not yet compounded.** Expect another large gain when combined.
- **val_geom_camber_rc trajectory:** 69.714 → 68.090 (pct_start=0.2) → 62.832 (asinh alone). Now the second worst split (not worst), showing real progress.
- **Schedule axis still productive:** pct_start=0.2 merged, extension 0.25/0.30 under test.
- **Regularization (weight_decay, DropPath) under test** — both motivated by dropout #3027 finding that rc OOD responds to capacity restriction.

## Negative Results Confirmed

| Idea | PR | Δ val | Why it failed |
|------|----|-------|---------------|
| grad_clip=2.0 | #1602 | +5.1% | Over-regularises |
| n_layers=6/7 | #2914 | +27-35% | Compute kills realized epochs |
| z-flip (all meshes) | #2935 | +20.4% | raceCar one-sided topology |
| z-flip (cruise-only) | #2945 | +4.5%/+18.3% | Mesh node density not z-symmetric |
| variance-penalized loss | #2963 | +5.7%/+17.8% | rc is extrapolation gap |
| EMA weights | #2915 | +2.1%/+251% | OneCycleLR cooldown = meaningful descent |
| surf_channel_weight cw=3 | #3000 | +3.81% | cw axis inverted-U |
| MLP dropout 0.05/0.10 | #3027 | +6.8%/+1.6% | Slows convergence; OOD gain dominated by ID regression |
| batch_size=8 | #2916 | +49.9% | compile OOMs; nocompile halves realized epochs |
| pct_start=0.05 | #2970 arm A | +2.5% | Fast warmup → bf16 instability; lower tail LR |

## Open Questions

- Does asinh + pct_start=0.2 compound? (#3062 will verify, ETA ~30 min)
- Does pct_start further improve at 0.25/0.30 on the asinh baseline? (#3060)
- Does final_div_factor improve on the asinh baseline? (#2987)
- Does DropPath improve rc on the asinh baseline? (#3055)
- Does cw<2 improve on the new asinh landscape? (#3051)

## Potential Next Directions (not yet assigned)

1. **asinh scale fine-tune** (350/300/250) — if compound confirms stacking, edge the scale down
2. **Camber-interpolation augmentation** — synthetic M=5-9 training samples; rc still the hardest split
3. **Per-domain normalization** — cruise vs raceCar pressure scales differ 4×; asinh partly addresses this but domain-level normalization is orthogonal
4. **Deeper-block-only regularization** — DropPath/dropout only on last 2-3 blocks
5. **asinh on Uy channel** — edward's follow-up suggestion; Uy std=9.74, moderate tails
