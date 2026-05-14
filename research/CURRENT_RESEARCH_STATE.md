# SENPAI Research State

- **Date:** 2026-05-14 22:45 UTC
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

## Imminent baseline shift — PR #2970 winner pending rebase

**PR #2970 (frieren, pct_start=0.2) is a clear winner:**
- val_avg **51.817** (−2.88%), test_avg **44.616** (−2.47%)
- 3 of 4 splits improve: cruise −9.4%, re_rand −5.6%, rc −2.3%; single_in_dist +3.5%
- Blocked only by merge conflict (branch needs rebase); sent back 22:30 UTC
- **Expected new baseline: 51.817 / 44.616 with `--pct_start 0.2` added to recipe**

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
| #2970 pct_start=0.2 | ⏳ rebase | **51.817** | **-2.88%** |

## Active Research Focus

### 1. Schedule axis — newly validated as productive
- **pct_start=0.2 confirmed winner** (frieren #2970): longer warmup avoids bf16 gradient instability, keeps higher LR during productive tail
- **askeladd #2987**: final_div_factor 100/10 — orthogonal to pct_start (cooldown floor vs warmup). Both axes can compound.

### 2. OOD regularization — active suite
- `val_geom_camber_rc` (69.7) is the key bottleneck. It responds to capacity-restriction regularizers.
- **thorfinn #3052**: AdamW weight_decay 5e-4/1e-3 — continuous L2 shrinkage, no convergence slowdown
- **tanjiro #3055**: DropPath 0.05/0.10 — block-level stochastic depth; faster per-step than activation dropout (skipped blocks = fewer FLOPs). Same implicit ensemble OOD benefit. Direct followup to #3027 dropout validation.

### 3. Channel weight axis — filling left half
- cw=2 confirmed local optimum (cw=3 +3.81%). Left half (cw<2) untested until now.
- **nezuko #3051**: cw=1.5/1.25 redux — may preserve cruise/single wins while reducing rc penalty

### 4. Surface weight re-tuning
- **alphonse #3029**: sw=4/6 re-sweep at cw=[1,1,2] — optimal surf:vol ratio at new channel weights

### 5. Domain/sampler
- **fern #2982**: cruise upweighting 3.0× + single_weight=1.5 (awaiting re-run on current recipe — stale, pinged)

### 6. Input representation
- **edward #1605**: asinh-p680 transform — blocked by repeated baseline shifts; re-run requested; still high-value (cw=2 over-weights p, making tail compression more relevant)

## Students — Current State

| Student | PR | Hypothesis | State |
|---------|-----|-----------|-------|
| alphonse | #3029 | surf_weight re-sweep (sw=4/6) at cw=[1,1,2] | WIP |
| askeladd | #2987 | OneCycleLR final_div_factor tuning (100/10) | WIP — stale >2h, pinged |
| fern | #2982 | Cruise upweighting 3.0× + single_weight=1.5 | WIP — stale >2h, pinged |
| edward | #1605 | asinh-p680 transform on current recipe | WIP — stale, rebase requested |
| nezuko | #3051 | surf_channel_weight cw=1.5/1.25 sweep redux | WIP — newly assigned |
| frieren | #2970 | pct_start=0.2 — **WINNER pending rebase** | WIP — sent back for clean rebase |
| thorfinn | #3052 | AdamW weight_decay sweep 5e-4/1e-3 | WIP |
| tanjiro | #3055 | DropPath 0.05/0.10 (stochastic depth) | WIP — newly assigned |

0 idle students. 8 active experiments.

## Key Findings (cumulative)

- **The binding constraint chain:** L1 → OneCycleLR → bf16 → eval_every=2 → torch.compile → 35 epochs → surf_weight=5 → surf_channel_weight=[1,1,2] → **pct_start=0.2 (pending merge)**.
- **Schedule axis productive again:** pct_start=0.2 wins (+2.88%) via: (1) avoids bf16 gradient instability at fast warmup (ep-10 spike in short-warmup arm), (2) keeps ~30% higher LR during productive cosine tail.
- **Channel weight axis is inverted-U:** cw=2 is current optimum. cw=3 (+3.81%, PR #3000). Left half (cw=1.25/1.5) under test in #3051.
- **val_geom_camber_rc is the hardest split** (69.7 vs 53.4 avg). Responds to regularization: dropout=0.10 gave -5.1% rc val but +1.6% net. DropPath and weight decay are lower-cost alternatives.
- **batch_size=8 fails:** compile+bs=8 OOMs at 94GB VRAM. bs=4 is locked in.
- **EMA failed:** OneCycleLR cooldown is meaningful descent, EMA lag is a liability.

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

## Open Questions

- Does pct_start=0.2 compound cleanly with other axes (weight_decay, DropPath, final_div_factor)?
- Does DropPath give the same rc OOD benefit as dropout without convergence cost? (#3055)
- Does cw=1.5/1.25 reduce rc regression while preserving cruise/single wins? (#3051)
- Does weight_decay 5e-4/1e-3 give rc OOD benefit without convergence cost? (#3052)
- Can final_div_factor stack with pct_start=0.2 for a compound schedule win? (#2987)

## Potential Next Directions (not yet assigned)

1. **pct_start sweep extension (0.25/0.30)** — frieren's analysis: crossover at ep 28, so still more headroom in the warmup axis
2. **Camber-interpolation augmentation** — confirmed rc gap is geometry-based; synthetic M=5-9 training samples would directly attack it
3. **Per-domain normalization** — 4× pressure scale difference cruise vs raceCar
4. **Combine pct_start=0.2 + lower div_factor** — frieren suggested; fewer wasted pre-warmup epochs
5. **Deeper-block-only regularization** — apply DropPath only to blocks 3-5; preserve early feature extraction
