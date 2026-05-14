# SENPAI Research State

- **Date:** 2026-05-14 22:25 UTC
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

### 1. Channel weight axis — filling left half of confirmed inverted-U
- cw axis confirmed inverted-U: cw=3 (+3.81%), cw=2 is local optimum
- **nezuko #3051**: cw=1.5/1.25 redux sweep — probes whether cw<2 reduces rc regression while preserving cruise/single wins

### 2. OOD regularization — building on dropout findings
- PR #3027 (thorfinn, dropout 0.05/0.10) validated: dropout=0.10 improved `val_geom_camber_rc` by -5.1% val/-2.8% test but cost +1.6% net (ID regressions dominated under 35-ep cap)
- **Key insight:** `val_geom_camber_rc` is responsive to capacity-restriction regularizers. Weight decay (continuous, no convergence slowdown) is the next axis to probe.
- **thorfinn #3052**: AdamW weight_decay sweep 5e-4/1e-3 (baseline=1e-4) — low-cost OOD regularization

### 3. Surface weight tuning at cw=2
- **alphonse #3029**: sw=4/6 re-sweep at cw=[1,1,2] — optimal surf:vol ratio at new channel weights

### 4. Schedule tail (near-exhausted)
- **askeladd #2987**: final_div_factor 100/10 — only remaining schedule efficiency lever

### 5. Domain/sampler
- **fern #2982**: cruise upweighting 3.0× + single_weight=1.5 (--epochs 35 re-run)

### 6. Input representation
- **edward #1605**: asinh-p680 transform — in-flight

### 7. Warmup tuning
- **frieren #2970**: pct_start 0.05/0.2

### 8. Batch size / extended schedule
- **tanjiro #2916**: bf16 batch_size=8 + extended schedule

## Students — Current State

| Student | PR | Hypothesis | State |
|---------|-----|-----------|-------|
| alphonse | #3029 | surf_weight re-sweep (sw=4/6) at cw=[1,1,2] baseline | WIP |
| askeladd | #2987 | OneCycleLR final_div_factor tuning (100/10) | WIP |
| fern | #2982 | Cruise upweighting 3.0× + single_weight=1.5 (--epochs 35) | WIP |
| edward | #1605 | asinh-p680 transform | WIP |
| nezuko | #3051 | surf_channel_weight cw=1.5/1.25 sweep redux | WIP — newly assigned |
| frieren | #2970 | pct_start warmup tuning (0.05/0.2) | WIP |
| thorfinn | #3052 | AdamW weight_decay sweep 5e-4/1e-3 | WIP — newly assigned |
| tanjiro | #2916 | bf16 batch_size=8 + extended schedule | WIP |

0 idle students. 8 active experiments.

## Key Findings (cumulative)

- **The binding constraint chain:** L1 → OneCycleLR → bf16 → eval_every=2 → torch.compile → 35 epochs → surf_weight=5 → surf_channel_weight=[1,1,2].
- **Channel weight axis is inverted-U:** cw=2 is current optimum. cw=3 (+3.81%, PR #3000) and cw=1 (old recipe) bracket it. Left half (cw=1.25/1.5) probed in #3051.
- **val_geom_camber_rc is the hardest split** (69.7 vs 53.4 avg). Responds to regularization: dropout=0.10 gave -5.1% val on rc but hurt ID. Weight decay is the lower-cost mechanism (#3052).
- **Dropout validates OOD mechanism but fails on primary metric:** Arm B (d=0.10) improved rc -5.1%/-2.8% but regressed primary +1.6%/+3.1%. Conv slowdown + 35-ep cap = insufficient budget.
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

## Open Questions

- Does cw=1.5/1.25 reduce rc regression while preserving cruise/single_in_dist gains? (#3051)
- Does weight_decay 5e-4/1e-3 give the same rc OOD benefit as dropout without convergence cost? (#3052)
- Does alphonse's sw=4/6 re-sweep find a better surf:vol ratio at the cw=2 baseline? (#3029)
- At val=53.35, how close are we to the physical floor for this architecture?

## Potential Next Directions (not yet assigned)

1. **Stochastic depth / DropPath** — drop entire residual branches (rate 0.05/0.10); less per-token disruption than MLP dropout
2. **Deeper-block-only dropout** — apply dropout only in blocks 3–5 (regularize high-level abstractions only)
3. **Camber-interpolation augmentation** — synthetic M=5-9 training samples to directly attack rc extrapolation gap
4. **Per-domain normalization** — 4× pressure scale difference cruise vs raceCar
5. **Larger batch_size + gradient accumulation** — batch=4 is conservative; batch=8 with grad accum may stabilize training
