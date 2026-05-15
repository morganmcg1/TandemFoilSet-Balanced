# SENPAI Research State

- **Date**: 2026-05-15 12:35
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 1 (first experiments on this advisor branch)
- **Most recent human research directive**: None received yet

## Current Research Focus

**Phase**: Round 1 — establishing baseline and probing primary levers.

We are working with a Transolver baseline on TandemFoilSet: predict (Ux, Uy, p) at every mesh node given tandem airfoil geometry and flow conditions. Primary metric is `val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across 4 val splits (lower is better).

The baseline model (n_hidden=128, 5 layers, 4 heads, slice_num=64, MSE loss) has **not yet been evaluated** on this branch. All 8 students are running first-round experiments.

**Round 1 hypothesis portfolio** (8 students, 8 hypotheses):

| PR | Student | Hypothesis | Category | Risk |
|----|---------|------------|----------|------|
| #3154 | alphonse | H5: Wider model (n_hidden 128→256, n_head 4→8) | Architecture | Med |
| #3156 | askeladd | H1: Per-channel surface-p loss upweight (3x, 5x) | Loss | Low |
| #3158 | edward | H2: EMA weight averaging (decay=0.999) | Training | Low |
| #3160 | fern | H4: Huber loss (delta=1.0, 0.5) | Loss | Low |
| #3163 | frieren | H3: Gradient clip + 5-epoch LR warmup | Optimization | Low |
| #3166 | nezuko | H7: FiLM Re/AoA conditioning on Transolver blocks | Architecture | Med-High |
| #3168 | tanjiro | H10: More slices (slice_num 64→128, 96) | Architecture | Low |
| #3170 | thorfinn | H11: Deeper model (5→7, 5→8 layers) | Architecture | Low-Med |

## Bottleneck Hypotheses

Based on analysis of train.py and program.md (no empirical results yet):

1. **Model capacity** (n_hidden=128 is below paper's default of 256) — H5 tests this
2. **Loss misalignment** (all channels weighted equally, but p is the only metric) — H1 tests this
3. **Training instability** (no gradient clipping, no LR warmup with variable-mesh batches) — H3 tests this
4. **EMA gap** (best checkpoint ≠ smoothed trajectory for OOD splits) — H2 tests this
5. **Slice count** (64 slices over 242K nodes may dilute surface node information) — H10 tests this
6. **Missing flow conditioning** (log(Re)/AoA treated as generic features, not global conditioning) — H7 tests this

## Key Discriminating Questions

1. **Capacity vs. loss**: Does H5 (wider) beat H1 (p-upweight) by >8%? If yes → capacity bottleneck. If they're similar → loss formulation bottleneck.
2. **Loss shape**: Does H4 (Huber) beat MSE baseline? If yes → heavy-tailed Re distribution is hurting.
3. **OOD geometry**: Which split benefits most — geom_camber_rc and geom_camber_cruise (FiLM/EMA candidates) or re_rand (gradient stability)?

## Potential Next Research Directions (Round 2)

Depending on Round 1 results:

- **If H5 (wider) wins big**: compound width + depth (n_hidden=256, n_layers=7) + more slices (slice_num=128)
- **If H1 (p-upweight) wins big**: try even higher p_surf_weight (7x, 10x); combine with Huber
- **If H7 (FiLM) works**: stack FiLM + wider model; try domain-specific FiLM per geometry type
- **If plateau already**: try completely different architectures (GNO/FNO, U-Net decoder, separate surface/volume streams)
- **Novel ideas for R2**: per-sample loss normalization (H8), WSD schedule (H9), separate p decoder head (H6)
