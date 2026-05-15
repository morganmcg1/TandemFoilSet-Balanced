# SENPAI Research State

- **Date:** 2026-05-15 12:35
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r2`
- **Target base branch:** `icml-appendix-willow`
- **W&B project:** `wandb-applied-ai-team/senpai-v1`
- **Hard limits:** `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`, 1 GPU / 96GB per student

## Most recent research direction from human researcher team

None yet — no human directives on this launch.

## Current research focus and themes

Round 1 (fresh start). The goal is to beat the vanilla Transolver baseline configured in `train.py` on `val_avg/mae_surf_p` (and the matching paper-facing `test_avg/mae_surf_p`). The four val tracks demand a common-recipe winner: changes that survive across in-distribution, two camber holdouts (raceCar M=6-8 and cruise M=2-4), and a stratified Re holdout.

**Identified bottlenecks driving Round 1 hypotheses:**

1. **Dynamic-range dominance in the loss.** Per-sample y_std varies ~13× within a split (high-Re samples drive extremes). Uniform MSE lets high-Re samples dominate gradients.
2. **Surface pressure is the scored channel.** The loss currently weights Ux, Uy, p equally on surface nodes (with a global `surf_weight=10` multiplier on the whole surface block).
3. **No LR warmup.** PhysicsAttention slice projection starts cold; the first epoch hits hard at `lr=5e-4`.
4. **Geometry generalization to unseen NACA camber.** Two of the four splits are full-file holdouts on front-foil camber (M=6-8 raceCar, M=2-4 cruise). Architectural geometry conditioning may help.
5. **Spatial-frequency content near foils.** Pressure gradients near leading/trailing edges are sharp; raw coordinates may underfit them without Fourier features.
6. **Possibly under-parameterized.** Baseline `n_hidden=128` is the smallest published Transolver config; larger may help if it fits in 30 min wall clock.

## Round 1 assignments (8 hypotheses, one per student)

| Student | Hypothesis | Angle |
|---|---|---|
| alphonse | Per-sample scale-normalizing loss (relative-L2 style) | loss |
| askeladd | 5-epoch linear warmup + cosine remainder | optimizer |
| edward | Per-channel surface loss weights (upweight `p`) | loss |
| fern | Fourier position features on (x, z) | features |
| frieren | Capacity scale-up: `n_hidden=256, n_head=8, slice_num=128` | arch-tweak |
| nezuko | PGOT-style geometry-conditioned slice assignment | arch-tweak |
| tanjiro | SmoothL1 (Huber) loss with `beta=0.05` for outlier-robust regression | loss |
| thorfinn | Stochastic depth (DropPath) on Transolver blocks | regularization |

Hypothesis details and references live in `research/RESEARCH_IDEAS_2026-05-15_12:35.md`.

## Potential next research directions (after Round 1)

If Round 1 surfaces a clear winner, Round 2 will compound and explore adjacent space. Anticipated follow-ups, depending on which lever moves the metric:

- **If H1 (scale loss) wins:** try relative-L2 per-channel (different denominators per channel), or per-domain re-weighting at the sampler level beyond the existing balanced sampler.
- **If H2 (warmup) wins:** stack warmup with all winning recipes. Try OneCycleLR or warmup-stable-decay (WSD).
- **If H3 (channel weighting) wins:** add `surface-only` direct head, or replace MSE on `p` with a calibrated surface-only loss.
- **If H4 (Fourier) wins:** sweep number of bands; try Gabor-like learned frequencies; add Fourier features per foil-relative coordinate.
- **If H5 (capacity) wins:** sweep `n_layers`, try `mlp_ratio=4`, try SwiGLU/GeGLU FFN.
- **If H7 (geom-conditioned slice) wins:** extend to per-block geometry conditioning, FiLM-like modulation of LayerNorm.
- **If H9 (stochastic depth) wins:** stack with weight decay schedule, try SWA / LAWA weight averaging.

**Plateau response (if 5+ experiments fail to improve):** Move to a different architecture entirely — GINO, Galerkin transformer, mesh GNN baseline, or spectral-conv hybrid — and reconsider the data normalization (per-sample relative scoring vs global stats).
