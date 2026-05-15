# SENPAI Research State

- **Last updated**: 2026-05-15 ~13:00 UTC
- **Branch**: `icml-appendix-charlie-pai2i-24h-r3`
- **Target**: TandemFoilSet 2D CFD surrogate; Transolver baseline
- **Primary metric**: `val_avg/mae_surf_p` (paper: `test_avg/mae_surf_p`) — lower is better
- **Per-run budget**: SENPAI_MAX_EPOCHS=50, SENPAI_TIMEOUT_MINUTES=30 (hard caps)

## Most recent human research direction
None — no human GitHub Issues are addressed to this branch / team in the current launch.

## Current research focus (Round 1)

Fresh launch. No experiments have been merged on this branch yet. Round 1 covers eight diverse, orthogonal directions chosen to maximize the chance that *something* improves on the Transolver baseline and to map the landscape of what works:

| # | Student | PR | Slug | Lever |
|---|---|---|---|---|
| 1 | alphonse | [#3177](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3177) | `per-sample-scale-norm` | Loss reweighting — equalize per-sample gradient magnitude across Re regimes |
| 2 | askeladd | [#3235](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3235) | `local-re-feature` | Input augmentation — `Re×|x|` boundary-layer feature gated to surface nodes |
| 3 | edward | [#3237](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3237) | `huber-loss` | Loss formulation — Huber (δ=1.0) instead of MSE to cap outlier gradients |
| 4 | fern | [#3238](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3238) | `dual-branch-heads` | Architecture — split final output into surface/volume MLPs |
| 5 | frieren | [#3239](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3239) | `fourier-pos-enc` | Architecture — log-spaced Fourier features over (x, z) |
| 6 | nezuko | [#3240](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3240) | `hflip-augment` | Augmentation — z-reflection symmetry of NS equations |
| 7 | tanjiro | [#3241](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3241) | `ema-weights` | Optimization — EMA model averaging (decay=0.9999) |
| 8 | thorfinn | [#3242](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3242) | `re-curriculum` | Curriculum — low-Re-first ordering for first 50% of epochs |

## Hypothesis themes

- **Re-regime balance** (alphonse, edward, thorfinn): Three different mechanisms to address the same observation — per-sample target variance spans an order of magnitude across Reynolds numbers, biasing MSE gradients toward high-Re extremes. Their interaction will tell us whether the bottleneck is loss-side or sampling-side.
- **Surface emphasis / physics priors** (askeladd, fern): Push the model toward better surface-pressure predictions, which is the primary metric.
- **Representational capacity** (frieren): Test whether the Transolver's linear (x, z) encoding bottlenecks fine spatial structure.
- **Data efficiency** (nezuko): Free 2× effective dataset via physical symmetry.
- **Training stability** (tanjiro): Low-risk, complementary gain via EMA.

## Potential next research directions (Round 2 candidates)

Will be selected based on Round 1 results, but currently in the queue:

1. **Composition winners** — stack the best 2–3 of Round 1 (e.g., per-sample-scale-norm + EMA + dual-heads if all three work)
2. **Per-channel loss weighting** — give pressure (`p`) a heavier weight than `Ux`, `Uy` since `mae_surf_p` is the metric of record
3. **Surface-only auxiliary loss** — wall pressure gradient regularization
4. **Larger model** — n_hidden=192 or 256, depth=6, slice_num=128 (timing permitting)
5. **Warmup + cosine** — short linear warmup before cosine decay
6. **Mixed-precision training** — bf16 to allow larger batches within 96GB VRAM
7. **Better sampler weighting** — refine the WeightedRandomSampler to include Re stratification (the natural follow-up if thorfinn's curriculum hypothesis is confirmed)
8. **Slice attention scaling** — slice_num sweep (32, 64, 128) if Round 1 shows model capacity is the bottleneck
9. **Per-domain normalization stats** — separate (y_mean, y_std) for raceCar / cruise / single
10. **GNN-style local message passing** — k-nearest-neighbor attention as an alternative to slice attention
11. **Physics-informed regularization** — divergence-free constraint on (Ux, Uy) in volume nodes

## Open uncertainties
- We do not yet have a measured baseline `val_avg/mae_surf_p` value on this exact (branch × GPU × code) configuration — Round 1 PRs will produce it.
- Whether the WeightedRandomSampler's domain balancing implicitly handles Re bias, or only domain bias. The trio of Re-targeted experiments (alphonse, edward, thorfinn) will disambiguate.
- Whether slice attention's learned routing already specializes on surface vs volume nodes — if yes, dual-heads (fern) is redundant; if no, it should win.

## Idle students
None.
