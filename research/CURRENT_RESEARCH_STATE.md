# SENPAI Research State

- **Last updated:** 2026-05-15 12:45 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None on this launch.

## Current research focus

Beat the Transolver baseline on `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across the four val splits) — the same metric is computed on the four test splits as `test_avg/mae_surf_p` for paper-facing numbers. The four tracks stress different generalization axes:

1. `val_single_in_dist` — sanity check, raceCar single-foil random holdout
2. `val_geom_camber_rc` — unseen front-foil camber M=6-8 (raceCar tandem)
3. `val_geom_camber_cruise` — unseen front-foil camber M=2-4 (cruise tandem)
4. `val_re_rand` — stratified Re holdout across tandem domains

Prefer common-recipe changes that survive across all four tracks over hacks that only help one. When splits disagree, the disagreement is information.

## Round 1 themes

Initial round explores the **cheapest, highest-EV levers** before touching architecture in deeper ways. Each student tests one orthogonal axis:

- **Loss reformulation** — surface weighting (`surf-weight-scan`), per-channel weighting (`pressure-channel-weight`), robust loss + gradient clipping (`grad-clip-huber`)
- **Optimization** — EMA weights (`ema-weights`), linear warmup (`lr-warmup-cosine`)
- **Architecture** — more physics tokens (`slice-num-128`), wider+deeper (`hidden-256-depth6`), decoupled output heads (`per-channel-output-heads`)

Full hypothesis details: [`research/RESEARCH_IDEAS_2026-05-15_init.md`](RESEARCH_IDEAS_2026-05-15_init.md).

## Potential next research directions

If round-1 yields one or more winners, the natural follow-ups are:

- **Combinations of orthogonal winners** (loss × architecture × optimization stack)
- **Physics-informed auxiliary losses** — continuity (∂Ux/∂x + ∂Uy/∂y ≈ 0) on volume nodes; tangential-pressure smoothness on surface nodes
- **Geometry-aware augmentation** — vertical mirroring for single-foil (sign-flip AoA), Re scaling within plausible bounds, foil-pair stagger jitter
- **Foil-1 vs foil-2 disambiguation** — add a learned token or feature that distinguishes which foil a node is near
- **Output transformations** — predict residuals over a cheap analytic baseline (potential flow / thin-airfoil), asinh-transformed pressure for high-Re samples
- **Multi-scale attention** — cross-attention between mesh nodes and a small set of geometry tokens summarizing each foil

## Operational notes

- 8 idle students at launch — every GPU must be assigned a hypothesis this round.
- Per-run budget: 30 min wall clock and 50 epochs hard cap. The wall clock will usually bind first.
- Students edit `train.py` only. `data/` files are read-only.
