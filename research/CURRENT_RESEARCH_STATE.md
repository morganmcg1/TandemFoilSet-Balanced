# SENPAI Research State

- **Date:** 2026-05-15
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues open at boot)_

## Current research focus

This is round 5 (r5) of the charlie 48h arm, a fresh advisor branch with no prior
experiment results. The dataset is TandemFoilSet — we predict (Ux, Uy, p) fields
on irregular CFD meshes of single and tandem airfoils. Primary metric is
`val_avg/mae_surf_p` (lower better), reported as the equal-weight average of
surface pressure MAE across four validation splits.

The baseline Transolver (~1–2 M params, 128 hidden, 5 layers, 4 heads, slice_num=64)
runs in ~30 min, leaving headroom to test architecture, loss, training-strategy,
and feature-engineering changes in parallel across 8 students.

## Themes being pursued in this round

1. **Heavy-tail / scale handling for pressure.** Per-sample pressure ranges span
   1–2 orders of magnitude across Re and domain. We test:
   - per-sample scale normalization (alphonse)
   - Huber loss on the normalized residuals (frieren)
   - gradient clipping (askeladd)

2. **Surface vs volume specialization.** Surface pressure is the dominant metric
   yet surface nodes are ~0.1% of total nodes. We test:
   - decoupled surface/volume decoder heads (fern)
   - surf_weight curriculum annealing 1→20 (thorfinn)

3. **Architecture capacity / multi-scale structure.**
   - multi-scale slice attention (32+128 slices) (tanjiro)

4. **Spatial / positional inductive biases.**
   - Fourier positional features on (x, z) coordinates (nezuko)

5. **Training stability / averaging.**
   - EMA over weights for validation checkpoint selection (edward)

## Potential next research directions (after wave 1 results)

- **Composite winners:** combine top-3 winners from wave 1 in a single PR.
- **Larger Transolver capacity:** n_hidden 192/256 and longer training schedule
  once a scaling-friendly recipe is identified.
- **Domain-conditioned MoE slice projections** (NESTOR analogue).
- **RoPE-2D in PhysicsAttention** if Fourier features show clear gain.
- **Physics-informed auxiliary losses:** divergence-free penalty on
  predicted (Ux, Uy) over volume nodes; pressure-velocity coupling.
- **Boundary-layer physics features** (Re_x proxy, gap×log(Re) interaction).
- **Stochastic depth** if larger architectures overfit on geometry holdouts.
- **Curriculum sampling on Re or on mesh size.**

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries with concrete code recipes).
