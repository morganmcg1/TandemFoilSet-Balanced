# SENPAI Research State

- **Date:** 2026-05-15
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.

## Current research focus

Round 3 of the willow-pai2i-48h cycle. The advisor branch was just freshly cut and no canonical baseline run exists yet on this branch state. The primary ranking metric is `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 validation tracks), with `test_avg/mae_surf_p` as the paper-facing number.

The fleet of 8 idle students is being assigned a first round of 8 single-variable experiments spanning four families of intervention:

1. **Model capacity** — does the small Transolver under-fit?
   - `alphonse`: wider hidden + more heads (128→192, 4→6)
   - `nezuko`: depth (5→8 layers)
   - `tanjiro`: MLP ratio (2→4)
2. **Optimization recipe**
   - `askeladd`: LR warmup + higher peak (5e-4→1e-3 with 3-epoch linear warmup)
3. **Loss formulation**
   - `edward`: per-channel loss weighting (upweight p by 3x)
   - `fern`: Huber loss instead of MSE
   - `frieren`: per-sample loss normalization (equal-weight per sample, not per node)
4. **Inputs**
   - `thorfinn`: Fourier position features on (x, z) + slice_num bump 64→96

Each PR runs **dual-arm** (baseline + variant in same wandb_group) so we simultaneously establish baseline metrics for this branch state AND get clean A/B attribution per hypothesis.

## Potential next research directions

After round-1 results come in, **stack winners** (compatible interventions can compound).

A separate researcher-agent has compiled a shortlist of 18 candidate hypotheses in `research/RESEARCH_IDEAS_2026-05-15_initial.md`. The top 8 by expected value are:

1. **SOAP optimizer** (S, −5% to −20%) — drop-in AdamW replacement; Hessian preconditioning resolves gradient conflicts between vol/surf loss terms. Highest single risk-adjusted EV.
2. **Ada-Temp slice reparameterization** (M, −8% to −15%) — per-point learned temperature offset in PhysicsAttention softmax; directly targets slice-collapse failure mode documented in Transolver++.
3. **AoA reflection augmentation** (S, −3% to −8%) — negate AoA + flip z + sign-flip Uy for raceCar samples; doubles effective raceCar training data, helps `val_geom_camber_rc`.
4. **Attention entropy regularization** (S, −3% to −7%) — diagnostic + light fix for slice uniformity.
5. **Log-Re sinusoidal embedding** (S, −3% to −8%) — 8-dim sinusoidal features replacing raw normalized log(Re); targets `val_re_rand` OOD.
6. **Divergence-free auxiliary loss** (M, −5% to −12%) — soft penalty on `‖∂Ux/∂x + ∂Uy/∂z‖` via finite differences on the irregular mesh; encodes incompressible continuity.
7. **Per-domain normalization** (S, −3% to −8%) — domain-conditioned y stats instead of global pooling of raceCar + cruise.
8. **Separate surface decoder head** (M, −4% to −10%) — dedicated 2× width MLP for surface nodes; orthogonal to loss-side surface upweighting.

These will be dispatched as students become idle after round 1.
