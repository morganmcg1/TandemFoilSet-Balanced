# SENPAI Research State

- **Date:** 2026-05-15 (updated 14:30 after PR #3140 close)
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline:** `val_avg/mae_surf_p = 135.30`, `test_avg/mae_surf_p (excl cruise) = 135.54` (W&B `xehwt9bi`).

## Tracked infrastructure issue: cruise-test NaN

`test_geom_camber_cruise/mae_surf_p` returns NaN on the baseline arm because at least one test-cruise sample produces a non-finite pressure prediction whose squared error propagates Inf through `data/scoring.py:accumulate_batch`. Affects all PRs identically. Validation cruise is finite — only test cruise is broken. Until fixed, every PR uses the 3-split `test_avg/mae_surf_p (excl cruise)` for paper-facing comparison.

Fix would land in `train.py:evaluate_split` (data/scoring.py is read-only per program.md): mask out samples whose pred contains non-finite values before calling `accumulate_batch`. Targeted as a dedicated small PR after round 1 finishes — not bundled into a hypothesis to keep single-variable comparisons clean.

## Round 1 → Round 2 hand-off

**Round 1 first result (PR #3140, alphonse, closed):** widening Transolver capacity is wall-clock-penalized under the 30-min cap. Width adds ~1.55× per-epoch cost, reducing reachable epochs by ~36%; the variant best-checkpointed early at epoch 8/9 and was uniformly worse on val and test.

**Updated strategic prior:** within a hard wall-clock cap, hypotheses that add per-step cost are intrinsically penalized. Round 2 should favor:
- Optimizer / training-recipe changes that don't slow each step (SOAP, Lion).
- Loss reformulations (already partially in flight: Huber, p-weight, per-sample).
- Data augmentation (AoA reflection).
- Input feature engineering (log-Re sinusoidal, per-domain normalization).
- Architecture *modifications* that don't add depth/width (Ada-Temp slice reparam, attention entropy reg, separate surface decoder head).

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
