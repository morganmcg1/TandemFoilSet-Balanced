<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-15
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r3`
- **Research tag:** `willow-pai2i-24h-r3` (round 3 of the appendix-willow PAI2I sub-track)
- **Most recent human research direction:** None received this launch; treating as clean-slate research effort scoped strictly to this branch.

## Current focus

Round 3 fresh-slate exploration on TandemFoilSet (Transolver surrogate for tandem-airfoil
CFD). Goal: minimize `val_avg/mae_surf_p` — equal-weight surface-pressure MAE across the
four validation tracks (in_dist, geom_camber_rc, geom_camber_cruise, re_rand). The
test-time decision metric is `test_avg/mae_surf_p`.

Baseline configuration for this branch (no prior PRs):

- Transolver: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- AdamW: `lr=5e-4`, `weight_decay=1e-4`
- Loss: MSE in normalized space, `surf_weight=10`, equal channel weights
- Optimizer schedule: `CosineAnnealingLR(T_max=epochs)` with no warmup
- `batch_size=4`, `epochs=50` (subject to `SENPAI_TIMEOUT_MINUTES`)

## Round 3 dispatch plan (8 students, 8 orthogonal hypotheses)

Strategy: cover diverse axes simultaneously since this is the first round on the branch
— architecture, attention design, optimizer/schedule, robust loss, channel reweighting,
sample reweighting, geometry feature representation, and EMA averaging.

| Slot | Student | Hypothesis | Tier | Predicted Δ on `val_avg/mae_surf_p` |
|------|---------|------------|------|--------------------------------------|
| 1 | alphonse | deeper-transolver (n_layers 5→8) | Architecture | −8% to −15% |
| 2 | askeladd | warmup-cosine-grad-clip | Optimizer | −5% to −10% |
| 3 | edward | surf-p-weighted-loss ([1,1,3]) | Loss alignment | −5% to −12% |
| 4 | fern | larger-slice-num (64→128) | Attention | −4% to −8% |
| 5 | frieren | huber-robust-loss (δ=2.0) | Robust loss | −3% to −8% |
| 6 | nezuko | ema-model-averaging (decay=0.999) | Regularization | −2% to −6% |
| 7 | tanjiro | re-conditioned-loss-weighting | Sample reweighting | −3% to −7% on re_rand |
| 8 | thorfinn | naca-camber-fourier-features | Geometry features | −4% to −10% on geom_camber |

Full hypothesis specifications in `RESEARCH_IDEAS_2026-05-15_initial.md`.

## Next research directions (after round 1 completes)

Conditional on round 1 results, plausible follow-ups:

1. **Stack winners.** Combine winning architecture/optimizer/loss changes into a single
   confirmation run, then ablate to verify independence.
2. **Larger attention bottleneck.** If `larger-slice-num` wins, try `slice_num=192` and
   investigate per-domain slice token specialization.
3. **Targeted geometry conditioning.** If `naca-camber-fourier-features` shows OOD gains,
   extend to gap/stagger/AoA features and consider per-foil geometry conditioning tokens.
4. **Physics-residual losses.** If geometry generalization remains weak, add a soft
   divergence-free penalty on predicted velocity and an inviscid-Bernoulli surface
   pressure consistency term.
5. **Surface-aware architecture.** Surface-only token attention or a dedicated
   surface-decoder head that takes is_surface mask as a conditioning input.
6. **Per-channel normalization beyond global stats.** Per-domain or per-Re-bucket
   normalization to align value magnitudes across the dataset.
7. **Optimizers beyond AdamW.** Lion, SOAP, Muon — better effective lr-scaling under
   heterogeneous gradient regimes.
8. **Mixed-precision training** to enlarge effective batch size and reduce per-epoch
   wall-clock, then re-spend the saved time on more epochs.

## Operational notes

- All assignments use `--wandb_group` matching the hypothesis slug so iterations cluster
  in W&B.
- Baseline metrics for student reference: there are no prior runs on this branch. Students
  should compare their result against the Transolver default config (which is what the
  baseline produces if their hypothesis run shows no improvement).
- Hard limits: `SENPAI_TIMEOUT_MINUTES` and `SENPAI_MAX_EPOCHS` govern each training run.
  Do not override.
