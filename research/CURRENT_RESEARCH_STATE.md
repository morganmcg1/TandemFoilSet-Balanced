# SENPAI Research State
- 2026-04-29 18:00 (branch: icml-appendix-charlie-pai2f-r4)
- No human researcher team directives received yet.

## Current Research Focus

**Target:** TandemFoilSet CFD surrogate — predict (Ux, Uy, p) at every mesh node.
**Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across 4 val splits (lower is better).
**Model:** Transolver with physics-aware attention over irregular meshes.
**Status:** Round 4+. Baseline now improved to 124.727 from PR #1128 (edward's per-sample Re-adaptive loss). 8 experiments currently in-flight exploring diverse improvement vectors.

## Baseline

| Metric | Value | PR |
|--------|-------|----|
| **val_avg/mae_surf_p** | **124.727** | #1128 (edward, per-sample Re-adaptive loss) |

Prior baseline history:
- 129.531 — PR #1112 (attention dropout=0.1)
- 124.727 — PR #1128 (per-sample Re-adaptive loss) CURRENT

Note: `test_avg/mae_surf_p = NaN` due to corrupt `test_geom_camber_cruise/000020.pt` (761 inf values in ground-truth). Valid test splits show 139.9 / 138.8 / 111.0 for single_in_dist, geom_camber_rc, re_rand respectively.

## Active Experiments (Round 4+)

| PR   | Student    | Status | Hypothesis |
|------|------------|--------|-----------|
| #1193 | tanjiro   | WIP | Random Fourier Features (n_rff=16, rff_scale=10.0) for multi-scale positional encoding |
| #1187 | fern      | WIP | Gradient clipping max_norm=1.0 + raised LR 8e-4 for faster convergence |
| #1186 | edward    | WIP | Combine surf_weight=5 with per-sample Re-adaptive loss |
| #1137 | nezuko    | WIP | Scale Transolver to n_hidden=256, n_layers=8 for high-Re splits |
| #1117 | thorfinn  | WIP | Re-conditioned output scale head for magnitude adaptation |
| #1114 | frieren   | WIP (sent back 2x) | Curriculum surf_weight ramp — ablating flat surf_weight=3/4 |
| #1111 | askeladd  | WIP | Layer-wise LR decay for geometry-stable representations |
| #1110 | alphonse  | WIP | Log-modulus transform on pressure channel loss |

Last checked: 2026-04-29 18:00.

## Research Themes Being Explored

1. **Loss formulation**: Curriculum surface weight ramp (sent back — too slow for budget); log-modulus pressure transform (PR #1110); Re-adaptive loss normalization (merged — current baseline PR #1128); sw=5 + adaptive combination (PR #1186)
2. **LR scheduling**: OneCycleLR (closed — budget mismatch); layer-wise LR decay (PR #1111); gradient clipping + higher LR (PR #1187)
3. **Architecture additions**: Re-conditioned output scale head (PR #1117); scaled model capacity n_hidden=256, n_layers=8 (PR #1137); Random Fourier Features for multi-scale geometry encoding (PR #1193)
4. **Regularization**: Attention dropout for OOD robustness (merged — prior baseline PR #1112)
5. **Closed/failed**: OneCycleLR (never peaks in ~30-min budget), slice-num-128 (per-epoch slowdown kills epoch count), learnable domain embedding (PR #1113 closed)

## Key Observations

- Attention dropout=0.1 achieves val_avg/mae_surf_p=129.531 at epoch 13/50 (training timed out ~14 epochs, ~133s/epoch)
- Per-sample Re-adaptive loss normalization (PR #1128) improved baseline to 124.727 — a 3.7% improvement
- Cruise split is much easier (~88-93) than raceCar splits (~150-160), suggesting multi-domain difficulty imbalance
- Single_in_dist val is consistently worse than geom_camber_rc — unusual for the "easiest" split
- VRAM usage peaked at 42.73 GB (well under 96 GB cap); significant room to scale model capacity
- Training timed out at epoch 14, best at epoch 13 — models under-trained; convergence speed is critical
- **slice-num-128 insight**: val_geom_camber_rc improved -11.6 MAE with 128 slices despite overall regression — physics token diversity helps raceCar tandem but per-epoch cost too high
- **Budget constraint**: ~30 min timeout means ~14 epochs at 133s/epoch. Per-epoch cost is as important as architecture quality
- **OneCycleLR lesson**: LR schedulers configured over full 50 epochs are fatally incompatible with the budget. Must configure over the realistic ~14 epoch window
- Re-adaptive loss works by normalizing per-sample pressure scale — suggests the model struggles with varying Re-dependent pressure magnitudes across the dataset

## Potential Next Research Directions

After round 4+ results are in, priority areas to explore:
- **Physics-informed losses**: Divergence-free velocity constraint (nabla u = 0 for incompressible); pressure-velocity coupling (Poisson-style penalty)
- **Spectral/frequency features**: Apply RFF to encode leading/trailing edge positions at multiple scales (being tested by tanjiro, PR #1193)
- **Convergence acceleration**: Warm restarts after first 5 epochs; gradient clipping + higher LR (being tested by fern, PR #1187)
- **Loss combination**: Re-adaptive + higher surf_weight (being tested by edward, PR #1186)
- **Ensemble / multi-head outputs**: Separate prediction heads per domain (raceCar single, raceCar tandem, cruise)
- **Higher resolution surface loss**: Weight loss proportional to local curvature (leading/trailing edge gets more weight)
- **Graph neural message passing**: Augment transformer with local neighborhood message passing for better surface gradient propagation
- **Stochastic weight averaging (SWA)**: Average weights over last few epochs to smooth out noise in under-trained models
- **Mixed precision / larger batches**: Increase batch size within VRAM budget to see more samples per epoch
