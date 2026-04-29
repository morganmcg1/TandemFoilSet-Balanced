# SENPAI Research State
- 2026-04-29 11:30
- No recent research directions from human researcher team (no GitHub issues found)
- Current research focus and themes: First round of experiments on the TandemFoilSet CFD surrogate modelling track. PR #1086 reviewed and sent back for revision. PR #1126 newly assigned. 6 other experiments still WIP.

## Current Baseline

**val_avg/mae_surf_p = 127.6661** (PR #1088, charliepai2f2-edward, surf_weight=10→25, epoch 13/50)

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 157.82 |
| val_geom_camber_rc | 135.65 |
| val_geom_camber_cruise | 99.26 |
| val_re_rand | 117.94 |

Note: NaN in test_geom_camber_cruise pressure due to corrupted GT sample `000020.pt`. Use `torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)` guard.

## Key Insight: Systematic Timeout/LR Mismatch

All Round 1 experiments use `CosineAnnealingLR(T_max=50)` but with a 30-min timeout, the baseline only completes ~14 epochs and wider models even fewer. This means:
- LR has only annealed ~5-28% from its initial value at end of training
- All Round 1 experiments are effectively running at near-peak LR throughout
- Fixing T_max to match actual epoch count could meaningfully improve convergence

## Active Experiments

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #1086 | alphonse | WIP (revision) | Width expansion: try n_hidden=192, n_head=6, surf_weight=25 + NaN guard |
| #1087 | askeladd | WIP | Slice count sweep: slice_num 64→128 |
| #1089 | fern | WIP | Deeper model: n_layers 5→8 with lr 3e-4 |
| #1090 | frieren | WIP | Per-field output heads: separate MLP decoder for Ux, Uy, p |
| #1091 | nezuko | WIP | Stochastic depth regularization: drop_path 0→0.1 |
| #1098 | tanjiro | WIP | Grad clip + higher LR (1e-3) |
| #1102 | thorfinn | WIP | MLP ratio 2→4: wider feedforward sublayer |
| #1126 | edward | WIP | Timeout-aware cosine LR: T_max=14 to fully anneal within 30-min cap |

## Potential Next Research Directions

After Round 1 results come in, consider:

1. **Timeout-aware hyperparameter tuning**: With T_max fixed (now being tested by edward), explore whether eta_min tuning, warmup schedules, or cyclic LR works better than vanilla cosine
2. **Loss formulation**: Laplacian/gradient smoothness penalty on pressure field, physics-informed boundary conditions
3. **Attention mechanism**: Cross-attention between surface and volume nodes explicitly; separate surface attention stream
4. **Multi-scale architecture**: Hierarchical pooling over mesh zones (background vs. dense foil zones)
5. **Data augmentation**: Reynolds number scaling, AoA perturbation, foil geometry perturbation
6. **Ensemble approaches**: Cheap model ensemble from multiple checkpoints
7. **Fourier features**: Random Fourier features for position encoding on irregular meshes
8. **Graph neural networks**: Message passing along mesh connectivity (GNN-based surrogate)
9. **Adaptive loss weighting**: Dynamic balancing between surface/volume/field losses during training
10. **Width expansion (revisited)**: n_hidden=192, n_head=6 (head_dim=32) to add capacity without exceeding timeout budget — alphonse's next experiment
