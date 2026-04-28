# SENPAI Research State

- 2026-04-28 19:15
- No directives yet from the human researcher team
- **Current research focus**: First-round sweep of Transolver architecture and training hyperparameter modifications on TandemFoilSet. All 8 students have been assigned initial experiments covering the most impactful single-variable changes from the baseline.

## Current Research Theme: Transolver Baseline Characterization

The research track is freshly started. The baseline Transolver configuration is:
- Architecture: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- Training: `lr=5e-4`, `surf_weight=10.0`, `weight_decay=1e-4`, `batch_size=4`
- Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across 4 val splits; lower is better)

## Active WIP Experiments (Round 1)

| PR | Student | Hypothesis |
|----|---------|------------|
| #735 | alphonse | Wider hidden dim 128→256 |
| #736 | askeladd | More PhysicsAttention slices 64→128 |
| #738 | edward   | Surface loss weight 10→20 |
| #740 | fern     | Deeper Transolver: 5→7 layers |
| #741 | frieren  | Wider FFN: mlp_ratio 2→4 |
| #744 | nezuko   | LR warmup: 5-epoch linear ramp then cosine decay |
| #746 | tanjiro  | More attention heads: 4→8 |
| #747 | thorfinn | Per-sample normalized loss for Re regime equalization |

## Potential Next Research Directions

Once Round 1 results come in, the next tier of experiments should explore:

1. **Combinations of winners** — if both wider hidden dim and more slices improve performance, combine them
2. **Loss reformulation** — physics-informed losses, per-channel weighting (p vs Ux/Uy), gradient-based surface emphasis
3. **Data augmentation** — Re-stratified sampling, geometry perturbation, domain-aware augmentation
4. **Attention mechanism** — cross-attention between surface and volume nodes, hierarchical attention
5. **Alternative optimizers** — Lion, SOAP, or schedule-free Adam
6. **Multi-scale features** — enriched input features (curvature, arc-length gradients, local normals)
7. **Graph neural approaches** — replace or augment Transolver with GNN layers for mesh connectivity
8. **Ensemble and test-time augmentation** — model averaging across Re regimes
9. **OOD specialization** — domain-specific heads for raceCar vs cruise domains
10. **Physics constraints** — enforce continuity equation softly via auxiliary loss
