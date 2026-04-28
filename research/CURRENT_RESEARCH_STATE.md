# SENPAI Research State

- 2026-04-28 19:30
- No recent research direction from human researcher team (no open GitHub issues found)
- Current research focus and themes:

## Current Focus

**Track: icml-appendix-charlie-pai2e-r2**

This is a fresh research track on TandemFoilSet. The baseline is the stock Transolver model (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2). The primary metric is `val_avg/mae_surf_p`.

### First Wave (in-flight, 8 student experiments)

The first wave tests fundamental architectural and optimization levers in parallel:

| PR | Student | Hypothesis | Key change |
|----|---------|------------|------------|
| #764 | charliepai2e2-alphonse | Larger model capacity | n_hidden 128→256 |
| #765 | askeladd | More physics slices | slice_num 64→128 |
| #766 | edward | Deeper network | n_layers 5→8 |
| #767 | fern | Higher surface weight | surf_weight 10→50 |
| #768 | frieren | Lower LR + warmup | lr 5e-4→1e-4, 500-step warmup |
| #772 | nezuko | Per-channel output affine | learnable scale+bias per output channel |
| #778 | tanjiro | Gradient clipping | clip_grad_norm 1.0 |
| #780 | thorfinn | Higher MLP ratio | mlp_ratio 2→4 |

## Potential Next Research Directions

1. **Surface-aware loss reformulation** — weight surface nodes more aggressively; explore per-sample normalization to handle Re variance
2. **Multi-scale attention** — combine coarse background zone features with dense foil-zone features
3. **Physics-informed constraints** — add continuity equation regularization (divergence-free velocity field)
4. **Ensemble / mixture-of-experts** — different sub-networks for different Re regimes or domain types
5. **Coordinate-aware positional encoding** — richer geometric embeddings using foil geometry directly
6. **Adaptive learning rate per parameter group** — different lr for spatial attention vs. MLP layers
7. **Test-time augmentation / ensembling** — averaging predictions over multiple augmented inputs
8. **Graph neural network approach** — exploit the mesh connectivity structure explicitly
