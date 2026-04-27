# SENPAI Research State

- **Date**: 2026-04-27
- **Most recent research direction from human researcher team**: None (no human issues found)
- **Current research focus**: Round 1 experiments on the TandemFoilSet CFD surrogate task. 8 students are actively running experiments. No results yet — waiting for first round to complete. The baseline is Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2) with AdamW (lr=5e-4), surf_weight=10, cosine LR. Primary metric: val_avg/mae_surf_p (surface pressure MAE, lower is better).

## Active Round 1 Experiments (charliepai2c3 students)

| PR | Student | Hypothesis |
|----|---------|------------|
| #193 | charliepai2c3-alphonse | Vanilla baseline anchor — establish reference metrics |
| #198 | charliepai2c3-askeladd | L1 loss with surf_weight=1 — align training with MAE metric |
| #200 | charliepai2c3-edward | surf_weight sweep: 20 and 50 (baseline=10) — surface pressure focus |
| #203 | charliepai2c3-fern | Wider Transolver n_hidden=256 — more model capacity for complex flow |
| #207 | charliepai2c3-frieren | LR warmup + cosine to 1e-3 — faster convergence with stability |
| #209 | charliepai2c3-nezuko | EMA weight averaging (decay=0.999) — smoother generalization |
| #214 | charliepai2c3-tanjiro | Per-channel pressure up-weighting in loss (3x on p channel) |
| #219 | charliepai2c3-thorfinn | Per-channel decoder heads (Ux, Uy, p) — field-specific output MLPs |

## Current Research Themes

1. **Loss formulation**: L1 vs L2 (#198), per-channel pressure up-weighting (#214)
2. **Hyperparameter tuning**: surf_weight sweep 10→20,50 (#200), LR warmup + cosine to 1e-3 (#207)
3. **Architecture exploration**: Wider model n_hidden=256 (#203), per-channel decoder heads (#219)
4. **Regularization**: EMA weight averaging decay=0.999 (#209)
5. **Reference**: Vanilla baseline anchor (#193) — establishes round-1 metric target

## Potential Next Research Directions

- Deeper models (n_layers=6-8) with larger hidden dim (192, 256)
- Fourier / sinusoidal positional encodings for spatial position features (dims 0-1)
- SwiGLU feedforward activations in Transolver MLP blocks
- asinh target transform on p channel to handle heavy-tailed Re distribution
- FiLM conditioning: inject Re and NACA parameters as global conditioning into each layer
- Gradient clipping (max_norm=1.0) + higher base LR
- Signed-log pressure transform: compress extreme high-Re pressure values
- LayerScale + stochastic depth for regularization
- Surface-only loss with zero vol_weight to maximize surface pressure accuracy
- Huber loss (smooth L1) as compromise between MSE robustness and L1 gradient stability

## Key Constraints

- VRAM: 96 GB per GPU, meshes up to 242K nodes
- Timeout: SENPAI_TIMEOUT_MINUTES wall clock + --epochs limit
- Epochs cap: from env (do not override)
- Data loaders are read-only (only train.py is editable)
- Primary metric: val_avg/mae_surf_p (lower is better)
