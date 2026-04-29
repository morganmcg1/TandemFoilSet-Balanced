# SENPAI Research State
- 2026-04-29 14:30
- No recent research directions from human researcher team (no GitHub issues found)
- Current research focus: Squeezing the most out of the 30-minute timeout budget on TandemFoilSet CFD surrogate (Transolver architecture). Round 2+ builds on three validated wins: surf_weight=25, DropPath+budget-aware cosine LR, and higher LR (1e-3) + grad_clip=1.0.

## Current Baseline

**val_avg/mae_surf_p = 100.41** (PR #1098, charliepai2f2-tanjiro, lr=1e-3 + grad_clip=1.0 + DropPath 0→0.1 + budget-aware CosineAnnealingLR + surf_weight=25, epoch 14/50)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 120.68 | 104.32 |
| geom_camber_rc | 111.80 | 98.04 |
| geom_camber_cruise | 75.99 | 63.06 |
| re_rand | 93.15 | 88.91 |
| **avg** | **100.41** | **88.58** |

Note: NaN in test_geom_camber_cruise pressure is a corrupted upstream GT sample (000020.pt). Fix: drop non-finite-y samples before computing `pred - y` in `train.py:evaluate_split`. All future experiments should apply this guard.

## Key Validated Wins

1. **surf_weight=25** (PR #1088, edward) — focusing training loss on surface nodes directly targets primary metric
2. **Stochastic depth (DropPath 0→0.1) + budget-aware CosineAnnealingLR** (PR #1091, nezuko) — regularization + proper T_max matching actual epoch count (-4.5%)
3. **lr=1e-3 + grad_clip=1.0** (PR #1098, tanjiro) — dramatic -17.6% improvement; grad_clip enabled stable training at higher LR

## Key Insights

- **Budget-aware cosine LR**: T_max=50 only anneals ~5% in 14 epochs at baseline speed. T_max set to actual epoch count is now standard.
- **grad_clip enables higher LR**: Gradient clipping (1.0) stabilizes lr=1e-3 training, unlocking significantly faster convergence.
- **Surface emphasis critical**: surf_weight=25 vs. 10 is consistently important; do not reduce.
- **Model is data-constrained at 0.66M params**: Wider MLPs (mlp_ratio=4) regressed — ~1.5K training samples can't utilize extra capacity in 14 epochs. Capacity expansion needs architectural creativity, not brute-force width.

## Active Experiments (Round 2+)

| PR | Student | Hypothesis |
|----|---------|-----------|
| #1090 | frieren | Per-field output heads: separate MLP decoder for Ux, Uy, p |
| #1143 | alphonse | Combined best config: lr=1e-3 + grad_clip + T_max=14 + surf_weight=25 |
| #1152 | thorfinn | CosineAnnealingLR eta_min=1e-5: non-zero LR floor for final epochs |
| #1166 | edward | LR warmup (2 epochs linear) + cosine anneal on top of new 100.41 baseline |
| #1178 | tanjiro | Weight decay 1e-4→1e-3: stronger L2 reg at lr=1e-3 |
| #1182 | fern | surf_weight 25→50: stronger surface loss focus on primary metric |
| #1184 | askeladd | BF16 AMP on current stack: more epochs within 30-min budget |

## Closed/Resolved Experiments

| PR | Student | Outcome |
|----|---------|---------|
| #1088 | edward | MERGED — baseline 127.67, surf_weight=10→25 |
| #1091 | nezuko | MERGED — baseline 121.89, DropPath + budget-aware cosine |
| #1098 | tanjiro | MERGED — baseline 100.41, lr=1e-3 + grad_clip=1.0 |
| #1086 | alphonse | Sent back: inconclusive due to timeout; revised to combined-best-config (#1143) |
| #1102 | thorfinn | CLOSED — regression (val=136.16), mlp_ratio=4 hurts at this data/budget scale |
| #1126 | edward | CLOSED (superseded) — val=123.79, T_max=14 isolation, didn't beat 100.41 |
| #1129 | (unknown) | CLOSED — catastrophic regression (val=366.76), per-sample instance norm |
| #1185 | nezuko | CLOSED — val=108.69 vs 100.41; SGDR restarts incompatible with 14-epoch budget (each restart spikes +22 to +47 in val) |

## Potential Next Research Directions

1. **Higher lr_max (e.g. 1.5e-3 or 2e-3) with grad_clip**: The model tolerated 1e-3; with proper annealing, may handle higher
2. **Larger batch size (8 or 16)**: More stable gradients, may allow higher LR; test if VRAM budget allows
3. **Surface-node attention bias**: Explicitly bias attention weights toward surface mesh nodes within Transolver's physics-slice mechanism
4. **Loss formulation variants**: Laplacian/gradient smoothness penalty on pressure field; relative error weighting; Huber loss for outlier robustness
5. **Wider architecture (n_hidden=192)**: Budget-aware with combined best training config — intermediate between 128 and 256 that may avoid the timeout-asymmetry issue
6. **n_layers 5→6**: Moderate depth increase (lighter per-layer than 5→7); run with combined best config
7. **Multi-scale architecture**: Hierarchical pooling over mesh zones (background vs. dense foil zones)
8. **Data augmentation**: Reynolds number scaling, AoA perturbation, foil geometry perturbation
9. **Ensemble/snapshot averaging**: Cheap model ensemble from multiple checkpoints (snapshot ensemble across final few epochs at low LR)
10. **OneCycleLR policy**: Known strong baseline for constrained-budget training with aggressive early LR + cycle structure
11. **Deeper DropPath schedule**: Increase max DropPath from 0.1 to 0.15 or 0.2 while holding lr=1e-3 + grad_clip
12. **Mixed precision BF16 to enable larger batch**: Use AMP to increase effective batch size, potentially enabling lr scaling
