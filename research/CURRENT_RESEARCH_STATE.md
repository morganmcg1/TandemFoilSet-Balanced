# SENPAI Research State
- 2026-04-29 13:00
- No recent research directions from human researcher team (no GitHub issues found)
- Current research focus: Squeezing the most out of the 30-minute timeout budget on TandemFoilSet CFD surrogate (Transolver architecture). Round 2 builds on validated wins (surf_weight=25, stochastic depth + budget-aware cosine LR).

## Current Baseline

**val_avg/mae_surf_p = 121.89** (PR #1091, charliepai2f2-nezuko, DropPath 0→0.1 + budget-aware CosineAnnealingLR + surf_weight=25, epoch 13/50)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 154.41 | 130.06 |
| geom_camber_rc | 128.38 | 118.20 |
| geom_camber_cruise | 95.98 | 79.45 |
| re_rand | 108.78 | 110.64 |
| **avg** | **121.89** | **109.59** |

Note: NaN in test_geom_camber_cruise pressure is a corrupted upstream GT sample (000020.pt). Fix: drop non-finite-y samples before computing `pred - y` in `train.py:evaluate_split`. All future experiments should apply this guard.

## Key Validated Wins

1. **surf_weight=25** (PR #1088, edward) — focusing training loss on surface nodes, directly targeting primary metric
2. **Stochastic depth (DropPath 0→0.1) + budget-aware CosineAnnealingLR** (PR #1091, nezuko) — regularization + proper schedule that fully anneals within actual epoch count

## Key Insight: Systematic Timeout/LR Mismatch

All default experiments use `CosineAnnealingLR(T_max=50)` but with a 30-min timeout, baseline only completes ~14 epochs. LR has only annealed ~5-28% from its initial value. Budget-aware T_max (matching actual epoch count) is now standard for new assignments.

## Active Experiments (Round 2)

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #1090 | frieren | WIP | Per-field output heads: separate MLP decoder for Ux, Uy, p |
| #1098 | tanjiro | WIP | Grad clip + higher LR (1e-3); earlier result val=114.62 but confounded by surf_weight=10 |
| #1126 | edward | WIP | Timeout-aware cosine LR: T_max=14 to fully anneal within 30-min cap |
| #1143 | alphonse | WIP | Combined best config: lr=1e-3 + grad_clip + T_max=14 + surf_weight=25 |
| #1144 | askeladd | WIP | BF16 AMP: more epochs per 30-min budget via mixed precision |
| #1152 | thorfinn | WIP | CosineAnnealingLR eta_min=1e-5: non-zero LR floor for final epochs |
| #1156 | nezuko | WIP | DropPath max rate 0.1→0.2: stronger stochastic depth regularization |
| #1161 | fern | WIP | n_layers 5→6: moderate depth + budget-aware LR + surf_weight=25 |

## Closed/Resolved Experiments

| PR | Student | Outcome |
|----|---------|---------|
| #1088 | edward | MERGED — new baseline, val=127.67, surf_weight=10→25 |
| #1091 | nezuko | MERGED — new baseline, val=121.89, DropPath + budget-aware cosine |
| #1086 | alphonse | Sent back (revision): inconclusive due to timeout; trying n_hidden=192/combined-best-config instead |
| #1102 | thorfinn | CLOSED — regression (val=136.16), mlp_ratio=4 hurts at this data scale |
| #1129 | (unknown) | CLOSED — catastrophic regression (val=366.76), per-sample instance norm |

## Potential Next Research Directions

1. **Warmup + cosine schedule**: LinearWarmup (2 epochs) before cosine decay — unexplored, may help with lr=1e-3 experiments
2. **Surface-node attention bias**: Explicitly bias attention weights toward surface mesh nodes within Transolver's physics-slice mechanism
3. **Loss formulation variants**: Laplacian/gradient smoothness penalty on pressure field; relative error weighting; Huber loss
4. **Wider but shallower model**: n_hidden=192, n_head=6 (budget-aware) — confounded by n_layers in prior PR #1086 attempt
5. **Multi-scale architecture**: Hierarchical pooling over mesh zones (background vs. dense foil zones)
6. **Data augmentation**: Reynolds number scaling, AoA perturbation, foil geometry perturbation
7. **Ensemble approaches**: Cheap model ensemble from multiple checkpoints (snapshot ensemble across final few epochs)
8. **Fourier features**: Random Fourier features for position encoding on irregular meshes
9. **Graph neural networks**: Message passing along mesh connectivity (GNN-based surrogate)
10. **AdamW with decoupled weight decay + OneCycleLR**: Higher peak LR early with aggressive decay — known strong baseline for constrained-budget training
