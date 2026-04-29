# SENPAI Research State
- 2026-04-29 12:30
- No recent research directions from human researcher team (no GitHub issues found)
- Current research focus and themes: Round 1 experiments on TandemFoilSet CFD surrogate. PR #1102 (MLP ratio) closed (regression). PR #1152 (eta_min sweep) newly assigned to thorfinn. Key outstanding result: PR #1098 shows val=114.62 with grad_clip+lr=1e-3 but with surf_weight=10 (confounded); PR #1143 (alphonse) will provide a clean ablation.

## Current Baseline

**val_avg/mae_surf_p = 127.6661** (PR #1088, charliepai2f2-edward, surf_weight=10→25, epoch 13/50)

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 157.82 |
| val_geom_camber_rc | 135.65 |
| val_geom_camber_cruise | 99.26 |
| val_re_rand | 117.94 |

Best unmerged result: **PR #1098** val_avg/mae_surf_p=**114.62** (tanjiro, WIP) — but confounded by surf_weight=10 vs baseline's surf_weight=25. PR #1143 (alphonse) will produce a clean combined-best test.

Note: NaN in test_geom_camber_cruise pressure due to corrupted GT sample `000020.pt`. Fix: drop non-finite-y samples before computing `pred - y` in `train.py:evaluate_split`. Should be applied to all future experiments.

## Key Insight: Systematic Timeout/LR Mismatch

All Round 1 experiments use `CosineAnnealingLR(T_max=50)` but with a 30-min timeout, the baseline only completes ~14 epochs and wider models even fewer. This means:
- LR has only annealed ~5-28% from its initial value at end of training
- All Round 1 experiments are effectively running at near-peak LR throughout
- Fixing T_max to match actual epoch count could meaningfully improve convergence
- Even with T_max=14, LR decays to zero at the final epoch — no sustained low-LR fine-tuning. `eta_min=1e-5` prevents this (now being tested by thorfinn, PR #1152)

## Active Experiments

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #1086 | alphonse | WIP (revision) | Width expansion: n_hidden=192, n_head=6, surf_weight=25 + NaN guard |
| #1087 | askeladd | WIP | Slice count sweep: slice_num 64→128 |
| #1089 | fern | WIP | Deeper model: n_layers 5→8 with lr 3e-4 |
| #1090 | frieren | WIP | Per-field output heads: separate MLP decoder for Ux, Uy, p |
| #1091 | nezuko | WIP | Stochastic depth regularization: drop_path 0→0.1 |
| #1098 | tanjiro | WIP | Grad clip + higher LR (1e-3); val=114.62 (confounded: surf_weight=10) |
| #1126 | edward | WIP | Timeout-aware cosine LR: T_max=14 to fully anneal within 30-min cap |
| #1143 | alphonse | WIP | Combined best config: lr=1e-3 + grad_clip + T_max=14 + surf_weight=25 |
| #1144 | askeladd | WIP | BF16 AMP: more epochs per 30-min budget |
| #1152 | thorfinn | WIP (new) | CosineAnnealingLR eta_min=1e-5: non-zero LR floor for final epochs |

## Closed/Resolved Experiments

| PR | Student | Outcome |
|----|---------|---------|
| #1088 | edward | MERGED — new baseline, val=127.67, surf_weight=10→25 |
| #1086 | alphonse | Sent back (revision): inconclusive due to timeout; try n_hidden=192, n_head=6 |
| #1102 | thorfinn | CLOSED — regression (val=136.16), mlp_ratio=4 hurts at this data scale |
| #1129 | (unknown) | CLOSED — catastrophic regression (val=366.76), per-sample instance norm |

## Potential Next Research Directions

1. **Combined best config validation**: PR #1143 (alphonse) will confirm whether lr=1e-3 + grad_clip + T_max=14 + surf_weight=25 is a clean winner
2. **eta_min floor for cosine annealing**: PR #1152 (thorfinn, new) — testing `eta_min=1e-5` to prevent LR hitting zero at final epoch
3. **Warmup + cosine schedule**: LinearWarmup (2 epochs) before cosine decay — unexplored
4. **Surface-node attention bias**: Explicitly bias attention weights toward surface mesh nodes within Transolver's physics-slice mechanism
5. **Loss formulation**: Laplacian/gradient smoothness penalty on pressure field, physics-informed boundary conditions
6. **Multi-scale architecture**: Hierarchical pooling over mesh zones (background vs. dense foil zones)
7. **Data augmentation**: Reynolds number scaling, AoA perturbation, foil geometry perturbation
8. **Ensemble approaches**: Cheap model ensemble from multiple checkpoints (snapshot ensemble across final few epochs)
9. **Fourier features**: Random Fourier features for position encoding on irregular meshes
10. **Graph neural networks**: Message passing along mesh connectivity (GNN-based surrogate)
