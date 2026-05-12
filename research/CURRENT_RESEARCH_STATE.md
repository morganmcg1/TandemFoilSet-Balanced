<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# SENPAI Research State — TandemFoilSet

- **Date**: 2026-05-12
- **Launch**: `willow-pai2g-24h-r3` (isolated 24h appendix experiment)
- **Advisor branch**: `icml-appendix-willow-pai2g-24h-r3`
- **W&B project**: `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r3`
- **Target metric** (lower is better): `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across 4 splits)
- **Paper-facing test metric**: `test_avg/mae_surf_p`
- **Hard caps**: 30 min wall-clock per training run, 50 epochs, 1 GPU (96GB) per student

## Most recent direction from human researcher team
None yet — no GitHub issues open at start of launch.

## Current research focus

This is a fresh research track on a clean baseline. Round 1 covers eight orthogonal "compound levers" that each address a different known limitation of the baseline training recipe, all testable inside a single 30-min run.

The baseline Transolver recipe has several obvious soft spots:

1. **Loss alignment with the metric.** Ranking is surface-pressure MAE only, but the loss uses equal channel weights and surf_weight=10 (modest).
2. **Heavy-tailed targets.** y std varies by ~10x within a single split (high-Re samples dominate MSE gradients).
3. **No gradient hygiene.** No clipping, no warmup, no EMA — all standard transformer-training practices are absent.
4. **Slow throughput.** No mixed precision, so 50 epochs at batch=4 is the budget ceiling.
5. **Small capacity.** n_hidden=128 may be capacity-limited across order-of-magnitude Re range.

## Round 1 hypothesis assignments (8 parallel)

| Slug | Mechanism | Risk |
|---|---|---|
| `surf-weight-50` | Raise surf_weight 10→50 (metric alignment) | Low–Med |
| `grad-clip-norm1` | clip_grad_norm_ at 1.0 (stabilize high-Re outliers) | Low |
| `p-channel-weight3x` | 3x weight on pressure channel in loss | Low–Med |
| `ema-decay999` | EMA model weights for val/test/checkpoint | Low |
| `warmup-5ep` | 5-epoch linear LR warmup before cosine | Low |
| `amp-bf16` | bfloat16 autocast — convert wall-clock to extra steps | Medium |
| `smooth-l1-beta01` | Replace MSE with SmoothL1(beta=0.1) in normalized space | Medium |
| `wider-n192` | n_hidden 128→192, n_head 4→6 | Medium |

Full hypothesis design lives in `research/RESEARCH_IDEAS_2026-05-12_initial.md`.

## Potential next directions (Round 2+)

After Round 1 results land, the next wave will likely come from:

- **Combos of Round 1 winners.** Stack the strongest of {`surf-weight`, `grad-clip`, `p-weight`, `ema`} which are mostly independent levers.
- **Loss reformulation deeper.** Per-Re loss reweighting (downweight low-Re where MAE is naturally tiny; upweight high-Re). Relative MAE in normalized space.
- **Architecture beyond width.** Deeper Transolver (n_layers 5→8) and slice_num sweep (64→128).
- **Geometric features.** Fourier positional features for (x, z), surface-distance features, or per-foil indicator channels.
- **Augmentation.** Horizontal mirroring (with AoA sign flip and rear-foil swap for tandem) — exact CFD symmetry that doubles effective dataset.
- **Optimizer.** AdamW → Lion, Adan; larger weight_decay; layer-wise LR decay.

The most promising single direction beyond Round 1 is likely a **best-checkpoint combo** of metric-aligned losses (`surf-weight` + `p-channel`) with training hygiene (`grad-clip` + `warmup` + `ema`) — stacking should compound to a multi-percent reduction in `val_avg/mae_surf_p`.

## Open questions to keep in view

- Are the val/test split distributions drifting? Worth comparing val and test surface-p MAE for each split after Round 1 to confirm val is a faithful predictor of test.
- Which split dominates the `val_avg`? If `val_re_rand` or `val_geom_camber_cruise` is much larger than the others, methods that help only the easy splits will be misleading.
- Does Transolver's `slice_num=64` saturate on 242K-node cruise meshes? Could be a capacity bottleneck specifically for cruise.
