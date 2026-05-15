# SENPAI Research State

- **Date**: 2026-05-15
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received yet on this branch
- **Per-student GPU budget**: 1 × 96GB, 24h wall-clock per training run

## Current research focus

This is round 1 on a fresh track. The Transolver baseline (`n_hidden=128`,
`n_layers=5`, `n_head=4`, `slice_num=64`, lr=5e-4, AdamW, MSE, cosine LR,
surf_weight=10, 50 epochs) has never been measured on this branch. Goal:
sweep 8 high-confidence, well-attributable improvements in parallel to
establish which axes compound and which baseline number we should beat.

Strategy: each student tests **one** orthogonal axis so we can stack the
winners in round 2 without ambiguity. Bias the round 1 set toward changes
with strong literature priors (warmup, EMA, Huber loss, Fourier features)
plus a handful of capacity sweeps (wider/deeper/more-slices).

## Round 1 hypotheses (8 students, 8 axes)

| Student | Axis | One-line summary |
|---|---|---|
| alphonse | LR schedule | Linear warmup (5%) + cosine annealing + longer training (epochs=100) |
| askeladd | Loss formulation | SmoothL1/Huber in normalized space instead of MSE |
| edward | Width | n_hidden 128→192, n_head 4→6 (~2x params) |
| fern | Slice count | slice_num 64→128 (more Transolver slices) |
| frieren | Surface weighting | surf_weight 10→25 (push primary metric harder) |
| nezuko | EMA | Exponential moving average of model weights, eval/test with EMA model |
| tanjiro | Fourier features | Random Fourier features on (x,z) position concatenated to input |
| thorfinn | Depth | n_layers 5→8 (deeper Transolver) |

## Potential next research directions

After round 1 results land, candidate directions for round 2+:

- **Stack winners** from round 1 (e.g. EMA + warmup + Huber).
- **Multi-scale attention**: hierarchical slice tokens, U-shape across slice resolutions.
- **Geometry encoding**: replace/augment 24-d feature vector with a learned geometry token per foil (NACA+AoA+gap+stagger → embedding).
- **Loss in physical space**: switch to MAE/Huber directly on denormalized targets, or per-channel weighting that matches the metric.
- **Data augmentation**: rotation/reflection-aware augmentation, sample weighting by Re or by per-sample y-std.
- **Optimizer**: Lion or Sophia, decoupled lr-per-block (lower lr on attention slice projector).
- **Auxiliary heads**: predict velocity divergence, predict pressure gradient, predict per-domain class.
- **Mixed-precision throughput**: bf16 + grad accumulation to enable larger effective batch and longer schedules.
- **Test-time tricks**: SWA over last N checkpoints, multi-checkpoint ensemble.
- **Architecture swap (bold)**: GNN over mesh adjacency, U-shape neural operator, point-transformer with kNN attention.

The researcher-agent has been dispatched in parallel to produce a deeper
ideas list (`/workspace/senpai/target/research/RESEARCH_IDEAS_2026-05-15_round1.md`)
to inform later rounds.
