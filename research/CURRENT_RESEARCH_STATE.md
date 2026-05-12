# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-12 (launch start)
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock. Treat experiments as isolated for git and experiment artifacts; do not cross-reference unrelated branches.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current research focus

This branch starts from the unmodified `train.py` baseline (Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2; AdamW lr=5e-4 wd=1e-4; CosineAnnealingLR; weighted MSE with surf_weight=10). With a hard 30-min per-run cap, we'll get only ~3–6 full epochs per arm, so the first round prioritizes **changes that show signal early and have low per-step overhead**:

1. **Loss-formulation levers** that re-weight surface pressure (which dominates the ranking metric) without changing capacity or schedule.
2. **Optimization-schedule levers** (LR shape, warmup) that change where the optimizer ends up after 3–6 epochs.
3. **Light architectural/feature additions** that fit in the 30-min budget — input encodings, dropout, small width/depth bumps only with care.
4. **Throughput levers** (mixed precision) that buy more epochs in the same wall-clock.

## Open hypothesis families (round 1)

| Student | Family | Why |
|---------|--------|-----|
| alphonse | `surf_weight=30` | Surface pressure is the ranking metric; the default 10 likely underweights it relative to volume. |
| askeladd | Huber loss (δ=1.0) replacing MSE | y has 4 orders of magnitude; Huber should be more robust on Re extremes (esp. re_rand). |
| edward   | OneCycleLR (max_lr=1e-3, pct_start=0.1) | Higher peak LR with warmup tends to escape sharp minima; cosine alone may under-explore. |
| fern     | Dropout=0.1 + grad-clip 1.0 | Light regularization to improve OOD generalization on the unseen-camber tracks. |
| frieren  | bf16 autocast | Free speed-up → more epochs in the 30-min cap → better convergence at fixed cost. |
| nezuko   | Fourier positional encoding on (x,z) | Spatial coords are continuous; Fourier features help models learn high-freq pressure patterns. |
| tanjiro  | Auxiliary surface-pressure head (separate MLP, λ=2) | Specialized capacity on the metric-driving channel without disturbing the main head. |
| thorfinn | Warmup 3 epochs then cosine for the rest | Smoother early steps with 5e-4 peak; cheap to test and orthogonal to OneCycleLR. |

## Potential next research directions

- **Output-space transforms:** asinh / log-magnitude on `p` to compress the high-Re tail before the loss.
- **Per-sample y normalization:** divide each sample's targets by its own y_std to remove the Re-driven scale variance.
- **Slice-count sweep:** Transolver's `slice_num` (64 default) controls the number of physical "tokens" — likely a high-leverage axis.
- **Sample weights tweak:** the balanced sampler equalizes domains; try domain-weighted loss with extra weight on cruise (largest meshes, smallest y).
- **EMA evaluation:** track an exponential moving average of weights and evaluate on it.
- **Residual / SwiGLU MLPs in TransolverBlock** if simpler wins land.
- **Sobolev / divergence-free penalty** on (Ux, Uy) — physics-informed aux loss.

The list above is the queue for round 2 (assignment after round 1 results land).
