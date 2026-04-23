# SENPAI Research State

- **Updated:** 2026-04-23 20:40
- **Research track:** test-launch-20260423-2036
- **Advisor branch:** testing2
- **W&B project:** wandb-applied-ai-team/senpai-testing

---

## Student Status

| Student | Status | Current PR |
|---------|--------|-----------|
| frieren | WIP | PR #1: Loss reformulation sweep |
| fern | WIP | PR #2: Capacity scaling sweep |

**Idle students:** none — all GPUs assigned.

---

## PRs Ready for Review

None.

---

## PRs In Progress (status:wip)

| PR | Student | Hypothesis | Branch |
|----|---------|-----------|--------|
| #1 | frieren | Huber/L1 loss sweep — close train/eval MSE→MAE metric gap | frieren/loss-reformulation-v1 |
| #2 | fern | Transolver capacity scaling (n_hidden, slice_num, n_layers) | fern/capacity-scaling-v1 |

---

## Baseline

**No baseline established yet.** Both PR #1 (GPU 0) and PR #2 (GPU 0) include vanilla baseline runs that will establish the anchor `val_avg/mae_surf_p` for this research track.

---

## Most Recent Research Direction from Human Team

No human issues received as of 2026-04-23 20:40.

---

## Current Research Focus and Themes

This is the **bootstrap round** — no prior experiments on this track. First-round priorities:

1. **Establish the baseline** `val_avg/mae_surf_p` via the vanilla Transolver run included in both PR sweeps.
2. **Loss reformulation** (PR #1): The train/eval mismatch (MSE training vs. MAE ranking) is a well-known source of suboptimality, especially for heavy-tailed CFD distributions. Huber δ ∈ {0.5, 1.0, 2.0} and L1 variants, crossed with surf_weight ∈ {5, 10, 20}.
3. **Capacity scaling** (PR #2): The ~0.7M-param baseline is almost certainly under-capacity for 242K-node meshes on 96GB GPUs. n_hidden 128→256→384, slice_num 64→128→192, n_layers 5→7→9.

These two dimensions are orthogonal — if both improve vs. baseline, they can be merged in round 2 for a compounding improvement.

---

## Potential Next Research Directions and Themes

(Generated from initial dataset analysis; researcher-agent report pending in RESEARCH_IDEAS file)

### High priority (expected high impact, low complexity)
- **Gradient clipping**: High-Re samples produce large normalized residuals and potentially exploding gradients. Adding `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before `optimizer.step()` may stabilize training on outlier batches.
- **EMA of model weights**: Exponential moving average of model parameters during training often improves generalization at no additional cost. Keeps a shadow model with polyak averaging (decay ~0.9999).
- **Cosine warmup + LR floor**: The current cosine annealing drops LR to 0 — a small LR floor (e.g., 5e-6) or a warmup phase (5 epochs linear ramp-up) may improve early training stability.
- **asinh normalization for p**: Pressure values span -29K to +2.6K with heavy tails. `asinh(p / scale)` is smooth, bounded, and handles large values more gracefully than linear normalization.

### Medium priority (architectural)
- **Surface-specific decoder head**: The ranking metric is surface pressure MAE. A dedicated higher-capacity decoder for surface nodes (separate MLP2 for is_surface nodes) could help.
- **Fourier positional encoding**: Replace raw (x,z) coordinates (dims 0–1) with random Fourier features or learnable Fourier encoding. This enriches the spatial signal and is known to help for physics-informed operators.
- **Relative position encoding in slice attention**: The current PhysicsAttention is position-agnostic in the slice-to-slice attention step. Adding relative position bias (RPE) between slice tokens could help localize physics patterns.
- **Per-domain normalization**: Cruise and raceCar have different Re ranges and p magnitudes. Separate y_mean/y_std per domain could reduce inter-domain interference.

### Longer-term (physics-informed, higher complexity)
- **Potential flow features**: Compute panel-method predictions (Ux_pot, Uy_pot, Cp_pot) at each node using the NACA geometry and AoA. These can be additional input features (residual learning baseline) or used as loss regularization.
- **Physics-consistency loss**: Incompressible 2D flow satisfies ∂Ux/∂x + ∂Uy/∂z = 0. Adding a divergence-free penalty at surface nodes could improve physical consistency.
- **Re-conditional normalization (FiLM)**: Use log(Re) to condition normalization layers (Feature-wise Linear Modulation) — the model learns per-Re scale and shift for each hidden layer, enabling sharper adaptation to the 10× value range difference between low-Re and high-Re samples.
- **Graph neural network layer**: Add a KNN-based message-passing step before or after Transolver attention to capture local mesh connectivity.
