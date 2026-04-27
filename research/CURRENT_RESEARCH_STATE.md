# SENPAI Research State
- 2026-04-27 — fresh advisor branch `icml-appendix-charlie-pai2d-r1`, no prior experiments on this track
- Human researcher direction: not yet received on this branch
- Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across the four val splits); ranking final metric is `test_avg/mae_surf_p`

## Baseline (default `train.py`)
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
- AdamW lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, cosine annealing over `epochs`
- No metrics measured yet on this branch — round 1 will establish the baseline numbers and start beating them.

## Round 1 hypothesis portfolio (8 experiments)
Coverage chosen so that each lever is moved independently and likely-orthogonal gains can compound across rounds:

| Student | Slug | Lever | Why |
|---|---|---|---|
| alphonse  | bigger-transolver-bf16   | Architecture (capacity)        | Wider model + bf16 — Transolver default may underfit 24-dim 3-output regression on 1499 train samples |
| askeladd  | surf-weight-50           | Loss balance                   | Surface pressure dominates the metric; raise weight from 10 → 50 |
| edward    | smoothl1-surface         | Loss form                      | SmoothL1 (Huber) on surface — eval is MAE, MSE over-penalizes outliers |
| fern      | warmup-cosine-1e3        | LR schedule                    | 5-epoch linear warmup, peak 1e-3, cosine to 1e-5 — transformers like warmup |
| frieren   | slice-128-heads-8        | Slice/head count               | More physics-aware slice tokens for irregular meshes up to 242K nodes |
| nezuko    | mlp-ratio-4              | MLP capacity                   | Default ratio=2 is conservative; 4 is the typical transformer setting |
| tanjiro   | ema-eval                 | Optimization smoothing         | Evaluate val on EMA(0.999) weights for smoother checkpoint selection |
| thorfinn  | channel-weighted-loss    | Per-channel loss weight        | Up-weight `p` channel in the surface loss — directly aligns with metric |

## Potential next directions (Round 2+)
- Combine the strongest two winners (capacity × loss alignment).
- Optimizer changes (Lion, Adan, SOAP).
- Mesh/sample augmentation (rotation, sub-sampling for larger effective batch).
- Physics-informed regularization (divergence-free / mass conservation auxiliary loss).
- Multi-scale slice attention (mix slice_num=32, 64, 128 across layers).
- Re-engineering of input features (log-Re bucketing, Fourier position features, distance-to-leading-edge).
- Per-domain conditioning (single vs raceCar tandem vs cruise tandem).
- Train/val mismatch diagnostics: which split is currently the worst and why?
