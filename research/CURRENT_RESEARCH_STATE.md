# SENPAI Research State

- **Last updated**: 2026-05-12 (start of round 4)
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging
  ablation. Each individual target training execution is capped at
  `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current research focus

This is a fresh round with no prior experiments on this advisor branch and
no merged winners. The default `train.py` Transolver config is the implicit
baseline — the goal of round 1 is to establish how much headroom exists
along several orthogonal axes simultaneously so the next rounds can
compound the strongest one.

The primary ranking metric is `val_avg/mae_surf_p` (paper-facing
comparison metric is `test_avg/mae_surf_p`). Each training run is wall-clock
capped at 30 minutes, which on the current 1499-sample training set with
`batch_size = 4` is enough for roughly a single-digit number of epochs.
Hypotheses for round 1 are picked to give meaningful signal under that
cap rather than require long convergence.

## Round 1 hypothesis matrix (one per student)

| Student | Slug | Axis | Hypothesis |
|---------|------|------|-----------|
| alphonse | `l1-loss` | Loss | Replace MSE with L1; aligns the training objective with the MAE metric. |
| askeladd | `surf-weight-30` | Loss balance | Raise `surf_weight` 10 → 30; primary metric is surface pressure. |
| edward | `wider-deeper` | Capacity | `n_hidden` 128 → 192, `n_layers` 5 → 7 within VRAM. |
| fern | `slice-num-128` | PhysicsAttention | `slice_num` 64 → 128 — finer mesh tokenization. |
| frieren | `lr-warmup-1e-3` | Optimization | `lr` 5e-4 → 1e-3 with 5-epoch linear warmup, cosine after. |
| nezuko | `ema-weights` | Regularization | EMA of model weights (decay 0.999), eval/test on EMA model. |
| tanjiro | `unified-pos` | Positional encoding | `unified_pos = True`, `ref = 8` — learned 3D grid embedding. |
| thorfinn | `swiglu-mlp` | Activation | Replace TransolverBlock MLP with SwiGLU. |

## Potential next research directions

See `research/RESEARCH_IDEAS_2026-05-12_round2.md` (commit `c4837e2`) for the
full ranked list of 11 hypotheses from the round-2 literature scan. Top
candidates for next assignment once round 1 results land:

- **H1 — Ada-Temp**: per-point adaptive slice temperature in
  PhysicsAttention (Transolver++, Feb 2025). 3-line change, no new HP,
  attacks documented head-collapse failure mode.
- **H4 — Per-channel pressure weight**: scale `p` channel by 3× in
  training loss to directly close the train-objective vs. eval-metric
  gap on `mae_surf_p`.
- **H5 — Gradient clipping**: `clip_grad_norm_(..., max_norm=1.0)` —
  trivial, often stabilizes high-dynamic-range targets.
- **H6 — Kendall uncertainty weighting** of vol/surf loss — learnable
  `log_sigma`s replace the manually tuned `surf_weight=10`.
- **H2 — Asymmetric Q/K slice projection** in PhysicsAttention
  (LinearNO, Nov 2024). Higher complexity / higher upside.

Remaining round-2 file entries cover: H3 remove `in_project_fx`, H7
Fourier coordinate augmentation, H8 stochastic depth, H9 Gumbel-Softmax
slicing, H10 FiLM global-condition injection, H11 log1p pressure target
reparameterization for high-Re.

Other useful directions if round 1 plateaus:

- Output head specialization (separate p / Ux / Uy heads, channel-balanced
  loss in physical units).
- Domain-aware sampler reweighting that better matches the val split
  weighting (3/4 tandem in val avg vs. 1/3 tandem in current sampler).
- Larger Transolver with longer training: confirm capacity gains hold
  beyond round 1's truncated epoch budget.
- Optimizer alternatives: Lion, Adafactor, Sophia at calibrated lr.
- Mesh-aware augmentations (geometry-preserving permutations of node
  ordering).
