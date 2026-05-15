# SENPAI Research State

- 2026-05-15 16:30 — updated after reviewing #3148 and #3149 (no winners) and assigning round-2 work to frieren and nezuko.
- No directives from the human researcher team yet on this launch.

## Current research focus

Round 1 on advisor branch `icml-appendix-willow-pai2i-24h-r1`. Two PRs reviewed
and closed (no winners). Implicit baseline: **`val_avg/mae_surf_p` ≈ 130 ± 3**
(mean of two baseline-equivalent arms from #3148 and #3149). Six round-1
hypotheses still in flight.

Key constraints identified so far:
- **30-min wall-clock cap is the binding budget** — all baseline-width runs hit it.
  Best val at epoch 13-14 of 50, meaning the model is still improving at cutoff.
- **Run-to-run variance ~3-4 mae_surf_p units** — improvements <3 units need
  multiple seeds to confirm.
- **Wider models underperform in this budget** — #3148 showed width hurts because
  wider models don't converge in 30 min; this is a wall-clock confound, not a
  capacity signal.
- **Per-channel loss weighting doesn't help** — #3149 showed surf-p upweighting
  degrades volume metrics due to shared-backbone capacity steal; the bottleneck
  is architectural (shared readout), not loss-level.
- **`test_avg/mae_surf_p` is None for all runs** due to Inf in cruise test GT
  (known multi-launch issue, #3292/#1569/#1567). Rank by `val_avg/mae_surf_p`.

## In-flight hypotheses (round 1)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3138 | alphonse  | PhysicsAttention slice count (`slice_num` ∈ {64,128,256}) | wip |
| #3142 | askeladd  | Surface-loss weight (`surf_weight` ∈ {10,30,80}) | wip |
| #3143 | edward    | Robust loss (Charbonnier ε=1e-3 vs MSE) | wip |
| #3145 | fern      | Depth (`n_layers` ∈ {5,8,10}) | wip |
| #3150 | tanjiro   | Warmup + cosine LR (`lr` ∈ {5e-4, 1e-3, 1.5e-3} with 3-epoch warmup) | wip |
| #3151 | thorfinn  | EMA model weights (`ema_decay` ∈ {0, 0.999, 0.9999}) | wip |
| #3330 | frieren   | bf16 AMP mixed precision (more epochs per 30-min budget) | wip (new) |
| #3331 | nezuko    | Separate per-channel output heads (break shared-readout bottleneck) | wip (new) |

## Closed this session

| PR | Student | Result |
|----|---------|--------|
| #3148 | frieren | Wider Transolver — wall-clock confound; wider models don't converge in 30 min |
| #3149 | nezuko  | Per-channel surf-p loss weighting — shared-backbone capacity steal; no improvement |

## Key research insight from round 1

The **wall-clock budget is the primary bottleneck**, not model capacity. The
baseline model is still improving at epoch 13-14 when the 30-min cap fires.
This points to two high-value orthogonal levers:
1. **Throughput** — anything that makes each step cheaper (AMP, bf16) → more
   steps in budget → better convergence. Frieren (#3330) is testing this.
2. **Architecture efficiency** — more expressive readout per parameter → squeeze
   more out of each step. Nezuko (#3331) is testing separate per-channel heads
   which adds negligible compute cost but breaks the shared-readout bottleneck.

## Potential next-round directions (post round-1 review)

- **Compound round-1 winners** (all levers are orthogonal by design).
- **Activation functions**: GELU → SwiGLU / GeGLU in the MLP blocks.
- **Fourier position encoding**: replace the raw (x, y) position with Fourier
  features for the `preprocess` MLP input.
- **Gradient clipping**: `max_norm=1.0` to stabilize early training (free).
- **RMSNorm**: replace LayerNorm in blocks — slightly faster, often comparable.
- **Data augmentation**: chord rotation + NACA-symmetric flip for raceCar domains.
- **Curriculum ordering**: train by ascending mesh size or Re to stabilize early.
- **Larger batch with AMP**: if bf16 frees VRAM, test batch_size 8 or 16.
- **Test-time augmentation**: average over symmetry-augmented forward passes.
- **Mesh-aware positional encoding**: signed-distance field, surface normals.
