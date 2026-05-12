# SENPAI Research State

- **Date**: 2026-05-12 18:00 UTC
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r3` (base `icml-appendix-charlie`)
- **Research tag**: `charlie-pai2g-24h-r3`
- **Students (8, all idle at boot)**: charliepai2g24h3-{alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn}
- **Per-run budget**: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50` (caps)
- **Logging**: local JSONL only (no W&B in this arm)

## Latest human direction

None received yet on this advisor branch.

## Current research focus

Fresh research track on TandemFoilSet. Primary metric is
`val_avg/mae_surf_p` — equal-weight mean surface pressure MAE over four
validation tracks. Round 1 is **broad coverage of orthogonal levers** to
locate where the headroom is on a 30-minute training budget.

Themes covered in round 1:

1. **Loss formulation** for heavy-tailed targets (Huber).
2. **Output structure / per-channel weighting** that prioritises surface
   pressure over velocity.
3. **Capacity scaling** (n_hidden 128→256, mlp_ratio 2→4, slice_num 64→128).
4. **Optimization hygiene** (gradient clipping + tuned weight decay).
5. **Physics conditioning** via FiLM on log(Re) for cross-regime generalization.
6. **Data augmentation** on AoA / NACA parameters for camber OOD splits.

These eight ideas are mutually orthogonal — the ranking of the round will tell
us whether the bottleneck is capacity, loss, conditioning, or augmentation.

## Round 1 assignments (planned)

| Student | Slug | Lever |
|---|---|---|
| alphonse | `huber-pressure-loss` | Loss for heavy tails |
| askeladd | `decoupled-channel-heads` | Per-channel surface weighting |
| edward | `scale-model-256` | Capacity (n_hidden 128→256, n_head 4→8) |
| fern | `grad-clip-adamw-tuned` | Optimization (clip + wd) |
| frieren | `mlp-ratio-4-wider-ffn` | FFN capacity (mlp_ratio 2→4) |
| nezuko | `more-slices-128` | Attention partitioning (slice_num 64→128) |
| tanjiro | `re-film-conditioning` | Re regime conditioning (FiLM) |
| thorfinn | `geometry-aoa-augmentation` | Camber/AoA jitter for OOD |

## Potential next research directions (round 2+)

- **Compose winners** sequentially: best architectural change × best loss × best
  conditioning.
- **Loss escalation**: Laplacian / per-channel relative MAE / log-cosh loss.
- **Per-channel decoders** with channel-specific depth (more layers for p).
- **Geometric augmentation in shape descriptor space** (dims 4–11), not just
  AoA / NACA scalars.
- **Larger slice counts paired with larger n_hidden** to expose the joint
  scaling frontier.
- **Mesh-aware positional encoding** (signed distance, normalized arc length
  re-encoded as Fourier features).
- **EMA model weights** for evaluation stability across the short training
  window.
- **Two-stage / curriculum training**: first fit volume MAE, then unfreeze a
  surface-only head and finetune with high `surf_weight_p`.
- **Mixup / sample-pair training** as a geometry regularizer.
- **Auxiliary heads** for `mae_surf_Ux`, `mae_surf_Uy`, with a primary
  surface-pressure loss term.
