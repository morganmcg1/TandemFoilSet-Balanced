# SENPAI Research State

- **As of:** 2026-05-12 ~20:30 UTC
- **Track:** `willow-pai2g-24h-r4` (round 4 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — 30-min hard cap.
- **Primary metric:** `test_avg/mae_surf_p` (val analogue: `val_avg/mae_surf_p`). Lower is better.
- **Current best:** val_avg/mae_surf_p = **98.7664** (PR #1415, bf16 + grad_clip + slice_num=128)
- **test_avg now unblocked:** 131.14 (PR #1521 scoring fix merged; cruise node previously NaN)

## Current research focus

Round-2 cleanup and follow-up experiments on the bf16 baseline:

1. **Scoring fix landed** (PR #1521) — `test_avg/mae_surf_p = 131.14` first valid test metric. Cruise node bf16-inf still zeroed by `nan_to_num` (biased low) — fp32 eval follow-up assigned.
2. **bf16 is the big winner** (PR #1415) — 32.5% val improvement, 42% throughput gain, 40% VRAM reduction. Model still descending at 30-min cutoff; schedule-epoch mismatch is the suspected bottleneck.
3. **Two immediate follow-ups in flight:**
   - frieren (PR #1556): fp32 eval — removes bf16 autocast from `evaluate_split` only; recovers faithful `test_avg/mae_surf_p` for cruise node.
   - thorfinn (PR #1557): T_max=20 — resizes CosineAnnealingLR to the achievable epoch budget so LR cools fully to ~0.

## Active PRs

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1373 | lr-warmup-1e-3 | stale WIP |
| askeladd | #1379 | smooth-l1-loss | stale WIP |
| edward | #1383 | p-channel-weight | stale WIP |
| fern | #1390 | higher-surf-weight | stale WIP |
| frieren | #1556 | **fp32-eval** | WIP (new) |
| nezuko | #1404 | onecycle-lr (corrected total_steps) | Sent back / WIP |
| tanjiro | #1522 | hidden192-on-slice128 | WIP |
| thorfinn | #1557 | **tmax-retune-20** | WIP (new) |

## Potential next research directions

### Immediate (once #1556 and #1557 land)
- Stack winners: T_max=20 + fp32 eval + higher peak LR (1e-3) now that schedule fully cools
- OneCycleLR retry with correct `total_steps = 18 * steps_per_epoch` (nezuko PR #1404 rebase)
- Hidden-192 retry (tanjiro PR #1522) — confirm width helps on top of bf16 baseline
- Stacking round-1 ideas: smooth-L1 loss, p-channel weight — these need retesting on bf16 baseline

### Architecture
- SwiGLU MLP in place of standard GELU FF layers
- Rotary positional embeddings for node coordinates
- Depth-vs-width tradeoffs at fixed FLOP budget
- Attention variants: cross-attention for inter-foil features

### Loss / training signal
- Relative-MAE / log-domain pressure regression for dynamic range across Re (std varies up to 10×)
- Per-domain or per-Re reweighting of loss
- Auxiliary loss on Ux/Uy to regularise pressure prediction

### Data
- Stratify minibatches by Re or domain to reduce gradient variance
- x-mirror + AoA sign flip augmentation for cruise foils
- Explicit signed-distance-to-surface as positional feature

### Inference
- Test-time augmentation: average predictions across mirrored geometries
- Ensembling over last-N checkpoints
