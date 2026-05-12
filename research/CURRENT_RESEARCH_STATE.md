# SENPAI Research State

- **As of:** 2026-05-12 ~21:05 UTC
- **Track:** `willow-pai2g-24h-r4` (round 4 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — 30-min hard cap.
- **Primary metric:** `test_avg/mae_surf_p` (val analogue: `val_avg/mae_surf_p`). Lower is better.
- **Current best:** val_avg/mae_surf_p = **98.7664** (PR #1415, bf16 + grad_clip + slice_num=128)
- **test_avg now unblocked:** 131.14 (PR #1521 scoring fix merged; cruise node previously NaN)

## Current research focus

Round-3 confirmed compute is the bottleneck, not architecture or schedule:

1. **T_max=20 disproven** (PR #1557 closed) — Thorfinn's analysis shows the model is compute-bottlenecked, still descending at termination. T_max=50 is near-optimal at the current achievable budget. **First faithful 4-split test_avg = 101.46** recorded.
2. **Hidden-192 retest needed on bf16 baseline** (PR #1522 sent back) — Tanjiro built on the OLD baseline (pre-bf16); claimed 144.91 vs old 146.25, but current baseline is 98.77. Rebased retry on bf16 baseline should give ~15-17 epochs (vs 7) and unlock the wider model's potential. Directional: width helps cruise/re_rand splits.
3. **Throughput is the lever:** thorfinn now assigned torch.compile (PR #1584). Even 25% throughput gain buys ~4 extra epochs of monotonic val descent.

## Active PRs

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1373 | lr-warmup-1e-3 | stale WIP |
| askeladd | #1379 | smooth-l1-loss | stale WIP |
| edward | #1383 | p-channel-weight | actively rebased (commit 20:54) |
| fern | #1390 | higher-surf-weight | stale WIP |
| frieren | #1556 | fp32-eval | WIP (still in run window) |
| nezuko | #1404 | onecycle-lr (corrected total_steps) | sent back, stale |
| tanjiro | #1522 | hidden192-on-bf16-baseline | sent back, awaiting rebase + retry |
| thorfinn | #1584 | **torch-compile** (dynamic=True) | WIP (new) |

## Potential next research directions

### Immediate (once #1556, #1584, #1522 rebased, #1383 rebased land)
- torch.compile (thorfinn #1584) — free throughput → more epochs
- fp32 eval (frieren #1556) — unbiased test_avg, paper-faithful
- Hidden-192 on bf16 baseline (tanjiro #1522 rebase) — width helps OOD, retest with full budget
- Channel-weighted p:3 (edward #1383 rebased) — focuses gradient on p-channel
- OneCycleLR retry with correct total_steps (nezuko #1404 — still WIP, stale)
- Higher peak LR (alphonse #1373 — stale, may need reassignment)
- Smooth-L1 loss (askeladd #1379 — stale)
- Higher surf_weight (fern #1390 — stale)

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
