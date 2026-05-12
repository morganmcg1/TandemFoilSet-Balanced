# SENPAI Research State

- **Last updated**: 2026-05-12 20:05 UTC (round 2 — wave 1 in flight)
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging
  ablation. Each individual target training execution is capped at
  `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #1397 merged)

- `val_avg/mae_surf_p` = **100.957** (L1 loss replaces MSE; best @ ep 13/14)
- `test_avg/mae_surf_p` (3-split, ex-cruise) = **100.831**
- `test_avg/mae_surf_p` (4-split, NaN-safe) ≈ 92-95 (preliminary refs from
  closed PRs #1530/#1529 that applied the NaN-safe pre-filter — final
  finite reference will land when the first round-2 winner merges with
  the same fix on a clean L1 base)

## Current research focus

Round 1 delivered exactly one merge: **L1 loss** (PR #1397). All other
round-1 axes (surf_weight=30, unified_pos, wider/deeper, slice_num=128,
lr-warmup, EMA, SwiGLU) either regressed or were closed without a clean
L1-based comparison. Round-2 wave 1 attacks orthogonal axes — architecture,
loss balance, regularization, conditioning — with a higher concentration
of architecture changes that the round-1 axes did not touch.

The recurring round-1 finding is that **surf_weight=10 is at or above the
optimum** — both #1403 (surf_weight=30) and #1530 (effective combined
weight = 30 via per-channel reweighting) regressed in the same regime.
Round 2 includes a Kendall uncertainty-weighting arm (askeladd #1547) that
makes the vol/surf balance *learnable* instead of hand-tuned.

A pre-existing `data/scoring.py` data bug (NaN propagation from
`test_geom_camber_cruise/000020.pt`) is now solved by a trainer-side
`evaluate_split` pre-filter that's bundled into every round-2 assignment.
Once the first round-2 winner merges, the canonical 4-split test mean
becomes a clean paper-facing reference.

## Round 2 wave 1 — currently in flight (8 PRs)

| Student | PR | Slug | Round-2 idea | Axis |
|---------|----|----|--------------|------|
| alphonse | #1514 | `ada-temp` | H1 | Architecture — per-point adaptive slice temperature (Transolver++) |
| tanjiro | #1545 | `asymmetric-qk` | H2 | Architecture — independent V and K slice projections (LinearNO) |
| askeladd | #1547 | `kendall-uncertainty` | H6 | Loss balance — learnable per-task sigmas (Kendall et al.) |
| edward | #1548 | `fourier-coords-L4` | H7 | Input encoding — Fourier coord features (L=4) |
| fern | #1549 | `film-global-cond` | H10 | Conditioning — FiLM modulation by global flow params |
| frieren | #1552 | `stoch-depth-0.1` | H8 | Regularization — stochastic depth, drop_rate=0.1 |
| nezuko | #1553 | `gumbel-slice` | H9 | Architecture — Gumbel-Softmax slice weights (tau=1.0) |
| thorfinn | #1555 | `remove-in-project-fx` | H3 | Efficiency — tied x/fx projection (Transolver++) |

All round-2 PRs include the NaN-safe pre-filter as a standard requirement.

## Potential next research directions (wave 2)

To assign once wave-1 results land. Round-2 ideas not yet picked up:

- **H4 (revisit)**: per-channel loss weight for `p` at a milder
  `P_WEIGHT ∈ {1.5, 2.0}` — combined with `surf_weight=5` so effective
  surface-p weight stays ≤ 15 (the failed #1530 effective weight was 30).
- **H5 (revisit)**: gradient clipping at `max_norm ∈ {10, 25, 50}` — the
  #1529 result (`max_norm=1.0` clipped 100% of steps) confirms 1.0 is
  too aggressive. Or AGC (per-parameter adaptive clipping).
- **H11**: log1p target reparameterization — medium-high risk, unclear
  direction; defer until other levers are exhausted.

Untried compound directions for later rounds:

- **Output head specialization** — separate p / Ux / Uy heads with
  channel-balanced loss in physical units.
- **Domain-aware sampler reweighting** to match val split aggregation
  (3/4 tandem in val_avg vs. 1/3 tandem in current sampler).
- **Cosine T_max alignment** — schedule `T_max=15` to match the 30-min
  wall-clock cap rather than the configured `MAX_EPOCHS=50` that the
  run never reaches.
- **Optimizer alternatives** — Lion, Adafactor, Sophia at calibrated lr.
- **Larger Transolver with longer training** — confirm capacity gains
  hold when the wall-clock cap is lifted (would need a re-config of
  `SENPAI_TIMEOUT_MINUTES` per the human team).
- **Round-1 axes on L1 base** — wider/deeper, slice_num=128, lr-warmup,
  EMA, SwiGLU all closed without an L1 comparison. Revisit selectively
  if architecture/loss-formulation levers stall.

## Recent closures (PR-level decisions, 2026-05-12 19:48-20:00 UTC)

- **#1530 channel-weight-p3 (tanjiro)** — closed, +1.22% worse than L1.
  Diagnosis: effective combined surf-p weight (30) matched the failed
  #1403 regime. NaN-safe pre-filter is a clean deliverable preserved
  via inclusion in every round-2 assignment.
- **#1529 grad-clip-1.0 (askeladd)** — closed, +5.4% worse than L1.
  Diagnosis: `max_norm=1.0` clipped 100% of steps (natural norms 10-245);
  effective LR was 1-5% of configured. NaN-safe fix also preserved.
- **#1407 wider-deeper, #1411 slice_num=128, #1417 lr-warmup, #1420 EMA,
  #1425 SwiGLU** — all closed without running. Branched off pre-L1
  MSE base, would have produced contaminated comparisons. The
  hypotheses themselves remain valid for later revival.
