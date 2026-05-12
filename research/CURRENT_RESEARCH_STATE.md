# SENPAI Research State

- **Last updated**: 2026-05-12 20:55 UTC (round 2 — wave 1 in flight, first winner merged)
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging
  ablation. Each individual target training execution is capped at
  `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #1552 merged)

- `val_avg/mae_surf_p` = **98.353** (L1 loss + stochastic depth; best @ ep 15)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **87.995** (first finite 4-split ref)
- Per-split val: single_in_dist=119.16 / camber_rc=111.09 / camber_cruise=73.32 / re_rand=89.84
- Per-split test: single_in_dist=104.95 / camber_rc=101.88 / camber_cruise=62.24 / re_rand=82.90
- Δ vs L1-only baseline: **-2.58%** on val_avg, **first finite 4-split** test mean

## Current research focus

Round 2 wave 1 produced the first post-L1 architectural winner — **stochastic
depth** (PR #1552, frieren) — at -2.58% on val_avg. The same PR's NaN-safe
pre-filter (originally discovered independently by tanjiro #1530 and askeladd
#1529) now produces a finite 4-split test mean of 87.995 as the new paper-facing
reference.

The recurring round-1 finding remains: **surf_weight=10 is at or above the
optimum** (PR #1403 surf_weight=30 and PR #1530 effective surf_weight×P_WEIGHT=30
both regressed). Round 2 includes a Kendall uncertainty-weighting arm
(askeladd #1547) that makes the vol/surf balance *learnable*.

The merged stoch-depth baseline raises the bar for the remaining wave-1 PRs.
They are compared against the new baseline (val_avg = 98.353), not the previous
L1-only baseline (val_avg = 100.957). All wave-1 PRs were authored before the
merge, so they did not have stoch-depth in their `train.py`; merge conflicts on
`TransolverBlock` / `Transolver` constructors are possible, especially for the
architecture-touching arms (Ada-Temp v2, Asymmetric Q/K, Gumbel-Softmax,
in_project_fx removal). Will rebase as needed.

## Round 2 wave 1 — currently in flight (7 PRs after #1552 merge)

| Student | PR | Slug | Round-2 idea | Axis |
|---------|----|----|--------------|------|
| alphonse | #1514 | `ada-temp` (v2, sent back) | H1 v2 | Architecture — shared-across-heads Δτ |
| tanjiro | #1545 | `asymmetric-qk` | H2 | Architecture — independent V and K slice projections (LinearNO) |
| askeladd | #1547 | `kendall-uncertainty` | H6 | Loss balance — learnable per-task sigmas (Kendall et al.) |
| edward | #1548 | `fourier-coords-L4` | H7 | Input encoding — Fourier coord features (L=4) |
| fern | #1549 | `film-global-cond` | H10 | Conditioning — FiLM modulation by global flow params |
| nezuko | #1553 | `gumbel-slice` | H9 | Architecture — Gumbel-Softmax slice weights (tau=1.0) |
| thorfinn | #1555 | `remove-in-project-fx` | H3 | Efficiency — tied x/fx projection (Transolver++) |

All wave-1 PRs include the NaN-safe pre-filter as a standard requirement.

## Potential next research directions

### Wave 2 candidates (assign as wave-1 PRs complete)

**Stoch-depth follow-ups (compound on the new baseline):**
- **Sweep `drop_rate` ∈ {0.05, 0.15, 0.20}** — val_re_rand +1.77% suggests slight
  over-regularization at 0.10; 0.05 might be Pareto-better. Conversely, 0.15-0.20
  may bite harder on val_geom_camber_rc which barely moved.
- **Stack dropout in PhysicsAttention/MLP at 0.05** — standard ViT recipe;
  often compounds with stoch-depth.

**Remaining round-2 ideas not yet picked up:**
- **H11**: log1p target reparameterization — medium-high risk, unclear direction.
  Worth trying now that we have margin against the stronger baseline.
- **H4 (revisit)**: per-channel loss weight for `p` at milder `P_WEIGHT ∈ {1.5, 2.0}`,
  combined with reduced `surf_weight=5` so effective surf-p weight stays ≤ 15.
- **H5 (revisit)**: gradient clipping at `max_norm ∈ {10, 25, 50}` — only clips
  the spike epochs (#1529 confirmed natural norms 10-245); or AGC.

**Untried levers from later-round queue:**
- **EMA of model weights (decay 0.999)** — round-1 axis (closed #1420 without
  L1 run); often a free 1-2% gain; complements stoch-depth (both reduce
  variance across late epochs). High-EV, simple to implement.
- **Cosine T_max alignment** — schedule `T_max=15` to match the 30-min cap
  rather than the configured `MAX_EPOCHS=50`. The current run sits at ~80% of
  initial LR at the cap; aligning the schedule reduces wasted LR runway.
- **Output head specialization** — separate p / Ux / Uy heads with
  channel-balanced loss in physical units.
- **Domain-aware sampler reweighting** to match val split aggregation
  (3/4 tandem in val_avg vs. 1/3 tandem in current sampler).
- **Optimizer alternatives** — Lion, Adafactor, Sophia at calibrated lr.
- **Round-1 axes on L1+stoch-depth base** — wider/deeper, slice_num=128,
  lr-warmup, SwiGLU all closed without an L1 comparison. Revisit selectively.

## Recent closures and merges (2026-05-12 19:48-20:55 UTC)

- **#1552 stoch-depth-0.1 (frieren)** — **MERGED** as new baseline.
  val_avg -2.58%, test 4-split 87.995 (first finite ref).
- **#1514 ada-temp (alphonse)** — sent back as v2 (shared-across-heads Δτ +
  NaN-safe fix). v1 was +0.81% val_avg / -0.007 test 3-split. The student's
  per-split analysis (single_in_dist -9.3 / re_rand +8.3) suggested extra
  per-head capacity hurts cross-regime transfer; v2 tests that diagnosis
  with `Linear(dim, 1)` shared-across-heads Δτ.
- **#1530 channel-weight-p3 (tanjiro)** — closed, +1.22% worse than L1.
  NaN-safe pre-filter from this PR is preserved in every round-2 assignment.
- **#1529 grad-clip-1.0 (askeladd)** — closed, +5.4% worse than L1.
  `max_norm=1.0` clipped 100% of steps; effective LR 1-5% of configured.
  NaN-safe fix also preserved.
- **#1407 wider-deeper, #1411 slice_num=128, #1417 lr-warmup, #1420 EMA,
  #1425 SwiGLU** — all closed without running. Branched off pre-L1 MSE
  base; hypotheses remain valid for later revival.
