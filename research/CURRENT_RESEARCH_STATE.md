# SENPAI Research State

- **Last updated**: 2026-05-12 21:00 UTC (round 2 — wave 1 mostly resolved, wave 2 spawning)
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

Round 2 wave 1 has now mostly resolved. The first post-L1 architectural winner
remains **stochastic depth** (PR #1552, frieren) — the new canonical baseline.
Three wave-1 PRs closed cleanly with diagnostic value but no metric gain:

- **Kendall uncertainty** (askeladd #1547) — clean negative result. Learned
  effective surf_weight converged to 1.52, ~7× lower than the hand-tuned
  optimum at 10. The Kendall MLE objective is fundamentally misaligned with
  the physical eval metric. Rules out the entire MLE-style balance-learning
  family (Kendall, GradNorm, DWA) unless constrained to optimize the eval
  surrogate.
- **Asymmetric Q/K** (tanjiro #1545) — compute-bound. Mechanism active
  (slice cos-sim = 0.097), but +40% per-step wall-clock cost truncated
  training to 10 epochs. Structural lesson: **future architectural changes
  must be parameter-additions, not compute-additions**.
- **Ada-Temp v1 + v2** (alphonse #1514) — both per-head and shared-heads
  variants exhausted. Shared-heads narrowed re_rand regression but broke
  camber_cruise (+8.12). Slice-collapse will be attacked instead via
  Gumbel-Softmax (nezuko #1553, WIP).

One wave-1 PR was **sent back with a re-tune plan**:

- **Tied projection / remove in_project_fx** (thorfinn #1555) — only +1.57%
  worse on val_avg, with three of four splits actually improving (single_in_dist
  was the only drag, +8.6%). Efficiency gains real: -12.5% params, -5.9% VRAM,
  identical wall time. **Re-tune spec: bump `n_hidden=128 → 144`** to reinvest
  freed capacity, expected to recover single_in_dist while preserving OOD gains.

The recurring round-1 finding holds firmly: **surf_weight=10 is at or above the
optimum**. PR #1403 (surf_weight=30) regressed by +5.1%, PR #1530 (effective
surf×P_WEIGHT=30) by +1.22%, and Kendall's learned weight (1.52) regressed by
+5.28% — three independent confirmations bracketing the optimum near 10.

## Round 2 wave 1 — final state

| Student | PR | Slug | Verdict | Δ vs baseline |
|---------|----|----|---------|---------------|
| frieren | #1552 | `stoch-depth-0.1` | **MERGED** (new baseline) | -2.58% |
| thorfinn | #1555 | `remove-in-project-fx` | **SENT BACK** for n_hidden=144 | +1.57% |
| alphonse | #1514 | `ada-temp` v2 | **CLOSED** | +3.4% (vs L1-only) / +6.1% (vs current) |
| askeladd | #1547 | `kendall-uncertainty` | **CLOSED** | +5.28% |
| tanjiro | #1545 | `asymmetric-qk` | **CLOSED** | +18.9% |
| nezuko | #1553 | `gumbel-slice` | WIP | — |
| fern | #1549 | `film-global-cond` | WIP | — |
| edward | #1548 | `fourier-coords-L4` | WIP | — |

After the round-2 wave-1 closures, **four students are idle**: frieren,
alphonse, askeladd, tanjiro. The researcher-agent has been spawned to refresh
the hypothesis pool with novel high-EV ideas that complement the in-flight
WIP PRs (Gumbel-Softmax, FiLM, Fourier coords, tied-projection retune).

## Wave 2 candidates (current pool, will be augmented by researcher-agent)

**Stoch-depth follow-ups (compound on the new baseline):**
- **Sweep `drop_rate` ∈ {0.05, 0.15, 0.20}** — val_re_rand +1.77% suggests slight
  over-regularization at 0.10; 0.05 might be Pareto-better. Conversely, 0.15-0.20
  may bite harder on val_geom_camber_rc which barely moved.
- **Stack dropout in PhysicsAttention/MLP at 0.05** — standard ViT recipe;
  often compounds with stoch-depth.

**Remaining round-2 ideas not yet picked up:**
- **H11**: log1p target reparameterization — medium-high risk, unclear direction.
  Worth trying now that we have margin against the stronger baseline.
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

**Constraints reaffirmed from wave-1 closures:**
- No more learnable loss-balance objectives (Kendall ruled out the family).
- No more architectural changes that add >10% per-step compute (asymmetric Q/K).
- surf_weight=10 is empirically at-or-near optimum (3 independent confirmations).

## Recent closures and merges (2026-05-12 19:48-21:00 UTC)

- **#1552 stoch-depth-0.1 (frieren)** — **MERGED** as new baseline.
  val_avg -2.58%, test 4-split 87.995 (first finite ref).
- **#1555 remove-in-project-fx (thorfinn)** — **SENT BACK** for n_hidden=144
  re-tune. Net +1.57% but efficiency gains real and direction worth iterating.
- **#1514 ada-temp v2 (alphonse)** — **CLOSED** at +3.4% vs L1-only base.
  Both per-head and shared-heads Δτ variants exhausted on this dataset/budget.
- **#1547 kendall-uncertainty (askeladd)** — **CLOSED** at +5.28%. Learned
  effective surf_weight=1.52 confirms MLE-balance objective mismatch.
- **#1545 asymmetric-qk (tanjiro)** — **CLOSED** at +18.9%. Compute-bound
  (40% per-step overhead, truncated to 10 epochs).
- **#1530 channel-weight-p3 (tanjiro)** — closed, +1.22% worse than L1.
- **#1529 grad-clip-1.0 (askeladd)** — closed, +5.4% worse than L1.
- **#1407 wider-deeper, #1411 slice_num=128, #1417 lr-warmup, #1420 EMA,
  #1425 SwiGLU** — all closed without running. Branched off pre-L1 MSE
  base; hypotheses remain valid for later revival.
