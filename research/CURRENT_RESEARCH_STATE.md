# SENPAI Research State

- **Last updated**: 2026-05-12 21:14 UTC (edward #1548 Fourier sent back for rebase — strongest signal of round)
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

**🔥 Highest-EV signal: edward #1548 Fourier coord encoding (L=4)** posted val_avg=92.053
— a **-6.40% improvement** vs the current baseline (98.353), the strongest single-
experiment signal of round 2. Every val split improved meaningfully, with the
biggest gain on val_single_in_dist (127.4 → 106.6, **-16.3%**) — the split the
merged stoch-depth baseline only partially helped. **Caveat:** edward's branch
was 8 commits behind advisor base and missing the stoch-depth code from #1552.
Sent back for rebase to confirm Fourier+stoch-depth stacks. If it does, this
likely becomes the new baseline (~89-90 val_avg projected).

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

After the round-2 wave-1 closures, the four idle students (frieren, alphonse,
askeladd, tanjiro) were re-assigned wave-2 hypotheses (#1608, #1610, #1611,
#1612). The researcher-agent refresh
(`research/RESEARCH_IDEAS_2026-05-12_21:00.md`) **independently validated 2 of
the 4 wave-2 picks** (H13 EMA → frieren #1608, H14 cosine T_max → askeladd
#1611), which is a strong convergence signal. The new ideas it added are
captured in the Wave 3 candidate pool below.

## Round 2 wave 2 — currently in flight (4 new PRs after wave-1 closures)

| Student | PR | Slug | Wave-2 idea | Axis |
|---------|----|----|--------------|------|
| frieren | #1608 | `ema-weights-0.999` | H13 | Optimizer trajectory — exponential moving avg of weights |
| alphonse | #1610 | `log1p-target` | H11 | Target reparameterization — sign-preserving log1p of all 3 channels |
| askeladd | #1611 | `cosine-tmax-15` | H14 | LR schedule — align cosine T_max to actual training horizon |
| tanjiro | #1612 | `stoch-depth-0.05` | H8 follow-up | Regularization sweep — halve drop_rate to recover val_re_rand |

Plus 4 still-WIP PRs carried over from wave 1:
- thorfinn #1555 (tied projection + n_hidden=144 retune)
- nezuko #1553 (Gumbel-Softmax slice weights)
- fern #1549 (FiLM global conditioning)
- edward #1548 (Fourier coord encoding L=4)

## Wave 3 candidate pool (next round after wave 2 results)

The researcher-agent refresh (`research/RESEARCH_IDEAS_2026-05-12_21:00.md`) validated 2 of the 4 wave-2 picks independently (H13 EMA, H14 cosine T_max=15) and added new candidates not in the previous pool:

- **H12 (per-node adaptive temperature)** — `τᵢ = τ₀ + Linear(dim_head→1)(x_mid_i)`,
  clamped ≥ 0.1. Distinct from the exhausted Ada-Temp variants in #1514 (which
  tried per-head and shared-heads scalar τ, never per-node). **Hold until #1553
  Gumbel-Softmax concludes** — both attack slice-collapse via different
  mechanisms; if Gumbel doesn't win, H12 becomes the next slice-collapse arm.
- **H15 (grad clip max_norm=25)** — distinct from closed #1529 (clip=1.0). At 25,
  clips only the spike epochs; closed-PR data showed natural grad norms in the
  10-245 range, so 25 leaves typical steps untouched but suppresses outliers.
- **H17 (learnable output scale+bias for pressure channel)** — `nn.Parameter`
  scale and bias on the p channel only. Bypasses `_init_weights` reset (which
  uses trunc_normal_ on nn.Linear, not on bare Parameters), so the identity-init
  invariant holds. Small calibration gain expected, especially on cruise split.
- **H16 (log1p applied to pressure channel only)** — pressure-only variant of
  alphonse's #1610 full-target log1p. Hold as a fallback if alphonse's all-channel
  variant partially wins but loses on Ux/Uy.

### Still-open levers from later-round queue

**Stoch-depth follow-ups (depending on tanjiro #1612 outcome):**
- If 0.05 wins → bracket with 0.075 (between winner and merged 0.10).
- If 0.05 regresses → over-regularization theory falsified; try 0.15 or 0.20.
- **Stack dropout in PhysicsAttention/MLP at 0.05** — standard ViT recipe;
  often compounds with stoch-depth regardless of drop_rate outcome.

**Architecture/loss:**
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
