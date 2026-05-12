# SENPAI Research State

- **Last updated**: 2026-05-12 21:30 UTC (wave 2 partly resolved — #1611 cosine merged as new baseline, wave 3 launched)
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging
  ablation. Each individual target training execution is capped at
  `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #1611 merged — cosine T_max=15 win, -4.21%)

- `val_avg/mae_surf_p` = **94.217** (cosine `T_max=15` + L1 + stoch-depth; best @ ep 15)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **84.859**
- Per-split val: single_in_dist=114.200 / camber_rc=102.157 / camber_cruise=73.321 / re_rand=87.188
- Δ vs PR #1552 baseline (98.353): **-4.21%** on val_avg, **-3.57%** on 4-split test
- Largest single-arm gain of wave 2 so far. Val MAE descended monotonically
  every epoch — model still improving at the 30-min cap, hinting at further
  headroom with more time or warm-started follow-ups.

## Current research focus

**Active leverage from the merged cosine T_max=15 baseline.** Three threads
are converging:

1. **Stack the cosine cooldown with EMA** — frieren #1608 was sent back for
   rebase. EMA averaged on top of the cosine cooldown phase should keep most
   of its 95.761 raw signal but now starts from a 94.2 floor (mechanistically
   orthogonal: EMA averages over the trajectory; cosine reshapes the trajectory).
2. **Outstanding extraordinary signal: fern #1549 FiLM = 81.291** — posted on
   a branch missing both stoch-depth and cosine T_max=15. **-17.3% vs current**
   baseline if it holds after rebase. This is the single highest-EV lever in
   flight; sent back for rebase to confirm.
3. **Outstanding strong signal: edward #1548 Fourier coords L=4 = 92.053** —
   posted on a branch missing stoch-depth. -6.40% vs old baseline. Sent back
   to verify it stacks with stoch-depth + cosine.

The recurring round-1 finding holds firmly: **surf_weight=10 is at or above
the optimum**. Three independent confirmations bracket the optimum near 10.

## Round 2 wave 2 — resolution

| Student | PR | Slug | Verdict | Δ vs `#1552` baseline |
|---------|----|----|---------|---------------|
| askeladd | #1611 | `cosine-tmax-15` | **MERGED** (new baseline 94.217) | -4.21% |
| frieren | #1608 | `ema-weights-0.999` | **SENT BACK** for rebase | -2.64% (vs old base) |
| alphonse | #1610 | `log1p-target` | **CLOSED** | +1.18% |
| tanjiro | #1612 | `stoch-depth-0.05` | WIP | — |

## Wave-1 carryover (still WIP / in rebase)

- **thorfinn #1555** — Tied projection + `n_hidden=144` retune (sent-back).
- **nezuko #1553** — Gumbel-Softmax slice weights (WIP).
- **fern #1549** — **FiLM global cond — sent back, MASSIVE -17.3% signal at 81.291**.
- **edward #1548** — Fourier coords L=4 — sent back for stoch-depth+cosine rebase (-6.40%).

## Round 2 wave 3 — currently in flight (2 new PRs after wave-2 closures)

| Student | PR | Slug | Wave-3 idea | Axis |
|---------|----|----|--------------|------|
| alphonse | #1636 | `log1p-p-only` | H16 | Target reparam — log1p only on p (heavy-tailed) channel |
| askeladd | #1637 | `grad-clip-25` | H15 | Optim trajectory — permissive grad clip on outlier spikes only |

### H16 motivation
alphonse's closed #1610 (full-target log1p) regressed +1.18% on val_avg but
revealed a clean per-split asymmetry: cruise/re_rand improved (the lower-peak
splits) while single_in_dist/camber_rc regressed (the high-peak splits). The
recomputed `log_y_std = [1.115, 1.531, 4.643]` showed only the pressure channel
is genuinely heavy-tailed (4× the others). H16 applies log1p ONLY to p,
isolating the compression where it matters.

### H15 motivation
askeladd's closed #1529 (`max_norm=1.0`) regressed +5.4% — but his own grad
norm diagnostic showed natural training norms in the 10-245 range. clip=1.0
fired on 100% of steps (effectively reducing LR); clip=25 fires only on
outlier spikes (~10-15% of steps based on his data), suppressing the rare
large-update events without touching typical descent. Compatible with both
stoch-depth (which can cause block-drop-induced grad spikes) and cosine
cooldown (where outlier spikes are especially damaging in the late
fine-tuning phase).

## Wave 4 candidate pool (held)

The researcher-agent refresh (`research/RESEARCH_IDEAS_2026-05-12_21:00.md`)
also surfaced these, gated on the wave-3 results and rebase outcomes above:

- **H12 (per-node adaptive temperature)** — `τᵢ = τ₀ + Linear(dim_head→1)(x_mid_i)`,
  clamped ≥ 0.1. Distinct from the exhausted Ada-Temp variants in #1514
  (which tried per-head and shared-heads scalar τ, never per-node).
  **Hold until #1553 Gumbel-Softmax concludes** — both attack slice-collapse
  via different mechanisms.
- **H17 (learnable output scale+bias for pressure channel)** — `nn.Parameter`
  scale and bias on the p channel only. Bypasses `_init_weights` reset
  (which uses trunc_normal_ on nn.Linear, not on bare Parameters), so the
  identity-init invariant holds. Small calibration gain expected on cruise.
- **H16-followup (learnable log compression α)** — if H16 wins, replace
  fixed `log1p(|y_p|)` with `log(1 + α·|y_p|) / α`, α learnable; α=0 →
  identity, α=1 → full log1p.
- **H15-followup (grad-clip bracket)** — if H15 wins, sweep `max_norm ∈
  {10, 50}` to bracket the optimum.

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

**Constraints reaffirmed from wave-1 + wave-2 closures:**
- No more learnable loss-balance objectives (Kendall ruled out the family).
- No more architectural changes that add >10% per-step compute (asymmetric Q/K).
- surf_weight=10 is empirically at-or-near optimum (3+ independent confirmations).
- Full-target log1p uniformly is wrong (channel-asymmetric); pressure-only
  may still work (H16 testing this now).

## Recent closures and merges (2026-05-12 19:48-21:30 UTC)

- **#1611 cosine-tmax-15 (askeladd)** — **MERGED** as new baseline 94.217.
  val_avg -4.21%, test 4-split 84.859 (-3.57%).
- **#1610 log1p-target (alphonse)** — **CLOSED** at +1.18%. Per-split
  asymmetry → motivates H16 (pressure-only variant) now in flight as #1636.
- **#1608 ema-weights-0.999 (frieren)** — **SENT BACK** for rebase onto
  new #1611 baseline. Raw signal 95.761 was -2.64% vs OLD baseline; should
  still stack with cosine.
- **#1549 film-global-cond (fern)** — **SENT BACK** for rebase. Posted
  val_avg=81.291 (-17.3% vs new baseline!) — extraordinary signal pending
  confirmation after rebase onto stoch-depth + cosine.
- **#1548 fourier-coords-L4 (edward)** — **SENT BACK** for stoch-depth +
  cosine rebase. Posted val_avg=92.053 (-6.40% vs old baseline).
- **#1552 stoch-depth-0.1 (frieren)** — **MERGED** (round-2 wave-1 winner).
- **#1555 remove-in-project-fx (thorfinn)** — **SENT BACK** for n_hidden=144
  re-tune. Net +1.57% but efficiency gains real and direction worth iterating.
- **#1514 ada-temp v2 (alphonse)** — **CLOSED** at +3.4% vs L1-only base.
- **#1547 kendall-uncertainty (askeladd)** — **CLOSED** at +5.28%.
- **#1545 asymmetric-qk (tanjiro)** — **CLOSED** at +18.9% (compute-bound).
- **#1530 channel-weight-p3 (tanjiro)** — closed, +1.22% worse than L1.
- **#1529 grad-clip-1.0 (askeladd)** — closed, +5.4% worse than L1.

## All 8 students currently assigned

| Student | PR | Status |
|---------|----|--------|
| alphonse | #1636 | WIP (H16 log1p-p-only) |
| askeladd | #1637 | WIP (H15 grad-clip-25) |
| edward | #1548 | rebasing (Fourier L=4) |
| fern | #1549 | rebasing (FiLM 81.291 signal) |
| frieren | #1608 | rebasing (EMA 0.999) |
| nezuko | #1553 | WIP (Gumbel-Softmax slices) |
| tanjiro | #1612 | WIP (stoch-depth 0.05) |
| thorfinn | #1555 | retuning (n_hidden=144) |

Zero idle students. Zero idle GPUs.
