# SENPAI Research State

- **Last updated**: 2026-05-12 23:58 UTC (#1555 thorfinn closed after rebase regression; wave 4 now full at 4 PRs)
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging
  ablation. Each individual target training execution is capped at
  `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #1637 merged — grad-clip max_norm=25, -4.16%)

- `val_avg/mae_surf_p` = **90.294** (grad-clip-25 + cosine-T_max-15 + L1 + stoch-depth; best @ ep 15)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **81.243**
- Per-split val: single_in_dist=109.497 / camber_rc=98.952 / camber_cruise=69.208 / re_rand=83.520
- Δ vs PR #1611 baseline (94.217): **-4.16%** on val_avg, **-4.26%** on 4-split test
- All four val splits improved uniformly (-3.14% to -5.61%)
- Compound progress: #1397 L1 → #1552 stoch-depth → #1611 cosine T_max=15 → #1637 grad-clip → val_avg has improved from 100.957 to 90.294 = **-10.6% over 4 merges**.

## Current research focus

**Stacking compound wins.** The merged baseline is now 90.294 with four
mechanisms stacked: L1 loss, stoch-depth schedule, cosine T_max=15, and
grad-clip max_norm=25. Each individual mechanism has been confirmed
beneficial on this dataset; the compound is now ~11% better than the
original L1-only baseline. Three threads in flight:

1. **Bracket grad-clip** (askeladd #1674, `max_norm=50`) — direct
   follow-up to merged #1637. Tests whether pure spike suppression
   (the 110-norm at ep8) is the active mechanism vs spike+heavy-step
   suppression (the typical 30-70 norms also getting clipped at 25).
   Mechanism check via the new `train/last_grad_norm` log.
2. **Per-channel output calibration** (alphonse #1675, H17) — clean pivot
   from closed log1p direction. Adds 6 learnable parameters (γ, β ∈ ℝ³)
   on the output head, identity-init. Attacks pressure-channel calibration
   without compression.
3. **Slice-collapse attack #3** (nezuko #1677, H12 per-node τ) — clean
   pivot from closed Gumbel-Softmax. Per-node deterministic temperature
   via small MLP, identity-init. Distinct from prior Ada-Temp and Gumbel
   attempts.

Two extraordinary signals still pending rebase confirmation:

- **fern #1549 FiLM = 81.291** posted on old base (missing stoch-depth, cosine, grad-clip). **-17.3% vs #1611 baseline, -10.0% vs current 90.294 baseline if it holds.** Highest-EV lever in flight.
- **edward #1548 Fourier coords L=4 = 92.053** posted on old base (missing stoch-depth and cosine). -6.40% vs #1611 baseline; would need to retest against 90.294.

The recurring round-1 finding holds: **surf_weight=10 is at or above the
optimum**. Three independent confirmations bracket the optimum near 10.

## Round 2 wave 3 — full resolution

| Student | PR | Slug | Verdict | Δ vs baseline-at-submission |
|---------|----|----|---------|---------------|
| askeladd | #1637 | `grad-clip-25` | **MERGED** (new baseline 90.294) | -4.16% |
| alphonse | #1636 | `log1p-p-only` | **CLOSED** (channel-attribution falsified) | +5.32% |
| nezuko | #1553 | `gumbel-slice` | **CLOSED** (3-run mean +4.4% vs old, hyp falsified) | +4.4% (3-run mean) |

## Round 2 wave 4 — currently in flight (4 PRs)

| Student | PR | Slug | Wave-4 idea | Axis |
|---------|----|----|--------------|------|
| askeladd | #1674 | `grad-clip-50` | H15 bracket | Optim — bracket grad-clip threshold above 25 |
| alphonse | #1675 | `out-scale-bias-h17` | H17 | Output head — learnable per-channel γ, β |
| nezuko | #1677 | `per-node-temp-h12` | H12 | Slice mechanism — per-node deterministic τ_i |
| thorfinn | #1699 | `attn-mlp-dropout-0.05` | new H18 | Regularization — fine-grained dropout on top of merged stoch-depth |

## Wave-1 / wave-2 carryover (still WIP)

| Student | PR | Slug | Status |
|---------|----|----|--------|
| fern | #1549 | `film-global-cond` | **REBASING** — extraordinary -17% signal pending re-run on current stack |
| edward | #1548 | `fourier-coords-L4` | **REBASING** — -6% signal needs re-run on current stack |
| frieren | #1608 | `ema-weights-0.999` | **REBASING** — -2.6% signal needs re-run on current stack |
| tanjiro | #1612 | `stoch-depth-0.05` | WIP (drop_rate=0.05 bracket below merged 0.10) |

## Wave 5 candidate pool (held)

The researcher-agent refresh `research/RESEARCH_IDEAS_2026-05-12_21:00.md`
surfaced these, gated on wave-4 results and pending-rebase outcomes:

- **H15-followup-bracket-low (grad-clip max_norm=10)** — only if #1674 at
  max_norm=50 regresses or matches 25. Pins 25 from below.
- **H17-followup (per-output-position γ, β with spatial freq bins)** —
  only if simpler H17 wins. More params, more risk.
- **H12-followup-floor-sweep (tau_floor ∈ {0.05, 0.2})** — only if #1677
  H12 wins. Bracket the floor.
- **Stoch-depth follow-ups (depending on tanjiro #1612 outcome):**
  - If 0.05 wins → bracket with 0.075 (between 0.05 winner and merged 0.10).
  - If 0.05 regresses → over-regularization theory falsified; try 0.15 or 0.20.
  - **Stack dropout in PhysicsAttention/MLP at 0.05** — standard ViT recipe;
    often compounds with stoch-depth regardless of drop_rate outcome.

### Other open levers

**Architecture/loss:**
- **Output head specialization** — separate p / Ux / Uy heads with
  channel-balanced loss in physical units. (Stronger version of H17.)
- **Domain-aware sampler reweighting** to match val split aggregation
  (3/4 tandem in val_avg vs. 1/3 tandem in current sampler).
- **Optimizer alternatives** — Lion, Adafactor, Sophia at calibrated lr.

**Constraints reaffirmed from wave-1 + wave-2 + wave-3 closures:**
- No more learnable loss-balance objectives (Kendall ruled out the family).
- No more architectural changes that add >10% per-step compute (asymmetric Q/K).
- surf_weight=10 is empirically at-or-near optimum (3+ independent confirmations).
- **Log compression is dead** on this dataset (full-channel #1610 regressed +1.18%, pressure-only #1636 regressed +5.32%; channel-attribution theory falsified).
- **Gumbel-Softmax-style noise injection is dead** in this 30-min budget regime (#1553 3-run mean +4.4%); slice-collapse must be attacked deterministically.

## Recent closures and merges (2026-05-12 19:48-23:00 UTC)

- **#1637 grad-clip-25 (askeladd)** — **MERGED** as new baseline 90.294. val_avg -4.16%, test 4-split 81.243 (-4.26%). Mechanism: clip on outlier-spike steps, leave typical 30-70 norms unscaled.
- **#1636 log1p-p-only (alphonse)** — **CLOSED** at +5.32%. Per-split asymmetry amplified from #1610: single_in_dist regressed +14.42%, camber_rc +9.23%. Channel-attribution theory falsified. Log-compression direction fully closed.
- **#1553 gumbel-slice (nezuko)** — **CLOSED** at +4.4% (3-run mean). All 3 runs underperform even the old L1 baseline. Sampling noise antagonistic to the current stoch-depth + cosine + grad-clip stack.
- **#1611 cosine-tmax-15 (askeladd)** — **MERGED** earlier today (new baseline 94.217 at the time, now superseded).
- **#1610 log1p-target (alphonse)** — **CLOSED** at +1.18%.
- **#1608 ema-weights-0.999 (frieren)** — **SENT BACK** for rebase.
- **#1549 film-global-cond (fern)** — **SENT BACK** for rebase. Extraordinary 81.291 signal pending.
- **#1548 fourier-coords-L4 (edward)** — **SENT BACK** for rebase.
- **#1552 stoch-depth-0.1 (frieren)** — **MERGED** (round-2 wave-1 winner).
- **#1555 remove-in-project-fx (thorfinn)** — **SENT BACK** for n_hidden=144.
- **#1514 ada-temp v2 (alphonse)** — **CLOSED** at +3.4% vs L1-only base.
- **#1547 kendall-uncertainty (askeladd)** — **CLOSED** at +5.28%.
- **#1545 asymmetric-qk (tanjiro)** — **CLOSED** at +18.9%.
- **#1530 channel-weight-p3 (tanjiro)** — closed, +1.22%.
- **#1529 grad-clip-1.0 (askeladd)** — closed, +5.4% — clip too aggressive at 1.0; the bracket success at 25 today validates the diagnostic.

## All 8 students currently assigned

| Student | PR | Status |
|---------|----|--------|
| alphonse | #1675 | WIP (H17 out-scale-bias) |
| askeladd | #1674 | WIP (grad-clip-50 bracket) |
| edward | #1548 | rebasing (Fourier L=4) |
| fern | #1549 | rebasing (FiLM 81.291 signal) |
| frieren | #1608 | rebasing (EMA 0.999) |
| nezuko | #1677 | WIP (H12 per-node temp) |
| tanjiro | #1612 | WIP (stoch-depth 0.05) |
| thorfinn | #1555 | retuning (n_hidden=144) |

Zero idle students. Zero idle GPUs.
