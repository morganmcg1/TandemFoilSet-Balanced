# SENPAI Research State

- **Last updated**: 2026-05-13 00:14 UTC (wave 4: #1675 H17 +3.24% closed, #1674 grad-clip-50 +3.32% closed; alphonse pivoted to #1711 H18 surf-loss weighting, askeladd pivoted to #1713 grad-clip-10)
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
original L1-only baseline.

**Output-side calibration is fully exhausted** after wave-4 results: #1610
(full log1p), #1636 (pressure-only log1p), #1675 (per-channel γ, β output
affine) all regressed. The structural lesson: the existing `y_std`
normalization is already calibrated; post-prediction adjustments either
compress the magnitude axis or let the optimizer drift away from a useful
init. Per-channel rebalancing must move **upstream to the loss layer**.

**Grad-clip bracket above (max_norm=50, #1674) closed at +3.32%** —
confirms Outcome B: bulk 30–70 norms are the active ingredient at 25, not
pure spike suppression. The 1.0 / 25 / 50 bracket points map to a clear
asymmetric profile around 25. **Lower bracket point (max_norm=10, #1713)
now in flight** to complete the curve.

Active threads:

1. **Grad-clip lower bracket** (askeladd #1713, `max_norm=10`) — direct
   follow-up to closed #1674. Completes the bracket around merged 25.
   If 10 wins → bracket below; if 10 regresses → 25 is at local optimum
   and fixed-threshold sweep ends.
2. **Per-channel surf-loss weighting** (alphonse #1711, H18,
   `[0.5, 0.5, 2.0]`) — clean pivot from closed H17. Attacks the same
   pressure-vs-velocity imbalance, but at the loss layer (mass-preserving
   reweighting) rather than the prediction layer. Direct gradient signal
   increase on the metric-defining channel.
3. **Slice-collapse attack #3** (nezuko #1677, H12 per-node τ) — clean
   pivot from closed Gumbel-Softmax. Per-node deterministic temperature
   via small MLP, identity-init. Distinct from prior Ada-Temp and Gumbel
   attempts.
4. **Fine-grained dropout stack** (thorfinn #1699, attn+MLP dropout p=0.05)
   — clean pivot from closed n_hidden=144 retune. Mechanistically
   orthogonal to merged block-level stoch-depth; zero compute overhead.

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

## Round 2 wave 4 — partial resolution + pivots

| Student | PR | Slug | Verdict | Δ vs baseline-at-submission |
|---------|----|----|---------|---------------|
| askeladd | #1674 | `grad-clip-50` | **CLOSED** (Outcome B: bulk 30–70 norms are active) | +3.32% |
| alphonse | #1675 | `out-scale-bias-h17` | **CLOSED** (γ-drift hurts large-magnitude p) | +3.24% |

## Round 2 wave 4 / 5 — currently in flight (6 PRs)

| Student | PR | Slug | Idea | Axis |
|---------|----|----|--------------|------|
| askeladd | #1713 | `grad-clip-10` | H15 lower-bracket | Optim — complete the bracket around 25 from below |
| alphonse | #1711 | `surf-ch-weight-h18` | H18 | Loss — per-channel surf-loss weighting [0.5, 0.5, 2.0] |
| nezuko | #1677 | `per-node-temp-h12` | H12 | Slice mechanism — per-node deterministic τ_i |
| thorfinn | #1699 | `attn-mlp-dropout-0.05` | new H18 | Regularization — fine-grained dropout on top of merged stoch-depth |
| tanjiro | #1612 | `stoch-depth-0.05` | H | Regularization — bracket stoch-depth below merged 0.10 |
| frieren | #1608 | `ema-weights-0.999` | H | Optim — EMA weights (rebasing) |

## Wave-1 / wave-2 carryover (still WIP)

| Student | PR | Slug | Status |
|---------|----|----|--------|
| fern | #1549 | `film-global-cond` | **REBASING** — extraordinary -17% signal pending re-run on current stack |
| edward | #1548 | `fourier-coords-L4` | **REBASING** — -6% signal needs re-run on current stack |

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

## Recent closures and merges (2026-05-12 19:48 → 2026-05-13 00:14 UTC)

- **#1675 out-scale-bias-h17 (alphonse)** — **CLOSED** at +3.24%. γ-drift verified (pressure scale +6.13% by ep 15) but uniformly regresses splits; largest hit on val_single_in_dist (+5.07%, the highest-p-magnitude split). Output-side calibration direction fully exhausted.
- **#1674 grad-clip-50 (askeladd)** — **CLOSED** at +3.32%. Outcome B confirmed: spike-only suppression captures pattern but not enough gain. Bulk 30–70 norms are the active ingredient at threshold 25.
- **#1555 remove-in-project-fx + n_hidden=144 (thorfinn)** — **CLOSED** at +13.71% after rebase. The cosine + grad-clip merges resolved the underfitting that motivated the retune; #1552-era diagnostic no longer applies.
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

## Wave 5 candidate pool (held)

The researcher-agent refresh `research/RESEARCH_IDEAS_2026-05-12_21:00.md`
surfaced these. Updated with wave-4 closure information:

- ~~**H15-followup-bracket-low (grad-clip max_norm=10)**~~ — **NOW IN FLIGHT** as #1713.
- ~~**H17-followup (per-output-position γ, β with spatial freq bins)**~~ — **DROPPED**:
  simpler H17 #1675 closed, so the more complex variant is closed by extension.
- **H12-followup-floor-sweep (tau_floor ∈ {0.05, 0.2})** — only if #1677
  H12 wins. Bracket the floor.
- **Stoch-depth follow-ups (depending on tanjiro #1612 outcome):**
  - If 0.05 wins → bracket with 0.075 (between 0.05 winner and merged 0.10).
  - If 0.05 regresses → over-regularization theory falsified; try 0.15 or 0.20.
- **H18-followup (per-channel surf-loss weight bracketing)** — only if #1711
  H18 wins. Try `[0.33, 0.33, 2.33]` (more aggressive) and `[0.75, 0.75, 1.5]`
  (gentler).
- **Domain-aware sampler reweighting** — if #1711 H18 regresses, attack the
  pressure-channel imbalance via sampler upweighting of high-p-magnitude
  samples instead of per-channel loss weighting.
- **Adaptive grad-clip (AGC or running-quantile)** — only if #1713
  `max_norm=10` confirms 25 is at local optimum of the fixed-threshold sweep.

## All 8 students currently assigned

| Student | PR | Status |
|---------|----|--------|
| alphonse | #1711 | WIP (H18 surf-ch-weight [0.5, 0.5, 2.0]) |
| askeladd | #1713 | WIP (grad-clip max_norm=10 — lower bracket) |
| edward | #1548 | rebasing (Fourier L=4) |
| fern | #1549 | rebasing (FiLM 81.291 signal) |
| frieren | #1608 | WIP (EMA 0.999 — post-rebase run) |
| nezuko | #1677 | WIP (H12 per-node temp) |
| tanjiro | #1612 | WIP (stoch-depth 0.05) |
| thorfinn | #1699 | WIP (attn+MLP dropout p=0.05) |

Zero idle students. Zero idle GPUs.
