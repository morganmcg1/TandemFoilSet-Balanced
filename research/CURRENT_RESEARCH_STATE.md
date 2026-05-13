# SENPAI Research State

- **Last updated**: 2026-05-13 01:05 UTC (wave 5: 3 closures #1713 grad-clip-10 +4.24%, #1677 H12 +3.11%, #1612 stoch-depth-0.05 +13.7%; 3 new assignments #1753 askeladd adaptive-clip, #1754 nezuko LR warmup, #1756 tanjiro stoch-depth-0.15; stale PRs #1549/#1608/#1548 sent back with rebase instructions)
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

**Three single-knob directions now fully bracketed and closed:**
- **Grad-clip fixed-threshold**: bracket {1.0, 10, 25, 50} complete. 25 is
  local optimum with asymmetric penalty (+5.4% / +4.24% / 0% / +3.32%).
  Direction closed; adaptive-clipping is the natural next step.
- **Stoch-depth single-knob**: 0.05 closed +13.7% (vs current baseline).
  The pre-registered 0.15 follow-up tests above-the-merged-0.10 direction.
  Split-asymmetric tradeoff observed: val_re_rand likes less drop,
  val_geom_camber_rc likes more.
- **Slice-collapse mechanism**: 3 arms (Ada-Temp #1514, Gumbel #1553, H12
  #1677) all failed. Direction closed. Slice mechanism gets no more
  attempts — pivoting to optimization-side levers.

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

Active threads (post wave-5 closures):

1. **Per-channel surf-loss weighting** (alphonse #1711, H18,
   `[0.5, 0.5, 2.0]`) — clean pivot from closed H17. Attacks the same
   pressure-vs-velocity imbalance, but at the loss layer (mass-preserving
   reweighting) rather than the prediction layer. Direct gradient signal
   increase on the metric-defining channel.
2. **Fine-grained dropout stack** (thorfinn #1699, attn+MLP dropout p=0.05)
   — clean pivot from closed n_hidden=144 retune. Mechanistically
   orthogonal to merged block-level stoch-depth; zero compute overhead.
3. **Adaptive grad-clip** (askeladd #1753, 1.5× running median over K=100
   step deque) — pre-registered pivot from closed fixed-threshold bracket.
   Tracks the gradient-norm distribution as it evolves; clips per-step
   relative to a recent population rather than a fixed value.
4. **Linear LR warm-up** (nezuko #1754, H19, 1-epoch warmup + cosine
   T_max=14) — pivot from closed slice-collapse direction. Addresses the
   consistent ep1 pre-clip grad-norm spike (60-100) observed in every
   recent grad-clip experiment. Standard transformer warmup recipe.
5. **Stoch-depth above-the-merged-optimum** (tanjiro #1756,
   `drop_rate=0.15`) — pre-registered follow-up. Tests whether hard OOD
   geometry splits want more regularization than merged 0.10.

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

## Round 2 wave 4 — final resolution

| Student | PR | Slug | Verdict | Δ vs baseline-at-submission |
|---------|----|----|---------|---------------|
| askeladd | #1674 | `grad-clip-50` | **CLOSED** (Outcome B: bulk 30–70 norms are active) | +3.32% |
| alphonse | #1675 | `out-scale-bias-h17` | **CLOSED** (γ-drift hurts large-magnitude p) | +3.24% |
| askeladd | #1713 | `grad-clip-10` | **CLOSED** (Outcome B: 25 is local optimum; bracket complete) | +4.24% |
| nezuko | #1677 | `per-node-temp-h12` | **CLOSED** (slice-collapse 3rd failure; mechanism learned, outcome rejected) | +3.11% |
| tanjiro | #1612 | `stoch-depth-0.05` | **CLOSED** (split-asymmetric; OOD geom wants more drop) | +13.7% vs current baseline |

## Round 2 wave 5 — currently in flight (5 PRs)

| Student | PR | Slug | Idea | Axis |
|---------|----|----|--------------|------|
| alphonse | #1711 | `surf-ch-weight-h18` | H18 | Loss — per-channel surf-loss weighting [0.5, 0.5, 2.0] |
| thorfinn | #1699 | `attn-mlp-dropout-0.05` | H18-thorfinn | Regularization — fine-grained dropout (orthogonal to stoch-depth) |
| askeladd | #1753 | `adaptive-grad-clip-q50-a1.5` | H15-adaptive | Optim — running-quantile clip (1.5× median over 100 steps) |
| nezuko | #1754 | `lr-warmup-h19` | H19 | Optim — linear warmup epoch 1 + cosine T_max=14 |
| tanjiro | #1756 | `stoch-depth-0.15` | H | Regularization — bracket above merged 0.10 |

## Wave-1 / wave-2 carryover (still WIP)

| Student | PR | Slug | Status |
|---------|----|----|--------|
| fern | #1549 | `film-global-cond` | **REBASING** — extraordinary -17% signal pending re-run on current stack |
| edward | #1548 | `fourier-coords-L4` | **REBASING** — -6% signal needs re-run on current stack |

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

## Recent closures and merges (2026-05-12 19:48 → 2026-05-13 01:00 UTC)

- **#1713 grad-clip-10 (askeladd)** — **CLOSED** at +4.24%. Lower-bracket fixed-threshold confirmation. Completes the {1.0, 10, 25, 50} bracket — 25 is the asymmetric local optimum. Adaptive-clip (#1753) is the natural pivot.
- **#1677 per-node-temp-h12 (nezuko)** — **CLOSED** at +3.11%. Slice-collapse mechanism's 3rd consecutive failure (after #1514 Ada-Temp v2 and #1553 Gumbel). Direction fully closed.
- **#1612 stoch-depth-0.05 (tanjiro)** — **CLOSED** at +13.7% vs current baseline 90.294 (but +4.4% vs pre-#1611/#1637 baseline that student used). Split-asymmetric finding: val_re_rand wants less drop, val_geom_camber_rc wants more. Pre-registered follow-up #1756 brackets above merged 0.10.
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

## Wave 6 candidate pool (held)

Updated with wave-5 closures. The researcher-agent refresh
`research/RESEARCH_IDEAS_2026-05-12_21:00.md` plus wave-4/5 findings surface
these gated on wave-5 results:

- ~~**H15-followup-bracket-low (grad-clip max_norm=10)**~~ — **CLOSED** (#1713 +4.24%).
- ~~**H17-followup (per-output-position γ, β with spatial freq bins)**~~ — **DROPPED**:
  simpler H17 #1675 closed.
- ~~**H12-followup-floor-sweep**~~ — **DROPPED**: H12 #1677 closed, slice-collapse
  mechanism direction fully closed.
- **Stoch-depth follow-ups (depending on tanjiro #1756 outcome):**
  - If 0.15 wins → bracket with 0.20 (push above merged optimum further).
  - If 0.15 regresses uniformly → single-knob direction exhausted at 0.10.
  - If 0.15 split-asymmetric (wins on val_geom_camber_rc, regresses on val_re_rand) →
    pivot to per-layer or per-block drop schedule (separate larger PR).
- **H18-followup (per-channel surf-loss weight bracketing)** — only if #1711
  H18 wins. Try `[0.33, 0.33, 2.33]` (more aggressive) and `[0.75, 0.75, 1.5]`
  (gentler).
- **Domain-aware sampler reweighting** — if #1711 H18 regresses, attack the
  pressure-channel imbalance via sampler upweighting of high-p-magnitude
  samples instead of per-channel loss weighting.
- **Adaptive grad-clip K/α bracket** — only if #1753 wins. K∈{50, 200}, α∈{1.25, 2.0}.
- **Warmup bracket** — only if #1754 H19 wins. Try 2-epoch warmup (T_max=13)
  and 0.5-epoch warmup (per-batch interpretation).
- **Optimizer betas refresh** — `betas=(0.9, 0.95)` (more responsive second-moment
  estimate, common in recent recipes). Standalone or combined with #1754 warmup.

## All 8 students currently assigned

| Student | PR | Status |
|---------|----|--------|
| alphonse | #1711 | WIP (H18 surf-ch-weight [0.5, 0.5, 2.0]) |
| askeladd | #1753 | WIP (adaptive grad-clip — 1.5× running median K=100) |
| edward | #1548 | rebasing (Fourier L=4) |
| fern | #1549 | rebasing (FiLM 81.291 signal) |
| frieren | #1608 | rebasing (EMA 0.999) |
| nezuko | #1754 | WIP (H19 LR warmup epoch 1 + cosine T_max=14) |
| tanjiro | #1756 | WIP (stoch-depth 0.15 — bracket above merged 0.10) |
| thorfinn | #1699 | WIP (attn+MLP dropout p=0.05) |

Zero idle students. Zero idle GPUs.
