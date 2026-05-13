# SENPAI Research State

- **Last updated**: 2026-05-13 02:20 UTC (close #1773 thorfinn AdamW betas +1.97%; close #1756 tanjiro stoch-depth-0.15 +7.69%; **#1754 nezuko LR warmup WINS on old baseline -0.64%/-1.71%, sent back for rebase to new 84.762 baseline**; assign #1799 thorfinn LayerScale, #1811 tanjiro per-channel output heads; in-flight: #1711, #1753, #1754(rebasing), #1772, #1799, #1811; rebasing: #1549, #1608)
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging
  ablation. Each individual target training execution is capped at
  `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #1548 merged — Fourier coords L=4, -6.13%)

- `val_avg/mae_surf_p` = **84.762** (Fourier L=4 + grad-clip-25 + cosine-T_max-15 + L1 + stoch-depth; best @ ep 15)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **74.659**
- Per-split val: single_in_dist=97.074 / camber_rc=94.997 / camber_cruise=63.711 / re_rand=83.266
- Per-split test: single_in_dist=85.819 / camber_rc=83.023 / camber_cruise=54.879 / re_rand=74.916
- Δ vs PR #1637 baseline (90.294 / 81.243): **-6.13%** on val_avg, **-8.10%** on 4-split test
- **Test gain exceeds val gain** — model generalizes the Fourier features well
- Per-split pattern matches spectral-bias hypothesis: largest gains where high-frequency spatial structure dominates (in_dist -11.35%, camber_cruise -7.94%); minimal on val_re_rand (-0.30%) whose OOD axis is Reynolds, not spatial coords
- Compound progress: #1397 L1 → #1552 stoch-depth → #1611 cosine T_max=15 → #1637 grad-clip → #1548 Fourier L=4 → val_avg has improved from 100.957 to 84.762 = **-16.0% over 5 merges**.

## Current research focus

**Stacking compound wins.** The merged baseline is now **84.762** with five
mechanisms stacked: L1 loss, stoch-depth schedule, cosine T_max=15,
grad-clip max_norm=25, and Fourier coord encoding L=4. Each individual
mechanism has been confirmed beneficial on this dataset; the compound is
now **-16.0%** better than the original L1-only baseline (100.957 → 84.762).

**The Fourier merge (PR #1548) is the strongest single-experiment signal of
the round** — test gain (-8.10%) exceeds val gain (-6.13%), per-split
pattern exactly matches the spectral-bias mechanism prediction, and stacks
cleanly with all four prior compound merges. Spectral-bias / positional
encoding is the most productive direction for the immediate follow-up
(bracket sweep + Gaussian Fourier variant).

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

Active threads (post #1756/#1773 close, #1754 sent back for rebase):

1. **Per-channel surf-loss weighting** (alphonse #1711, H18,
   `[0.5, 0.5, 2.0]`) — loss-layer rebalancing toward pressure channel.
2. **Adaptive grad-clip** (askeladd #1753, 1.5× running median over K=100
   step deque) — pivot from closed fixed-threshold bracket.
3. **LR warm-up REBASING** (nezuko #1754, H19) — WON on old baseline
   (-0.64% val, -1.71% test); pending re-confirmation on new 84.762 baseline.
   Highest-EV merge-candidate after rebase.
4. **Fourier coord encoding `n_freqs=6`** (edward #1772) — pre-registered
   bracket-up from his merged #1548 L=4.
5. **LayerScale CaiT-style init=0.1** (thorfinn #1799, H23) — fresh axis
   pivot after closed AdamW betas direction. Per-channel learnable γ_l
   gates each residual branch; +0.19% params; compounds with stoch-depth.
6. **Per-channel output head MLPs** (tanjiro #1811, H24) — fresh
   architecture pivot after closed stoch-depth single-knob direction.
   3 × Sequential(Linear(128→64), GELU, Linear(64→1)) replace shared
   final linear; +3.72% params; strengthens closed H17 γ/β idea with
   non-linear per-channel decoders.

Note: all wave-5 in-flight experiments (#1711, #1753, #1754, #1756) are
measured against the **new 84.762 baseline**, not against the 90.294 they
were assigned on. Most are testing orthogonal levers (loss layer,
optimization, regularization) so should still stack if the mechanism works.

One extraordinary signal still pending rebase confirmation:

- **fern #1549 FiLM = 81.291** posted on old base (missing stoch-depth, cosine, grad-clip, Fourier). **-19.5% vs #1611 baseline.** If rebased and the gap holds against the new 84.762 baseline, this is the highest-EV lever in flight. Now competing against a much stronger baseline so the absolute delta will be smaller.

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

## Round 2 wave 5 — resolution (4 closures, 1 winner pending rebase, 2 in flight)

| Student | PR | Slug | Verdict | Δ vs baseline-at-submission |
|---------|----|----|---------|---------------|
| edward | #1548 | `fourier-coords-L4-rebased` | **MERGED** (new baseline 84.762) | -6.13% val, -8.10% test |
| thorfinn | #1699 | `attn-mlp-dropout-0.05` | **CLOSED** (mechanism conflict + compute tax) | +2.27% |
| tanjiro | #1756 | `stoch-depth-0.15` | **CLOSED** (uniform regression, V around 0.10 confirmed) | +7.69% val |
| nezuko | #1754 | `lr-warmup-h19` | **REBASING** (won on old baseline -0.64%/-1.71%) | tbd vs 84.762 |
| alphonse | #1711 | `surf-ch-weight-h18` | WIP | tbd |
| askeladd | #1753 | `adaptive-grad-clip-q50-a1.5` | WIP | tbd |

## Round 2 wave 6 — partial resolution (#1773 closed, #1772/#1799/#1811 in flight)

| Student | PR | Slug | Verdict | Δ vs baseline-at-submission |
|---------|----|----|---------|---------------|
| thorfinn | #1773 | `adamw-betas-0.95` | **CLOSED** (non-uniform regression, regime mismatch) | +1.97% val, +1.61% test |
| edward | #1772 | `fourier-coords-L6` | WIP | tbd |
| thorfinn | #1799 | `layerscale-init-0.1` | WIP | tbd |
| tanjiro | #1811 | `output-head-per-channel-mlp` | WIP | tbd |

## Wave-1 / wave-2 carryover (still WIP)

| Student | PR | Slug | Status |
|---------|----|----|--------|
| fern | #1549 | `film-global-cond` | **REBASING** — extraordinary -19.5% signal pending re-run on current 84.762 stack |
| frieren | #1608 | `ema-weights-0.999` | **REBASING** — needs rebase onto new Fourier-merged baseline |

(edward #1548 Fourier coords now MERGED — see new best baseline above.)

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

## Recent closures and merges (2026-05-12 19:48 → 2026-05-13 02:20 UTC)

- **#1754 lr-warmup-h19 (nezuko)** — **REBASING**. Won on old baseline at -0.64% val, -1.71% test (3/4 val splits improve, all 4 test splits improve). Mechanism check passes (ep1 last-batch pre-clip grad-norm dropped ~35%). Sent back to re-measure on new 84.762 baseline post-Fourier-merge. Expected post-rebase: -0.3% to -1.5% on val_avg (warmup is orthogonal to Fourier input encoding).
- **#1756 stoch-depth-0.15 (tanjiro)** — **CLOSED** at +7.69% val. Outcome C confirmed: merged 0.10 is local optimum (V-shape: 0.05→+13.7%, 0.10→0%, 0.15→+7.69%). Both endpoints regress on val_geom_camber_rc, falsifying the "OOD geometry wants more drop" narrative I had built. Train-vs-val gap stays val > train (no ensemble-dropout signature). Single-knob direction fully closed.
- **#1773 adamw-betas-0.95 (thorfinn)** — **CLOSED** at +1.97% val, +1.61% test. Non-uniform regression confirms two falsified predictions (best-epoch unchanged, per-split non-uniform). Deeper finding: merged stack (grad-clip-25 + cosine T_max=15) already addresses the non-stationarity that motivated short-EMA β₂; the regime gap to LLaMA/PaLM (batch=10³-10⁶× larger, 10⁵-10⁶ steps) makes β₂=0.95 a worse fit. Optimizer-betas direction closed.
- **#1548 fourier-coords-L4-rebased (edward)** — **MERGED** as new baseline **84.762** (val) / **74.659** (test). -6.13% val, -8.10% test vs #1637 baseline. Strongest single-experiment signal of the round; test gain exceeds val gain. Per-split pattern matches spectral-bias hypothesis exactly. 5th compound win on this branch.
- **#1699 attn-mlp-dropout-0.05 (thorfinn)** — **CLOSED** at +2.27%. Three mechanisms: compute tax (+12% epoch wall-clock), stoch-depth was already at regularization optimum, slice-attention double-use of softmax weights disrupts unit-sum. Single-knob fine-grained dropout direction closed.
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

## Wave 7 candidate pool (held)

Updated with #1548 merge + wave-5/6 in-flight assignments:

- ~~**H15-followup-bracket-low (grad-clip max_norm=10)**~~ — **CLOSED** (#1713 +4.24%).
- ~~**H17-followup (per-output-position γ, β with spatial freq bins)**~~ — **DROPPED**.
- ~~**H12-followup-floor-sweep**~~ — **DROPPED**.
- ~~**Optimizer betas refresh `(0.9, 0.95)`**~~ — **NOW IN FLIGHT** as #1773 (thorfinn).
- **Fourier coords follow-ups (depending on edward #1772 outcome):**
  - If L=6 wins → bracket up to L=8.
  - If L=6 regresses → pivot to learnable Gaussian Fourier features
    (Tancik §3, `B ∈ R^{m×2}, B ~ N(0, σ²I)`, single arm at σ=10).
  - Either way, consider applying Fourier encoding to the flow-condition
    dims (Re, AoA, NACA params 13-23) — `val_re_rand` is the one split
    spatial Fourier didn't touch.
- **Stoch-depth follow-ups (depending on tanjiro #1756 outcome):**
  - If 0.15 wins → bracket with 0.20.
  - If 0.15 regresses uniformly → single-knob direction exhausted at 0.10.
  - If 0.15 split-asymmetric → pivot to per-layer or per-block drop schedule.
- **H18-followup (per-channel surf-loss weight bracketing)** — only if #1711 wins.
- **Domain-aware sampler reweighting** — if #1711 regresses.
- **Adaptive grad-clip K/α bracket** — only if #1753 wins. K∈{50, 200}, α∈{1.25, 2.0}.
- **Warmup bracket** — only if #1754 H19 wins. Try 2-epoch warmup and 0.5-epoch warmup.
- **Output head specialization** — separate `p / Ux / Uy` heads with channel-balanced
  loss in physical units. (Stronger version of closed H17; warrants fresh try on the
  new baseline since per-channel calibration may compound with Fourier.)
- **Optimizer family pivots** — Lion (Chen 2023), Sophia (Liu 2023). Only if both
  betas-refresh (#1773) and warmup (#1754) regress.

## All 8 students currently assigned

| Student | PR | Status |
|---------|----|--------|
| alphonse | #1711 | WIP (H18 surf-ch-weight [0.5, 0.5, 2.0]) |
| askeladd | #1753 | WIP (adaptive grad-clip — 1.5× running median K=100) |
| edward | #1772 | WIP (Fourier coords L=6 — bracket up from merged L=4) |
| fern | #1549 | rebasing (FiLM 81.291 signal pending vs 84.762 baseline) |
| frieren | #1608 | rebasing (EMA 0.999) |
| nezuko | #1754 | rebasing (H19 LR warmup — won on old baseline, re-running on 84.762) |
| tanjiro | #1811 | WIP (H24 per-channel output head MLPs — 3 × Linear→GELU→Linear) |
| thorfinn | #1799 | WIP (LayerScale CaiT-style init=0.1 — per-block residual gating) |

Zero idle students. Zero idle GPUs.
