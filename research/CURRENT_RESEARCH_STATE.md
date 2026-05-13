# SENPAI Research State

- **Last updated**: 2026-05-13 11:45 UTC (Wave 13: MERGE #2175 tanjiro SwiGLU inner_dim=256 (val=67.381 −2.08%, test=57.800 −2.71% — 13th compound win); ASSIGN #2200 tanjiro swiglu-inner-dim-384; SEND-BACK #2098 frieren Lion rebase+re-run on post-#2175 baseline; #2075 thorfinn just picked up after rate-limit clear)
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging
  ablation. Each individual target training execution is capped at
  `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #2175 merged — SwiGLU inner_dim=256, 13th compound win)

- `val_avg/mae_surf_p` = **67.381** (SwiGLU MLP inner_dim=256 + LR warmup + LayerScale init=0.025 + surf-ch-weight [0.5,0.5,2.0] + Fourier L=6 + grad-clip-25 + cosine-T_max-14 + L1 + stoch-depth; best @ ep 13)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **57.800**
- Per-split val: single_in_dist=73.341 / camber_rc=80.673 / camber_cruise=48.675 / re_rand=66.834
- Per-split test: single_in_dist=64.685 / camber_rc=69.035 / camber_cruise=40.356 / re_rand=57.121
- Δ vs PR #2105 baseline (68.812 / 59.410): **−2.08%** val_avg, **−2.71%** test_avg
- **Mechanism (SwiGLU inner_dim=256)**: expanding from 176 to 256 removes the 2/3-ratio budget constraint from the Shazeer SwiGLU recipe. Every epoch was a new best at ep 13; model is capacity-limited not overfitting. All 4 test splits improve; val_camber_rc lone regression (+1.74%) but its matched test split wins (−0.39%) → noise. n_params: 831,191 (+22.6% vs 677,591).
- Compound progress: #1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896→#2018→#1754→#2105→**#2175** → val_avg has improved from 100.957 to **67.381** = **−33.3% over 13 merges**.

## Current research focus

**Wave 13 — SwiGLU capacity sweep + orthogonal diversified probes.**

The compound stack has 13 merged wins (100.957 → 67.381 = **−33.3%**): L1 loss, stoch-depth [0,0.025,0.05,0.075,0.1], cosine T_max=14, LR warmup epoch-1, grad-clip max_norm=25, Fourier L=6, LayerScale CaiT init=0.025, per-channel surf-weight [0.5,0.5,2.0], SwiGLU MLP inner_dim=176 → **inner_dim=256** (latest merged win).

**SwiGLU capacity is still the strongest open lever.** Both inner_dim=176 (−7.53%/−9.33%) and inner_dim=256 (−2.08%/−2.71%) won. The capacity-limited signature ("every epoch a new best") persists at 256, so inner_dim=384 (#2200) is the live next bet. The SwiGLU width sweep is an active research axis with confirmed upside.

**Wave 13 active threads** (all 8 students assigned, zero idle GPUs):

| Student | PR | Slug | Hypothesis | Status |
|---------|----|----|---------|--------|
| alphonse | #2136 | n_head=2-wider | n_head=4→2 (dim_head=64): wider heads preserve OOD cross-geom features | WIP |
| askeladd | #2102 | rmsnorm | LayerNorm→RMSNorm: no mean-centering, avoids LayerScale coupling | WIP |
| edward | #2135 | gaussian-σ-calibrated | Gaussian RFF σ=1.0 + σ=0.5 (calibrated for std-normalized coords) | WIP |
| fern | #2157 | vol-ch-weight-pressure | vol_loss per-channel weight [1,1,2]: pressure emphasis in volume domain | WIP |
| frieren | #2098 | lion-optimizer | Lion (rebase → re-run on post-#2175 baseline 67.381); strong signal on old baseline (67.846 < 68.812) | WIP (rebase) |
| nezuko | #2137 | lr-3e-4-bracket | Peak lr=5e-4→3e-4 with merged warmup+cosine stack | WIP |
| tanjiro | #2200 | swiglu-inner-dim-384 | SwiGLU inner_dim 256→384: next bisect in capacity sweep (OOM fallback to 320) | NEW |
| thorfinn | #2075 | layerscale-init-0.0125 | Test floor of LayerScale operating-point sweep (just picked up after rate-limit clear) | WIP |

**Merged in Wave 12/13:**
- **#2105 tanjiro SwiGLU** — MERGED (val=68.812 −6.96% vs 73.958; 12th compound win; largest single-PR gain)
- **#2175 tanjiro SwiGLU inner_dim=256** — MERGED (val=67.381 −2.08% vs 68.812; 13th compound win; capacity-limited signal)
- **#2099 fern wd-3x** — CLOSED (+2.24%/+2.67% vs new baseline; wd axis confirmed at 1e-4)

**Frieren Lion (#2098) context:** pre-rebase val=67.846 was already below current baseline 67.381 by only 0.46; barely misses. After rebase+re-run with SwiGLU inner_dim=256, Lion+SwiGLU composition needs testing. Predicted outcome: ~63–65 val if fully orthogonal, ~67 if partially redundant.

**Prioritized research themes for Wave 14** (after in-flight results):

1. **SwiGLU width sweep**: if #2200 (inner_dim=384) wins, next bisect is 512 or check capacity vs compute; if plateaus at 384, inner_dim=256 is the optimum.
2. **Lion optimizer** (#2098 rebase): strongest orthogonal candidate — operates on different axis from all merged stack components.
3. **SwiGLU + RMSNorm**: LLaMA/Mistral recipe is SwiGLU + RMSNorm — if #2102 wins, combine with SwiGLU.
4. **Attention depth**: if Lion wins, test n_layers=6 — SwiGLU wider per-layer may compound with depth.
5. **Gaussian Fourier σ-calibration** (#2135): camber_cruise test now 40.356; σ-calibrated RFF may further improve OOD generalization.
6. **n_head=2** (#2136): wider heads + SwiGLU wider MLP may be synergistic.
7. **vol-ch-weight** (#2157): pressure weighting in volume domain, orthogonal to surf-ch-weight already merged.
8. **lr=3e-4 bracket** (#2137): LR may need re-tuning with wider SwiGLU model.
9. **GeGLU comparison**: replace SiLU with GELU in the gate — test whether SiLU's non-zero negative activations add value vs GELU's smooth approximation.

**Closed axes (do not re-test):**
- Dyadic Fourier: L=4 merged, L=6 merged, L=8 closed (val_re_rand × surf-ch-weight interaction) → Gaussian Fourier in-flight
- Per-channel decoder output heads (both half-capacity #1811 and full-capacity #2020)
- Per-channel surf-weight: 4× is local optimum; 7× and 1× both worse
- Coord jitter: std=0.005 direction-inverted, std=0.002 amplified regression; axis fully closed
- SmoothL1 / loss-landscape smoothing: absorbed by LayerScale γ_l gating; all β values plateau
- Adaptive grad-clip: over-clips on LayerScale-attenuated stack (static max_norm=25 is optimum)
- EMA weights: fights Fourier high-freq feature sharpening
- Output-side calibration: log1p (#1610, #1636), γ-bias (#1675) — all regressed
- Gumbel/Ada-Temp slice collapse (3 tests, mechanism learned but outcome rejected)
- Fixed-threshold grad-clip: bracket {1.0, 10, 25, 50} complete, 25 is optimum
- Stoch-depth single-knob: V-shape; 0.05 (+13.7%) and 0.15 (+7.69%) both worse; 0.10 is optimum
- Attn-mlp-dropout: compute tax + stoch-depth redundancy
- AdamW betas (0.9, 0.95): non-uniform regression, regime mismatch
- n_head=8: +7.81% regression, head fragmentation (dim_head=16) destroys OOD cross-geom features, +42% wall-time incompatible with 30-min cap
- Gaussian Fourier σ=10: +36.6% regression, σ scale mismatch for std-normalized coords (direction NOT closed — σ-calibration in-flight at #2135)
- Weight decay 3×: wd=3e-4 regressed +2.24%/+2.67% (vs baseline 73.958); wd axis confirmed at 1e-4 optimum; fit/generalization gap already tight

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

Active threads (post #1811 close → #1852 coord jitter pivot):

1. **Per-channel surf-loss weighting** (alphonse #1711, H18,
   `[0.5, 0.5, 2.0]`) — loss-layer rebalancing toward pressure channel.
2. **Adaptive grad-clip** (askeladd #1753, 1.5× running median over K=100
   step deque) — pivot from closed fixed-threshold bracket.
3. **LR warm-up REBASING** (nezuko #1754, H19) — WON on old baseline
   (-0.64% val, -1.71% test); pending re-confirmation on new 82.311 baseline.
   Highest-EV merge-candidate after rebase.
4. **Fourier coord encoding `n_freqs=8`** (edward #1830, H26) —
   pre-registered bracket-up from just-merged #1772 L=6. Plateau-probe;
   Tancik's curve predicts plateau at L=8-10. Three clean outcomes:
   win (continue to L=10), plateau (pivot to Gaussian Fourier),
   regress (locate plateau just below L=8, pivot to Gaussian).
5. **LayerScale CaiT-style init=0.1 MERGED** (thorfinn #1799, H23) — 7th
   compound win. Final result on rebased L=6 stack: val=78.260 (-4.92%),
   test=69.903 (-4.67%) vs #1772. Mechanism preserved across rebase:
   per-channel γ_l std 38.8% block-0 attn, depth-decreasing trend on mlp
   branch. Follow-ups assigned: init=0.05 bracket (this iteration).
6. **Coord jitter augmentation std=0.005** (tanjiro #1852, H27) — fresh
   data-augmentation pivot after closed decoder-side direction. Add
   Gaussian noise to normalized spatial coords (x, z) before Fourier
   encoding, masked to non-padded nodes, train-only. Zero param cost.
   Mechanism: forces invariance to mesh-level coord perturbations,
   regularizes high-freq Fourier feature responses (L=6 features at
   `sin(32π·x)` are highly sensitive to small δx).
7. **SmoothL1 (Huber) loss with β=0.01 REBASING** (frieren #1828, H25) —
   ran on **stale stack** (Fourier L=4) at val=83.938 (-0.97%), test=73.300
   (-1.82%) vs 84.762/74.659. But +1.98% regression vs current 82.311
   baseline (and essentially flat on test, 73.300 vs 73.330). Mechanism
   solid: all 4 test splits improve, late-cooldown grad-norm trace
   (ep14=13.9, ep15=16.4 vs ep10-13 range 31-49) directly demonstrates
   smooth-near-zero gradient effect during cosine cooldown. Sent back to
   re-run on current 82.311 stack with Fourier L=6 — open question:
   does Fourier L=6 already address some of what SmoothL1 was fixing?

Note: all in-flight experiments (#1549, #1711, #1753, #1754, #1828, #1830,
#1852) were assigned/rebased against the **prior 82.311 baseline** (post #1772
merge). The new baseline is **78.260** after #1799 merged. Most mechanisms
are orthogonal to LayerScale (per-channel residual gating):
- SmoothL1, surf-ch-weight, FiLM-via-extracted-z, LR warmup, adaptive
  grad-clip, Fourier L=8, coord jitter — all should stack with LayerScale.
- The one mechanistic concern is FiLM (#1549) which also modulates hidden
  states via gamma/beta — could partially overlap with LayerScale's per-channel
  gating.

When in-flight experiments post terminal results, compare against the
**new 78.260 baseline**, not 82.311. Wins likely smaller in magnitude but
still merge-eligible if positive.

One extraordinary signal still pending rebase confirmation:

- **fern #1549 FiLM = 81.291** posted on old base (missing stoch-depth, cosine, grad-clip, Fourier). **-19.5% vs #1611 baseline.** If rebased and the gap holds against the new 78.260 baseline (now with LayerScale included), this is the highest-EV lever in flight. Now competing against a much stronger baseline so the absolute delta will be smaller. Mechanism overlap with LayerScale is the new risk.

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

## Round 2 wave 6 — resolution (#1772 MERGED, #1773/#1811 closed, #1799 in flight)

| Student | PR | Slug | Verdict | Δ vs baseline-at-submission |
|---------|----|----|---------|---------------|
| edward | #1772 | `fourier-coords-L6` | **MERGED** (new baseline 82.311) | -2.89% val, -1.78% test |
| thorfinn | #1773 | `adamw-betas-0.95` | **CLOSED** (non-uniform regression, regime mismatch) | +1.97% val, +1.61% test |
| thorfinn | #1799 | `layerscale-init-0.1` | **MERGED** (new baseline 78.260; rebased on L=6, all 8 splits improve, per-channel std preserved 38.8% block-0 attn) | -4.92% val, -4.67% test |
| tanjiro | #1811 | `output-head-per-channel-mlp` | **CLOSED** (confound: baseline had shared MLP, experiment effectively split capacity to half width; val_single_in_dist regressed +5.34%, opposite predicted direction) | +1.99% val, +0.89% test |

## Round 2 wave 7 — kick-off (#1608/#1811 closed → fresh #1828/#1852; #1830 follow-up to merged #1772)

| Student | PR | Slug | Verdict | Δ vs baseline-at-submission |
|---------|----|----|---------|---------------|
| frieren | #1608 | `ema-weights-0.999` | **CLOSED** (rebased: weight-space smoothing fights Fourier sharpening, inverse-correlates per-split) | +2.93% val, +4.12% test |
| tanjiro | #1811 | `output-head-per-channel-mlp` | **CLOSED** (confound: split capacity at half width; val_single_in_dist regressed +5.34% inverse to prediction) | +1.99% val |
| frieren | #1828 | `smooth-l1-loss-beta-001` | **REBASING** (won on stale stack: val=83.938, test=73.300; -0.97%/-1.82% pre-#1772; +1.98% vs current 82.311; mechanism solid via late-cooldown grad-norm trace; all 4 test splits improve) | tbd vs 78.260 |
| edward | #1830 | `fourier-coords-L8` | WIP — plateau-probe follow-up to merged #1772 L=6 | tbd vs 78.260 |
| tanjiro | #1852 | `coord-jitter-aug-0.005` | WIP — fresh data-aug axis after closed decoder-side direction | tbd vs 78.260 |
| thorfinn | #1896 | `layerscale-init-0.05` | WIP — init bracket from merged #1799 LayerScale init=0.1 (tests per-channel granularity hypothesis) | tbd vs 78.260 |

## Wave-1 / wave-2 carryover (still WIP)

| Student | PR | Slug | Status |
|---------|----|----|--------|
| fern | #1549 | `film-global-cond` | **REBASING** — extraordinary -19.5% signal pending re-run on current 82.311 stack (rebased at 03:04:51 UTC after 2 nudges; training not yet re-run; fun_dim=22 metrics still in tree from May 12 pre-everything run) |

(edward #1548 Fourier coords now MERGED — see new best baseline above; frieren #1608 EMA now CLOSED — see wave 7 above.)

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

## Recent closures and merges (2026-05-12 19:48 → 2026-05-13 03:10 UTC)

- **#1811 output-head-per-channel-mlp (tanjiro)** — **CLOSED** at +1.99% val, +0.89% test. Student found a critical confound in the PR spec: the baseline `mlp2` was already `Sequential(Linear(128→128), GELU, Linear(128→3))` (a shared 128-hidden MLP), not a single Linear projection. The 3 × Sequential(Linear(128→64), GELU, Linear(64→1)) replacement effectively *halves* per-channel hidden capacity (128 → 64) while only adding +8K params. Per-split direction inverted prediction: `val_single_in_dist` (predicted to gain most) regressed +5.34% (largest hit). Cross-channel features the shared decoder learns (pressure-velocity correlations from physics) appear more valuable than per-channel specialization at reduced width. **Axis-wide finding: decoder-side per-channel direction is closed regardless of capacity setting; merged ~666K stack is well-balanced at the decoder.** Picked coord jitter augmentation as the fresh data-augmentation pivot.
- **#1772 fourier-coords-L6 (edward)** — **MERGED** as new baseline **82.311** val / **73.330** test. -2.89% val, -1.78% test vs #1548 baseline. 6th compound win on this branch. All 8 splits improve (-0.91% to -4.10% on val, -1.17% to -2.91% on test). Surprise -4.10% on val_re_rand (pre-registered as ~flat) — mechanism finding: at L=4 the network was over-spending preprocess MLP capacity on low-freq geometry; L=6 frees capacity for Reynolds-dependent features. `val_geom_camber_cruise` only -0.91% (was -7.94% at L=4) — leading-edge plateau indicator. No overfit signature (best ep=15, wall time unchanged). Pre-registered follow-up: #1830 L=8 plateau-probe.
- **#1608 ema-weights-0.999 (frieren)** — **CLOSED** at +2.93% val, +4.12% test (rebased onto current 84.762 baseline). Pre-rebase had won -2.64% val on old 98.353 baseline, but the rebase flipped the sign: EMA's low-pass smoothing on weights fights Fourier's high-frequency feature responses. Per-split: `val_single_in_dist` (Fourier's biggest gain at -11.35%) regressed +6.17% under EMA; `test_single_in_dist` regressed +7.80%. Clean inverse correlation between Fourier-gain magnitude and EMA-regression magnitude. Two compounding mechanisms: (a) decay=0.999 effective window ≈ 2.7 epochs is too long for T_max=15 cosine cooldown, so the EMA copy structurally trails the live model into the final cooldown; (b) EMA's spectral smoothing undoes Fourier's spectral sharpening. **Axis-wide finding: weight-space smoothing is closed on this compound; future variance-reduction PRs must target loss-landscape or trajectory features instead. Picked SmoothL1 (β=0.01) as the loss-landscape pivot.**
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

## All 8 students assigned — Wave 11

| Student | PR | Status |
|---------|----|--------|
| alphonse | #2059 | WIP — n_head=4→8 attention diversity at zero param cost |
| askeladd | #2102 | NEW Wave 11 — RMSNorm replacing LayerNorm in all TransolverBlocks |
| edward | #2078 | WIP — Gaussian Fourier σ=10 pivot from dyadic L=8 |
| fern | #2099 | NEW Wave 11 — weight decay 3× bracket (wd=1e-4→3e-4) |
| frieren | #2098 | NEW Wave 11 — Lion optimizer (sign-momentum, lr=1e-4, wd=3e-4) |
| nezuko | #1754 | WIP — LR warmup H19 rebase onto post-#2018 stack |
| tanjiro | #2105 | NEW Wave 11 — SwiGLU activation (gated nonlinearity in MLP blocks) |
| thorfinn | #2075 | WIP — LayerScale init=0.0125 bracket-down from merged 0.025 |

Zero idle students. Zero idle GPUs.

## Wave 8 candidate pool (held — depending on in-flight outcomes)

- **H18 bracket** (depending on #1962 outcome):
  - If [0.33, 0.33, 2.33] (p:vel 7×) wins → bracket to [0.25, 0.25, 2.5] (p:vel 10×)
  - If [0.33, 0.33, 2.33] plateaus → 4× was the optimum; close bracket
  - If [0.33, 0.33, 2.33] regresses → mid-bracket [0.4, 0.4, 2.2] (p:vel 5.5×) for fine-grain
- **Fourier follow-ups** (depending on #1830 outcome):
  - If L=8 wins → bracket to L=10 (aliasing risk on dense meshes)
  - If L=8 plateaus → Gaussian Fourier features (Tancik §3, σ ∈ {3, 5, 10})
  - Either way → consider Fourier on flow-condition dims (Re, AoA — what `val_re_rand` cares about)
- **LayerScale bracket** (depending on #1896 outcome):
  - If init=0.05 wins → init=0.025 (Fixup-style)
  - If init=0.05 plateaus → init bracket up to 0.15
  - Either way → consider per-block init schedule (Fixup, Saint-style)
- **Coord jitter bracket** (depending on #1852 outcome):
  - If std=0.005 wins → bracket {0.002, 0.01} to find optimum noise scale
  - If std=0.005 plateaus → try jitter on flow-condition dims, or input-mask augmentation
- **SmoothL1 directional** (depending on #1828 outcome):
  - If β=0.005 wins → close loss-landscape with smooth-near-zero confirmed
  - If β=0.005 plateaus → close bracket; pivot to Charbonnier/Tukey biweight
- **Architecture pivots** (held if simpler levers exhaust):
  - **Separate p/Ux/Uy decoder heads with full capacity** (stronger H17, not the half-width confound)
  - **Optimizer family**: Lion (Chen 2023), Sophia (Liu 2023)
  - **Domain-aware sampler reweighting** to match val split aggregation
  - **Learnable Re/AoA Fourier features** on flow-condition dims (since val_re_rand benefits most from spatial Fourier per #1772)
