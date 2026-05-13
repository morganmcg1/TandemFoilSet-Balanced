# SENPAI Research State — charlie-pai2g-48h-r5

- **As of:** 2026-05-13 01:15 (round-11: closed #1727 fern weight_decay=5e-4 (+4.06%, under-fit on 36-epoch budget); closed #1701 alphonse batch=8 (+16.1%, step-count starvation −54% grad updates); sent back #1653 askeladd grad-clip for β=0.5 rebase (result on stale β=1.0 base, lever is real −14.92% on old baseline); assigned #1774 alphonse lr=7.5e-4 (faster steps per same epoch count); assigned #1775 fern weight_decay=5e-5 (DOWN sweep to bracket WD optimum). Baseline still 64.07.)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r5` (advisor) — Charlie no-W&B logging ablation, round 5
- **Most recent human-team direction:** None yet on this branch; instructions
  scoped to the launch (treat experiments as isolated, no W&B logging,
  `SENPAI_TIMEOUT_MINUTES=30` cap per training execution).

## Round-11 research focus

4 merged winners → baseline 64.07 (from 110.76 at round-1 start, -42%). Primary focus:
1. **Stacking sampler 2x on β=0.5** — nezuko #1619 (highest-confidence win).
2. **Stacking warmup-500 on β=0.5** — frieren #1652 (OOD lever confirmed on old base).
3. **Loss shape sweep β=0.25/L1** — thorfinn #1700 (monotone β signal: 2.0>1.0>0.5).
4. **Grad-clip β=0.5 rebase** — askeladd #1653 (lever is real: -14.92% on old β=1.0 base; must rebase to test stacking).
5. **FFN capacity** — edward #1741 (mlp_ratio=3: targeted capacity vs uniform widening).
6. **LR bump** — alphonse #1774 (lr=7.5e-4: step-size increase orthogonal to schedule).
7. **WD bracket DOWN** — fern #1775 (weight_decay=5e-5: completes the 3-point bracket around 1e-4 default).
8. **EMA eval** — tanjiro #1660 (pod rate-limited but alive, will resume when quota clears).

**Architecture axis status:** Depth (n_layers) CLOSED by #1413+#1588. Uniform width (n_hidden) CLOSED by #1398+#1587+#1688. Only mlp_ratio (FFN-only) and n_head untested.

## Fleet status

### Merged winners
| PR | Student | Hypothesis | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|---|---|---|---|---|---|
| #1633 ✓ | charliepai2g48h5-thorfinn | Huber β=0.5 (sharper loss) | **64.07** | **55.50** | Epoch 37 of 37; still descending; -8.2% vs #1568; wins all 4 splits |
| #1568 ✓ | charliepai2g48h5-thorfinn | torch.compile(dynamic=True) + bf16 AMP | 69.83 | 61.87 | Epoch 36 of 36; still improving; -30.9% vs #1532; 2× throughput |
| #1532 ✓ | charliepai2g48h5-thorfinn | bf16 AMP + scoring-NaN fix | 101.12 | 91.50 | Epoch 17 of 19; -8.7% vs #1444 |
| #1444 ✓ | charliepai2g48h5-thorfinn | MSE → Smooth-L1 (Huber, β=1.0) | 110.76 | NaN (bug) | Prior baseline |

**Current baseline: val_avg/mae_surf_p = 64.0705, test_avg/mae_surf_p = 55.4961 (PR #1633)**

> Current advisor branch has: Smooth-L1 (β=0.5) + bf16 AMP + torch.compile(dynamic=True)
> + scoring-NaN workaround. All new PRs inherit these. Epoch budget at 30-min cap: **~37 epochs
> at ~49.5 s/epoch**. Peak GPU memory: ~24 GB (abundant headroom on 96 GB).
> **Key signal:** β monotone decreasing — β=2.0 (77.81) > β=1.0 (69.83) > β=0.5 (64.07). Sweep continues toward β=0.25 / L1.

### Closed (not winners)
| PR | Student | Hypothesis | val_avg/mae_surf_p | Reason |
|---|---|---|---|---|
| #1727 ✗ | charliepai2g48h5-fern | weight_decay 1e-4 → 5e-4 | 66.6723 | +4.06% regression. Under-fit on 36-epoch budget: trajectory lagged baseline by 3-5 epochs. WD-UP axis closed at 5×. |
| #1701 ✗ | charliepai2g48h5-alphonse | batch_size 4 → 8 + compile | 74.4033 | +16.1% regression. Step-count starvation: −54% grad updates (6,392 vs 13,875). Batch scaling confirmed dead-end at TandemFoil scale. |
| #1688 ✗ | charliepai2g48h5-edward | `n_hidden` 128 → 160 + compile | 73.6658 | Width ruled out at 30-min cap (+5.49% vs 69.83 baseline). Same compute-starvation pattern as depth. |
| #1676 ✗ | charliepai2g48h5-fern | AdamW β2 0.999 → 0.95 | 69.9029 | Near-baseline wash (+0.10%); lever doesn't transfer to small encoder-only Transolver. β2 axis closed. |
| #1560 ✗ | charliepai2g48h5-alphonse | T_max=36 cosine (re-run on compile) | 69.598 | Marginal (+0.23 MAE vs 69.83 compile baseline); lever characterized — gap closes as epoch budget grows. T_max=36 no longer beats new 64.07 baseline |
| #1588 ✗ | charliepai2g48h5-fern | `n_layers` 5 → 6 + bf16 | 111.058 | +9.83% vs bf16 baseline; depth lever ruled out (surface regressed more than volume) |
| #1590 ✗ | charliepai2g48h5-frieren | `slice_num` 64 → 96 + bf16 | 105.024 | +3.86% vs bf16 baseline; slice lever now well-characterized (64 best) |
| #1587 ✗ | charliepai2g48h5-edward | `n_hidden` 128 → 160 + bf16 (stale, no commits) | — | Pod stalled; reassigned on compile baseline as #1688 |
| #1561 ✗ | charliepai2g48h5-askeladd | Grad clip max_norm=1.0 (stale, no commits) | — | Pod stalled before training; reassigned on compile baseline |
| #1535 ✗ | charliepai2g48h5-tanjiro | EMA weights eval decay=0.999 (stale, no commits) | — | Pod stalled before training; reassigned on compile baseline |
| #1428 ✗ | charliepai2g48h5-nezuko | Per-channel weights [1,1,3] | 135.531 | All 4 splits worse; Ux/Uy coupling degraded |
| #1422 ✗ | charliepai2g48h5-frieren | `slice_num` 64 → 128 (fp32) | 145.971 | Wall-clock binding (11 epochs); reassigned slice_num=96 + bf16 |
| #1413 ✗ | charliepai2g48h5-fern | `n_layers` 5 → 7 (fp32) | 144.904 | Wall-clock binding (10 epochs); reassigned n_layers=6 + bf16 |
| #1398 ✗ | charliepai2g48h5-edward | `n_hidden` 128 → 192 (fp32) | 138.138 | Wall-clock binding (10 epochs); reassigned n_hidden=160 + bf16 |
| #1388 ✗ | charliepai2g48h5-askeladd | 5-epoch warmup + `lr` 5e-4 → 1e-3 | 152.033 | Higher peak lr overshoots |
| #1375 ✗ | charliepai2g48h5-alphonse | `surf_weight` 10 → 30 | 120.394 | Biases away from volume manifold |
| #1439 ✗ | charliepai2g48h5-tanjiro | `batch_size` 4 → 8 | 155.504 | Wall-clock binding at fp32; now irrelevant |

### In-flight (WIP)
| PR | Student | Hypothesis | Theme |
|---|---|---|---|
| #1619 | charliepai2g48h5-nezuko | Sampler 2× — **3rd rebase onto β=0.5** (highest-confidence next win) | Data/sampler |
| #1652 | charliepai2g48h5-frieren | Warmup-500 — **2nd rebase onto β=0.5** (OOD lever confirmed, expects compounding) | Schedule |
| #1700 | charliepai2g48h5-thorfinn | Huber β=0.25 + pure L1 sweep (continue from β=0.5 win) | Loss shape |
| #1653 | charliepai2g48h5-askeladd | Grad clip max_norm=1.0 — **β=0.5 REBASE** (was on stale β=1.0 baseline) | Optimization × diagnostic |
| #1660 | charliepai2g48h5-tanjiro | EMA weights eval (decay=0.999) on compile baseline | Eval ensemble |
| #1700 | charliepai2g48h5-thorfinn | Huber β=0.25 + pure L1 sweep (continue from β=0.5 win) | Loss shape |
| #1619 | charliepai2g48h5-nezuko | Sampler 2× — **3rd rebase onto β=0.5** | Data/sampler |
| #1652 | charliepai2g48h5-frieren | Warmup-500 — **2nd rebase onto β=0.5** | LR schedule |
| #1741 | charliepai2g48h5-edward | `mlp_ratio` 2 → 3: targeted FFN capacity | Architecture |
| #1774 | charliepai2g48h5-alphonse | lr 5e-4 → 7.5e-4: faster steps per epoch | Optimizer/LR |
| #1775 | charliepai2g48h5-fern | weight_decay 1e-4 → 5e-5: WD bracket DOWN sweep | Regularization |

> **Note on #1700:** Sweeping β downward from 0.5 (merged winner). β=0.25 narrows
> the quadratic region to |e|<0.25, further L1-ifying the loss. Arm B tests pure L1.
> Clear monotone signal: β=2.0 (77.81) > β=1.0 (69.83) > β=0.5 (64.07).

> **Note on #1701:** Retesting batch=8 from PR #1439 in compile era. At fp32, batch=8
> hit ~340 s/epoch (5 epochs). At compile+bf16, batch=8 should give ~65-75 s/epoch
> (~24-27 epochs). Quality vs. epoch-count trade-off.

> **Note on #1688:** n_hidden=160+compile. Per-epoch cost ~65-75 s → ~24-27 epochs in
> 30 min. Depth ruled out by #1413+#1588; width is the last untested capacity axis.
> If #1688 wins, follow up with n_hidden=192+compile.

## Open research questions

1. **T_max matched to compile-era epoch budget.** PR #1560 (alphonse) proved
   the mechanism at T_max=18 (90.32, -10.7% vs bf16 baseline). Re-running at
   T_max=36 to match the post-compile 36-epoch budget. Expected further gain:
   ~6-10 MAE off the 69.83 baseline.

2. **`val_single_in_dist` still the hardest split.** #1619 (nezuko) CONFIRMED:
   sampler boost 2× → -10.7% on val_single_in_dist, -2.8% on val_avg vs bf16
   baseline (98.29). Lever is real; sent back to rebase onto compile baseline
   (69.83). Expected: ~67-68 after compounding.

3. **Width capacity (#1587 edward) still pending.** n_hidden=160+bf16 not yet
   compile-era. If it wins vs bf16 baseline, natural follow-up is n_hidden=160+compile.
   (Depth ruled out by #1413 fp32 + #1588 bf16 — surface metric regressed more
   than volume, opposite of slice-attention mechanism prediction.)

4. **Gradient-norm characterization.** PR #1653 (askeladd, grad clip on compile)
   will log per-epoch grad-norm stats. Diagnostic value: if β2=0.95 (#1676 fern)
   also in flight, both optimizer-area experiments inform whether AdamW defaults
   need tuning.

5. **Huber β optimum.** β=1.0 was arbitrary. PR #1633 tests β=0.5 (sharper)
   and β=2.0 (smoother). Result will orient further β sweeps or confirm β=1.0
   is near-optimal.

## Constraints (post-compile)

- **Epoch budget:** ~36 epochs in 30 min at baseline config (n_hidden=128,
  n_layers=5, slice_num=64) with compile + bf16. 2× faster than bf16-only.
- **Memory:** ~24 GB peak at baseline with compile + bf16. Significant headroom
  on 96 GB GPUs — capacity scale-ups now viable.
- **Wall-clock binding:** Every arm is still wall-clock-bound. Best epoch = 36
  (terminal) in PR #1568 — model still descending. More epochs = more gain.
- **Schedule-terminal effect confirmed:** Alphonse's data shows cosine reaching
  LR→0 gains ~8 MAE in the last 4 epochs. T_max=36 on compiled model is the
  highest-confidence cheap win queued.

## Themes

1. **Loss reformulation** (cheap, high expected value):
   - Smooth-L1 vs MSE: **WON (PR #1444, merged)** — Huber β=1.0 baseline.
   - Surface weight tuning: refuted (PR #1375).
   - Per-channel reweighting [1,1,3]: refuted (PR #1428).
   - **Huber β sweep (β=0.5 / β=2.0): in flight (PR #1633).**

2. **Throughput** (cheap, very high expected value):
   - bf16 AMP + scoring fix: **WON (PR #1532, merged)** — 19 epochs in 30 min.
   - torch.compile(dynamic=True): **WON (PR #1568, merged)** — 36 epochs in 30 min; -30.9%.

3. **Optimization schedule** (cheap, high expected value):
   - Higher peak lr (1e-3 + warmup): refuted (PR #1388).
   - **T_max matched to compile budget (T_max=36): in flight (PR #1560 rerun, alphonse).**

4. **Capacity / model topology** (moderate cost, moderate expected value):
   - **n_hidden=160 + compile: in flight (PR #1688, edward).**
   - n_hidden=160 + bf16 (stale): closed, reassigned → #1688.
   - n_layers=6 + bf16: **refuted (PR #1588, closed)** — +9.83% vs bf16 baseline.
   - n_layers=7 + fp32: **refuted (PR #1413, closed)** — wall-clock bound.
   - slice_num=96 + bf16: **refuted (PR #1590, closed)** — monotone-worse.
   - **Depth lever ruled out.** Width (#1688) first test on compile baseline.

5. **Regularization / stabilization**:
   - EMA weights for eval (decay=0.999) on compile: in flight (PR #1660, tanjiro).
   - Gradient clipping (max_norm=1.0) on compile + logging: in flight (PR #1653, askeladd).

6. **Data / sampler** (targeting val_single_in_dist):
   - RaceCar single 2× confirmed (+2.8% val_avg vs bf16 baseline): **re-running on compile baseline (PR #1619, nezuko rebase).**

7. **Optimizer**:
   - AdamW β2=0.95: **REFUTED (PR #1676, closed)**. β2 axis closed — doesn't transfer to this small encoder-only Transolver on 1499 samples.

8. **Regularization** (new):
   - **weight_decay 1e-4 → 5e-4: in flight (PR #1727, fern)**. Predicts -1% to -3% on OOD splits.

## What has been ruled out

- **Higher peak lr (1e-3 + warmup):** refuted by PR #1388.
- **Higher surf_weight (10 → 30):** refuted by PR #1375.
- **Higher batch (4 → 8):** fully ruled out at both fp32 (#1439) AND compile+bf16 (#1701, −54% step count, +16.1% loss). Batch scaling is dead at TandemFoil scale regardless of compute regime.
- **Larger capacity at fp32:** wall-clock-bound (PRs #1398, #1413, #1422). n_layers=6+bf16 also ruled out (#1588, +9.83%).
- **n_layers scaling (depth):** fully ruled out. Both n_layers=7+fp32 (#1413) and n_layers=6+bf16 (#1588) lost.
- **n_hidden width scaling:** fully ruled out. n_hidden=192+fp32 (#1398), n_hidden=160+bf16-stale (#1587), n_hidden=160+compile (#1688, +5.49%). All fail under 30-min cap via compute starvation.
- **weight_decay UP (1e-4→5e-4):** refuted by PR #1727 (+4.06%). Stronger L2 under-fits on this 36-epoch budget. WD-up axis closed.
- **Per-channel loss weights [1,1,3]:** refuted by PR #1428. All 4 splits worsened.
- **AdamW β2=0.95:** refuted by PR #1676. Mechanism mismatch: transformer-recipe β2 suited for large-scale LM, not 1499-sample Transolver. β2 axis closed.

## Potential next research directions (post round-5)

- **Compound schedule + compile:** T_max=36 is the highest-confidence next win.
- **n_hidden=192 + compile:** if #1587 (n_hidden=160+bf16) wins, the natural next step.
- **Step-based warmup at lower peak lr** (queued from PR #1388 analysis).
- **AoA reflection augmentation:** flip AoA/z/Uy/saf for data augmentation. Complex but high value.
- **Spectral / Fourier feature lifting on (x, z)** — high-frequency surface features.
- **Re-aware output head** — log-magnitude normalization for high-Re pressure extremes.
- **Auxiliary divergence loss** — incompressible flow physics regularizer.
- **Optimizer swaps** — Lion or AdaFactor vs AdamW.
- **Sampler sweep** — if 2× wins, try 1.5× and 3× to find the optimum.
- **Domain-aware loss reweighting** (separate from sampler) — per-sample loss weight based on domain membership.

This document is a living summary — update after each PR cycle.
