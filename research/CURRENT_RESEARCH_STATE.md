# SENPAI Research State — charlie-pai2g-48h-r5

- **As of:** 2026-05-12 22:45 (round-6: closed #1590 slice_num=96, stale #1561/#1535; assigned #1652 frieren warmup, #1653 askeladd grad-clip on compile, tanjiro EMA pending PR creation post-rate-limit-reset)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r5` (advisor) — Charlie no-W&B logging ablation, round 5
- **Most recent human-team direction:** None yet on this branch; instructions
  scoped to the launch (treat experiments as isolated, no W&B logging,
  `SENPAI_TIMEOUT_MINUTES=30` cap per training execution).

## Round-5 research focus

We have now merged 3 stacked winners, establishing a strong baseline at
val_avg/mae_surf_p=69.83 (from 110.76 at round-1 start). The focus shifts to:
1. **Compounding cheap orthogonal levers** on top of the compile baseline.
2. **Testing capacity arms** now that compile makes 36 epochs available.
3. **Data/sampler side** (nezuko, PR #1619) targeting val_single_in_dist.

## Fleet status

### Merged winners
| PR | Student | Hypothesis | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|---|---|---|---|---|---|
| #1568 ✓ | charliepai2g48h5-thorfinn | torch.compile(dynamic=True) + bf16 AMP | **69.83** | **61.87** | Epoch 36 of 36; still improving; -30.9% vs #1532; 2× throughput |
| #1532 ✓ | charliepai2g48h5-thorfinn | bf16 AMP + scoring-NaN fix | 101.12 | 91.50 | Epoch 17 of 19; -8.7% vs #1444 |
| #1444 ✓ | charliepai2g48h5-thorfinn | MSE → Smooth-L1 (Huber, β=1.0) | 110.76 | NaN (bug) | Prior baseline |

**Current baseline: val_avg/mae_surf_p = 69.8316, test_avg/mae_surf_p = 61.8652 (PR #1568)**

> Current advisor branch has: Smooth-L1 (β=1.0) + bf16 AMP + torch.compile(dynamic=True)
> + scoring-NaN workaround. All new PRs inherit these. Epoch budget at 30-min cap: **~36 epochs
> at ~49.5 s/epoch**. Peak GPU memory: ~24 GB (abundant headroom on 96 GB).

### Closed (not winners)
| PR | Student | Hypothesis | val_avg/mae_surf_p | Reason |
|---|---|---|---|---|
| #1439 ✗ | charliepai2g48h5-tanjiro | `batch_size` 4 → 8 | 155.504 | Wall-clock binding at fp32; now irrelevant |
| #1375 ✗ | charliepai2g48h5-alphonse | `surf_weight` 10 → 30 | 120.394 | Biases away from volume manifold |
| #1388 ✗ | charliepai2g48h5-askeladd | 5-epoch warmup + `lr` 5e-4 → 1e-3 | 152.033 | Higher peak lr overshoots |
| #1398 ✗ | charliepai2g48h5-edward | `n_hidden` 128 → 192 (fp32) | 138.138 | Wall-clock binding (10 epochs); reassigned n_hidden=160 + bf16 |
| #1413 ✗ | charliepai2g48h5-fern | `n_layers` 5 → 7 (fp32) | 144.904 | Wall-clock binding (10 epochs); reassigned n_layers=6 + bf16 |
| #1422 ✗ | charliepai2g48h5-frieren | `slice_num` 64 → 128 (fp32) | 145.971 | Wall-clock binding (11 epochs); reassigned slice_num=96 + bf16 |
| #1428 ✗ | charliepai2g48h5-nezuko | Per-channel weights [1,1,3] | 135.531 | All 4 splits worse; Ux/Uy coupling degraded |
| #1590 ✗ | charliepai2g48h5-frieren | `slice_num` 64 → 96 + bf16 | 105.024 | +3.86% vs bf16 baseline; slice lever now well-characterized (64 best) |
| #1561 ✗ | charliepai2g48h5-askeladd | Grad clip max_norm=1.0 (stale, no commits) | — | Pod stalled before training; reassigned on compile baseline |
| #1535 ✗ | charliepai2g48h5-tanjiro | EMA weights eval decay=0.999 (stale, no commits) | — | Pod stalled before training; reassigned on compile baseline |

### In-flight (WIP)
| PR | Student | Hypothesis | Theme |
|---|---|---|---|
| #1619 | charliepai2g48h5-nezuko | RaceCar single sampler boost 2× (50% share) | Data/sampler |
| #1560 | charliepai2g48h5-alphonse | T_max=36 cosine matched to compile budget (re-run) | Schedule × compile |
| #1633 | charliepai2g48h5-thorfinn | Huber β sweep (β=0.5 and β=2.0) | Loss shape |
| #1652 | charliepai2g48h5-frieren | Step-based linear warmup (500 steps) + cosine | Schedule warmup |
| #1653 | charliepai2g48h5-askeladd | Grad clip max_norm=1.0 + per-epoch grad-norm logging | Optimization × diagnostic |
| (tanjiro pending) | charliepai2g48h5-tanjiro | EMA weights eval (decay=0.999) on compile baseline | Regularization |
| #1587 | charliepai2g48h5-edward | `n_hidden` 128 → 160 + bf16 (pre-compile) | Width |
| #1588 | charliepai2g48h5-fern | `n_layers` 5 → 6 + bf16 (pre-compile) | Depth |

> **Note on #1560:** Alphonse's original T_max=18 result (90.32) proved the
> schedule-completion mechanism clearly. PR sent back because (a) no code change was
> made and (b) post-compile budget is now 36 epochs, making T_max=36 the correct value.
> Student re-running with --epochs 36 on the updated advisor branch.

> **Note on #1587, #1588, #1590:** These pre-date PR #1568 (compile). They'll test
> capacity + bf16 (without compile). Results will tell us whether the capacity lever
> is real; if it is, we follow up with capacity + compile.

## Open research questions

1. **T_max matched to compile-era epoch budget.** PR #1560 (alphonse) proved
   the mechanism at T_max=18 (90.32, -10.7% vs bf16 baseline). Re-running at
   T_max=36 to match the post-compile 36-epoch budget. Expected further gain:
   ~6-10 MAE off the 69.83 baseline.

2. **`val_single_in_dist` still the hardest split.** At compile baseline:
   val_single_in_dist=77.10 vs 50.64-83.49 for other splits. Still ~13% above
   val_avg. PR #1619 (nezuko) tests domain-aware sampler reweighting (RaceCar
   single 2×) to directly target coverage of this split.

3. **Capacity arms haven't had compile throughput yet.** PRs #1587, #1588,
   #1590 test n_hidden=160 / n_layers=6 / slice_num=96 paired with bf16 but
   NOT compile. If any of these show improvement, the natural follow-up is the
   same intervention + compile (which now allows 20-28 epochs for wider/deeper
   models).

4. **Gradient-norm characterization.** PR #1561 (askeladd, grad clip) will
   log per-step norms. Diagnostic value: if gradients are well-behaved,
   optimization is the wrong lever; if spiky, clip or AdamW β tweaks help.

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
   - n_hidden=160 + bf16: in flight (PR #1587, edward).
   - n_layers=6 + bf16: in flight (PR #1588, fern).
   - slice_num=96 + bf16: in flight (PR #1590, frieren).
   - *(If any of these win → follow-up with compile compounding.)*

5. **Regularization / stabilization**:
   - EMA weights for eval (decay=0.999): in flight (PR #1535, tanjiro).
   - Gradient clipping (max_norm=1.0): in flight (PR #1561, askeladd).

6. **Data / sampler** (targeting val_single_in_dist):
   - **RaceCar single 2× sampler boost: in flight (PR #1619, nezuko).**

## What has been ruled out

- **Higher peak lr (1e-3 + warmup):** refuted by PR #1388.
- **Higher surf_weight (10 → 30):** refuted by PR #1375.
- **Higher batch (4 → 8) at fp32:** refuted by PR #1439. (Retestable at compile+bf16 if needed.)
- **Larger capacity at fp32:** wall-clock-bound (PRs #1398, #1413, #1422). Verdict pending with bf16+compile.
- **Per-channel loss weights [1,1,3]:** refuted by PR #1428. All 4 splits worsened.

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
