# SENPAI Research State — charlie-pai2g-48h-r5

- **As of:** 2026-05-12 21:00 (round-3 closures: capacity arms; round-3 assignments out for edward, fern, frieren)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r5` (advisor) — Charlie no-W&B logging ablation, round 5
- **Most recent human-team direction:** None yet on this branch; instructions
  scoped to the launch (treat experiments as isolated, no W&B logging,
  `SENPAI_TIMEOUT_MINUTES=30` cap per training execution).

## Round-5 research focus

We start clean: 8 idle students, 1 GPU each, and no prior round-5 winners
recorded on this branch. The aim of round 5 is to **stake out the highest-value
single-axis levers on the Transolver baseline** for `val_avg/mae_surf_p` so the
fleet can lock in a real baseline number and identify which levers compound.

## Round-1 fleet status

### Merged winners
| PR | Student | Hypothesis | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|---|---|---|---|---|---|
| #1532 ✓ | charliepai2g48h5-thorfinn | bf16 AMP + scoring-NaN fix | **101.12** | **91.50** | Epoch 17 of 19; still improving; -8.7% vs #1444 |
| #1444 ✓ | charliepai2g48h5-thorfinn | MSE → Smooth-L1 (Huber, β=1.0) | 110.76 | NaN (bug) | Prior baseline |

**Round-5 baseline floor: val_avg/mae_surf_p = 101.1212, test_avg/mae_surf_p = 91.5013**

> Scoring-NaN workaround merged via PR #1532. New PRs inheriting the advisor
> branch will have the fix. PRs created before #1532 may still see NaN on
> `test_geom_camber_cruise` — those report 3-split partial average.

### Closed (not winners)
| PR | Student | Hypothesis | val_avg/mae_surf_p | Reason |
|---|---|---|---|---|
| #1439 ✗ | charliepai2g48h5-tanjiro | `batch_size` 4 → 8 | 155.504 | Wall-clock binding, not gradient noise |
| #1375 ✗ | charliepai2g48h5-alphonse | `surf_weight` 10 → 30 | 120.394 | Biases away from volume manifold |
| #1388 ✗ | charliepai2g48h5-askeladd | 5-epoch warmup + `lr` 5e-4 → 1e-3 | 152.033 | Higher peak lr overshoots good basins |
| #1398 ✗ | charliepai2g48h5-edward | `n_hidden` 128 → 192 | 138.138 | Wall-clock binding (10/50 epochs); reassigned milder n_hidden=160 + bf16 |
| #1413 ✗ | charliepai2g48h5-fern | `n_layers` 5 → 7 | 144.904 | Wall-clock binding (10/50 epochs); reassigned milder n_layers=6 + bf16 |
| #1422 ✗ | charliepai2g48h5-frieren | `slice_num` 64 → 128 | 145.971 | Wall-clock binding (11/50 epochs); reassigned milder slice_num=96 + bf16 |

### In-flight (WIP)
| PR | Student | Hypothesis | Theme |
|---|---|---|---|
| #1428 | charliepai2g48h5-nezuko | Per-channel weights [1,1,3] favoring pressure | Loss channel |
| #1535 | charliepai2g48h5-tanjiro | EMA model weights for eval (decay 0.999) | Regularization |
| #1560 | charliepai2g48h5-alphonse | T_max=14 cosine matched to actual epochs | Schedule |
| #1561 | charliepai2g48h5-askeladd | Gradient norm clipping (max_norm=1.0) | Optimization |
| #1568 | charliepai2g48h5-thorfinn | `torch.compile` + bf16 AMP | Throughput |
| #1587 | charliepai2g48h5-edward | `n_hidden` 128 → 160 + bf16 (inherited) | Width × throughput |
| #1588 | charliepai2g48h5-fern | `n_layers` 5 → 6 + bf16 (inherited) | Depth × throughput |
| #1590 | charliepai2g48h5-frieren | `slice_num` 64 → 96 + bf16 (inherited) | Slice × throughput |

## Open research questions

Insights surfaced across closed PRs that are now informing in-flight or queued
arms:

1. **`T_max` matched to actual epoch budget.** Surfaced by PRs #1439, #1375,
   and #1388 (three students independently raised it). The
   `CosineAnnealingLR(T_max=50)` schedule never reaches its late-phase low LR
   because only ~14 epochs fit under the 30-min cap. **Now in flight as
   PR #1560 (alphonse): `--epochs 14` so `T_max=14`.**

2. **`val_single_in_dist` dominates the cross-split mean.** Its surf_p MAE is
   roughly 2× the others' (135 vs 77-101 at baseline; PR #1375 confirmed
   raising surf_weight makes it *worse*, going to 148). A loss/sampler tweak
   that improves only that split would dominate `val_avg/mae_surf_p`. Worth
   probing later with single-domain-aware loss reweighting or domain-aware
   sampler tweaks (`sample_weights` override in `train.py` since `data/loader.py`
   is read-only).

3. **Step-based warmup at lower peak lr.** Surfaced by PR #1388. The 5-epoch
   warmup in PR #1388 used ~36% of the 14-epoch budget. A short step-based
   warmup over the first ~500 steps at a modest peak (closer to baseline
   lr=5e-4 than 1e-3) is a smaller perturbation worth a future try.

4. **Gradient-norm characterization.** PR #1561 (askeladd) will log per-step
   gradient norms — informative diagnostic regardless of whether the clip
   intervention helps. Tells us whether the baseline trajectory has
   high-variance gradients (warranting clipping, layer-wise lr decay, or
   AdamW-beta tweaks) or well-behaved gradients (where regularization or
   data augmentation are the better swings).

5. **Capacity scale-ups need throughput compensation.** Three students
   (edward #1398, fern #1413, frieren #1422) independently demonstrated that
   capacity scale-ups at fp32 only fit 10-11 epochs in the 30-min cap (vs
   baseline's 19), and val curves are still descending at timeout. The
   round-3 reassignments (PRs #1587, #1588, #1590) test the milder bumps
   paired with bf16 inheritance. **If these still don't beat baseline, then
   any model-architecture lever requires `torch.compile` (PR #1568) or
   matched LR schedule (PR #1560) to be testable in this budget.**

Constraints shape what we can sensibly try in 30-minute training executions:

- A single execution is wall-clock-bound; counterfactuals must show signal
  within the first ~30 wall-clock minutes (≈ 15-30 epochs for the default
  Transolver on this dataset, depending on mesh sizes).
- Reference architecture (n_hidden=128, n_layers=5, slice_num=64, mlp_ratio=2)
  is well-shaped for fast iteration — moderate scale-ups still fit on 96 GB.
- Test-time metric `test_avg/mae_surf_p` always evaluated at the end from the
  best val checkpoint — every PR must report both val + test averages, and the
  per-split breakdown, with finite values across all four val and four test
  splits.

## Themes that drove round-1 and round-2 assignments

1. **Loss reformulation** (cheap, high expected value):
   - Smooth-L1 vs MSE: **WON (PR #1444, merged)** — this is the baseline now.
   - Surface weight tuning: refuted (PR #1375).
   - Per-channel reweighting: in flight (PR #1428).

2. **Optimization schedule** (cheap, high expected value):
   - Higher peak lr (1e-3 + warmup): refuted (PR #1388).
   - T_max matched to actual epochs: in flight (PR #1560 round-2 reassign).

3. **Capacity / model topology** (moderate cost, moderate expected value):
   - Wider Transolver (`n_hidden` 128 → 192) — in flight (PR #1398).
   - Deeper Transolver (`n_layers` 5 → 7) — in flight (PR #1413).
   - Larger slice count (`slice_num` 64 → 128) — in flight (PR #1422).

4. **Effective batch size** (cheap, moderate expected value):
   - Batch 4 → 8: refuted (PR #1439).

5. **Throughput / regularization** (round-2 additions):
   - bf16 AMP + scoring fix: in flight (PR #1532).
   - EMA weights for eval: in flight (PR #1535).
   - Gradient clipping (max_norm=1.0): in flight (PR #1561 round-2 reassign).

## What has been ruled out (rounds 2 + 3 closures)

- **Higher peak lr (1e-3 with warmup) — refuted by PR #1388.** Schedule is well
  tuned at lr=5e-4. Pushing lr higher is unproductive at this wall-clock budget.
- **Higher surf_weight (10 → 30) — refuted by PR #1375.** Biases gradients away
  from the volume manifold; `val_single_in_dist` (the hardest split) regresses.
- **Higher batch (4 → 8) — refuted by PR #1439.** Wall-clock is the binding
  constraint, not gradient noise. Memory was at 84/96 GB at batch=8.
- **Larger capacity at fp32 — wall-clock-bound (PRs #1398, #1413, #1422).**
  Width=192, depth=7, slice_num=128 each fit only 10-11 epochs vs baseline's
  19 — undertrained at timeout. Capacity lever requires bf16 throughput
  compensation (in flight: #1587, #1588, #1590) before final verdict.

Cheap lever space is mostly explored. Remaining cheap-lever candidates:
schedule shape (T_max #1560), gradient stabilization (#1561), per-channel
loss weighting (#1428), EMA (#1535).

## Potential next research directions (round 3+ once current arms settle)

- **Compound winners.** If T_max=14 (#1560), grad-clip (#1561), bf16 (#1532),
  or EMA (#1535) each win independently, stack the orthogonal ones in a
  single follow-up PR.
- **Step-based warmup at lower peak** (queued from PR #1388 analysis).
- **AoA reflection augmentation.** Cheap, physics-grounded data augmentation.
  Requires careful handling of `saf` (signed arc-length) and `dsdf` channels
  in the input feature layout.
- **Spectral / Fourier feature lifting on (x, z)** before the preprocess MLP —
  helps high-frequency surface features that drive pressure.
- **Boundary-aware attention masking or extra surface-aware blocks at the
  end of the encoder** — pressure peaks live at the foil surface; biasing
  attention toward surface slices could help.
- **Re-aware feature normalization or log-magnitude head** — high-Re samples
  drive extremes; explicit Re-conditioning of the output scale may reduce
  loss-channel coupling.
- **Domain-aware sampler tweak** — up-weight `RaceCar single` training samples
  (currently equal-weighted in `sample_weights`) since `val_single_in_dist` is
  the hardest split. Override `sample_weights` in `train.py` (read-only data/).
- **Huber β sweep.** Try β=0.5 (sharper) and β=2.0 (smoother) to find the
  Huber transition that matches the dataset's residual distribution.
- **Auxiliary divergence or gradient penalty losses** — incompressible flow
  divergence-free constraint as a physics regularizer.
- **Optimizer swaps**: Lion or AdaFactor as alternatives to AdamW.

This document is a living summary — update after each PR cycle to reflect
which themes compounded and which directions have been ruled out.
